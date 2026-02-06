import random
import torch
import torch.nn.functional as F
from scipy.fft import idctn
from torch import nn

import numpy as np
import copy

from misc import utils
from misc.utils import is_using_distributed
from text_utils.tokenizer import tokenize
# from .loss import compute_simclr_loss
from .visual_transformer import visual_transformer
from .text_transformer import text_transformers
from .eda import EDA

from .base_transformer import Transformer, LayerNorm, QuickGELU

from .shared_modules import AllGather
from collections import OrderedDict

from .ronghe import TextFeatureGatedFusion,CrossModalAttentionFusion,ClassifierHead
from .ronghe import CrossAttentionTextQueryCLIP,CrossAttentionImageQueryCLIP
from .fusion_model import Fusion,Fusion2

class CLIP(nn.Module):
    def __init__(self, config, image_encode, text_encode, num_classes=11003, eps=1e-2):
        super().__init__()
        self.visual = image_encode
        self.encode_text = text_encode
        self.embed_dim = config.model.embed_dim # Embedding dimension for image and text features, used for aligning image and text features

        # self.use_gather = config.model.use_gather
        self.logit_scale = nn.Parameter(torch.ones([]))# Scale similarity score: Multiply the calculated similarity score matrix by logit_scale
        # nn.init.constant_(self.logit_scale, np.log(1 / 0.07))# self.logit_scale: A trainable parameter used to scale the similarity score between image and text features. This initialization is equivalent to setting the initial temperature parameter to 0.07, because exp(log(1/0.07)) = 1/0.07. The temperature parameter is used to control the distribution of similarity scores in contrastive learning. A smaller temperature value makes the similarity scores sharper, thereby enhancing the contrastive effect.
        self.config = config
        self.eda = EDA()# During training, EDA may be used to augment input data, such as random cropping, color jittering, etc., to improve model generalization.
        self.eps = eps # In normalization operations or loss functions, eps can prevent division by zero or numerical instability.

        # Gating module
        # self.gate = TextFeatureGatedFusion(hidden_size=512, dropout_prob=0.3).to(config.device)

        # # Attention mechanism fusion module
        # self.fusion = CrossModalFusion(dim=512)
        # Dual cross
        # self.cross_fusion = CrossModalAttentionFusion(embed_dim=512).to(self.config.device)
        # self.pvd_proj = nn.Linear(1024, 512)

        # # PVD part
        # self.pvd_fc1 = nn.Linear(1024, 256)  # Reduce input features to 256 dimensions
        # self.pvd_fc2 = nn.Linear(256, 128)  # Second projection layer of PVD module
        # self.pvd_fc3 = nn.Linear(128, 64)  # Final output layer
        # # Adjust PVD output to 512 dimensions
        # self.pvd_proj = nn.Linear(64, 512)

        self.classifier1 = ClassifierHead(input_dim=512, num_classes=num_classes, dropout_prob=0.3)
        # self.classifier2 = ClassifierHead(input_dim=512, hidden_dim=1024, num_classes=num_classes, dropout_prob=0.3)

        # # Loss function
        # self.criterion = torch.nn.CrossEntropyLoss()

        # Respective dual cross
        # self.cross_text = CrossAttentionTextQueryCLIP(dim=512)
        # self.cross_image = CrossAttentionImageQueryCLIP(dim=512)

        # Direct connection
        # self.fusion = nn.Linear(1024, 512)
        # self.activation = nn.ReLU()  # Optional, can also be changed to GELU, Tanh, etc.

        # transformer_decoder
        self.fusion = Fusion(config)
        self.fusion2 = Fusion2(config)
        self.liner = nn.Linear(1024, 512)
        self.relu = nn.ReLU()


    def forward(self, input,alpha):
        # ret = dict() # Initialize an empty dictionary ret to store various loss values

        images = input['image'].to(self.config.device) # Move input image data images and images_1 to the specified device (e.g., GPU)
        # images = input['aug_ss_1'].to(self.config.device)
        # caption_1 = input['caption_1'] # Get text data texts and back-translated text texts_bt
        # caption_2 = input['caption_2']
        captions = input['caption']
        labels = input['id']

        # if 'caption_bt' in input:
        #     texts_bt = input['caption_bt']
        #     # print(texts.shape)
            # texts contains all captions
        # if 'caption_bt_1' in input:
        # # back translation
        #     if self.config.experiment.back_trans:# Fixed in each epoch, but changes across different epochs, improving generalization this way
        #         caption_bt_1=input['caption_bt_1']
        #         caption_bt_2=input['caption_bt_2']
        #         for i in range(len(caption_1)):
        #             if random.random() < self.config.experiment.backtrans_p:
        #                 caption_1[i] = caption_bt_1[i]
        #         for i in range(len(caption_2)):
        #             if random.random() < self.config.experiment.backtrans_p:
        #                 caption_2[i] = caption_bt_2[i]
        # random deletion
        # cap_new = [] # Create an empty list cap_new to store text after random deletion operation
        # for text in caption_1:
        #     eda_alpha = self.config.experiment.eda_alpha # eda_alpha usually controls the intensity of random deletion, e.g., representing the probability of deleting words
        #     cap_new.append(self.eda.random_deletion(text, eda_alpha))
        # caption_1 = cap_new
        # cap_new = []  # Create an empty list cap_new to store text after random deletion operation
        # for text in caption_2:
        #     eda_alpha = self.config.experiment.eda_alpha  # eda_alpha usually controls the intensity of random deletion, e.g., representing the probability of deleting words
        #     cap_new.append(self.eda.random_deletion(text, eda_alpha))
        # caption_2 = cap_new


        # caption_1 = self.augment_batch(caption_1, self.eda,method='sr', n=2)
        # caption_2 = self.augment_batch(caption_2, self.eda,method='sr', n=2)

        # MLM
        if self.config.experiment.mlm:# False here
            text_tokens, mlm_labels = tokenize(captions, context_length=self.config.experiment.text_length,
                                               mask_type='MLM')
            text_tokens = text_tokens.to(self.config.device)
            mlm_labels = mlm_labels.to(self.config.device)
        else:
            # text_tokens1 = tokenize(caption_1, context_length=self.config.experiment.text_length).to(self.config.device)
            # text_tokens2 = tokenize(caption_2, context_length=self.config.experiment.text_length).to(self.config.device)
            text_tokens = tokenize(captions, context_length=self.config.experiment.text_length).to(self.config.device)

        visual_features, image_seq_embeddings = self.encode_image(images, return_dense=True) # When return_dense is set to True, the encode_image method returns two results: image_features and image_seq_embedding
        # text_features1, text_seq_embeddings = self.encode_text(text_tokens1, return_dense=True)
        # text_features2, text_seq_embeddings = self.encode_text(text_tokens2, return_dense=True)
        text_features, text_seq_embeddings = self.encode_text(text_tokens, return_dense=True)

        # Gating module
        # text_features = self.gate(text_features1, text_features2)

        # Attention fusion
        # fused_features = self.fusion(visual_features, text_features)  # Use attention fusion

        # Cross-attention module
        # out_text, out_image, fused = self.cross_fusion(text_features, visual_features)

        # # PVD module
        # pvd_features = torch.cat((visual_features, text_features), dim=1)  # [batch_size, 1024]
        # pvd_out = torch.relu(self.pvd_fc1(pvd_features))  # [batch_size, 256]
        # pvd_out = torch.relu(self.pvd_fc2(pvd_out))  # [batch_size, 128]
        # pvd_out = torch.relu(self.pvd_fc3(pvd_out))  # [batch_size, 64]
        # pvd_out = torch.relu(self.pvd_proj(pvd_out))  # [batch_size, 512]
        # # Ensure consistent data types
        # # visual_features = visual_features.float()
        # # # text_features = text_features.float()
        # # # pvd_out = pvd_out.float()
        # # Add features
        # combined_features = visual_features + text_features + pvd_out  # [batch_size, 512]


        # combined_features = visual_features + fused_features # [batch_size, 512]
        # Respective dual cross
        # fusion_text = self.cross_text(text_features, visual_features)
        # fusion_image = self.cross_image(visual_features,fusion_text )
        # fused_features =fusion_text + fusion_image

        # Direct connection
        # fused = torch.cat([visual_features, text_features], dim=1)  # (batch_size, 1024)
        # output = self.activation(self.fusion(fused))
        # fused_features = visual_features + text_features

        # transformer_decoder
        text_fused = self.fusion2(text_features, visual_features, text_tokens)
        image_fused = self.fusion(visual_features, text_features)
        # fused = text_fused + image_fused
        fused = torch.cat((text_fused, image_fused), dim=-1)
        fused = self.liner(fused)
        # fused=self.relu(fused)



        # fused_features = visual_features + text_fused
        # logit_scale = self.logit_scale.exp()
        # loss = self.clip_contrastive_loss(visual_features, text_features,labels,logit_scale)

        x = self.classifier1(fused)
        # x = self.classifier1(text_fused)

        return x

    def cross_former(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q),
            self.ln_pre_i(k),
            self.ln_pre_i(v),
            need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x


    @property
    def dtype(self):
        try:
            return self.visual.conv1.weight.dtype
        except:
            try:
                return self.visual.head.weight.dtype
            except:
                try:
                    return self.visual.stem[0].weight.dtype
                except:
                    return self.encode_text.text_projection.weight.dtype

    def encode_image(self, image, return_dense=False):
        if return_dense: # Passed True
            output = self.visual(image.type(self.dtype), return_dense=return_dense)# return_dense: A boolean parameter, default is False. When set to True, indicates that dense feature representation needs to be returned; when set to False, returns default feature representation.
            return output
        output = self.visual(image.type(self.dtype))
        return output

    def augment_batch(self, text_list, eda, method=None, n=1, p=0.1):
        """
        Batch augment a list of texts

        """
        # print(f"eda type: {type(eda)}")
        # print(f"text_list type: {type(text_list)}")

        return [eda.augment(text, method=method, n=n, p=p) for text in text_list]

    def clip_contrastive_loss(self,image_features, text_features,labels, logit_scale=100.0):
        # Ensure normalization
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Calculate similarity matrix: [B, B]
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T  # Symmetric

        # Generate labels (0~B-1) as ground truth
        # batch_size = image_features.size(0)
        # labels = torch.arange(batch_size, device=image_features.device)

        # Bidirectional cross entropy: image->text, text->image
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)

        return (loss_i2t + loss_t2i) / 2

def clip_vitb(config, num_classes=11003): # This function will initialize the image encoder and text encoder separately, and then combine them into a CLIP model to return
    image_encode = visual_transformer(config)
    text_encode = text_transformers(config)
    model = CLIP(config, image_encode, text_encode, num_classes, config.experiment.ritc_eps)
    return model

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
        self.embed_dim = config.model.embed_dim #图像和文本特征的嵌入维度，用于对齐图像和文本特征

        # self.use_gather = config.model.use_gather
        self.logit_scale = nn.Parameter(torch.ones([]))#缩放相似度得分：将计算得到的相似度得分矩阵乘以 logit_scale
        # nn.init.constant_(self.logit_scale, np.log(1 / 0.07))#self.logit_scale：一个可训练的参数，用于缩放图像和文本特征之间的相似性分数。这种初始化方式相当于将初始的温度参数设置为 0.07，因为 exp(log(1/0.07)) = 1/0.07。温度参数在对比学习中用于控制相似性分数的分布，较小的温度值会使相似性分数更加尖锐，从而增强对比效果
        self.config = config
        self.eda = EDA()#在训练过程中，EDA 可能用于对输入数据进行增强，例如随机裁剪、颜色抖动等，以提高模型的泛化能力
        self.eps = eps #一化操作或损失函数中，eps 可以防止除以零或数值不稳定的情况

        #控制门模块
        # self.gate = TextFeatureGatedFusion(hidden_size=512, dropout_prob=0.3).to(config.device)

        # #注意力机制融合模块
        # self.fusion = CrossModalFusion(dim=512)
        #双交叉
        # self.cross_fusion = CrossModalAttentionFusion(embed_dim=512).to(self.config.device)
        # self.pvd_proj = nn.Linear(1024, 512)

        # # PVD部分
        # self.pvd_fc1 = nn.Linear(1024, 256)  # 输入特征降维到256
        # self.pvd_fc2 = nn.Linear(256, 128)  # PVD模块的第二个投影层
        # self.pvd_fc3 = nn.Linear(128, 64)  # 最终输出层
        # #调整 PVD 输出到 512 维
        # self.pvd_proj = nn.Linear(64, 512)

        self.classifier1 = ClassifierHead(input_dim=512, num_classes=num_classes, dropout_prob=0.3)
        # self.classifier2 = ClassifierHead(input_dim=512, hidden_dim=1024, num_classes=num_classes, dropout_prob=0.3)

        # #损失函数
        # self.criterion = torch.nn.CrossEntropyLoss()

        #各自双交叉
        # self.cross_text = CrossAttentionTextQueryCLIP(dim=512)
        # self.cross_image = CrossAttentionImageQueryCLIP(dim=512)

        #直接连接
        # self.fusion = nn.Linear(1024, 512)
        # self.activation = nn.ReLU()  # 可选，也可以换成 GELU、Tanh 等

        #transfoemer_decoder
        self.fusion = Fusion(config)
        self.fusion2 = Fusion2(config)
        self.liner = nn.Linear(1024, 512)
        self.relu = nn.ReLU()


    def forward(self, input,alpha):
        # ret = dict() #初始化一个空字典 ret 用于存储各种损失值

        images = input['image'].to(self.config.device) #输入的图像数据 images 和 images_1 移动到指定设备（如 GPU）
        # images = input['aug_ss_1'].to(self.config.device)
        # caption_1 = input['caption_1'] #获取文本数据 texts 和反向翻译后的文本 texts_bt
        # caption_2 = input['caption_2']
        captions = input['caption']
        labels = input['id']

        # if 'caption_bt' in input:
        #     texts_bt = input['caption_bt']
        #     # print(texts.shape)
            #texts包含了所有的caption
        # if 'caption_bt_1' in input:
        # # back translation
        #     if self.config.experiment.back_trans:#在每一个epoch中都是固定的，但是在不同epoch是会改变的，通过这种方式提高泛化能力
        #         caption_bt_1=input['caption_bt_1']
        #         caption_bt_2=input['caption_bt_2']
        #         for i in range(len(caption_1)):
        #             if random.random() < self.config.experiment.backtrans_p:
        #                 caption_1[i] = caption_bt_1[i]
        #         for i in range(len(caption_2)):
        #             if random.random() < self.config.experiment.backtrans_p:
        #                 caption_2[i] = caption_bt_2[i]
        # random deletion
        # cap_new = [] #创建一个空列表 cap_new，用于存储经过随机删除操作后的文本
        # for text in caption_1:
        #     eda_alpha = self.config.experiment.eda_alpha #eda_alpha 通常控制着随机删除操作的强度，例如表示删除单词的概率
        #     cap_new.append(self.eda.random_deletion(text, eda_alpha))
        # caption_1 = cap_new
        # cap_new = []  # 创建一个空列表 cap_new，用于存储经过随机删除操作后的文本
        # for text in caption_2:
        #     eda_alpha = self.config.experiment.eda_alpha  # eda_alpha 通常控制着随机删除操作的强度，例如表示删除单词的概率
        #     cap_new.append(self.eda.random_deletion(text, eda_alpha))
        # caption_2 = cap_new


        # caption_1 = self.augment_batch(caption_1, self.eda,method='sr', n=2)
        # caption_2 = self.augment_batch(caption_2, self.eda,method='sr', n=2)

        # MLM
        if self.config.experiment.mlm:#这里为false
            text_tokens, mlm_labels = tokenize(captions, context_length=self.config.experiment.text_length,
                                               mask_type='MLM')
            text_tokens = text_tokens.to(self.config.device)
            mlm_labels = mlm_labels.to(self.config.device)
        else:
            # text_tokens1 = tokenize(caption_1, context_length=self.config.experiment.text_length).to(self.config.device)
            # text_tokens2 = tokenize(caption_2, context_length=self.config.experiment.text_length).to(self.config.device)
            text_tokens = tokenize(captions, context_length=self.config.experiment.text_length).to(self.config.device)

        visual_features, image_seq_embeddings = self.encode_image(images, return_dense=True) #当return_dense 被设置为 True 时，encode_image 方法会返回两个结果，即 image_features 和 image_seq_embedding
        # text_features1, text_seq_embeddings = self.encode_text(text_tokens1, return_dense=True)
        # text_features2, text_seq_embeddings = self.encode_text(text_tokens2, return_dense=True)
        text_features, text_seq_embeddings = self.encode_text(text_tokens, return_dense=True)

        #门控模块
        # text_features = self.gate(text_features1, text_features2)

        #注意力融合
        # fused_features = self.fusion(visual_features, text_features)  # 使用注意力融合

        #交叉注意力模块
        # out_text, out_image, fused = self.cross_fusion(text_features, visual_features)

        # # PVD模块
        # pvd_features = torch.cat((visual_features, text_features), dim=1)  # [batch_size, 1024]
        # pvd_out = torch.relu(self.pvd_fc1(pvd_features))  # [batch_size, 256]
        # pvd_out = torch.relu(self.pvd_fc2(pvd_out))  # [batch_size, 128]
        # pvd_out = torch.relu(self.pvd_fc3(pvd_out))  # [batch_size, 64]
        # pvd_out = torch.relu(self.pvd_proj(pvd_out))  # [batch_size, 512]
        # # 确保数据类型一致
        # # visual_features = visual_features.float()
        # # # text_features = text_features.float()
        # # # pvd_out = pvd_out.float()
        # # 特征相加
        # combined_features = visual_features + text_features + pvd_out  # [batch_size, 512]


        # combined_features = visual_features + fused_features # [batch_size, 512]
        #各自双交叉
        # fusion_text = self.cross_text(text_features, visual_features)
        # fusion_image = self.cross_image(visual_features,fusion_text )
        # fused_features =fusion_text + fusion_image

        #直接连接
        # fused = torch.cat([visual_features, text_features], dim=1)  # (batch_size, 1024)
        # output = self.activation(self.fusion(fused))
        # fused_features = visual_features + text_features

        #transformer_decoder
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
        if return_dense: #传过来true
            output = self.visual(image.type(self.dtype), return_dense=return_dense)#return_dense：一个布尔类型的参数，默认值为 False。当设置为 True 时，表示需要返回密集的特征表示；当设置为 False 时，返回默认的特征表示。
            return output
        output = self.visual(image.type(self.dtype))
        return output

    def augment_batch(self, text_list, eda, method=None, n=1, p=0.1):
        """
        对一个文本列表批量增强

        """
        # print(f"eda type: {type(eda)}")
        # print(f"text_list type: {type(text_list)}")

        return [eda.augment(text, method=method, n=n, p=p) for text in text_list]

    def clip_contrastive_loss(self,image_features, text_features,labels, logit_scale=100.0):
        # 保证归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # 计算相似度矩阵：[B, B]
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T  # 对称的

        # 生成标签（0~B-1）作为 ground truth
        # batch_size = image_features.size(0)
        # labels = torch.arange(batch_size, device=image_features.device)

        # 双向 cross entropy：图像->文本，文本->图像
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)

        return (loss_i2t + loss_t2i) / 2

def clip_vitb(config, num_classes=11003): #该函数会分别初始化图像编码器和文本编码器，然后将它们组合到一个 CLIP 模型中返回
    image_encode = visual_transformer(config)
    text_encode = text_transformers(config)
    model = CLIP(config, image_encode, text_encode, num_classes, config.experiment.ritc_eps)
    return model

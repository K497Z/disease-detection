import os
import torch
import torch.nn.functional as F
from torch import nn
from .base_transformer import Transformer, LayerNorm


class TextTransformer(nn.Module):
    def __init__(self, config,
                 embed_dim: int,
                 context_length: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 positional_embedding_flag: bool,
                 checkpoint: bool,
                 bpe_path=None,
                 ):
        super().__init__() # Subclasses may need to inherit some initialization operations from the parent class and may also have their own additional initialization logic. Using super().__init__() ensures that the parent class's initialization code is executed before the subclass-specific initialization code.
        self.config = config
        self.context_length = context_length
        self.positional_embedding_flag = positional_embedding_flag

        self.transformer = Transformer( # Transformer is a class typically used to implement the Transformer model architecture. Transformer is a deep learning model architecture widely used in natural language processing and other fields, with powerful sequence modeling capabilities.
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),# Obtain the mask matrix to mask future information
            checkpoint=checkpoint,
            dropout=config.experiment.dropout
        )
        self.token_embedding = nn.Embedding(49408, transformer_width)# nn.Embedding is a class in PyTorch used to convert discrete tokens (such as word indices) into continuous vector representations. 49408 represents the vocabulary size, i.e., the number of different tokens the model can handle. transformer_width is the dimension of the embedding vector.
        self.positional_embedding = nn.Parameter(
            torch.normal(mean=0, std=0.02, size=(self.context_length, transformer_width)))# Vectorize a sentence into a tensor where each element is randomly sampled from a normal distribution with a mean of 0 and a standard deviation of 0.02.
        self.ln_final = LayerNorm(transformer_width)# LayerNorm is a normalization layer used to normalize input features, helping to improve model training stability and convergence speed.
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))# self.text_projection is a trainable projection matrix used to project the model's output features to the specified dimension embed_dim.
        self.initialize_parameters()
        # Positional extension
        # self.extend_positional_embedding(160)

    def train(self, mode=True):# torch.nn.Module already has a built-in train method similar to this.
        self.training = mode # The training attribute is typically used to indicate whether the model is in training mode.
        for module in self.children():# self.children() is a method that typically returns all sub-modules (sub-layers) of the current module.
            module.train(mode)
        return self

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)# This standard deviation is mainly used for parameter initialization of certain projection layers. Its calculation takes into account the width and number of layers of the hidden layers. The more layers, the smaller the initialization value of the parameters should be.
        attn_std = self.transformer.width ** -0.5 # Parameter initialization for the attention module.
        fc_std = (2 * self.transformer.width) ** -0.5 # Parameter initialization for the fully connected layer.
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)# in_proj_weight is the weight matrix of the input projection layer of the attention module.
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)# out_proj_weight is the weight matrix of the output projection layer of the attention module.
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)# block.mlp represents the multi-layer perceptron module in the current residual block, and c_fc is the first fully connected layer in the multi-layer perceptron.
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)# c_proj is the second fully connected layer in the multi-layer perceptron.
        if self.text_projection is not None:
            # nn.init.normal_(self.text_projection.weight, std=self.transformer.width ** -0.5)  # todo
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    # def extend_positional_embedding(self, new_context_length):
    #     old_embedding = self.positional_embedding  # [77, 512]
    #
    #     # Only interpolate sequence length, does not affect hidden_dim
    #     old_embedding = old_embedding.unsqueeze(0)  # [1, 77, 512]
    #     new_embedding = F.interpolate(
    #         old_embedding.permute(0, 2, 1),  # Become [1, 512, 77]
    #         size=new_context_length,  # Only change sequence length
    #         mode='linear',
    #         align_corners=False
    #     ).permute(0, 2, 1).squeeze(0)  # Change back to [new_length, 512]
    #
    #     self.positional_embedding = nn.Parameter(new_embedding)
    #
    #     print("After extending:", self.positional_embedding.shape)  # Ensure it is [new_length, 512]

    @property
    def dtype(self):
        return self.positional_embedding.dtype

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)# The function is used to create an uninitialized tensor with a shape of (self.context_length, self.context_length), i.e., a square matrix. This matrix will serve as the attention mask.
        mask.fill_(float("-inf"))# Used to fill all elements in the tensor with the specified value.
        mask.triu_(1)  # zero out the lower diagonal. triu_ is an in-place operation method in PyTorch used to get the upper triangular part of a matrix. Parameter 1 means keeping elements starting from the first diagonal above the main diagonal, setting the main diagonal and elements below it to 0.
        return mask

    def forward(self, texts, mask_type=None, return_dense=False):# This code is the forward method of the TextTransformer class, defining how the model processes input data and generates output. Below is a detailed analysis of this code, including data flow processing, the role of key operations, and the meaning of the output.
        if mask_type is not None:
            texts, labels = texts
        x = self.token_embedding(texts).type(self.dtype)  # [batch_size, n_ctx, d_model] The passed texts are not the original text content but should be integer sequences mapped to corresponding indices in the vocabulary after tokenization.
        if self.positional_embedding_flag:
            x = x + self.positional_embedding.type(self.dtype)  # Fix!!!
            # print("Positional Embedding Shape:", self.positional_embedding.shape)
        x = x.permute(1, 0, 2)  # NLD -> LND Adjust data dimensions from [batch_size, n_ctx, d_model] to [n_ctx, batch_size, d_model].
        x = self.transformer(x)# After processing by the Transformer encoder, the output shape remains [n_ctx, batch_size, d_model].
        # x = x.permute(1, 0, 2)  # LND -> NLD Restore data dimensions from [n_ctx, batch_size, d_model] to [batch_size, n_ctx, d_model].
        x = self.ln_final(x).type(self.dtype)# Normalization processing x.shape = [batch_size, seq_len, d_model].
        x = x @ self.text_projection # text_projection is a trainable matrix with shape [d_model, embed_dim], used to project feature vectors from d_model dimension to target dimension embed_dim.

        if mask_type is not None or return_dense:
            words_feat = x

        # x = x[torch.arange(x.shape[0]), texts.argmax(dim=-1)]

        if mask_type is not None:
            return x, words_feat, labels

        if return_dense:
            return x, words_feat

        return x

def text_transformers(config):
    model_config = config.model
    kwargs = {
        'context_length': config.experiment.text_length,# Used to specify the context length of text input, i.e., the maximum length of input text.
        'transformer_width': 512, # Set to 512, representing the hidden layer dimension size in the Transformer model.
        'transformer_heads': 8,
        'transformer_layers': 12,
        'positional_embedding_flag': True, # Indicates whether to use positional embedding; positional embedding is enabled here.
        'checkpoint': False,
        'embed_dim': model_config.embed_dim,# Used to specify the embedding dimension of the model, i.e., the word vector dimension.
    }
    model = TextTransformer(config, **kwargs)
    return model

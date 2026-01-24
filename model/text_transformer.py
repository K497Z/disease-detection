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
        super().__init__() #子类可能需要继承父类的某些初始化操作，同时可能还会有自己额外的初始化逻辑。使用 super().__init__() 可以确保父类的初始化代码被执行，然后再执行子类特有的初始化代码
        self.config = config
        self.context_length = context_length
        self.positional_embedding_flag = positional_embedding_flag

        self.transformer = Transformer( #Transformer 是一个类，通常用于实现 Transformer 架构的模型。Transformer 是一种在自然语言处理和其他领域广泛应用的深度学习模型架构，具有强大的序列建模能力
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),#获得掩膜矩阵，遮住后面信息
            checkpoint=checkpoint,
            dropout=config.experiment.dropout
        )
        self.token_embedding = nn.Embedding(49408, transformer_width)#nn.Embedding 是 PyTorch 中的一个类，用于将离散的标记（如单词的索引）转换为连续的向量表示。49408 表示词汇表的大小，即模型能够处理的不同标记的数量。transformer_width 是嵌入向量的维度
        self.positional_embedding = nn.Parameter(
            torch.normal(mean=0, std=0.02, size=(self.context_length, transformer_width)))#将一个句子向量化， 的张量，其中每个元素都是从均值为 0、标准差为 0.02 的正态分布中随机采样得到的
        self.ln_final = LayerNorm(transformer_width)#LayerNorm 是一种归一化层，用于对输入的特征进行归一化处理，有助于提高模型的训练稳定性和收敛速度
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))#self.text_projection 是一个可训练的投影矩阵，用于将模型的输出特征投影到指定的维度 embed_dim
        self.initialize_parameters()
        #位置扩展
        # self.extend_positional_embedding(160)

    def train(self, mode=True):#torch.nn.Module 已经内置了类似的 train 方法
        self.training = mode #training 属性通常用于指示模型是否处于训练模式
        for module in self.children():#self.children() 是一个方法，通常返回当前模块的所有子模块（子层）
            module.train(mode)
        return self

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)#这个标准差主要用于某些投影层（projection layer）的参数初始化。它的计算综合考虑了隐藏层的宽度和层数。层数越多，参数的初始化值也应该越小
        attn_std = self.transformer.width ** -0.5 #注意力模块（attention module）的参数初始化
        fc_std = (2 * self.transformer.width) ** -0.5 #于全连接层（fully connected layer）的参数初始化
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)#in_proj_weight 是注意力模块的输入投影层的权重矩阵
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)#out_proj_weight 是注意力模块的输出投影层的权重矩阵
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)#block.mlp 表示当前残差块中的多层感知机模块，c_fc 是多层感知机中的第一个全连接层
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)#c_proj 是多层感知机中的第二个全连接层
        if self.text_projection is not None:
            # nn.init.normal_(self.text_projection.weight, std=self.transformer.width ** -0.5)  # todo
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    # def extend_positional_embedding(self, new_context_length):
    #     old_embedding = self.positional_embedding  # [77, 512]
    #
    #     # 只插值序列长度，不影响 hidden_dim
    #     old_embedding = old_embedding.unsqueeze(0)  # [1, 77, 512]
    #     new_embedding = F.interpolate(
    #         old_embedding.permute(0, 2, 1),  # 变成 [1, 512, 77]
    #         size=new_context_length,  # 只改变序列长度
    #         mode='linear',
    #         align_corners=False
    #     ).permute(0, 2, 1).squeeze(0)  # 变回 [new_length, 512]
    #
    #     self.positional_embedding = nn.Parameter(new_embedding)
    #
    #     print("After extending:", self.positional_embedding.shape)  # 确保是 [new_length, 512]

    @property
    def dtype(self):
        return self.positional_embedding.dtype

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)#函数用于创建一个未初始化的张量，其形状为 (self.context_length, self.context_length)，即一个正方形矩阵。这个矩阵将作为注意力掩码
        mask.fill_(float("-inf"))#用于将张量中的所有元素填充为指定的值
        mask.triu_(1)  # zero out the lower diagonaltriu_ 是 PyTorch 中的原地操作方法，用于获取矩阵的上三角部分。参数 1 表示从主对角线之上的第一条对角线开始保留元素，将主对角线和主对角线以下的元素置为 0
        return mask

    def forward(self, texts, mask_type=None, return_dense=False):#这段代码是 TextTransformer 类的 forward 方法，定义了模型如何处理输入数据并生成输出。以下是对这段代码的详细分析，包括数据流的处理过程、关键操作的作用以及输出的含义
        if mask_type is not None:
            texts, labels = texts
        x = self.token_embedding(texts).type(self.dtype)  # [batch_size, n_ctx, d_model]传入的 texts 并非原始的文本内容，而应当是经过分词和词元化处理后，被映射为词汇表中对应索引的整数序列
        if self.positional_embedding_flag:
            x = x + self.positional_embedding.type(self.dtype)  # Fix!!!
            # print("Positional Embedding Shape:", self.positional_embedding.shape)
        x = x.permute(1, 0, 2)  # NLD -> LND将数据维度从 [batch_size, n_ctx, d_model] 调整为 [n_ctx, batch_size, d_model]
        x = self.transformer(x)#经过 Transformer 编码器处理后，输出的形状仍为 [n_ctx, batch_size, d_model]
        # x = x.permute(1, 0, 2)  # LND -> NLD将数据维度从 [n_ctx, batch_size, d_model] 恢复为 [batch_size, n_ctx, d_model]
        x = self.ln_final(x).type(self.dtype)#归一化处理x.shape = [batch_size, seq_len, d_model]
        x = x @ self.text_projection #text_projection 是一个可训练的矩阵，形状为 [d_model, embed_dim]，用于将特征向量从 d_model 维度投影到目标维度 embed_dim

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
        'context_length': config.experiment.text_length,#用于指定文本输入的上下文长度，即输入文本的最大长度
        'transformer_width': 512, #设置为 512，代表 Transformer 模型中隐藏层的维度大小
        'transformer_heads': 8,
        'transformer_layers': 12,
        'positional_embedding_flag': True, #表示是否使用位置编码，这里开启了位置编码
        'checkpoint': False,
        'embed_dim': model_config.embed_dim,#用于指定模型的嵌入维度，即词向量维度
    }
    model = TextTransformer(config, **kwargs)
    return model

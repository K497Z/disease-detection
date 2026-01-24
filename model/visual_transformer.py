import torch
from torch import nn
from .base_transformer import Transformer, LayerNorm
from typing import Tuple, Union


class VisualTransformer(nn.Module):#VisualTransformer 模块通过将输入图像分割成图像块，添加分类标记和位置编码，然后使用 Transformer 编码器进行特征提取，最终输出图像的特征表示
    def __init__(self, input_resolution: Union[int, Tuple[int, int]], patch_size: int, width: int, layers: int, heads: int, embed_dim: int,
                 checkpoint: bool, dropout: float = 0, emb_dropout: float = 0):
        super().__init__()
        if isinstance(input_resolution, int):
            input_resolution = (input_resolution, input_resolution)
        self.input_resolution = input_resolution
        self.num_x = (input_resolution[1] - patch_size) // patch_size + 1
        self.num_y = (input_resolution[0] - patch_size) // patch_size + 1
        num_patches = self.num_x * self.num_y #图像分割部分

        #卷积层和嵌入层初始化
        output_dim = embed_dim
        self.output_dim = output_dim
        self.freeze_conv1 = True #冻结层意味着在训练过程中，该层的权重不会更新，从而保持其初始值或预训练值不变
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width,
                               kernel_size=patch_size, stride=patch_size, bias=False)#初始化卷积层

        scale = width ** -0.5 #缩放因子，是为了初始化时保持数值稳定，避免梯度爆炸或消失
        self.class_embedding = nn.Parameter(scale * torch.randn(width))#torch.randn(width)：生成一个形状为 [width] 的随机向量，其元素服从标准正态分布（均值为 0，标准差为 1）
        #nn.Parameter：将生成的向量包装为一个可训练的参数，使其在训练过程中可以更新
        self.positional_embedding = nn.Parameter(scale * torch.randn(num_patches + 1, width))#初始化位置编码，num_patches + 1：表示位置编码的总数，包括所有 patch 和一个额外的分类标记
        self.ln_pre = LayerNorm(width)#初始化层归一化层


        self.transformer = Transformer(width, layers, heads, checkpoint=checkpoint, dropout=dropout,
                                       emb_dropout=emb_dropout)#初始化 Transformer 编码器

        self.ln_post = LayerNorm(width)#初始化层归一化层
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))#这行代码定义了一个可训练的投影矩阵 self.proj，用于将输入特征从维度 width 投影到目标维度 output_dim
        self.initialize_parameters()#调用 initialize_parameters 方法进行参数初始化

    def initialize_parameters(self):#该方法的主要作用是对模型中的参数进行初始化
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)# 投影层的标准差
        #self.transformer.width ** -0.5：基于特征维度的缩放因子。(2 * self.transformer.layers) ** -0.5：基于层数的缩放因子，确保随着层数增加，初始化的方差逐渐减小。目的：这种缩放方式有助于在多层 Transformer 中保持梯度的稳定性。
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:#self.transformer.resblocks 通常是一个包含多个残差块的列表或可迭代对象。残差块是 Transformer 架构中的核心组件，它一般由多头注意力机制（Multi-Head Attention）和前馈神经网络（Feed Forward Network）组成，并且使用了残差连接（Residual Connection）来缓解梯度消失问题。通过这个循环，代码会依次对每个残差块中的参数进行初始化
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)#in_proj_weight 是多头注意力层中输入投影矩阵的权重，用于将输入的特征投影到多个头的低维空间中
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)#out_proj_weight 是多头注意力层中输出投影矩阵的权重，用于将多个头的输出合并并投影回原始的特征维度
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)#前馈神经网络是残差块（Residual Block）的重要组成部分，通常由两个线性层（全连接层）和一个非线性激活函数（如 ReLU）构成。其作用是对多头注意力机制的输出进行进一步的特征变换和信息处理，增强模型的表达能力
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)

        if self.freeze_conv1:#检查是否需要冻结第一个卷积层（conv1
            for layer in [self.conv1]:
                layer.eval() #将 conv1 切换到评估模式
                for param in layer.parameters():#遍历 conv1 的所有参数，并将它们的 requires_grad 属性设置为 False。这意味着在训练过程中，这些参数不会计算梯度，也不会更新。冻结层的权重在整个训练过程中保持不变。
                    param.requires_grad = False
        return self

    def forward(self, x: torch.Tensor, return_dense=False, return_feature=False):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1) #x.shape[0] 是 batch_size，x.shape[1] 是 width，-1 表示自动计算该维度的大小，即 grid × grid，即patch 的总数
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x) #它对每个样本的特征进行归一化，使得每个特征的均值为 0，标准差为 1

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x) #输入张量 x 应用一个层归一化
        dense_feat = x

        if self.proj is not None:
            dense_feat = x @ self.proj
            # x = dense_feat[:, 0, :]
            x=dense_feat

        if return_dense:
            return x, dense_feat
        if return_feature:
            return dense_feat
        return x


def visual_transformer(config): #VisualTransformer 通常是一种用于处理视觉数据的 Transformer 架构模型
    vision_width = 768 #视觉 Transformer 模型中特征的维度
    vision_layers = 12 #Transformer 编码器层的数量，这里设置为 12
    vision_heads = vision_width // 64 #多头注意力机制中的头数，通过将 vision_width 除以 64 得到

    kwargs = {
        'layers': vision_layers,
        'heads': vision_heads,
        'input_resolution': config.experiment.input_resolution,#input_resolution 表示输入图像的分辨率
        'patch_size': 16,#是指将输入图像分割成的图像块（patch）的大小
        'width': vision_width,#width 每个向量的特征维度
        'checkpoint': False,#checkpoint 是一个布尔值，用于指定是否使用检查点机制。检查点机制是一种内存优化技术，通过在反向传播过程中重新计算一些中间结果，减少内存的使用
        'embed_dim': config.model.embed_dim,#embed_dim 模型最终输出的特征维度,与width不同
    }

    model = VisualTransformer(**kwargs)
    return model

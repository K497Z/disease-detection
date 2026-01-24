import torch
import torch.nn as nn
from .transformer_decoder import TransformerDecoderLayer, TransformerDecoder
import numpy as np
from sklearn import metrics

import ipdb


class Fusion(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        decoder_layer = TransformerDecoderLayer(cfg.model.fusion.d_model, cfg.model.fusion.H, 1024,
                                                0.1, 'relu', normalize_before=True)  # 定义了解码器层的具体结构
        decoder_norm = nn.LayerNorm(cfg.model.fusion.d_model)
        self.decoder = TransformerDecoder(decoder_layer, cfg.model.fusion.N, decoder_norm,
                                          return_intermediate=False)  # cfg.model.fusion.N：表示解码器堆叠的层数，return_intermediate：这是一个布尔值，用于指定是否返回中间层的输出

        # Learnable Queries
        # self.query_embed = nn.Embedding(config['num_queries'] ,self.d_model)
        self.dropout_feas = nn.Dropout(cfg.model.fusion.dropout)

        # Attribute classifier
        # self.classifier = nn.Linear(cfg.model.fusion.d_model, cfg.model.fusion.state_prob)

    def forward(self, query_embed, features):
        features, ws = self.decoder(query_embed, features,
                                    memory_key_padding_mask=None, pos=None,
                                    query_pos=None)  # query_embed：这是查询嵌入，代表解码器的输入。features：作为编码器的输出，也就是解码器要处理的记忆（memory）
        out = self.dropout_feas(features)
        out = out.permute(1, 0, 2)
        out = out[:, 0, :]
        return out

class Fusion2(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        decoder_layer = TransformerDecoderLayer(cfg.model.fusion.d_model, cfg.model.fusion.H, 1024,
                                                0.1, 'relu', normalize_before=True)  # 定义了解码器层的具体结构
        decoder_norm = nn.LayerNorm(cfg.model.fusion.d_model)
        self.decoder = TransformerDecoder(decoder_layer, cfg.model.fusion.N, decoder_norm,
                                          return_intermediate=False)  # cfg.model.fusion.N：表示解码器堆叠的层数，return_intermediate：这是一个布尔值，用于指定是否返回中间层的输出

        # Learnable Queries
        # self.query_embed = nn.Embedding(config['num_queries'] ,self.d_model)
        self.dropout_feas = nn.Dropout(cfg.model.fusion.dropout)

        # Attribute classifier
        # self.classifier = nn.Linear(cfg.model.fusion.d_model, cfg.model.fusion.state_prob)

    def forward(self, query_embed, features, texts):
        features, ws = self.decoder(query_embed, features,
                                    memory_key_padding_mask=None, pos=None,
                                    query_pos=None)  # query_embed：这是查询嵌入，代表解码器的输入。features：作为编码器的输出，也就是解码器要处理的记忆（memory）
        out = self.dropout_feas(features)
        out = out.permute(1, 0, 2)
        out = out[torch.arange(out.shape[0]), texts.argmax(dim=-1)]
        return out

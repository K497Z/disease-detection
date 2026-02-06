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
                                                0.1, 'relu', normalize_before=True)  # Defines the specific structure of the decoder layer
        decoder_norm = nn.LayerNorm(cfg.model.fusion.d_model)
        self.decoder = TransformerDecoder(decoder_layer, cfg.model.fusion.N, decoder_norm,
                                          return_intermediate=False)  # cfg.model.fusion.N: Indicates the number of stacked decoder layers; return_intermediate: This is a boolean value indicating whether to return the output of intermediate layers

        # Learnable Queries
        # self.query_embed = nn.Embedding(config['num_queries'] ,self.d_model)
        self.dropout_feas = nn.Dropout(cfg.model.fusion.dropout)

        # Attribute classifier
        # self.classifier = nn.Linear(cfg.model.fusion.d_model, cfg.model.fusion.state_prob)

    def forward(self, query_embed, features):
        features, ws = self.decoder(query_embed, features,
                                    memory_key_padding_mask=None, pos=None,
                                    query_pos=None)  # query_embed: This is the query embedding, representing the input to the decoder. features: Serves as the output of the encoder, i.e., the memory to be processed by the decoder
        out = self.dropout_feas(features)
        out = out.permute(1, 0, 2)
        out = out[:, 0, :]
        return out

class Fusion2(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        decoder_layer = TransformerDecoderLayer(cfg.model.fusion.d_model, cfg.model.fusion.H, 1024,
                                                0.1, 'relu', normalize_before=True)  # Defines the specific structure of the decoder layer
        decoder_norm = nn.LayerNorm(cfg.model.fusion.d_model)
        self.decoder = TransformerDecoder(decoder_layer, cfg.model.fusion.N, decoder_norm,
                                          return_intermediate=False)  # cfg.model.fusion.N: Indicates the number of stacked decoder layers; return_intermediate: This is a boolean value indicating whether to return the output of intermediate layers

        # Learnable Queries
        # self.query_embed = nn.Embedding(config['num_queries'] ,self.d_model)
        self.dropout_feas = nn.Dropout(cfg.model.fusion.dropout)

        # Attribute classifier
        # self.classifier = nn.Linear(cfg.model.fusion.d_model, cfg.model.fusion.state_prob)

    def forward(self, query_embed, features, texts):
        features, ws = self.decoder(query_embed, features,
                                    memory_key_padding_mask=None, pos=None,
                                    query_pos=None)  # query_embed: This is the query embedding, representing the input to the decoder. features: Serves as the output of the encoder, i.e., the memory to be processed by the decoder
        out = self.dropout_feas(features)
        out = out.permute(1, 0, 2)
        out = out[torch.arange(out.shape[0]), texts.argmax(dim=-1)]
        return out

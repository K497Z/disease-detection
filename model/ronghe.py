import torch
import torch.nn as nn

class TextFeatureGatedFusion(nn.Module):
    def __init__(self, hidden_size=512, dropout_prob=0.3):
        super(TextFeatureGatedFusion, self).__init__()

        # Two text feature transformations
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        # Gating layer
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(p=dropout_prob)

    def forward(self, feat1, feat2):
        # feat1 and feat2 are features of two captions obtained using CLIP text encoder
        f1 = self.linear1(feat1)
        f2 = self.linear2(feat2)

        gate_weights = self.sigmoid(self.gate(f1 * f2))
        fused = gate_weights * f1 + (1 - gate_weights) * f2

        fused = self.drop(fused)
        return fused

class ClassifierHead(nn.Module):
    def __init__(self, input_dim=512, num_classes=100, dropout_prob=0.3):
        super(ClassifierHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
        # self.fc2 = nn.Linear(hidden_dim, num_classes)
        # self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x)
        # x = self.relu(x)
        x = self.dropout(x)
        # x = self.fc2(x)
        return x

# Self-attention
# class CrossModalFusion(nn.Module):
#     def __init__(self, dim=512, n_heads=8, dropout=0.1, n_layers=1):
#         super().__init__()
#
#         encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, dropout=dropout, batch_first=True)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
#
#     def forward(self, image_feat, text_feat):
#         """
#         image_feat: [B, D]
#         text_feat: [B, D]
#         return: fused_feat: [B, D]
#         """
#         # Concatenate -> [B, 2, D]
#         x = torch.stack([image_feat, text_feat], dim=1)
#
#         # Optional positional embedding (can be added or not)
#         # x = x + self.pos_embedding[:, :2, :]
#
#         x = self.transformer(x)  # [B, 2, D]
#
#         # Fusion strategy: mean pooling / take the first token / weighted
#         fused_feat = x.mean(dim=1)  # [B, D]
#
#         return fused_feat

import torch.nn.functional as F

# Fused cross-attention
class CrossModalAttentionFusion(nn.Module):
    def __init__(self, embed_dim=512, fusion_method="sum", dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_method = fusion_method

        if fusion_method == "concat":
            fusion_dim = embed_dim * 2
        elif fusion_method == "sum":
            fusion_dim = embed_dim
        else:
            raise ValueError("fusion_method must be 'concat' or 'sum'")


        self.image_to_fused = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=dropout, batch_first=True)
        self.text_to_fused = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=dropout, batch_first=True)

        # Map to unified dimension
        self.fuse_proj = nn.Linear(fusion_dim, embed_dim)

    def forward(self, text_feat, image_feat):
        """
        text_feat: (B, D) from CLIP text encoder
        image_feat: (B, D) from CLIP image encoder
        """
        # B, D = text_feat.shape

        # Simple fusion (concat or sum)
        if self.fusion_method == "concat":
            fused = torch.cat([text_feat, image_feat], dim=-1)  # (B, 2D)
        else:  # sum
            fused = text_feat + image_feat  # (B, D)

        fused = self.fuse_proj(fused).unsqueeze(1)  # (B, 1, D)
        text_q = text_feat.unsqueeze(1)  # (B, 1, D)
        image_q = image_feat.unsqueeze(1)  # (B, 1, D)

        # Text as Q, fused as K, V
        attn_text, _ = self.text_to_fused(query=text_q, key=fused, value=fused)  # (B, 1, D)

        # Image as Q, fused as K, V
        attn_image, _ = self.image_to_fused(query=image_q, key=fused, value=fused)  # (B, 1, D)

        # Flatten output
        out_text = attn_text.squeeze(1)  # (B, D)
        out_image = attn_image.squeeze(1)  # (B, D)

        return out_text, out_image, fused.squeeze(1)  # Can choose which fusion to use

# Direct guidance
class CrossAttentionTextQueryCLIP(nn.Module):
    def __init__(self, dim=512, num_heads=8):
        super(CrossAttentionTextQueryCLIP, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, text_feat, image_feat):
        # Inputs are both (B, D), transform to (B, 1, D)
        text_feat = text_feat.unsqueeze(1)
        image_feat = image_feat.unsqueeze(1)

        attn_output, _ = self.cross_attn(query=text_feat, key=image_feat, value=image_feat)  # (B, 1, D)
        out = self.norm(attn_output + text_feat)  # Residual + norm
        return out.squeeze(1)  # Return (B, D)


class CrossAttentionImageQueryCLIP(nn.Module):
    def __init__(self, dim=512, num_heads=8):
        super(CrossAttentionImageQueryCLIP, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, image_feat, text_feat):
        # Inputs are both (B, D), transform to (B, 1, D)
        image_feat = image_feat.unsqueeze(1)
        text_feat = text_feat.unsqueeze(1)

        attn_output, _ = self.cross_attn(query=image_feat, key=text_feat, value=text_feat)  # (B, 1, D)
        out = self.norm(attn_output + image_feat)  # Residual + norm
        return out.squeeze(1)  # Return (B, D)

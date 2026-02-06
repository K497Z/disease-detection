import torch
from torch import nn
from .base_transformer import Transformer, LayerNorm
from typing import Tuple, Union


class VisualTransformer(nn.Module):# The VisualTransformer module processes the input image by dividing it into patches, adding classification tokens and positional embeddings, then using a Transformer encoder for feature extraction, and finally outputting the feature representation of the image.
    def __init__(self, input_resolution: Union[int, Tuple[int, int]], patch_size: int, width: int, layers: int, heads: int, embed_dim: int,
                 checkpoint: bool, dropout: float = 0, emb_dropout: float = 0):
        super().__init__()
        if isinstance(input_resolution, int):
            input_resolution = (input_resolution, input_resolution)
        self.input_resolution = input_resolution
        self.num_x = (input_resolution[1] - patch_size) // patch_size + 1
        self.num_y = (input_resolution[0] - patch_size) // patch_size + 1
        num_patches = self.num_x * self.num_y # Image segmentation part

        # Convolutional layer and embedding layer initialization
        output_dim = embed_dim
        self.output_dim = output_dim
        self.freeze_conv1 = True # Freezing a layer means its weights will not be updated during training, keeping its initial or pre-trained values unchanged.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width,
                               kernel_size=patch_size, stride=patch_size, bias=False)# Initialize convolutional layer

        scale = width ** -0.5 # Scale factor to maintain numerical stability during initialization and avoid gradient explosion or disappearance.
        self.class_embedding = nn.Parameter(scale * torch.randn(width))# torch.randn(width): Generates a random vector of shape [width], with elements following a standard normal distribution (mean 0, standard deviation 1).
        # nn.Parameter: Wraps the generated vector as a trainable parameter so it can be updated during training.
        self.positional_embedding = nn.Parameter(scale * torch.randn(num_patches + 1, width))# Initialize positional embedding. num_patches + 1: Total number of positional embeddings, including all patches and an extra classification token.
        self.ln_pre = LayerNorm(width)# Initialize Layer Normalization layer


        self.transformer = Transformer(width, layers, heads, checkpoint=checkpoint, dropout=dropout,
                                       emb_dropout=emb_dropout)# Initialize Transformer encoder

        self.ln_post = LayerNorm(width)# Initialize Layer Normalization layer
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))# This line defines a trainable projection matrix self.proj, used to project input features from dimension `width` to target dimension `output_dim`.
        self.initialize_parameters()# Call initialize_parameters method to initialize parameters.

    def initialize_parameters(self):# The main function of this method is to initialize the parameters in the model.
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)# Standard deviation for projection layers.
        # self.transformer.width ** -0.5: Scale factor based on feature dimension. (2 * self.transformer.layers) ** -0.5: Scale factor based on the number of layers, ensuring variance gradually decreases as layers increase. Purpose: This scaling helps maintain gradient stability in deep Transformers.
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:# self.transformer.resblocks is typically a list or iterable containing multiple residual blocks. Residual blocks are core components in Transformer architecture, usually consisting of Multi-Head Attention and Feed Forward Network, using Residual Connection to alleviate gradient disappearance. This loop initializes parameters in each residual block sequentially.
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)# in_proj_weight is the weight of the input projection matrix in the multi-head attention layer, used to project input features into lower-dimensional spaces of multiple heads.
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)# out_proj_weight is the weight of the output projection matrix in the multi-head attention layer, used to merge outputs from multiple heads and project back to the original feature dimension.
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)# The Feed Forward Network is an important part of the Residual Block, usually consisting of two linear layers (fully connected layers) and a non-linear activation function (like ReLU). Its function is to further transform features and process information from the multi-head attention output, enhancing the model's expressive power.
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)

        if self.freeze_conv1:# Check if the first convolutional layer (conv1) needs to be frozen.
            for layer in [self.conv1]:
                layer.eval() # Switch conv1 to evaluation mode.
                for param in layer.parameters():# Iterate through all parameters of conv1 and set their requires_grad attribute to False. This means these parameters will not calculate gradients or update during training. Weights of the frozen layer remain unchanged throughout training.
                    param.requires_grad = False
        return self

    def forward(self, x: torch.Tensor, return_dense=False, return_feature=False):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1) # x.shape[0] is batch_size, x.shape[1] is width, -1 indicates automatic calculation of this dimension's size, i.e., grid x grid, which is total number of patches.
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x) # Applies normalization to features of each sample, making mean 0 and standard deviation 1.

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x) # Applies Layer Normalization to input tensor x.
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


def visual_transformer(config): # VisualTransformer is typically a Transformer architecture model for processing visual data.
    vision_width = 768 # Feature dimension in the Vision Transformer model.
    vision_layers = 12 # Number of Transformer encoder layers, set to 12 here.
    vision_heads = vision_width // 64 # Number of heads in multi-head attention, obtained by dividing vision_width by 64.

    kwargs = {
        'layers': vision_layers,
        'heads': vision_heads,
        'input_resolution': config.experiment.input_resolution,# input_resolution indicates the resolution of the input image.
        'patch_size': 16,# Refers to the size of patches the input image is divided into.
        'width': vision_width,# width: Feature dimension of each vector.
        'checkpoint': False,# checkpoint is a boolean indicating whether to use checkpoint mechanism. Checkpoint mechanism is a memory optimization technique that reduces memory usage by recomputing some intermediate results during backpropagation.
        'embed_dim': config.model.embed_dim,# embed_dim: Final output feature dimension of the model, different from width.
    }

    model = VisualTransformer(**kwargs)
    return model

import os
import torch
import numpy as np
import math
import torch.nn.functional as F


def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    posemb = posemb.unsqueeze(0)
    posemb_new = posemb_new.unsqueeze(0)

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb.squeeze(0)


def interpolate_text(pos_embed_checkpoint, target_dim=77):
    # (n_ctx, n_feat) for pos_embed_checkpoint, including SOT and EOT
    if pos_embed_checkpoint.size(0) == target_dim:
        return pos_embed_checkpoint
    start_token = pos_embed_checkpoint[:1, :]
    end_token = pos_embed_checkpoint[-1:, :]
    pos_tokens = pos_embed_checkpoint[1:-1, :].unsqueeze(0).permute(0, 2, 1)
    pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=target_dim - 2, mode='linear')
    pos_tokens = pos_tokens.squeeze(0).t()
    pos_tokens = torch.cat([start_token, pos_tokens, end_token], dim=0)
    return pos_tokens


def load_checkpoint(model, config):# This code defines a load_checkpoint function to load pre-trained model weights and adapt them to the current model structure. This function supports two types of checkpoints: original_clip and saved
    if config.model.ckpt_type == 'original_clip':
        with open(config.model.checkpoint, 'rb') as opened_file:
            model_tmp = torch.jit.load(opened_file, map_location="cpu")# Used to load the official CLIP pre-trained TorchScript .pt model and place it on the CPU
            state = model_tmp.state_dict()# Extract model weights (parameters)
        for key in ["input_resolution", "context_length", "vocab_size"]:# The state_dict of the CLIP .pt model contains some unnecessary keys
                del state[key]

        # 2 towers in new_state: visual, encode_text
        new_state = {}
        for name, params in state.items():# When loading pre-trained model parameters, adapt the visual positional embedding parameters. The shape of visual positional embeddings may vary under different models or training settings. The resize_pos_embed function adjusts the pre-trained positional embedding parameters to match the shape of the current model's parameters, ensuring they can be correctly loaded.
            if name == 'visual.positional_embedding' and params.shape != model.visual.positional_embedding.shape:
                params = resize_pos_embed(params, model.visual.positional_embedding, model.visual.num_y, model.visual.num_x)

            if name == 'positional_embedding':
                new_state['encode_text.' + name] = interpolate_text(params, config.experiment.text_length)
                # new_state['encode_text.' + name] = interpolate_text(params, 160)
            elif name.startswith('transformer') or name in ['positional_embedding', 'token_embedding.weight',
                                                            'ln_final.weight', 'ln_final.bias', 'text_projection']:
                new_state['encode_text.' + name] = params
            else:
                new_state[name] = params
    elif config.model.ckpt_type == 'saved':
        ckpt = torch.load(os.path.join(config.model.saved_path, 'checkpoint_best.pth'), map_location='cpu')
        new_state = ckpt['model']
    else:
        raise KeyError

    load_result = model.load_state_dict(new_state, strict=False)# model.load_state_dict(): This is a method of the PyTorch model object used to load the specified state dictionary into the model.
    # load_result: This method returns a namedtuple object containing two attributes: missing_keys (a list of parameter names present in the model but missing in the state dictionary) and unexpected_keys (a list of parameter names present in the state dictionary but missing in the model).
    return model, load_result # Return the model loaded with the new state dictionary and the load_result as a tuple


def cosine_scheduler(config):# Define a function named cosine_scheduler, whose main function is to generate a cosine annealing learning rate schedule, which can include a warmup phase.
    # This function generates a learning rate schedule array based on the passed configuration object `config`. This array specifies the learning rate for each iteration step throughout the training process. The schedule combines a warmup phase and a cosine annealing phase. In the warmup phase, the learning rate gradually increases from a lower value to the base value, and in the cosine annealing phase, the learning rate gradually decreases from the base value to the final value.
    schedule_config = config.schedule
    base_value = schedule_config.lr # Base learning rate, i.e., the learning rate used after the warmup phase
    start_warmup_value = schedule_config.lr_start # Learning rate at the start of the warmup phase
    final_value = schedule_config.lr_end # Final learning rate at the end of the cosine annealing phase
    epochs = schedule_config.epoch
    warmup_epochs = schedule_config.epoch_warmup # warmup_epochs: Number of epochs for the warmup phase
    niter_per_ep = schedule_config.niter_per_ep # niter_per_ep: Number of iterations per training epoch

    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep # Get the total number of iterations for the entire warmup phase
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)# warmup_schedule: Generated arithmetic progression containing warmup_iters elements, starting from start_warmup_value and ending at base_value, where each element represents the learning rate for an iteration step.

    iters = np.arange(epochs * niter_per_ep - warmup_iters)# Integer array `iters` from 0 to epochs * niter_per_ep - warmup_iters - 1, representing all iteration steps in the cosine annealing phase.
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters))) # Finally obtain an array `schedule`, where each element represents the learning rate for each iteration step in the cosine annealing phase.

    schedule = np.concatenate((warmup_schedule, schedule)) # This line concatenates the warmup phase learning rate schedule array and the cosine annealing phase learning rate schedule array to form a complete learning rate schedule array.
    assert len(schedule) == epochs * niter_per_ep # assert is an assertion statement in Python used to check if a condition is true. If the condition is false, an AssertionError is raised.
    return schedule


# def build_optimizer(config, model):
#     p_wd, p_non_wd = [], []
#     for n, p in model.named_parameters():
#         if not p.requires_grad:
#             continue  # frozen weights
#         if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
#             p_non_wd.append(p)
#         else:
#             p_wd.append(p)
#
#     schedule_config = config.schedule
#     optim_params = [{"params": p_wd, "weight_decay": schedule_config.weight_decay, "ratio": 1.},
#                     {"params": p_non_wd, "weight_decay": 0, "ratio": 1.}]
#
#     optimizer = torch.optim.AdamW(optim_params, lr=schedule_config.lr, betas=schedule_config.betas,
#                                   eps=schedule_config.eps, weight_decay=schedule_config.weight_decay)
#     return optimizer


def build_optimizer(config, model):# Build an AdamW optimizer based on the given configuration object `config` and model `model`.
    # This function iterates through all trainable parameters of the model, sets different weight decay and learning rate adjustment ratios based on parameter attributes (such as dimensions, names, etc.), and then creates an AdamW optimizer using these configurations.
    params = [] # Used to store configuration information for each trainable parameter, eventually serving as the parameter list for the AdamW optimizer.
    schedule_config = config.schedule
    for n, p in model.named_parameters():# model.named_parameters() method iterates through all parameters of the model, where n is the parameter name and p is the parameter tensor.
        if not p.requires_grad:# If the parameter's requires_grad attribute is False, it means the parameter is frozen and does not need training, so skip it.
            continue  # frozen weights
        weight_decay = schedule_config.weight_decay # Initialize weight_decay to the value specified in the configuration file. In the L2 regularization formula, λ is the weight decay coefficient.
        # λ: Weight decay coefficient, a hyperparameter that needs to be set manually. It controls the impact of the regularization term; the larger λ is, the stronger the regularization.
        ratio = 1. # Initialize learning rate adjustment ratio to 1.0. In a model containing multiple sub-modules, some sub-modules may need to learn features faster, while others need more stable adjustments. By setting different learning rate ratios, different learning rates can be assigned to these different sub-modules or parameter groups, improving model training efficiency and performance.

        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:# If the parameter dimension is less than 2 (e.g., scalar or vector), or the parameter name contains 'bias', 'ln' (likely layer normalization), or 'bn' (likely batch normalization), set weight decay to 0. This is because weight decay is usually not needed for these types of parameters to avoid affecting model stability.
            weight_decay = 0.
        if "cross" in n or "classifier" in n or "mlm_head" in n: # If the parameter name contains 'cross', 'classifier', or 'mlm_head', multiply the learning rate adjustment ratio by the ratio_factor specified in the configuration file (default is 5.0). This means the learning rate for these parameters will be higher than others to speed up training for these parts.
            ratio = ratio * schedule_config.ratio_factor  # default 5.0

        params += [{"params": [p], "weight_decay": weight_decay, "ratio": ratio}] # Encapsulate the current parameter's configuration information (parameter tensor, weight decay, and learning rate ratio) into a dictionary and add it to the params list.

    optimizer = torch.optim.AdamW(params, lr=schedule_config.lr, betas=schedule_config.betas,
                                  eps=schedule_config.eps, weight_decay=schedule_config.weight_decay)# Create an AdamW optimizer using torch.optim.AdamW.

    return optimizer

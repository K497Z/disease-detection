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


def load_checkpoint(model, config):#这段代码定义了一个 load_checkpoint 函数，用于加载预训练模型的权重，并将其适配到当前模型的结构中。这个函数支持两种类型的检查点（checkpoint）：original_clip 和 saved
    if config.model.ckpt_type == 'original_clip':
        with open(config.model.checkpoint, 'rb') as opened_file:
            model_tmp = torch.jit.load(opened_file, map_location="cpu")#用于加载 CLIP 官方预训练的 TorchScript .pt 模型，并放到 CPU 上
            state = model_tmp.state_dict()#提取 模型权重（参数）
        for key in ["input_resolution", "context_length", "vocab_size"]:#CLIP .pt 模型的 state_dict 里 包含一些不必要的键
                del state[key]

        # 2 towers in new_state: visual, encode_text
        new_state = {}
        for name, params in state.items():#在加载预训练模型参数时，对视觉位置编码（visual.positional_embedding）参数进行形状适配处理。在不同的模型或者不同的训练设置下，视觉位置编码的形状可能会有所不同，通过 resize_pos_embed 函数可以对预训练的位置编码参数进行调整，使其形状与当前模型的视觉位置编码参数相匹配，从而能够正确地加载到当前模型中
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

    load_result = model.load_state_dict(new_state, strict=False)#model.load_state_dict()：这是 PyTorch 模型对象的一个方法，其作用是把指定的状态字典加载到模型里。
   #load_result：该方法会返回一个 namedtuple 对象，包含两个属性：missing_keys：这是一个列表，里面存储着模型中有但状态字典里没有的参数名称。unexpected_keys：这也是一个列表，存储着状态字典中有但模型里不存在的参数名称
    return model, load_result #载了新状态字典的模型 model 以及加载结果 load_result 作为元组返回


def cosine_scheduler(config):#定义一个名为 cosine_scheduler 的函数，其主要功能是生成一个余弦退火学习率调度计划，并且可以包含预热（warmup）阶段
    #该函数会根据传入的配置对象 config 生成一个学习率调度数组，此数组规定了在整个训练过程里每个迭代步骤对应的学习率。这个调度计划结合了预热阶段和余弦退火阶段，预热阶段学习率从较低值逐渐增加到基础值，而余弦退火阶段学习率会从基础值逐渐降低到最终值
    schedule_config = config.schedule
    base_value = schedule_config.lr #基础学习率，也就是预热阶段结束后使用的学习率
    start_warmup_value = schedule_config.lr_start #预热阶段开始时的学习率
    final_value = schedule_config.lr_end #余弦退火阶段结束时的最终学习率
    epochs = schedule_config.epoch
    warmup_epochs = schedule_config.epoch_warmup #warmup_epochs：预热阶段的轮数
    niter_per_ep = schedule_config.niter_per_ep #niter_per_ep：每个训练轮次的迭代次数

    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep #得到整个热身阶段的总迭代次数
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)#warmup_schedule：生成的等差数列，包含 warmup_iters 个元素，从 start_warmup_value 开始，到 base_value 结束，每个元素代表一个迭代步骤的学习率

    iters = np.arange(epochs * niter_per_ep - warmup_iters)#从 0 到 epochs * niter_per_ep - warmup_iters - 1 的整数数组 iters，这个数组代表了余弦退火阶段的所有迭代步骤
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters))) #最终得到一个数组 schedule，其中每个元素代表余弦退火阶段每个迭代步骤对应的学习率

    schedule = np.concatenate((warmup_schedule, schedule)) #这行代码将预热阶段的学习率调度数组和余弦退火阶段的学习率调度数组连接起来，形成一个完整的学习率调度数组
    assert len(schedule) == epochs * niter_per_ep #assert 是 Python 中的一个断言语句，用于检查某个条件是否为真。如果条件为假，则会抛出 AssertionError 异常
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


def build_optimizer(config, model):#根据给定的配置对象 config 和模型 model 构建一个 AdamW 优化器
    #该函数会遍历模型的所有可训练参数，根据参数的属性（如维度、名称等）为不同的参数设置不同的权重衰减（weight decay）和学习率调整比例（ratio），然后使用这些参数配置创建一个 AdamW 优化器
    params = [] #用于存储每个可训练参数的配置信息，最终会作为 AdamW 优化器的参数列表
    schedule_config = config.schedule
    for n, p in model.named_parameters():#model.named_parameters() 方法遍历模型的所有参数，n 是参数的名称，p 是参数的张量
        if not p.requires_grad:#如果参数的 requires_grad 属性为 False，说明该参数是冻结的，不需要进行训练，因此跳过该参数
            continue  # frozen weights
        weight_decay = schedule_config.weight_decay #初始化权重衰减 weight_decay 为配置文件中指定的值，L2正则化公式中λ是权重衰减系数
        #λ ：权重衰减系数，是一个超参数，需要人为设定。它对正则化项的影响程度起到控制作用，λ越大，正则化的力度就越强
        ratio = 1. #初始化学习率调整比例 ratio 为 1.0在一个包含多个子模块的模型中，某些子模块可能需要更快地学习到数据的特征，而另一些子模块则需要更稳定地进行调整。通过设置不同的学习率调整比例，可以为这些不同的子模块或参数组分配不同的学习率，从而提高模型的训练效率和性能

        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:#如果参数的维度小于 2（例如标量或向量），或者参数名称中包含 'bias'、'ln'（可能表示层归一化层）或 'bn'（可能表示批归一化层），则将权重衰减设置为 0。这是因为对于这些类型的参数，通常不需要进行权重衰减，以避免影响模型的稳定性
            weight_decay = 0.
        if "cross" in n or "classifier" in n or "mlm_head" in n: #如果参数名称中包含 'cross'、'classifier' 或 'mlm_head'，则将学习率调整比例 ratio 乘以配置文件中指定的 ratio_factor（默认值为 5.0）。这意味着这些参数的学习率会比其他参数更高，以加快这些部分的训练速度
            ratio = ratio * schedule_config.ratio_factor  # default 5.0

        params += [{"params": [p], "weight_decay": weight_decay, "ratio": ratio}] #将当前参数的配置信息（参数张量、权重衰减和学习率调整比例）封装成一个字典，并添加到 params 列表中

    optimizer = torch.optim.AdamW(params, lr=schedule_config.lr, betas=schedule_config.betas,
                                  eps=schedule_config.eps, weight_decay=schedule_config.weight_decay)#使用 torch.optim.AdamW 创建一个 AdamW 优化器

    return optimizer

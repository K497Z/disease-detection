import os
import random
import time
from pathlib import Path

import torch

from misc.build import load_checkpoint, cosine_scheduler, build_optimizer
from misc.data import build_pedes_data
from misc.eval import test
from misc.utils import parse_config, init_distributed_mode, set_seed, is_master, is_using_distributed, \
    AverageMeter
from model.tbps_model import clip_vitb
from options import get_args


def run(config):
    print(config)

    # data
    dataloader = build_pedes_data(config)  # 调用 build_pedes_data 函数，构建训练和测试数据加载器（DataLoader）
    train_loader = dataloader['train_loader']
    num_classes = len(train_loader.dataset.person2text)

    meters = {
        "loss": AverageMeter(),
        # "nitc_loss": AverageMeter(),
        # "ss_loss": AverageMeter(),
        # "citc_loss": AverageMeter(),
        # "ritc_loss": AverageMeter(),
        # "mlm_loss": AverageMeter(),
        # "id_loss": AverageMeter(),
    }
    best_rank_1 = 0.0
    best_epoch = 0

    # model
    model = clip_vitb(config, num_classes)  # 用于构建一个基于 CLIP 架构的模型，其中视觉部分使用 Vision Transformer (ViT-B)，文本部分使用 Transformer
    model.to(config.device)

    model, load_result = load_checkpoint(model, config)

    if is_using_distributed():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.device],
                                                          find_unused_parameters=True)

    # schedule
    config.schedule.niter_per_ep = len(train_loader)
    lr_schedule = cosine_scheduler(config)  # cosine_scheduler 函数的主要作用是生成一个余弦退火学习率调度计划

    # optimizer
    optimizer = build_optimizer(config, model)

    # train
    it = 0
    scaler = torch.cuda.amp.GradScaler()  # cuda只适用于GPU环境，这里先注释
    # accum_steps = 2  # 累积2个batch的梯度

    for epoch in range(config.schedule.epoch):
        print()
        if is_using_distributed():
            dataloader['train_sampler'].set_epoch(epoch)

        start_time = time.time()  # time.time() 是 Python 标准库 time 模块中的函数，用于返回当前时间的时间戳（从 1970 年 1 月 1 日午夜开始的秒数）。这里将当前时间保存到 start_time 变量中，可能是为了后续计算每个 epoch 的训练时间
        for meter in meters.values():
            meter.reset()  # 使得每个轮次的训练，meters这些参数都重新设置为0
        model.train()

        for i, batch in enumerate(train_loader):
            # 将数据加载到 CPU
            # batch = {k: v.to(config.device) for k, v in batch.items()},这个是我自己加的，但是数据集本来不是在cpu上面吗

            # 学习率调度
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[it] * param_group['ratio']
            # 软标签比例
            if epoch == 0:
                alpha = config.model.softlabel_ratio * min(1.0, i / len(train_loader))
            else:
                alpha = config.model.softlabel_ratio

            # MixGen 数据增强
            if config.experiment.mixgen:
                if random.random() < config.experiment.mixgen_p:
                    import model.mixgen as mg
                    if config.experiment.mixgen_type == 'cat':
                        mixgen_func = mg.concatgen
                    else:
                        mixgen_func = mg.mixgen
                    img, cap = mixgen_func(batch['image'], batch['caption'],
                                           num=int(config.experiment.mixgen_ratio * len(batch['caption'])))
                    batch.update({
                        'image': img,
                        'caption': cap,
                    })

            with torch.autocast(device_type='cuda'):  # 这里使用gpu：cuda所以我把它注释掉
                ret = model(batch, alpha)  # 这里一开始是不行的，但是我把batch_size改小之后就可以运行了
                loss = sum([v for k, v in ret.items() if "loss" in k])

            # 更新损失记录
            batch_size = batch['image'].shape[0]
            meters['loss'].update(loss.item(), batch_size)
            meters['nitc_loss'].update(ret.get('nitc_loss', 0), batch_size)
            meters['ss_loss'].update(ret.get('ss_loss', 0), batch_size)
            meters['citc_loss'].update(ret.get('citc_loss', 0), batch_size)
            meters['ritc_loss'].update(ret.get('ritc_loss', 0), batch_size)
            meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)
            meters['id_loss'].update(ret.get('id_loss', 0), batch_size)

            # 反向传播和优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()#新加的
            # optimizer.step()#新加的
            model.zero_grad()
            optimizer.zero_grad()
            it += 1

            if (i + 1) % config.log.print_period == 0:  # 这里是打印批次
                info_str = f"Epoch[{epoch + 1}] Iteration[{i + 1}/{len(train_loader)}]"
                # log loss
                for k, v in meters.items():
                    if v.val != 0:
                        info_str += f", {k}: {v.val:.4f}"
                info_str += f", Base Lr: {param_group['lr']:.2e}"
                print(info_str)

        if is_master():
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (i + 1)  # 计算每个 batch 的平均耗时和吞吐量，监控训练效率。
            print("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                  .format(epoch + 1, time_per_batch, train_loader.batch_size / time_per_batch))

            eval_result = test(model.module, dataloader['test_loader'], 77,
                               config.device)  # 这里之前没有用并行化处理过，即没有被包装在并行化容器中，所以直接传model即可
            # eval_result = test(model, dataloader['test_loader'], 77, config.device)
            rank_1, rank_5, rank_10, map = eval_result['r1'], eval_result['r5'], eval_result['r10'], eval_result['mAP']
            print('Acc@1 {top1:.5f} Acc@5 {top5:.5f} Acc@10 {top10:.5f} mAP {mAP:.5f}'.format(top1=rank_1, top5=rank_5,
                                                                                              top10=rank_10, mAP=map))
            torch.cuda.empty_cache()
            if best_rank_1 < rank_1:
                best_rank_1 = rank_1
                best_epoch = epoch

                save_obj = {
                    'model': model.module.state_dict(),
                    # 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                }
                torch.save(save_obj, os.path.join(config.model.saved_path, 'checkpoint_best.pth'))

    print(f"best Acc@1: {best_rank_1} at epoch {best_epoch + 1}")


if __name__ == '__main__':

    config_path = 'config/config.yaml'

    args = get_args()
    if args.simplified:
        config_path = 'config/s.config.yaml'
    config = parse_config(config_path)

    Path(config.model.saved_path).mkdir(parents=True, exist_ok=True)  # 保存路径

    init_distributed_mode(config)  # 是一个用于初始化分布式训练模式的函数

    set_seed(config)

    run(config)

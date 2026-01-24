import os
import random
import time
from pathlib import Path

import torch
from torchmetrics.functional import accuracy

from misc.build import load_checkpoint, cosine_scheduler, build_optimizer
from misc.data import build_pedes_data
from misc.eval import test
from misc.utils import parse_config, init_distributed_mode, set_seed, is_master, is_using_distributed, \
    AverageMeter
from model.tbps_model import clip_vitb
from options import get_args
import matplotlib.pyplot as plt



def run(config):
    print(config)

    # data
    dataloader = build_pedes_data(config)  # 调用 build_pedes_data 函数，构建训练和测试数据加载器（DataLoader）
    train_loader = dataloader['train_loader']
    num_classes = len(train_loader.dataset.pairs)


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

    # #定义损失函数
    criterion = torch.nn.CrossEntropyLoss()

    #绘图
    test_loss_list = []
    test_acc_list = []

    #early stopping
    best_acc = 0.0
    # patience = 30
    # counter = 0
    #训练
    for epoch in range(config.schedule.epoch):
        print()
        if is_using_distributed():
            dataloader['train_sampler'].set_epoch(epoch)

        start_time = time.time()
        model.train()

        for i, batch in enumerate(train_loader):
            # 学习率调度
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[it] * param_group['ratio']
                # param_group['lr'] = 1.0e-5
            # 软标签比例
            if epoch == 0:
                alpha = config.model.softlabel_ratio * min(1.0, i / len(train_loader))
            else:
                alpha = config.model.softlabel_ratio

            with torch.autocast(device_type='cuda'):  # 这里使用gpu：cuda所以我把它注释掉
                outputs = model(batch, alpha)  # 这里一开始是不行的，但是我把batch_size改小之后就可以运行了
                ids = batch['id'].long().to(config.device)  # 这行代码的主要功能是从输入数据字典 input 中提取样本的 ID 信息，
                loss = criterion(outputs, ids) # 这里的ids不知道和学姐代码里面的label一个不

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
                info_str += f", loss: {loss.item():.4f}"
                info_str += f", Base Lr: {param_group['lr']:.2e}"
                print(info_str)

        if is_master():#分布式情况下用，可以注释掉
            # end_time = time.time()
            # time_per_batch = (end_time - start_time) / (i + 1)  # 计算每个 batch 的平均耗时和吞吐量，监控训练效率。
            # print("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
            #       .format(epoch + 1, time_per_batch, train_loader.batch_size / time_per_batch))

            # eval_result = test(model.module, dataloader['test_loader'], 77,config.device)  # 这里之前没有用并行化处理过，即没有被包装在并行化容器中，所以直接传model即可
            loss,accuracy, precision, recall, f1 = test(model, dataloader['test_loader'], config.experiment.text_length, config.device)
            # rank_1, rank_5, rank_10, map = eval_result['r1'], eval_result['r5'], eval_result['r10'], eval_result['mAP']
            # print('Acc@1 {top1:.5f} Acc@5 {top5:.5f} Acc@10 {top10:.5f} mAP {mAP:.5f}'.format(top1=rank_1, top5=rank_5,

            print(f"Epoch: {epoch+1}, "f"Test Accuracy: {accuracy:.4f}, "f"Test Precision: {precision:.4f}, "f"Test Recall: {recall:.4f}, "f"Test F1 Score: {f1:.4f}")
            test_loss_list.append(loss)
            test_acc_list.append(accuracy)

            torch.cuda.empty_cache()
            if best_rank_1 < accuracy:
                best_rank_1 = accuracy
                best_epoch = epoch
                counter = 0
                save_obj = {
                    # 'model': model.module.state_dict(),
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                }
                torch.save(save_obj, os.path.join(config.model.saved_path, 'checkpoint_best.pth'))
            # else:
                # counter += 1
                # if counter >= patience:
                #     print(f"Early stopping at epoch {epoch}")
                #     break

    print(f"best Acc@1: {best_rank_1} at epoch {best_epoch + 1}")
    # ==== 画拟合曲线 ====
    test_loss_list = [x.item() if isinstance(x, torch.Tensor) else x for x in test_loss_list]
    plt.figure(figsize=(12, 5))

    # === Loss 曲线 ===
    plt.subplot(1, 2, 1)
    plt.plot(test_loss_list, label='Test Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)

    # === Accuracy 曲线 ===
    plt.subplot(1, 2, 2)
    plt.plot(test_acc_list, label='Test Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

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

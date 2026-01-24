import torch
import torch.nn.functional as F

#这段代码实现了 SimCLR损失函数的计算。SimCLR 是一种无监督学习方法，用于学习图像的特征表示。其核心思想是通过对比同一图像的不同增强视图（正样本对）和不同图像的视图（负样本对）来优化模型，使得正样本对在特征空间中更接近，负样本对更远离
def compute_simclr_loss(logits_a, logits_b, logits_a_gathered, logits_b_gathered, labels, temperature):
    sim_aa = logits_a @ logits_a_gathered.t() / temperature #除以 temperature（温度参数）主要是为了调整相似度得分的分布
    #这里用点积计算相似度，类似于余弦相似度，表征图像的距离远近
    sim_ab = logits_a @ logits_b_gathered.t() / temperature
    sim_ba = logits_b @ logits_a_gathered.t() / temperature
    sim_bb = logits_b @ logits_b_gathered.t() / temperature
    #用于对比第一组增强视图内不同样本之间的相似度，在后续损失计算中，需要通过掩码操作排除自身与自身的相似度得分，避免模型将样本自身作为正样本进行错误学习
    #正样本对：同一图像的两个增强视图（例如，样本 1 的 logits_a 与 logits_b 中的对应视图）的相似度得分位于 sim_ab 的对角线上（如 sim_ab_11）。负样本对：其他所有样本的相似度得分（包括组内和组间）。
    masks = torch.where(F.one_hot(labels, logits_a_gathered.size(0)) == 0, 0, float('-inf'))#当独热编码矩阵中的元素为 0（即 condition 为 True）时，掩码矩阵对应位置的值为 0；当独热编码矩阵中的元素为 1（即 condition 为 False）时，掩码矩阵对应位置的值为负无穷大 float('-inf'),在后续的计算中，将这个掩码矩阵加到相似度得分矩阵上，那些对应负无穷大的位置在进行 softmax 等操作时，会使得这些位置的概率趋近于 0，从而达到屏蔽样本自身相似度得分的目的
    #在进行对比学习时，模型需要区分正样本对（同一图像的不同增强视图）和负样本对（不同图像的视图）
    sim_aa += masks
    sim_bb += masks
    sim_a = torch.cat([sim_ab, sim_aa], 1)#这样做也是为了将正样本对（sim_ba 中包含同一图像不同增强视图的相似度得分）和负样本对（sim_bb 中包含不同图像第二组增强视图之间的相似度得分）的相似度得分整合，用于后续损失计算，让模型更好地区分正样本对和负样本对
    sim_b = torch.cat([sim_ba, sim_bb], 1)
    loss_a = F.cross_entropy(sim_a, labels)#是标记正样本对的标签。在 SimCLR 中，正样本对是指同一图像的不同增强视图，labels 用于告诉模型哪些是正样本对
    #交叉熵损失（Cross-Entropy Loss）常用于分类任务，衡量预测概率分布与真实标签分布之间的差异。在对比学习中，它被巧妙地用于迫使模型将正样本对的相似度得分最大化，负样本对的得分最小化
    #标记正样本对的位置：对于每个样本 i，其正样本对的相似度得分在 sim_ab 中位于第 i 列（假设 logits_b_gathered 包含所有进程的样本
    loss_b = F.cross_entropy(sim_b, labels)
    return (loss_a + loss_b) * 0.5

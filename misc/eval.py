import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch.nn.functional as F
# import clip
from text_utils.tokenizer import tokenize


# @torch.no_grad()
def test(model,test_loader , max_length, device):
    # switch to evaluate mode

    model.eval()
    all_preds = []
    all_labels = []
    # 定义损失函数
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():  # 只在需要的地方关闭梯度计算
        for batch in test_loader:
            # images = batch['image'].to(device)
            # text_inputs = batch['text'].to(device)
            # print("Batch size:", len(batch))
            # print("Batch content:", batch)

            labels = batch['id'].to(device)
            # 前向传播
            outputs = model(batch,1)
            loss = criterion(outputs, labels)
            # 获取预测结果
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())



    # 计算各项评估指标
    print(f"loss: {loss}")
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)



    return loss,accuracy, precision, recall, f1

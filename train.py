# 权重初始化，默认xavier

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
import time

from sklearn.metrics import precision_score, recall_score, f1_score

from utils import get_time_dif, multilabel_accuracy
from torch import nn

from pytorch_pretrained import BertAdam


def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    print('Training...')
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    train_loss = []
    train_accuracy =[]
    test_accuracy = []
    test_loss = []
    model.train()
    print('Training...start')
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            criterion = nn.BCELoss()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 10 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = (outputs >= 0.5).float()
                train_accuracy.append(multilabel_accuracy(labels, predic))
                print(predic[:3])
                print(true[:3])
                train_f1 = metrics.f1_score(true, predic, average='micro')
                dev_f1, dev_loss, accuracy, precision, recall = evaluate(config, model, dev_iter)
                train_loss.append(loss.item())
                test_accuracy.append(accuracy)
                test_loss.append(dev_loss)
                if dev_loss < dev_best_loss:
                    print(f"dev_loss :{dev_loss}<-{dev_best_loss}")
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train F1: {2:>6.2%}, accuracy: {3:>6.2%}, precision: {4:>6.2%}, recall: {5:>6.2%}, Dev Loss: {6:>5.2},  Dev F1: {7:>6.2%},  Time: {8} {9}'
                print(msg.format(total_batch, loss.item(), train_f1, accuracy, precision, recall, dev_loss, dev_f1, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    plt.figure(figsize=(10, 5))
    # 绘制 loss 值
    plt.plot(train_loss, label='Training Loss')
    # 绘制 accuracy 值
    plt.plot(test_accuracy, label='Test Accuracy')
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.plot(test_loss, label='Test Loss')
    # 添加标签和标题
    plt.xlabel('Batch Number')
    plt.ylabel('Value')
    plt.title('Training Test')
    # 添加图例
    plt.legend()
    # 显示图表
    plt.show()
    print('Training...end')
    # test(config, model, test_iter)


# def test(config, model, test_iter):
#     # test
#     model.load_state_dict(torch.load(config.save_path))
#     model.eval()
#     start_time = time.time()
#     test_f1, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
#     msg = 'Test Loss: {0:>5.2},  Test F1: {1:>6.2%}'
#     print(msg.format(test_loss, test_f1))
#     print("Precision, Recall and F1-Score...")
#     print(test_report)
#     print("Confusion Matrix...")
#     print(test_confusion)
#     time_dif = get_time_dif(start_time)
#     print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int).reshape(0, config.num_classes)  # 假设你有一个 num_classes 配置
    labels_all = np.array([], dtype=int).reshape(0, config.num_classes)  # 同上
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            criterion = nn.BCELoss()
            loss = criterion(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = (outputs >= 0.5).float()
            labels_all = np.vstack((labels_all, labels))
            predict_all = np.vstack((predict_all, predic))


    train_f1 = f1_score(labels_all, predict_all, average='micro', zero_division=0)
    precision = precision_score(labels_all, predict_all, average='macro', zero_division=0)
    recall = recall_score(labels_all, predict_all, average='macro', zero_division=0)
    accuracy = multilabel_accuracy(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion_matrices = []
        for i in range(labels_all.shape[1]):
            confusion = metrics.confusion_matrix(labels_all[:, i], predict_all[:, i])
            confusion_matrices.append(confusion)
        return train_f1, loss_total / len(data_iter), accuracy, precision, recall, report, confusion_matrices
    return train_f1, loss_total / len(data_iter), accuracy, precision, recall
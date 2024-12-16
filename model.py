import os

import torch
from torch import nn
from transformers import BertModel, BertTokenizer

from dataset import topic_categories

os.environ['TRANSFORMERS_CACHE'] = '/Users/zuohao/PycharmProjects/StudyBertClassify/study/cache'

class Config(object):

    """配置参数"""
    def __init__(self, dataset=''):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'                                # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = topic_categories                                           # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        print(self.device)
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                          # epoch数 设置为1个，害怕跑不动
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 128                                             # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-6 * 0.1                                # 学习率
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.hidden_size = 768

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        for param in self.bert.parameters():
            param.requires_grad = True
        # 转化为类别维度的数据
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子

        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        outputs = self.bert(context, attention_mask=mask, output_hidden_states=False)
        # 获取最后一层的隐藏状态和池化输出
        pooled = outputs.pooler_output
        # dropout_output = self.dropout(pooled)
        out = self.fc(pooled)
        return torch.sigmoid(out)
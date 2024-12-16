# 1.获取数据 训练 测试
# 2.模型选择使用
# 3.训练和验证、并可视化损失函数，保存模型参数
# 4.验证
# 5.使用
import torch

from model import Config, Model

if __name__ == '__main__':
    # 假设你有自己的模型文件 'bert.ckpt'
    config = Config("/Users/zuohao/PycharmProjects/StudyBertClassify/study")
    model = Model(config).to(config.device)
    model.load_state_dict(torch.load("/saved_dict/bert.ckpt", weights_only=True))
    text = "这辆车价格挺便宜"
    inputs = config.tokenizer.tokenize(text)
    topic_categories = ["动力", "价格", "内饰", "配置", "安全性", "外观", "操控", "油耗", "空间", "舒适性"]
    # 获取模型输出
    print(inputs)
    outputs = model(**inputs)
    predic = (outputs >= 0.5).float()
    res = []
    index = 0
    for p in predic:
        if p == 1:
            res.append(topic_categories[index])
        index += 1
    print(outputs.logits.shape)
    print(outputs)

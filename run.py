from model import Config, Model
from train import train
from utils import build_dataset, build_iterator
import numpy as np
import torch


if __name__ == '__main__':
    config = Config("/Users/zuohao/PycharmProjects/StudyBertClassify/study")
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    #数据集
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = []
    test_iter = build_iterator(test_data, config)

    model = Model(config).to(config.device)
    model.load_state_dict(
        torch.load("/saved_dict/bert.ckpt", weights_only=True))
    train(config, model, train_iter, test_iter, dev_iter)


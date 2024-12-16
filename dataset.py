from torch.utils.data import Dataset

# 类别 emotion_categories[0, 1, -1]对应[中立、正向、负向]
topic_categories = ["动力", "价格", "内饰", "配置", "安全性", "外观", "操控", "油耗", "空间", "舒适性"]
topic_map = {category: index for index, category in enumerate(topic_categories)}
emotion_categories = [0, 1, -1]

# 加载数据
class MyDataset(Dataset):

    def __init__(self, data_file):

        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f.read().split('\n')):
                if not line:
                    break
                parts = line.split('	')
                sentence = parts[0].replace(' ', '')
                tags = parts[1].split('#')
                labels = tags[0]
                Data[idx] = {
                    'sentence': sentence,
                    'labels': labels
                }
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

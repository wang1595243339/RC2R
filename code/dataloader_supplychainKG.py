import pickle
from os import listdir
from os.path import join

import dgl
import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, file_paths):
        super().__init__()
        self.datasets = []
        for file_path in file_paths:
            with open(file_path, 'rb') as fr:
                self.datasets.append(pickle.load(fr))

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, item):
        return self.datasets[item]


def collate(samples):
    graphs, texts, node_feats, label01s, nce_labels = [], [], [], [], []
    for sample in samples:
        g = sample['graph']
        g = dgl.add_self_loop(g)
        graphs.append(g)

        # node_feat = sample['node_feature']
        # node_feats.append(node_feat)
        text = sample['text']
        texts.append(text)

        label01 = sample['label']
        label01s.append(label01)

        nce_label = sample['nce_label']
        nce_labels.append(nce_label)

    # return dgl.batch(graphs), torch.cat(node_feats, dim=0), texts
    return dgl.batch(graphs), texts, label01s, nce_labels


def random_split(data_dir="E:/Projects/casual_reasoning/data/1000triples/text_g_node_label/", limit=2,
                 train_ratio=0.5):
    data_files = listdir(data_dir)[:limit]
    np.random.shuffle(data_files)
    train_size = int(limit * train_ratio)
    train_files = [join(data_dir, _) for _ in data_files[:train_size]]
    test_files = [join(data_dir, _) for _ in data_files[train_size:]]
    return MyDataset(train_files), MyDataset(test_files)

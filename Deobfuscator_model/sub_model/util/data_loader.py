"""

    加载自制数据集

"""

import csv
import torch
from conf import batch_size
from torch.utils.data import Dataset, DataLoader
import copy # you'd better to use deep copy when you want copy a list 

csv.field_size_limit(131072*64) # default: 131072
csv_file_dataset = "/home/pingmu123/torch-mlir/Deobfuscator_model/sub_model/dataset/dataset.csv"


csv_reader = csv.reader(open(csv_file_dataset, encoding='utf-8'))
next(csv_reader) # jump headLine

data = []
label = []
for row in csv_reader:
    data.append(row[0])
    label.append(row[1])


Data = copy.deepcopy(data)
Label = copy.deepcopy(label)

# myDataset
class myDataset(Dataset):
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label

    def __len__(self):
        return len(self.Data)
    
    def __getitem__(self, index):
        data = self.Data[index]
        label = self.Label[index]
        return data, label
dataset = myDataset(Data, Label)
print('dataset size:', dataset.__len__())


# Partition
train_size = int(len(dataset) * 0.6)
valid_size = int(len(dataset) * 0.2)
test_size = len(dataset) - train_size - valid_size
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

# data_loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

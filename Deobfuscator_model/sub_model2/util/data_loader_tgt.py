"""

    加载自制数据集

"""
import torch
from torch.utils.data import Dataset, DataLoader
import copy
import csv
csv.field_size_limit(2**30) # default: 131072

# file path
csv_file_path = "/home/pingmu123/torch-mlir/Deobfuscator_model/dataSet/"
# csv_file2 = csv_file_path + "token_no_ob_info.csv"
csv_file2 = csv_file_path + "token_with_ob_info.csv"


csv_reader = csv.reader(open(csv_file2, encoding='utf-8'))
next(csv_reader) # jump headLine



data = []
for row in csv_reader:
    data.append(row[1])


Data = copy.deepcopy(data)
# myDataset
class myDataset(Dataset):
    def __init__(self, Data):
        self.Data = Data

    def __len__(self):
        return len(self.Data)
    
    def __getitem__(self, index):
        data = self.Data[index]
        return data
dataset = myDataset(Data)
print('dataset size:', dataset.__len__())


# Partition
train_size = int(len(dataset) * 0.6)
valid_size = int(len(dataset) * 0.2)
test_size = len(dataset) - train_size - valid_size
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

# data_loader
train_loader_tgt = DataLoader(train_dataset, batch_size=1, shuffle=False)
valid_loader_tgt = DataLoader(valid_dataset, batch_size=1, shuffle=False)
test_loader_tgt = DataLoader(test_dataset, batch_size=1, shuffle=False)



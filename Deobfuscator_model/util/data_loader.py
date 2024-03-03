"""

    加载自制数据集

"""

from conf import *
import csv
import torch
from torch.utils.data import Dataset, DataLoader
import copy # you'd better to use deep copy when you want copy a list 

csv.field_size_limit(131072*64) # default: 131072
csv_file_path = "/home/pingmu123/torch-mlir/Deobfuscator_model/dataSet/"
csv_file1 = csv_file_path + "data.csv"
csv_file2_origin = csv_file_path + "token_origin.csv"
csv_file2_no_ob_info = csv_file_path + "token_no_ob_info.csv"
csv_file2_with_ob_info = csv_file_path + "token_with_ob_info.csv"



csv_reader = csv.reader(open(csv_file2_with_ob_info, encoding='utf-8'))
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
valid_size = int(len(dataset) * 0.01)
test_size = len(dataset) - train_size - valid_size
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

# data_loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# coding=gbk
"""
author(作者): Channing Xie(谢琛)
time(时间): 2020/5/31 11:54
filename(文件名): parallel_training.py
function description(功能描述):
...
    此代码实现pytorch的并行计算。当有多个GPU可供计算时，可以使用 model = nn.DataParallel(model) 并行计算
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

input_size = 5
output_size = 2

batch_size = 30
data_size = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.len


rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size), batch_size=batch_size, shuffle=True)


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, Input: torch.Tensor):
        output = self.fc(Input)
        print("\tIn Model: input size", Input.size(),
              "output size", output.size())  # 单个模型batch_size/n组数据
        return output


model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)  # n个GPU相当于复制了n个模型，每个模型跑batch_size/n组数据
model.to(device)

for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())  # 总的模型batch_size组数据

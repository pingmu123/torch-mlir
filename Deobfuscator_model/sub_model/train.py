import math
import time
import sys

sys.path.append("/home/pingmu123/torch-mlir/Deobfuscator_model/")

import torch
from torch import nn, optim
from torch.optim import Adam
import numpy as np

from conf import *
from sub_model.model.de_obfuscator import de_obfuscator
from util.epoch_timer import epoch_time
from sub_model.util.data_loader import train_loader, valid_loader, test_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


model = de_obfuscator().to(device)
print(
    f"The model has {count_parameters(model):,} trainable parameters"
)  # f'{num:控制格式}'
model.apply(init_weights)

optimizer = Adam(
    params=model.parameters(), lr=init_lr, weight_decay=weight_decay, eps=adam_eps
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    verbose=True,  # print a message
    factor=factor,
    patience=patience,
)

criterion = nn.CrossEntropyLoss()


def train(model, train_loader, optimizer, criterion, clip):  # iterator==dataloader?
    model.train()
    epoch_loss = 0
    for i, data in enumerate(train_loader):
        src_tensor, tgt_label = data
        optimizer.zero_grad()
        # both of src and tgt is a tuple which the string of number is batch_size
        # to tensor:
        src = []
        tgt = []
        for j in range(0, len(src_tensor)):
            src_t = (
                src_tensor[j]
                .replace("[", "")
                .replace("]", "")
                .replace("(", "")
                .replace(")", "")
                .replace(" ", "")
                .split(",")
            )
            for num in src_t:
                if len(num) > 0:  # num = ''
                    src.append(float(num))
            for c in tgt_label[j]:
                if c == "0":
                    tgt.append(int(0))
                if c == "1":
                    tgt.append(int(1))
        # src = torch.tensor(src).reshape(padding_shape)
        src = torch.tensor(src).reshape((len(src_tensor), 3, 224, 224))
        tgt = torch.tensor(tgt)
        # end of to tensor
        output = model(src)  # output: (batch_size, 2)
        # out_label = output.max(dim=1)[1] # (batch_size, 1)
        loss = criterion(output, tgt)
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        if i % 3 == 0:
            print(
                "step:",
                round((i / len(train_loader)) * 100, 2),
                "%, loss:",
                loss.item(),
            )

    return epoch_loss / len(train_loader)  # 求平均


def evaluate(model, valid_loader, criterion):
    model.eval()
    epoch_loss = 0
    correct = 0.0
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            src_tensor, tgt_label = data
            optimizer.zero_grad()
            # both of src and tgt is a tuple which have a string element
            # to tensor:
            src = []
            tgt = []
            for j in range(0, len(src_tensor)):
                src_t = (
                    src_tensor[j]
                    .replace("[", "")
                    .replace("]", "")
                    .replace("(", "")
                    .replace(")", "")
                    .replace(" ", "")
                    .split(",")
                )
                for num in src_t:
                    if len(num) > 0:  # num = ''
                        src.append(float(num))
                for c in tgt_label[j]:
                    if c == "0":
                        tgt.append(int(0))
                    if c == "1":
                        tgt.append(int(1))
            # src = torch.tensor(src).reshape(padding_shape)
            src = torch.tensor(src).reshape((len(src_tensor), 3, 224, 224))
            tgt = torch.tensor(tgt)
            # end of to tensor
            output = model(src)  # output: (batch_size, 2)
            out_label = output.max(dim=1)[1]  # (batch_size, 1)
            correct += out_label.eq(tgt.view_as(out_label)).sum().item()
            loss = criterion(output, tgt)
            epoch_loss += loss.item()

        acc_rate = 100.0 * correct / len(valid_loader.dataset)
        print(
            "Accuracy : {:.3f}%\n".format(100.0 * correct / len(valid_loader.dataset))
        )  # valid_loader.dataset: records count
        return epoch_loss / len(valid_loader), acc_rate


def run(total_epoch, best_loss):
    train_losses, test_losses, acc_rates = [], [], []
    for step in range(0, total_epoch):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, clip)
        valid_loss, acc_rate = evaluate(model, valid_loader, criterion)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)  # check and update lr

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        acc_rates.append(acc_rate)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            # save model all parameters:
            #   dict: model-valid_loss
            torch.save(model.state_dict(), "saved/model-{0}.pt".format(valid_loss))

        # save records of result
        f = open("result/train_loss.txt", "w")
        f.write(str(train_losses))
        f.close()

        f = open("result/acc_rate.txt", "w")
        f.write(str(lers))
        f.close()

        f = open("result/test_loss.txt", "w")
        f.write(str(test_losses))
        f.close()

        print(f"Epoch:{step+1} | Time: {epoch_mins}min(s)  {epoch_secs}second(s)")
        # \t: Tab
        # 长度为7，保留3位小数
        # PPL: 困惑度(perplexity), 可直接根据loss计算得到
        print(f"\tTrain Loss: {train_loss:.3f}")
        print(f"\tVal Loss: {valid_loss:.3f}")
        # print(f'\tLER: {ler:.3f}')


if __name__ == "__main__":
    run(total_epoch=30, best_loss=inf)

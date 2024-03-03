import math
import time

from torch import nn, optim
from torch.optim import Adam
import numpy as np

from conf import *
from models.model.transformer import Transformer

from util.epoch_timer import epoch_time
from util.get_words_dict import model_token_padding, model_token_beginning, model_token_ending, vocab_size, word2idx
from util.data_loader import train_loader, valid_loader, test_loader
from util.ler import get_ler

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)
padding_idx = word2idx[model_token_padding]
beginning_idx = word2idx[model_token_beginning]
ending_idx = word2idx[model_token_ending]

model = Transformer(padding_idx,
                    beginning_idx,
                    ending_idx,
                    tgt_voc_size=vocab_size,
                    d_model=d_model,
                    n_head=n_head,
                    max_len_src = max_len_src,
                    max_len_tgt = max_len_tgt,
                    ffn_hidden=ffn_hidden,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device) # 送入GPU

print(f'The model has {count_parameters(model):,} trainable parameters') # f'{num:控制格式}' 
model.apply(init_weights)
# model.load_state_dict(torch.load('saved/de_ob_model.pt'))

optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True, # print a message
                                                 factor=factor,
                                                 patience=patience)

criterion = nn.CrossEntropyLoss(ignore_index = padding_idx) # ignore_index = src_pad_idx


def train(model, train_loader, optimizer, criterion, clip): # iterator==dataloader?
    model.train()
    epoch_loss = 0
    for i, data in enumerate(train_loader):
        src, tgt = data
        optimizer.zero_grad()

        # both of src and tgt is a tuple which have a string element
        src_idx = []
        tgt_idx = []
        for batch_num in range(0, len(src)):
            temp_src_idx = []
            temp_tgt_idx = []
            src_words = src[batch_num].replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace("'", '').split(', ')
            tgt_words = tgt[batch_num].replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace("'", '').split(', ')
            for word in src_words:
                temp_src_idx.append(word2idx[word])
                if len(temp_src_idx) == max_len_src: break
            for word in tgt_words:
                temp_tgt_idx.append(word2idx[word])
                if len(temp_tgt_idx) == max_len_tgt: break
            while(len(temp_src_idx)<max_len_src):
                temp_src_idx.append(padding_idx)
            while(len(temp_src_idx)>max_len_src):
                del(temp_src_idx[len(temp_src_idx)-1])
            while(len(temp_tgt_idx)<max_len_tgt):
                temp_tgt_idx.append(padding_idx)
            src_idx.append(temp_src_idx)
            tgt_idx.append(temp_tgt_idx)
        while len(src_idx) < batch_size:
            temp = [padding_idx for _ in range(0, max_len_src)]
            src_idx.append(temp)
        while len(tgt_idx) < batch_size:
            temp = [padding_idx for _ in range(0, max_len_tgt)]
            tgt_idx.append(temp)
        src = torch.tensor(src_idx).contiguous().view(batch_size, max_len_src)
        tgt = torch.tensor(tgt_idx).contiguous().view(batch_size, max_len_tgt)

        # print(tgt[0][1].dtype == torch.int64)
        # exit()

        output = model(src, tgt)
        output_reshape = output.contiguous().view(-1, output.shape[-1]) # [batch, length, tgt_voc_size] -> [batch * length, tgt_voc_size]

        loss = criterion(output_reshape, torch.tensor(tgt_idx).flatten())
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        for j in range(0, batch_size):
            try:
                out_idx = output[j].max(dim=1)[1] # [1]: 概率最大的词所在的索引（维度编号就是idx) out_idx: [length, ]
                # print(out_idx[0:50])
                # print(tgt[j:j+1, 0:50])
            except:
                pass # pass while error
        print('step:', round((i/len(train_loader))*100, 2), '%, loss:', loss.item())

    return epoch_loss / len(train_loader) # 求平均


def evaluate(model, valid_loader, criterion):
    model.eval()
    epoch_loss = 0
    batch_ler = []
    with torch.no_grad():
        for data in valid_loader:
            src, tgt = data
            # both of src and tgt is a tuple which have a string element
            src_idx = []
            tgt_idx = []
            for batch_num in range(0, len(src)):
                temp_src_idx = []
                temp_tgt_idx = []
                src_words = src[batch_num].replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace("'", '').split(', ')
                tgt_words = tgt[batch_num].replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace("'", '').split(', ')
                for word in src_words:
                    temp_src_idx.append(word2idx[word])
                    if len(temp_src_idx) == max_len_src: break
                for word in tgt_words:
                    temp_tgt_idx.append(word2idx[word])
                    if len(temp_tgt_idx) == max_len_tgt: break
                while(len(temp_src_idx)<max_len_src):
                    temp_src_idx.append(padding_idx)
                while(len(temp_src_idx)>max_len_src):
                    del(temp_src_idx[len(temp_src_idx)-1])
                while(len(temp_tgt_idx)<max_len_tgt):
                    temp_tgt_idx.append(padding_idx)
                src_idx.append(temp_src_idx)
                tgt_idx.append(temp_tgt_idx)
            while len(src_idx) < batch_size:
                temp = [padding_idx for _ in range(0, max_len_src)]
                src_idx.append(temp)
            while len(tgt_idx) < batch_size:
                temp = [padding_idx for _ in range(0, max_len_tgt)]
                tgt_idx.append(temp)
            src = torch.tensor(src_idx).contiguous().view(batch_size, max_len_src)
            tgt = torch.tensor(tgt_idx).contiguous().view(batch_size, max_len_tgt)

            output = model(src, tgt)
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            
            loss = criterion(output_reshape, torch.tensor(tgt_idx).flatten())
            epoch_loss += loss.item()
            
            # LER
            total_ler = []
            for j in range(0, batch_size):
                try:
                    pre_idx = output[j].max(dim=1)[1] # [1]: 概率最大的词所在的索引（维度编号就是idx
                    tgt_idx = tgt[j:j+1, :].flatten()
                    print(pre_idx[0:100])
                    print(tgt_idx[0:100])
                    ler = get_ler(pre_idx, tgt_idx)
                    total_ler.append(ler)
                except:
                    print('ler compute error!') 
            total_ler = sum(total_ler) / (len(total_ler)+0.00000000001) # average LER of this batch
            batch_ler.append(total_ler)
        batch_ler = sum(batch_ler) / len(batch_ler) # average LER of all batches

        return epoch_loss / len(valid_loader), batch_ler
    
def run(total_epoch, best_loss):
    train_losses, test_losses, lers = [], [], []
    for step in range(0, total_epoch):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, clip)
        valid_loss, ler = evaluate(model, valid_loader, criterion)
        # test_loader?
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss) # check and update lr
        
        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        lers.append(ler)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if(train_loss < best_loss):
            best_loss = train_loss
            # save model all parameters:
            #   dict: model-valid_loss
            torch.save(model.state_dict(), 'saved/de_ob_model.pt')
        
        
        # save records of result
        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/LER.txt', 'w')
        f.write(str(lers))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print(f'Epoch:{step+1} | Time: {epoch_mins}min(s)  {epoch_secs}second(s)')
        # \t: Tab
        # 长度为7，保留3位小数
        # PPL: 困惑度(perplexity), 可直接根据loss计算得到
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tLER: {ler:.3f}')

if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)


        
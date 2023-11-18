import math
import time

from torch import nn, optim
from torch.optim import Adam
import numpy as np

from conf import *
from models.model.transformer import Transformer

from util.epoch_timer import epoch_time
from util.tokenizer import model_token_padding
from util.token2vec import origin_model_vec2token, origin_model_token2idx, origin_model_token2vec_size
from util.data_loader import train_loader, valid_loader, test_loader
from util.ler import get_ler

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)

model = Transformer(# tgt_sos_idx=tgt_sos_idx,
                    d_model=d_model,
                    tgt_voc_size=origin_model_token2vec_size,
                    max_len_src = max_len_src,
                    max_len_tgt = max_len_tgt,
                    ffn_hidden=ffn_hidden,
                    n_head=n_head,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device) # 送入GPU
print(f'The model has{count_parameters(model):,} trainable parameters') # f'{num:控制格式}' 
model.apply(init_weights)

optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True, # print a message
                                                 factor=factor,
                                                 patience=patience)

tgt_pad_vec = origin_model_token2idx[model_token_padding]
criterion = nn.CrossEntropyLoss(ignore_index = tgt_pad_vec) # ignore_index = src_pad_idx


def train(model, train_loader, optimizer, criterion, clip): # iterator==dataloader?
    model.train()
    epoch_loss = 0
    for i, data in enumerate(train_loader):
        src, tgt = data
        optimizer.zero_grad()
        # both of src and tgt is a tuple which have a string element
        # to tensor:
        src_str_list = src[0].replace('[', '').replace(']', '').replace(' ', '').split(',')
        tgt_str_list = tgt[0].replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace(' ', '').split(',')
        src = []
        for num in src_str_list:
            if len(num)>0: # num = ''
                src.append(float(num))
        tgt = []
        for num in tgt_str_list:
            if len(num)>0: # num = ''
                tgt.append(int(num))
        src = torch.tensor(src).contiguous().view(batch_size, -1, d_model)
        tgt = torch.tensor(tgt).contiguous().view(batch_size, -1, d_model)
        # end of to tensor

        output = model(src, tgt[:, :-1, :]) # 去掉'<EOS>'
        output_reshape = output.contiguous().view(-1, output.shape[-1]) # [batch, length, tgt_voc_size] -> [batch * length, tgt_voc_size]

        # tgt[batch, length, d_model] -> tgt[batch, length]: idx
        b, l, d = tgt.size()
        tgt_idx = torch.zeros((b, l)).contiguous().view(-1)
        tgt = tgt.contiguous().view(-1)
        for index in range(0, b*l):
            vec = tgt[index*d: index*d+d]
            vec = tuple(np.array(vec))
            tgt_idx[index] = origin_model_token2idx[origin_model_vec2token[vec]]
        tgt_idx = tgt_idx.contiguous().view(b, l)
        tgt_idx = tgt_idx[:, 1:].contiguous().view(-1)

        temp_idx = [origin_model_token2idx['<EOS>'] for _ in range(0, max_len_tgt)]
        temp_idx = torch.tensor(temp_idx)
        for index in range(0,len(tgt_idx)):
            temp_idx[index] = tgt_idx[index]
        tgt_idx = temp_idx
        
        loss = criterion(output_reshape, tgt_idx)
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        print('step:', round((i/len(train_loader))*100, 2), '%, loss:', loss.item())

    return epoch_loss / len(train_loader) # 求平均


def evaluate(model, valid_loader, criterion):
    model.eval()
    epoch_loss = 0
    batch_ler = [] # todo: use LER
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            src, tgt = data
            optimizer.zero_grad()
            # both of src and tgt is a tuple which have a string element
            # to tensor:
            src_str_list = src[0].replace('[', '').replace(']', '').replace(' ', '').split(',')
            tgt_str_list = tgt[0].replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace(' ', '').split(',')
            src = []
            for num in src_str_list:
                if len(num)>0: # num = ''
                    src.append(float(num))
            tgt = []
            for num in tgt_str_list:
                if len(num)>0: # num = ''
                    tgt.append(int(num))
            src = torch.tensor(src).contiguous().view(batch_size, -1, d_model)
            tgt = torch.tensor(tgt).contiguous().view(batch_size, -1, d_model)
            output = model(src, tgt[:, :-1, :])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            b, l, d = tgt.size()
            tgt_idx = torch.zeros((b, l)).contiguous().view(-1)
            tgt = tgt.contiguous().view(-1)
            for index in range(0, b*l):
                vec = tgt[index*d: index*d+d]
                vec = tuple(np.array(vec))
                tgt_idx[index] = origin_model_token2idx[origin_model_vec2token[vec]]
            tgt_idx = tgt_idx.contiguous().view(b, l)
            tgt_idx = tgt_idx[:, 1:].contiguous().view(-1)

            temp_idx = [origin_model_token2idx['<EOS>'] for _ in range(0, max_len_tgt)]
            temp_idx = torch.tensor(temp_idx)
            for index in range(0,len(tgt_idx)):
                temp_idx[index] = tgt_idx[index]
            tgt_idx = temp_idx
            
            loss = criterion(output_reshape, tgt_idx)
            epoch_loss += loss.item()

            # LER
            total_ler = []
            for j in range(0, batch_size):
                try:
                    out_words = output[j].max(dim=1)[1] # 概率在最大的词序列（维度编号就是idx）
                    tgt_words = tgt_idx
                    ler = get_ler(out_words, tgt_words)
                    total_ler.append(ler)
                except:
                    pass # pass while error
            total_ler = sum(total_ler) / len(total_ler) # average LER of this batch
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
    epoch_mins, epoch_secs = epoch_time(start_time, end_time) # todo

    if(valid_loss < best_loss):
        best_loss = valid_loss
        # save model all parameters:
        #   dict: model-valid_loss
        torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))
    
    
    # save records of result
    f = open('result/train_loss.txt', 'w')
    f.write(str(train_losses))
    f.close()

    f = open('result/LER.txt', 'w')
    f.write(str(lers))
    f.close()

    f = open('result/train_loss.txt', 'w')
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


        
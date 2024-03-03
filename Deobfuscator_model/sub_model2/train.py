"""
    word2vec:

        Train vector of word in token.csv.

"""
from torch import nn, optim
from torch.optim import Adam
import sys
sys.path.append('/home/pingmu123/torch-mlir/Deobfuscator_model/')
from util.get_words_dict import word2one_hot, word2idx, idx2word, model_token_padding, vocab_size
from sub_model2.util.data_loader import train_loader, valid_loader, test_loader
from sub_model2.util.data_loader_tgt import train_loader_tgt, valid_loader_tgt, test_loader_tgt
from sub_model2.model.my_word2vec import my_word2vec
from conf import *
import time
from util.epoch_timer import epoch_time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)

# word2vec
vocab_size = vocab_size
embedding_size = d_model
print(vocab_size, embedding_size)

word2vec_model = my_word2vec(vocab_size=vocab_size, embedding_size=embedding_size)

print(f'The model has {count_parameters(word2vec_model):,} trainable parameters') # f'{num:控制格式}' 
word2vec_model.apply(init_weights)

optimizer = Adam(params=word2vec_model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True, # print a message
                                                 factor=factor,
                                                 patience=patience)

pad_idx = word2idx[model_token_padding]
criterion = nn.CrossEntropyLoss() # ignore_index = src_pad_idx

def train(model, train_loader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, data in enumerate(train_loader):
        src_batch = data
        optimizer.zero_grad()
        src = []
        for batch_num in range(0, len(src_batch)):
            src_s = src_batch[batch_num].replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace("'", '').split(', ')
            src_idx = []
            for word in src_s:
                src_idx.append(word2idx[word])
            length = len(src_idx)
            for _ in range(0, max_len-length):
                src_idx.append(word2idx[model_token_padding])
            if length <= max_len:
                src.append(src_idx)
        if len(src) == 0: continue
        src = torch.tensor(src)
        sub_loss = 0

        for pos in range(0, max_len):
            cur_word = src[:, pos:pos+1].tolist()
            src_one_hot = []
            for j in range(0, len(cur_word)):
                src_one_hot.append(word2one_hot[idx2word[cur_word[j][0]]])
            # while count of left words of this pos less than n_gram, padding with '<PAD>'
            # right side is similar to left side
            left_words = src[:, max(pos - n_gram, 0): pos].tolist()
            for j in range(0, len(left_words)): # len(left_words) == batch_size
                while len(left_words[j]) < n_gram: 
                    left_words[j].insert(0, word2idx[model_token_padding])
            right_words = src[:, pos+1: min(max_len, pos + n_gram)].tolist()
            for j in range(0, len(right_words)):
                while len(right_words[j]) < n_gram:
                    right_words[j].append(word2idx[model_token_padding])
            other_words = []
            for j in range(0, len(src_batch)):
                other_words.append(left_words[j] + right_words[j])
            output = word2vec_model(torch.tensor(src_one_hot))  # out_put: [batch, vocab_size]
                                                                # other_words: [batch, 2*n_gram]
            tgt = torch.tensor(other_words)
            loss = 0.0

            for j in range(0, 2*n_gram):   # other_words is more than 1
                tgt_idx = tgt[:, j:j+1].contiguous().view(-1) # [batch, 1]
                loss += criterion(output, tgt_idx)

            loss /= (2*n_gram) # average
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            sub_loss += loss.item()
        sub_loss /= max_len
        print('step:', round((i/len(train_loader))*100, 2), '%, sub_loss:', sub_loss)

        epoch_loss += sub_loss

    return epoch_loss / len(train_loader) # 求平均


def evaluate(model, valid_loader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for data in valid_loader:
            src_batch = data
            src = []
            for batch_num in range(0, len(src_batch)):
                src_s = src_batch[batch_num].replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace("'", '').split(', ')
                src_idx = []
                for word in src_s:
                    src_idx.append(word2idx[word])
                length = len(src_idx)
                for _ in range(0, max_len-length):
                    src_idx.append(word2idx[model_token_padding])
                if length <= max_len:
                    src.append(src_idx)
            if len(src) == 0: continue
            src = torch.tensor(src)
            sub_loss = 0.0
            for pos in range(0, max_len):
                cur_word = src[:, pos:pos+1].tolist()
                src_one_hot = []
                for j in range(0, len(cur_word)):
                    src_one_hot.append(word2one_hot[idx2word[cur_word[j][0]]])
                # while count of left words of this pos less than n_gram, padding with '<PAD>'
                # right side is similar to left side
                left_words = src[:, max(pos - n_gram, 0): pos].tolist()
                for j in range(0, len(left_words)): # len(left_words) == batch_size
                    while len(left_words[j]) < n_gram: 
                        left_words[j].insert(0, word2idx[model_token_padding])
                right_words = src[:, pos+1: min(max_len, pos + n_gram)].tolist()
                for j in range(0, len(right_words)):
                    while len(right_words[j]) < n_gram:
                        right_words[j].append(word2idx[model_token_padding])
                other_words = []
                for j in range(0, len(src_batch)):
                    other_words.append(left_words[j] + right_words[j])
                output = word2vec_model(torch.tensor(src_one_hot))  # out_put: [batch, vocab_size]
                                                                    # other_words: [batch, 2*n_gram]
                tgt = torch.tensor(other_words)
                
                loss = 0.0
                for j in range(0, 2*n_gram):   # other_words is more than 1
                    tgt_idx = tgt[:, j:j+1].contiguous().view(-1) # [batch, 1]
                    loss += criterion(output, tgt_idx)
                loss /= (2*n_gram) # average

                sub_loss += loss.item()
            
            sub_loss /= max_len
        epoch_loss += sub_loss
    return epoch_loss / (len(valid_loader))
    
def run(total_epoch, best_loss):
    train_losses, test_losses = [], []
    for step in range(0, total_epoch):
        start_time = time.time()
        train_loss = train(word2vec_model, train_loader, optimizer, criterion, clip)
        valid_loss = evaluate(word2vec_model, valid_loader, criterion)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss) # check and update lr
        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if(valid_loss < best_loss):
            best_loss = valid_loss
            # save model all parameters:
            #   dict: model-valid_loss
            torch.save(word2vec_model.state_dict(), 'saved/model_word2vec.pt')
        
        # save records of result
        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        # f = open('result/LER.txt', 'w')
        # f.write(str(lers))
        # f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print(f'Epoch:{step+1} | Time: {epoch_mins}min(s)  {epoch_secs}second(s)')
        # \t: Tab
        # 长度为7，保留3位小数
        # PPL: 困惑度(perplexity), 可直接根据loss计算得到
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tVal Loss: {valid_loss:.3f}')
        # print(f'\tLER: {ler:.3f}')

        # W1 is the word embedding

    return

def run_tgt(total_epoch, best_loss):
    train_losses, test_losses = [], []
    for step in range(0, total_epoch):
        start_time = time.time()
        train_loss = train(word2vec_model, train_loader_tgt, optimizer, criterion, clip)
        valid_loss = evaluate(word2vec_model, valid_loader_tgt, criterion)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss) # check and update lr
        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if(valid_loss < best_loss):
            best_loss = valid_loss
            # save model all parameters:
            #   dict: model-valid_loss
            torch.save(word2vec_model.state_dict(), 'saved/model_word2vec_tgt.pt')
        
        # save records of result
        f = open('result/train_loss_tgt.txt', 'w')
        f.write(str(train_losses))
        f.close()

        # f = open('result/LER.txt', 'w')
        # f.write(str(lers))
        # f.close()

        f = open('result/test_loss_tgt.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print(f'Epoch:{step+1} | Time: {epoch_mins}min(s)  {epoch_secs}second(s)')
        # \t: Tab
        # 长度为7，保留3位小数
        # PPL: 困惑度(perplexity), 可直接根据loss计算得到
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tVal Loss: {valid_loss:.3f}')
        # print(f'\tLER: {ler:.3f}')

        # W1 is the word embedding

    return
if __name__ == '__main__':
    run(total_epoch=2, best_loss=inf)
    run_tgt(total_epoch=2, best_loss=inf)



        
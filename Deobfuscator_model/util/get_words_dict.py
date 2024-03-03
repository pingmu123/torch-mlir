import csv
csv.field_size_limit(2**30) # default: 131072

csv_file_path = "/home/pingmu123/torch-mlir/Deobfuscator_model/dataSet/"
csv_file2_no_ob_info = csv_file_path + "token_no_ob_info.csv"
csv_file2_with_ob_info = csv_file_path + "token_with_ob_info.csv"

csv_reader = csv.reader(open(csv_file2_with_ob_info, encoding='utf-8'))
next(csv_reader) # jump headLine

# dict
word2idx = {}
idx2word = {}
word2one_hot = {}

# special symbol
model_token_padding = '<PAD>' # padding
model_token_beginning = '<BOS>' # begin symbol
model_token_ending = '<EOS>' # end symbol

words = [model_token_padding]

for row in csv_reader:
    tokens = row[0].replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace("'", '').split(', ')
    for token in tokens:
        if token not in words:
            words.append(token) 


for id in range(0, len(words)):
    word2idx[words[id]] = id
for k, v in word2idx.items():
    idx2word[v] = k
for i in range(0, len(words)):
    one_hot_size = len(words)
    zero = [0 for _ in range(one_hot_size)]
    zero[i] = 1
    word2one_hot[words[i]] = tuple(zero)
vocab_size = len(words)
print('vocab_size: ', vocab_size)
'''
Data Engineering
'''

'''
D1. Import Libraries for Data Engineering
'''
import os
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import unicodedata

import torch
import random
# Setup seeds
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

'''
D2. Import Raw Dataset
'''

! wget http://www.manythings.org/anki/fra-eng.zip
! unzip fra-eng.zip

"""
# Raw Data Download Option

import urllib3
import zipfile
import shutil

http = urllib3.PoolManager()
url ='http://www.manythings.org/anki/fra-eng.zip'
filename = 'fra-eng.zip'
path = os.getcwd()
zipfilename = os.path.join(path, filename)
with http.request('GET', url, preload_content=False) as r, open(zipfilename, 'wb') as out_file:       
    shutil.copyfileobj(r, out_file)

with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
    zip_ref.extractall(path)
"""

# for using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
D3. [PASS] Tokenizer Install & import
''' 
# Keras Tokenizer is a tokenizer provided by default in tensorflow 2.X and is a word level tokenizer. It does not require a separate installation.

'''
D4. Define Hyperparameters for Data Engineering
'''
ENCODER_LEN  = 41
DECODER_LEN  = ENCODER_LEN
BATCH_SIZE   = 128
num_examples = 1024*12

'''
D5. Load and modifiy to pandas dataframe
'''
import pandas as pd

pd.set_option('display.max_colwidth', None)

train_df = pd.read_csv('fra.txt', names=['SRC', 'TRG', 'lic'], sep='\t')
del train_df['lic']
print(len(train_df))

train_df = train_df.loc[:, 'SRC':'TRG']
    
train_df.head()

train_df["src_len"] = ""
train_df["trg_len"] = ""
train_df.head()

# [OPT] Count the number of words
for idx in range(len(train_df['SRC'])):
    # initialize string
    text_eng = str(train_df.iloc[idx]['SRC'])

    # default separator: space
    result_eng = len(text_eng.split())
    train_df.at[idx, 'src_len'] = int(result_eng)

    text_fra = str(train_df.iloc[idx]['TRG'])
    # default separator: space
    result_fra = len(text_fra.split())
    train_df.at[idx, 'trg_len'] = int(result_fra)

print('Translation Pair :',len(train_df)) # Print Dataset Size

'''
D6. [OPT] Delete duplicated data
'''
train_df = train_df.drop_duplicates(subset = ["SRC"])
print('Translation Pair :',len(train_df)) # Print Dataset Size

train_df = train_df.drop_duplicates(subset = ["TRG"])
print('Translation Pair :',len(train_df)) # Print Dataset Size


'''
D7. [OPT] Select samples
'''
# Assign the result to a new variable.
is_within_len = (8 < train_df['src_len']) & (train_df['src_len'] < 20) & (8 < train_df['trg_len']) & (train_df['trg_len'] < 20)
# Filter the data that meets the condition and store it in a new variable.
train_df = train_df[is_within_len]

dataset_df_8096 = train_df.sample(n=num_examples, # number of items from axis to return.
          random_state=1234) # seed for random number generator for reproducibility

print('Translation Pair :',len(dataset_df_8096))   # Print Dataset Size

'''
D8. Preprocess and build list
'''
# Source Data
raw_src = []
for sentence in dataset_df_8096['SRC']:
    sentence = sentence.lower().strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # removing contractions
    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"he's", "he is", sentence)
    sentence = re.sub(r"she's", "she is", sentence)
    sentence = re.sub(r"it's", "it is", sentence)
    sentence = re.sub(r"that's", "that is", sentence)
    sentence = re.sub(r"what's", "that is", sentence)
    sentence = re.sub(r"where's", "where is", sentence)
    sentence = re.sub(r"how's", "how is", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can't", "cannot", sentence)
    sentence = re.sub(r"n't", " not", sentence)
    sentence = re.sub(r"n'", "ng", sentence)
    sentence = re.sub(r"'bout", "about", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    raw_src.append(sentence)

# Target Data
raw_trg = []

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')
    
for sentence in dataset_df_8096['TRG']:
    # 위에서 구현한 함수를 내부적으로 호출
    sentence = unicode_to_ascii(sentence.lower())

    # 단어와 구두점 사이에 공백을 만듭니다.
    # Ex) "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,¿])", r" \1", sentence)

    # (a-z, A-Z, ".", "?", "!", ",") 이들을 제외하고는 전부 공백으로 변환합니다.
    sentence = re.sub(r"[^a-zA-Z!.?]+", r" ", sentence)

    sentence = re.sub(r"\s+", " ", sentence)

    raw_trg.append(sentence)

print(raw_src[:5])
print(raw_trg[:5])

'''
D9. Add <SOS>, <EOS> for source and target
'''
SRC_df = pd.DataFrame(raw_src)
TRG_df = pd.DataFrame(raw_trg)

SRC_df.rename(columns={0: "SRC"}, errors="raise", inplace=True)
TRG_df.rename(columns={0: "TRG"}, errors="raise", inplace=True)
train_df = pd.concat([SRC_df, TRG_df], axis=1)

print('Translation Pair :',len(train_df)) # 리뷰 개수 출력
train_df.sample(10)

raw_src_df  = train_df['TRG']
raw_trg_df  = train_df['SRC']

src_sentence  = raw_src_df.apply(lambda x: "<SOS> " + str(x) + " <EOS>")
trg_sentence  = raw_trg_df.apply(lambda x: "<SOS> "+ x + " <EOS>")

'''
D10. Define tokenizer
'''
filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
oov_token = '<unk>'

# Define tokenizer
SRC_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters = filters, oov_token=oov_token)
TRG_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters = filters, oov_token=oov_token)

SRC_tokenizer.fit_on_texts(src_sentence)
TRG_tokenizer.fit_on_texts(trg_sentence)

src_to_index = SRC_tokenizer.word_index
index_to_src = SRC_tokenizer.index_word

tar_to_index = TRG_tokenizer.word_index
index_to_tar = TRG_tokenizer.index_word

n_enc_vocab = len(SRC_tokenizer.word_index) + 1
n_dec_vocab = len(TRG_tokenizer.word_index) + 1

print('Word set size of Encoder :',n_enc_vocab)
print('Word set size of Decoder :',n_dec_vocab)

'''
D11. Tokenizer test
'''
# Source Tokenizer
lines = [
  "It is winter and the weather is very cold.",
  "Will this Christmas be a white Christmas?",
  "Be careful not to catch a cold in winter and have a happy new year."
]
for line in lines:
    txt_2_ids = SRC_tokenizer.texts_to_sequences([line])
    ids_2_txt = SRC_tokenizer.sequences_to_texts(txt_2_ids)
    print("Input     :", line)
    print("txt_2_ids :", txt_2_ids)
    print("ids_2_txt :", ids_2_txt[0],"\n")

# Target Tokenizer
lines = [
  "C'est l'hiver et il fait très froid.",
  "Ce Noël sera-t-il un Noël blanc ?",
  "Attention à ne pas attraper froid en hiver et bonne année."
]
for line in lines:
    txt_2_ids = TRG_tokenizer.texts_to_sequences([line])
    ids_2_txt = TRG_tokenizer.sequences_to_texts(txt_2_ids)
    print("Input     :", line)
    print("txt_2_ids :", txt_2_ids)
    print("ids_2_txt :", ids_2_txt[0],"\n")

'''
D12. Tokenize
'''
# tokenize / encode integers / add start and end tokens / padding
tokenized_inputs      = SRC_tokenizer.texts_to_sequences(src_sentence)
tokenized_outputs     = TRG_tokenizer.texts_to_sequences(trg_sentence)

'''
D13. [EDA] Explore the tokenized datasets
'''

len_result = [len(s) for s in tokenized_inputs]

print('Maximum length of source : {}'.format(np.max(len_result)))
print('Average length of source : {}'.format(np.mean(len_result)))

plt.subplot(1,2,1)
plt.boxplot(len_result)
plt.subplot(1,2,2)
plt.hist(len_result, bins=50)
plt.show()


len_result = [len(s) for s in tokenized_outputs]

print('Maximum length of target : {}'.format(np.max(len_result)))
print('Average length of target : {}'.format(np.mean(len_result)))

plt.subplot(1,2,1)
plt.boxplot(len_result)
plt.subplot(1,2,2)
plt.hist(len_result, bins=50)
plt.show()

'''
D14. Pad sequences
'''

from tensorflow.keras.preprocessing.sequence import pad_sequences
tkn_sources = pad_sequences(tokenized_inputs,  maxlen=ENCODER_LEN, padding='post', truncating='post')
tkn_targets = pad_sequences(tokenized_outputs, maxlen=DECODER_LEN, padding='post', truncating='post')


'''
D15. Data type define
'''

tensors_src   = torch.tensor(tkn_sources).to(device)
tensors_trg   = torch.tensor(tkn_targets).to(device)

'''
D16. [EDA] Explore the Tokenized datasets
'''
print('Size of source language data(shape) :', tkn_sources.shape)
print('Size of target language data(shape) :', tkn_targets.shape)

# Randomly output the 0th sample
print(tkn_sources[0])
print(tkn_targets[0])

'''
D17. [PASS] Split Data
'''

'''
D18. Build dataset
'''

from torch.utils.data import TensorDataset   # 텐서데이터셋
from torch.utils.data import DataLoader      # 데이터로더

dataset    = TensorDataset(tensors_src, tensors_trg)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

'''
D19. [PASS] Define some useful parameters for further use
'''

'''
Model Engineering
'''

'''
M01. Import Libraries for Model Engineering
'''
from tqdm import tqdm, tqdm_notebook, trange

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

'''
M02. Define Hyperparameters for Model Engineering
'''
n_layers  = 2     # 6
hid_dim   = 256
pf_dim    = 1024
n_heads   = 8
dropout   = 0.3
pe_input  = 512
pe_target = 512
layer_norm_epsilon = 1e-12
N_EPOCHS  = 20

'''
M03. [PASS] Load datasets
'''

'''
M04. Build Transformer model
'''

""" 
C01. Sinusoid position encoding
"""
class get_sinusoid_encoding_table(nn.Module):
    '''The position of the entered word is added to the original vector information'''
    def __init__(self, position, hid_dim):
        super().__init__()

        self.hid_dim = hid_dim  # the original number of dimensions of the word vector

        # 입력 문장에서의 임베딩 벡터의 위치（pos）임베딩 벡터 내의 차원의 인덱스（i）
        pe = torch.zeros(position, hid_dim)
        pe = pe.to(device)

        for pos in range(position):
            for i in range(0, hid_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/hid_dim)))
                pe[pos, i + 1] = math.cos(pos /
                                          (10000 ** ((2 * i)/hid_dim)))

        # pe의 선두에 미니배치 차원을 추가한다
        self.pe = pe.unsqueeze(0)

        self.pe.requires_grad = False

    def forward(self, x):
        # 입력x와 Positonal Encoding을 더한다
        ret = math.sqrt(self.hid_dim)*x + self.pe[:, :x.size(1)]
        return ret


"""
C02. Scaled dot product attention
"""
class ScaledDotProductAttention(nn.Module):
    """Calculate the attention weights.
    query, key, value must have matching leading dimensions.
    key, value must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.
    
    query, key, value의 leading dimensions은 동일해야 합니다.
    key, value 에는 일치하는 끝에서 두 번째 차원이 있어야 합니다(예: seq_len_k = seq_len_v).
    MASK는 유형에 따라 모양이 다릅니다(패딩 혹은 미리보기(=look ahead)).
    그러나 추가하려면 브로드캐스트할 수 있어야 합니다.

    Args:
        query: query shape == (batch_size, n_heads, seq_len_q, depth)
        key: key shape     == (batch_size, n_heads, seq_len_k, depth)
        value: value shape == (batch_size, n_heads, seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
              to (batch_size, n_heads, seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights
    """
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask):

        # 1. MatMul Q, K-transpose. Attention score matrix.
        matmul_qk = torch.matmul(query, torch.transpose(key,2,3))

        # 2. scale matmul_qk
        # Divide by the root of dk.
        dk = key.shape[-1]
        scaled_attention_logits = matmul_qk / math.sqrt(dk)

        # 3. add the mask to the scaled tensor.
        # 마스킹. 어텐션 스코어 행렬의 마스킹 할 위치에 매우 작은 음수값을 넣는다.
        # 매우 작은 값이므로 소프트맥스 함수를 지나면 행렬의 해당 위치의 값은 0이 된다.
        # Masking. Put a very small negative value in the position to be masked in the attention score matrix.
        # Since it is a very small value, the value at the corresponding position in the matrix becomes 0 after passing the softmax function.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # 4. softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
        # attention weight : (batch_size, n_heads, sentence length of query, sentence length of key)
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        
        # 5. MatMul attn_prov, V
        # output : (batch_size, n_heads, sentence length of query, hid_dim/n_heads)
        output = torch.matmul(attention_weights, value)

        return output, attention_weights

"""
C03. Multi head attention
"""
class MultiHeadAttentionLayer(nn.Module):
    
    def __init__(self, hid_dim, n_heads):
        super(MultiHeadAttentionLayer, self).__init__()
        self.n_heads = n_heads
        assert hid_dim % self.n_heads == 0
        self.hid_dim = hid_dim
        
        # hid_dim divided by n_heads.
        self.depth = int(hid_dim/self.n_heads)
        
        # Define dense layers corresponding to WQ, WK, and WV
        self.q_linear = nn.Linear(hid_dim, hid_dim)
        self.k_linear = nn.Linear(hid_dim, hid_dim)
        self.v_linear = nn.Linear(hid_dim, hid_dim)

        self.scaled_dot_attn = ScaledDotProductAttention()
        
        # the original number of dimensions of the word vector
        self.out = nn.Linear(hid_dim, hid_dim)

    def split_heads(self, inputs, batch_size):
        """Split the last dimension into (n_heads, depth).
        Transpose the result such that the shape is (batch_size, n_heads, seq_len, depth)
        """
        inputs = torch.reshape(
            inputs, (batch_size, -1, self.n_heads, self.depth))
        return torch.transpose(inputs, 1,2)

    def forward(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = query.shape[0]
        
        # 1. Pass through the dense layer corresponding to WQ
        # q : (batch_size, sentence length of query, hid_dim)
        query = self.q_linear(query)
        
        # split head
        # q : (batch_size, n_heads, sentence length of query, hid_dim/n_heads)
        query = self.split_heads(query, batch_size)
        
        # 2. Pass through the dense layer corresponding to WK
        # k : (batch_size, sentence length of key, hid_dim)
        key   = self.k_linear(key)
        
        # split head
        # k : (batch_size, n_heads, sentence length of key, hid_dim/n_heads)
        key   = self.split_heads(key, batch_size)
        
        # 3. Pass through the dense layer corresponding to WV
        # v : (batch_size, sentence length of value, hid_dim)
        value = self.v_linear(value)
        
        # split head
        # v : (batch_size, n_heads, sentence length of value, hid_dim/n_heads)
        value = self.split_heads(value, batch_size)
        
        # 4. Scaled Dot Product Attention. Using the previously implemented function
        # (batch_size, n_heads, sentence length of query, hid_dim/n_heads)
        # attention_weights.shape == (batch_size, n_heads, seq_len_q, seq_len_k)
        scaled_attention, _ = self.scaled_dot_attn(query, key, value, mask)
        
        # (batch_size, sentence length of query, n_heads, hid_dim/n_heads)
        scaled_attention = torch.transpose(scaled_attention, 1,2)
        
        # 5. Concatenate the heads
        # (batch_size, sentence length of query, hid_dim)
        concat_attention = torch.reshape(scaled_attention,
                                      (batch_size, -1, self.hid_dim))
        
        # 6. Pass through the dense layer corresponding to WO
        # (batch_size, sentence length of query, hid_dim)
        outputs = self.out(concat_attention)

        return outputs

"""
C04. Positionwise Feedforward Layer
"""
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim):
        super(PositionwiseFeedforwardLayer, self).__init__()
        self.linear_1 = nn.Linear(hid_dim, pf_dim)
        self.linear_2 = nn.Linear(pf_dim, hid_dim)

    def forward(self, attention):
        output = self.linear_1(attention)
        output = F.relu(output)
        output = self.linear_2(output)
        return output

"""
C05. Encoder layer
"""
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        
        self.attn = MultiHeadAttentionLayer(hid_dim, n_heads)
        self.ffn  = PositionwiseFeedforwardLayer(hid_dim, pf_dim)
        
        self.layernorm1 = nn.LayerNorm(hid_dim)
        self.layernorm2 = nn.LayerNorm(hid_dim)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, inputs, padding_mask):
        
        # 1. Encoder mutihead attention is defined
        attention   = self.attn({'query': inputs, 'key': inputs, 'value': inputs, 'mask': padding_mask})
        attention   = self.dropout1(attention)
        
        # 2. 1 st residual layer
        attention   = self.layernorm1(inputs + attention)  # (batch_size, input_seq_len, hid_dim)
        
        # 3. Feed Forward Network
        ffn_outputs = self.ffn(attention)  # (batch_size, input_seq_len, hid_dim)
        
        ffn_outputs = self.dropout2(ffn_outputs)
        
        # 4. 2 nd residual layer
        ffn_outputs = self.layernorm2(attention + ffn_outputs)  # (batch_size, input_seq_len, hid_dim)

        # 5. Encoder output of each encoder layer
        return ffn_outputs

"""
C06. Encoder
"""
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.embedding    = nn.Embedding(n_enc_vocab, hid_dim)
        self.pos_encoding = get_sinusoid_encoding_table(pe_input, hid_dim)

        self.enc_layers   = EncoderLayer()
        self.dropout1     = nn.Dropout(dropout)

    def forward(self, x, padding_mask):

        # adding embedding and position encoding.
        # 1. Token Embedding
        emb = self.embedding(x)  # (batch_size, input_seq_len, hid_dim)
        emb *= math.sqrt(hid_dim)
        
        # 2. Sinusoidal positional Encoding
        emb = self.pos_encoding(emb)
        output = self.dropout1(emb)
        
        # 3. Self padding Mask is created from encoder input

        # 4. Encoder layers are stacked
        for i in range(n_layers):
            output = self.enc_layers(output, padding_mask)
            
        # 5. Final layer's output is the encoder output

        return output  # (batch_size, input_seq_len, hid_dim)
    
"""
C07. Decoder layer
"""
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()

        self.attn   = MultiHeadAttentionLayer(hid_dim, n_heads)
        self.attn_2 = MultiHeadAttentionLayer(hid_dim, n_heads)

        self.ffn = PositionwiseFeedforwardLayer(hid_dim, pf_dim)

        self.layernorm1 = nn.LayerNorm(hid_dim)
        self.layernorm2 = nn.LayerNorm(hid_dim)
        self.layernorm3 = nn.LayerNorm(hid_dim)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, inputs, enc_outputs, padding_mask, look_ahead_mask):
        
        # 1. 1st Encoder mutihead attention is defined. Q,K,V is same and it's comes from decoder input or previous decoder output
        attention1 = self.attn(
            {'query': inputs, 'key': inputs, 'value': inputs, 'mask': look_ahead_mask})
        attention1 = self.dropout1(attention1)
        
        # 2. 1st residual layer
        attention1 = self.layernorm1(inputs + attention1)
        
        # 3. 2nd Encoder mutihead attention is defined. Q comes from Multi-Head attention. K,V are same and comes from encoder output
        attention2 = self.attn_2(
            {'query': attention1, 'key': enc_outputs, 'value': enc_outputs, 'mask': padding_mask})
        attention2 = self.dropout2(attention2)

        # 4. 2nd residual layer
        attention2 = self.layernorm2(attention1 + attention2)  # (batch_size, target_seq_len, hid_dim)

        # 5. Feed Forward Network
        ffn_outputs = self.ffn(attention2)  # (batch_size, target_seq_len, hid_dim)
        ffn_outputs = self.dropout3(ffn_outputs)
        
        # 6. 3 rd residual layer
        ffn_outputs = self.layernorm3(attention2 + ffn_outputs)  # (batch_size, target_seq_len, hid_dim)

        # 7. Decoder output of each decoder layer
        return ffn_outputs  

"""
C08. Decoder
"""
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.embedding    = nn.Embedding(n_dec_vocab, hid_dim)
        self.pos_encoding = get_sinusoid_encoding_table(pe_target, hid_dim)
        self.dec_layers = DecoderLayer()
        self.dropout      = nn.Dropout(dropout)
        
    def forward(self, enc_output, dec_input, padding_mask, look_ahead_mask):

        # 1. Decoder input Token Embedding
        emb = self.embedding(dec_input)
        emb *= math.sqrt(hid_dim)
        
        # 2. Sinusoidal positional Encoding
        emb = self.pos_encoding(emb)
        output = self.dropout(emb)

        # 3. Padding mask is created from **encoder inputs** in this implementation
        # 4. Look ahead Mask is created from **decoder inputs**
        # 5. Decoder layers are stacked
        for i in range(n_layers):
            output = self.dec_layers(output, enc_output, padding_mask, look_ahead_mask)

    
        # 6. Final layer's output is the decoder output
        
        return output
    
"""
C09. Attention pad mask
"""
def create_padding_mask(x):
    input_pad = 0
    mask = (x == input_pad).float()
    mask = mask.unsqueeze(1).unsqueeze(1)
    # (batch_size, 1, 1, key의 문장 길이)
    return mask

""" 
C10. Attention decoder mask (Look Ahead Mask)
"""
def create_look_ahead_mask(seq):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = seq.shape[1]
    look_ahead_mask = torch.ones(seq_len, seq_len)
    
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1).to(device)
    # padding_mask = create_padding_mask(seq).to(device) # 패딩 마스크도 포함
    # return torch.maximum(look_ahead_mask, padding_mask)

    return look_ahead_mask

"""
C12. Transformer Class
"""
class Transformer(nn.Module):
    def __init__(self, n_enc_vocab, n_dec_vocab,
                 n_layers, pf_dim, hid_dim, n_heads,
                 pe_input, pe_target, dropout):
        super(Transformer, self).__init__()
        
        # Ecoder and Decoder
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fin_output = nn.Linear(hid_dim, n_dec_vocab)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, enc_inputs, dec_inputs):
        # 1. Encoder Padding Mask
        enc_padding_mask = create_padding_mask(enc_inputs)
        # 2. Dec Padding Mask
        dec_padding_mask = create_padding_mask(enc_inputs)
        # 3. Look ahead Mask
        look_ahead_mask  = create_look_ahead_mask(dec_inputs)
        dec_target_padding_mask = create_padding_mask(dec_inputs).to(device) # 패딩 마스크도 포함
        look_ahead_mask  = torch.maximum(dec_target_padding_mask, look_ahead_mask)

        enc_output = self.encoder(enc_inputs, enc_padding_mask)
        # 4. Encoder outputs are created from encoder model and it is given to the "Key" and "Value" of Multi-Head Attention 2 of decoder layers 
        dec_output = self.decoder(enc_output, dec_inputs, dec_padding_mask, look_ahead_mask)
        # 5. Decoder output is creadted from decoder model
        
        final_output = self.fin_output(dec_output)
        
        # 6. Final outputs are created. Then it is used or Language Model. In the official tutorial "Softmax" was missed
        return final_output

# Model Define for Training
model = Transformer(
    n_enc_vocab = n_enc_vocab,
    n_dec_vocab = n_dec_vocab,
    n_layers  = n_layers,
    pf_dim      = pf_dim,
    hid_dim     = hid_dim,
    n_heads     = n_heads,
    pe_input    = 512,
    pe_target   = 512,
    dropout     = dropout)

model.to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

# 네트워크 초기화
def initialize_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # Liner층의 초기화
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# TransformerBlock모듈의 초기화 설정
model.apply(initialize_weights)

import os.path

if os.path.isfile('./checkpoints/transformermodel.pt'):
    model.load_state_dict(torch.load('./checkpoints/transformermodel.pt'))

print('네트워크 초기화 완료')

# 손실 함수의 정의
criterion = nn.CrossEntropyLoss()

# 최적화 설정
# learning_rate = 2e-4
learning_rate = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

from IPython.display import clear_output
import datetime

Model_start_time = time.time()

# 학습 정의
def train(epoch, model, dataloader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    with tqdm_notebook(total=len(dataloader), desc=f"Train {epoch+1}") as pbar:    
        for batch_idx, samples in enumerate(dataloader):
            src_inputs, trg_outputs = samples

            with torch.set_grad_enabled(True):
                # Transformer에 입력
                logits_lm = model(src_inputs, trg_outputs)

                pad = torch.LongTensor(trg_outputs.size(0), 1).fill_(0).to(device)
                preds_id = torch.transpose(logits_lm,1,2)
                labels_lm = torch.cat((trg_outputs[:, 1:], pad), -1)

                optimizer.zero_grad()
                loss = criterion(preds_id, labels_lm)  # loss 계산
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                epoch_loss +=loss.item()

            pbar.update(1)
            # pbar.set_postfix_str(f"Loss {epoch_loss.result():.4f} Accuracy {train_accuracy.result():.4f}")
            # pbar.set_postfix_str(f"Loss {loss.result():.4f}")
            
    return epoch_loss / len(dataloader)

CLIP = 0.5

epoch_ = []
epoch_train_loss = []
# 네트워크가 어느정도 고정되면 고속화
torch.backends.cudnn.benchmark = True
# epoch 루프
best_epoch_loss = float("inf")

for epoch in range(N_EPOCHS):
    
    train_loss = train(epoch, model, dataloader, optimizer, criterion, CLIP)
    
    if train_loss < best_epoch_loss:
        if not os.path.isdir("checkpoints"):
            os.makedirs("checkpoints")
        best_epoch_loss = train_loss
        torch.save(model.state_dict(), './checkpoints/transformermodel.pt')

    epoch_.append(epoch)
    epoch_train_loss.append(train_loss)
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    
    # print('Epoch {0}/{1} Average Loss: {2}'.format(epoch+1, N_EPOCHS, epoch_loss))
    # clear_output(wait = True)

fig = plt.figure(figsize=(8,8))
fig.set_facecolor('white')
ax = fig.add_subplot()
ax.plot(epoch_,epoch_train_loss, label='Average loss')

ax.legend()
ax.set_xlabel('epoch')
ax.set_ylabel('loss')

plt.show()

# Predict the trained model
trained_model = Transformer(
    n_enc_vocab = n_enc_vocab,
    n_dec_vocab = n_dec_vocab,
    n_layers    = n_layers,
    pf_dim      = pf_dim,
    hid_dim     = hid_dim,
    n_heads     = n_heads,
    pe_input    = 512,
    pe_target   = 512,
    dropout     = dropout).to(device)
trained_model.load_state_dict(torch.load('./checkpoints/transformermodel.pt'))

def evaluate(text):
    text = SRC_tokenizer.texts_to_sequences([text])
    text = pad_sequences(text, maxlen=ENCODER_LEN, padding='post', truncating='post')
    decoder_input = [TRG_tokenizer.word_index['<sos>']]
    input  = torch.tensor(text).to(device)
    output = torch.tensor([decoder_input]).to(device)
    
    for i in range(DECODER_LEN):
        predictions = trained_model(input, output)
        predictions = predictions[:, -1:, :]
                            
        # PAD, UNK, START 토큰 제외
        predicted_id = torch.argmax(predictions[:,:,3:], axis=-1) + 3
        if predicted_id == 3:  # token id of <EOS>
            predicted_id = predicted_id
            break
        output = torch.cat((output, predicted_id),-1)
    return output

def predict(text):
    prediction = evaluate(text)
    out_list = prediction.tolist()
    out_list[0].pop(0)
    output_indexes = out_list[0]
    predicted_sentence = TRG_tokenizer.sequences_to_texts([output_indexes])
    
    return predicted_sentence

for idx in (11, 21, 31, 41, 51):
    print("Input        :", raw_src_df[idx])
    print("Prediction   :", predict(raw_src_df[idx]))
    print("Ground Truth :", raw_trg_df[idx],"\n")

'''
M13. [PASS] Explore the training result with test dataset
'''
    

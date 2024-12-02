!pip install tensorflow==2.15.0
import tensorflow as tf
print(tf.__version__)

!pip install sentencepiece

data_dir = "/content"

! pip list | grep sentencepiece

import sentencepiece as spm

'''
D1. Import Libraries for Data Engineering
'''
import csv
import os
import re
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import unicodedata

import torch
import random
from sklearn.model_selection import train_test_split

from IPython.display import display

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
D3. Tokenizer Install & import
''' 
# Keras Tokenizer is a tokenizer provided by default in tensorflow 2.X and is a word level tokenizer. It does not require a separate installation.

'''
D4. Define Hyperparameters for Data Engineering
'''
ENCODER_LEN = 41            # json_encode_length
DECODER_LEN = ENCODER_LEN   # json_decode_length
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
D9. Define dataframe
'''
SRC_df = pd.DataFrame(raw_src)
TRG_df = pd.DataFrame(raw_trg)

SRC_df.rename(columns={0: "SRC"}, errors="raise", inplace=True)
TRG_df.rename(columns={0: "TRG"}, errors="raise", inplace=True)
total_df = pd.concat([SRC_df, TRG_df], axis=1)

print('Translation Pair :',len(total_df)) # 리뷰 개수 출력

'''
D10. Define tokenizer
'''

with open('corpus_src.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(total_df['SRC']))

with open('corpus_trg.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(total_df['TRG']))

# This is the folder to save the data. Modify it to suit your environment.
data_dir = "/content"

corpus = "corpus_src.txt"
prefix = "nmt_src_vocab"
vocab_size = 4000
spm.SentencePieceTrainer.train(
    f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" + 
    " --model_type=bpe" +
    " --max_sentence_length=999999" +               # max sentence length
    " --pad_id=0 --pad_piece=[PAD]" +               # pad (0)
    " --unk_id=1 --unk_piece=[UNK]" +               # unknown (1)
    " --bos_id=2 --bos_piece=[BOS]" +               # begin of sequence (2)
    " --eos_id=3 --eos_piece=[EOS]" +               # end of sequence (3)
    " --user_defined_symbols=[SEP],[CLS],[MASK]")   # other additional tokens


corpus = "corpus_trg.txt"
prefix = "nmt_trg_vocab"
vocab_size = 4000
spm.SentencePieceTrainer.train(
    f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" + 
    " --model_type=bpe" +
    " --max_sentence_length=999999" +               # max sentence length
    " --pad_id=0 --pad_piece=[PAD]" +               # pad (0)
    " --unk_id=1 --unk_piece=[UNK]" +               # unknown (1)
    " --bos_id=2 --bos_piece=[BOS]" +               # begin of sequence (2)
    " --eos_id=3 --eos_piece=[EOS]" +               # end of sequence (3)
    " --user_defined_symbols=[SEP],[CLS],[MASK]")   # other additional tokens

for f in os.listdir("."):
    print(f)

vocab_src_file = f"{data_dir}/nmt_src_vocab.model"
vocab_src = spm.SentencePieceProcessor()
vocab_src.load(vocab_src_file)

vocab_trg_file = f"{data_dir}/nmt_trg_vocab.model"
vocab_trg = spm.SentencePieceProcessor()
vocab_trg.load(vocab_trg_file)

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
    print("Input        :", line)
    txt_2_tkn = vocab_src.encode_as_pieces(line)
    print("txt --> tkn  :", txt_2_tkn)
    tkn_2_txt = vocab_src.DecodePieces(txt_2_tkn)
    print("tkn --> txt  :", tkn_2_txt)

    txt_2_ids = vocab_src.encode_as_ids(line)
    print("txt --> ids  :", txt_2_ids)
    ids_2_txt = vocab_src.DecodeIds(txt_2_ids)
    print("ids --> txt  :", ids_2_txt)

    ids2 = vocab_src.piece_to_id(txt_2_tkn)
    print("Piece --> id :", ids2)
    print("Id --> piece :", vocab_src.id_to_piece(ids2))
    print("\n")

# Target Tokenizer
lines = [
  "C'est l'hiver et il fait très froid.",
  "Ce Noël sera-t-il un Noël blanc ?",
  "Attention à ne pas attraper froid en hiver et bonne année."
]
for line in lines:
    print("Input        :", line)
    txt_2_tkn = vocab_trg.encode_as_pieces(line)
    print("txt --> tkn  :", txt_2_tkn)
    tkn_2_txt = vocab_trg.DecodePieces(txt_2_tkn)
    print("tkn --> txt  :", tkn_2_txt)

    txt_2_ids = vocab_trg.encode_as_ids(line)
    print("txt --> ids  :", txt_2_ids)
    ids_2_txt = vocab_trg.DecodeIds(txt_2_ids)
    print("ids --> txt  :", ids_2_txt)

    ids2 = vocab_trg.piece_to_id(txt_2_tkn)
    print("Piece --> id :", ids2)
    print("Id --> piece :", vocab_trg.id_to_piece(ids2))
    print("\n")


train_df, test_df = train_test_split(total_df, test_size=0.2)

# 구분자 변경
train_df.to_csv('/content/ratings_train.txt', sep = '\t', index = False)
test_df.to_csv('/content/ratings_test.txt', sep = '\t', index = False)

train_df[:5]
test_df[:5]

""" train data 준비 """
def prepare_train(vocab_src, vocab_trg, infile, outfile):
    df = pd.read_csv(infile, sep="\t", engine="python")
    with open(outfile, "w") as f:
        for index, row in df.iterrows():

            src_document = row["SRC"]
            if type(src_document) != str:
                continue
            temp_src_sent = vocab_src.encode_as_pieces(src_document)
            if len(temp_src_sent)>256:
                temp_src_sent = temp_src_sent[:256]
            
            trg_document = row["TRG"]
            if type(trg_document) != str:
                continue
            temp_trg_sent = vocab_trg.encode_as_pieces(trg_document)
            if len(temp_trg_sent)>256:
                temp_trg_sent = temp_trg_sent[:256]

            instance = {"SRC": temp_src_sent, "TRG": temp_trg_sent }
            f.write(json.dumps(instance))
            f.write("\n")

prepare_train(vocab_src, vocab_trg, f"{data_dir}/ratings_train.txt", f"{data_dir}/ratings_train.json")
prepare_train(vocab_src, vocab_trg, f"{data_dir}/ratings_test.txt", f"{data_dir}/ratings_test.json")
for f in os.listdir(data_dir):
    print(f)

data = [json.loads(line) for line in open('/content/ratings_train.json', 'r')]
print(data[0])

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
n_enc_vocab = len(vocab_src)
n_dec_vocab = len(vocab_trg)
n_layers  = 2     # 6
hid_dim   = 256
pf_dim    = 1024
i_pad     = 0
n_heads   = 8
d_head    = 64
dropout   = 0.3
layer_norm_epsilon = 1e-12
N_EPOCHS  = 20
n_output = n_dec_vocab

'''
M03. [PASS] Load datasets
'''

'''
M04. Build Transformer model
'''

""" 
C01. Sinusoid position encoding
"""
def get_sinusoid_encoding_table(n_seq, hid_dim):
    def cal_angle(position, i_hidn):
        return position / np.power(10000, 2 * (i_hidn // 2) / hid_dim)
    def get_posi_angle_vec(position):
        return [cal_angle(position, i_hidn) for i_hidn in range(hid_dim)]

    sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # even index sin 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # odd index cos

    return sinusoid_table


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
    
    def __init__(self):
        super(MultiHeadAttentionLayer, self).__init__()
        
        # Define dense layers corresponding to WQ, WK, and WV
        self.q_linear = nn.Linear(hid_dim, n_heads * d_head)
        self.k_linear = nn.Linear(hid_dim, n_heads * d_head)
        self.v_linear = nn.Linear(hid_dim, n_heads * d_head)
        self.scaled_dot_attn = ScaledDotProductAttention()
        self.output_MHA = nn.Linear(n_heads * d_head, hid_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)
        
        # 1. Pass through the dense layer corresponding to WQ
        # q : (bs, n_heads, n_q_seq, d_head)
        query = self.q_linear(Q).view(batch_size, -1, n_heads, d_head).transpose(1,2)
        
        # 2. Pass through the dense layer corresponding to WK
        # k : (bs, n_heads, n_k_seq, d_head)
        key   = self.k_linear(K).view(batch_size, -1, n_heads, d_head).transpose(1,2)
        
        # 3. Pass through the dense layer corresponding to WV
        # v : (bs, n_heads, n_v_seq, d_head)
        value = self.v_linear(V).view(batch_size, -1, n_heads, d_head).transpose(1,2)

        # 4. Scaled Dot Product Attention. Using the previously implemented function
        # (bs, n_heads, n_q_seq, n_k_seq)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # (bs, n_heads, n_q_seq, d_head), (bs, n_heads, n_q_seq, n_k_seq)
        scaled_attention, attn_prob = self.scaled_dot_attn(query, key, value, attn_mask)
        
        # 5. Concatenate the heads
        # (bs, n_heads, n_q_seq, h_head * d_head)
        concat_attention = scaled_attention.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_head)
        
        # 6. Pass through the dense layer corresponding to WO
        # (bs, n_heads, n_q_seq, e_embd)
        outputs = self.output_MHA(concat_attention)
        outputs = self.dropout(outputs)
        # (bs, n_q_seq, hid_dim), (bs, n_heads, n_q_seq, n_k_seq)
        return outputs, attn_prob

"""
C04. Positionwise Feedforward Layer
"""
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self):
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
        
        self.attn = MultiHeadAttentionLayer()
        self.ffn = PositionwiseFeedforwardLayer()
        
        self.layernorm1 = nn.LayerNorm(hid_dim)
        self.layernorm2 = nn.LayerNorm(hid_dim)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, inputs, padding_mask):
        
        # 1. Encoder mutihead attention is defined
        attention, attn_prob = self.attn(inputs, inputs, inputs, padding_mask)
        attention   = self.dropout1(attention)
        
        # 2. 1 st residual layer
        attention   = self.layernorm1(inputs + attention)  # (batch_size, input_seq_len, hid_dim)
        
        # 3. Feed Forward Network
        ffn_outputs = self.ffn(attention)  # (batch_size, input_seq_len, hid_dim)
        
        ffn_outputs = self.dropout2(ffn_outputs)
        
        # 4. 2 nd residual layer
        ffn_outputs = self.layernorm2(attention + ffn_outputs)  # (batch_size, input_seq_len, hid_dim)

        # 5. Encoder output of each encoder layer
        return ffn_outputs, attn_prob

"""
C06. Encoder
"""
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.embedding    = nn.Embedding(n_enc_vocab, hid_dim)
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(ENCODER_LEN + 1, hid_dim))
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)

        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
    
    def forward(self, inputs):
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).expand(inputs.size(0), inputs.size(1)).contiguous() + 1
        pos_mask = inputs.eq(i_pad)
        positions.masked_fill_(pos_mask, 0)

        # (bs, ENCODER_LEN, hid_dim)
        outputs = self.embedding(inputs) + self.pos_emb(positions)

        # (bs, ENCODER_LEN, ENCODER_LEN)
        attn_mask = create_padding_mask(inputs, inputs, i_pad)

        attn_probs = []
        for layer in self.layers:
            # (bs, ENCODER_LEN, hid_dim), (bs, n_heads, ENCODER_LEN, ENCODER_LEN)
            outputs, attn_prob = layer(outputs, attn_mask)
            attn_probs.append(attn_prob)
        # (bs, ENCODER_LEN, hid_dim), [(bs, n_heads, ENCODER_LEN, ENCODER_LEN)]
        return outputs, attn_probs
    
"""
C07. Decoder layer
"""
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()

        self.attn   = MultiHeadAttentionLayer()
        self.attn_2 = MultiHeadAttentionLayer()

        self.ffn = PositionwiseFeedforwardLayer()

        self.layernorm1 = nn.LayerNorm(hid_dim)
        self.layernorm2 = nn.LayerNorm(hid_dim)
        self.layernorm3 = nn.LayerNorm(hid_dim)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, inputs, enc_outputs, self_attn_mask, dec_enc_attn_mask):
        # 1. 1st Encoder mutihead attention is defined. Q,K,V is same and it's comes from decoder input or previous decoder output
        # (bs, DECODER_LEN, hid_dim), (bs, n_heads, DECODER_LEN, DECODER_LEN)
        attention1, self_attn_prob = self.attn(inputs, inputs, inputs, self_attn_mask)
        attention1 = self.dropout1(attention1)
        
        # 2. 1st residual layer
        attention1 = self.layernorm1(inputs + attention1)
        
        # 3. 2nd Encoder mutihead attention is defined. Q comes from Multi-Head attention. K,V are same and comes from encoder output
        # (bs, DECODER_LEN, hid_dim), (bs, n_heads, DECODER_LEN, ENCODER_LEN)
        attention2, dec_enc_attn_prob = self.attn_2( attention1, enc_outputs, enc_outputs, dec_enc_attn_mask)
        attention2 = self.dropout2(attention2)

        # 4. 2nd residual layer
        attention2 = self.layernorm2(attention1 + attention2)  # (batch_size, raw_trgeq_len, hid_dim)

        # 5. Feed Forward Network
        ffn_outputs = self.ffn(attention2)  # (batch_size, raw_trgeq_len, hid_dim)
        ffn_outputs = self.dropout3(ffn_outputs)
        
        # 6. 3 rd residual layer
        ffn_outputs = self.layernorm3(attention2 + ffn_outputs)  # (batch_size, raw_trgeq_len, hid_dim)
        
        # (bs, DECODER_LEN, hid_dim), (bs, n_heads, DECODER_LEN, DECODER_LEN), (bs, n_heads, DECODER_LEN, ENCODER_LEN)
        return ffn_outputs, self_attn_prob, dec_enc_attn_prob

"""
C08. Decoder
"""
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.embedding    = nn.Embedding(n_dec_vocab, hid_dim)
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(DECODER_LEN + 1, hid_dim))
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)

        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
    
    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        positions = torch.arange(dec_inputs.size(1), device=dec_inputs.device, dtype=dec_inputs.dtype).expand(dec_inputs.size(0), dec_inputs.size(1)).contiguous() + 1
        pos_mask = dec_inputs.eq(i_pad)
        positions.masked_fill_(pos_mask, 0)
    
        # (bs, DECODER_LEN, hid_dim)
        dec_outputs = self.embedding(dec_inputs) + self.pos_emb(positions)

        # (bs, DECODER_LEN, DECODER_LEN)
        dec_attn_pad_mask = create_padding_mask(dec_inputs, dec_inputs, i_pad)
        
        # (bs, DECODER_LEN, DECODER_LEN)
        dec_attn_decoder_mask = create_look_ahead_mask(dec_inputs)
        
        # (bs, DECODER_LEN, DECODER_LEN)
        dec_self_attn_mask = torch.gt((dec_attn_pad_mask + dec_attn_decoder_mask), 0)
        
        # (bs, DECODER_LEN, ENCODER_LEN)
        dec_enc_attn_mask = create_padding_mask(dec_inputs, enc_inputs, i_pad)

        self_attn_probs, dec_enc_attn_probs = [], []
        for layer in self.layers:
            # (bs, DECODER_LEN, hid_dim), (bs, DECODER_LEN, DECODER_LEN), (bs, DECODER_LEN, ENCODER_LEN)
            dec_outputs, self_attn_prob, dec_enc_attn_prob = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            self_attn_probs.append(self_attn_prob)
            dec_enc_attn_probs.append(dec_enc_attn_prob)
        # (bs, DECODER_LEN, hid_dim), [(bs, DECODER_LEN, DECODER_LEN)], [(bs, DECODER_LEN, ENCODER_LEN)]S
        return dec_outputs, self_attn_probs, dec_enc_attn_probs

"""
C09. Attention pad mask
"""
def create_padding_mask(seq_q, seq_k, i_pad):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    mask = seq_k.data.eq(i_pad).unsqueeze(1).expand(batch_size, len_q, len_k)  # <pad>
    return mask

""" 
C10. Attention decoder mask (Look Ahead Mask)
"""
def create_look_ahead_mask(seq):
    look_ahead_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
    look_ahead_mask = look_ahead_mask.triu(diagonal=1) # upper triangular part of a matrix(2-D)
    return look_ahead_mask

"""
C12. Transformer Class
"""
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        
        # Ecoder and Decoder
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, enc_inputs, dec_inputs):
        
        enc_outputs, enc_self_attn_probs = self.encoder(enc_inputs)
        
        dec_outputs, dec_self_attn_probs, dec_enc_attn_probs = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        
        return dec_outputs, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs
    
    def save(self, epoch, loss, path):
        torch.save({
            "epoch": epoch,
            "loss": loss,
            "state_dict": self.state_dict()
        }, path)
    
    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save["state_dict"])
        return save["epoch"], save["loss"]

""" Define Language Model Head """
class Language_Model_Head(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.transformer = Transformer()
        
        # lm
        self.projection_lm = nn.Linear(hid_dim, n_output, bias=False)
    
    def forward(self, enc_inputs, dec_inputs):
        
        dec_outputs, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs = self.transformer(enc_inputs, dec_inputs)

        logits_lm = F.log_softmax(self.projection_lm(dec_outputs), dim = 2)
        
        return logits_lm, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs

""" Language Model Dataset """
class Language_M_Dataset(torch.utils.data.Dataset):
    def __init__(self, vocab_src, vocab_trg, infile):
        self.vocab_src = vocab_src
        self.vocab_trg = vocab_trg
        self.src_sentences = []
        self.trg_sentences = []

        line_cnt = 0
        with open(infile, "r") as f:
            for line in f:
                line_cnt += 1

        with open(infile, "r") as f:
            for i, line in enumerate(tqdm(f, total=line_cnt, desc=f"Loading {infile}", unit=" lines")):
                data = json.loads(line)
                self.src_sentences.append([self.vocab_src.piece_to_id(p) for p in data["SRC"]])
                self.trg_sentences.append([self.vocab_trg.piece_to_id("[BOS]")] + [self.vocab_trg.piece_to_id(p) for p in data["TRG"]] + [self.vocab_trg.piece_to_id("[EOS]")])
    
    def __len__(self):
        assert len(self.src_sentences) == len(self.trg_sentences)
        return len(self.src_sentences)
    
    def __getitem__(self, item):
        return (torch.tensor(self.src_sentences[item]),
                torch.tensor(self.trg_sentences[item]))

""" Language Model data collate_fn """
def L_M_collate(inputs):
    enc_inputs, dec_inputs = list(zip(*inputs))

    enc_inputs = torch.nn.utils.rnn.pad_sequence(enc_inputs, batch_first=True, padding_value=0)
    dec_inputs = torch.nn.utils.rnn.pad_sequence(dec_inputs, batch_first=True, padding_value=0)

    batch = [
        enc_inputs,
        dec_inputs,
    ]
    return batch

""" 데이터 로더 """
batch_size = 64  #128
train_dataset = Language_M_Dataset(vocab_src, vocab_trg, f"{data_dir}/ratings_train.json")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=L_M_collate)
test_dataset = Language_M_Dataset(vocab_src, vocab_trg, f"{data_dir}/ratings_test.json")
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=L_M_collate)

print(train_dataset[0])

""" 모델 epoch 학습 """
def train_epoch(epoch, model, criterion, optimizer, train_loader):
    losses = []
    model.train()
    
    with tqdm_notebook(total=len(train_loader), desc=f"Train {epoch+1}") as pbar:
        for i, value in enumerate(train_loader):
            enc_inputs, dec_inputs = map(lambda v: v.to(device), value)

            optimizer.zero_grad()
            outputs = model(enc_inputs, dec_inputs[:,:-1])
            logits_lm = outputs[0]
            output_dim = logits_lm.shape[-1]
            
            logits_lm = logits_lm.contiguous().view(-1, output_dim)
            dec_inputs = dec_inputs[:,1:].contiguous().view(-1)
            
            loss_lm = criterion(logits_lm, dec_inputs)
            loss_val = loss_lm.item()
            losses.append(loss_val)

            loss_lm.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")
    return np.mean(losses)

learning_rate = 5e-5

model = Language_Model_Head()
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_epoch, best_loss, best_score = 0, 0, 0
losses, scores = [], []
for epoch in range(N_EPOCHS):
    loss = train_epoch(epoch, model, criterion, optimizer, train_loader)
    # score = eval_epoch(model, test_loader)

    losses.append(loss)
    # scores.append(score)

    # if best_score < score:
    #     best_epoch, best_loss, best_score = epoch, loss, score
# print(f">>>> epoch={best_epoch}, loss={best_loss:.5f}, socre={best_score:.5f}")

"""
# table
data = {
    "loss": losses,
}
df = pd.DataFrame(data)
display(df)

# graph
plt.figure(figsize=[12, 4])
plt.plot(losses, label="loss")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.show()
"""

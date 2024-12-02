!pip install tensorflow==2.15.0
import tensorflow as tf
print(tf.__version__)

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

print("Tensorflow version {}".format(tf.__version__))
import random
# Setup seeds
SEED = 1234
tf.random.set_seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE

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
BUFFER_SIZE  = 20000
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

raw_src_df  = train_df['SRC']
raw_trg_df  = train_df['TRG']

src_sentence  = raw_src_df.apply(lambda x: "<SOS> " + str(x) + " <EOS>")
trg_sentence  = raw_trg_df.apply(lambda x: "<SOS> "+ x + " <EOS>")

'''
D10. Define tokenizer
'''
filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
oov_token = '<unk>'

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

tkn_sources = tf.cast(tkn_sources, dtype=tf.int64)
tkn_targets = tf.cast(tkn_targets, dtype=tf.int64)

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

dataset = tf.data.Dataset.from_tensor_slices((tkn_sources, tkn_targets))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

print("Inpute Vocabulary Size: {}".format(len(SRC_tokenizer.word_index)))
print("Target Vocabulary Size: {}".format(len(TRG_tokenizer.word_index)))

'''
D19. Define some useful parameters for further use
'''

example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape

max_length_input  = example_input_batch.shape[1]
max_length_output = example_target_batch.shape[1]

'''
Model Engineering
'''

'''
M01. Import Libraries for Model Engineering
'''
from tqdm import tqdm, tqdm_notebook, trange

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

'''
M02. Define Hyperparameters for Model Engineering
'''
n_layers  = 2     # 6
hid_dim   = 256
pf_dim    = 1024
n_heads   = 8
dropout   = 0.3
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
def get_sinusoid_encoding_table(position, hid_dim):
    # angle_rads = get_angles(np.arange(position)[:, np.newaxis],
    #                         np.arange(hid_dim)[np.newaxis, :],
    #                         hid_dim)
    position = np.arange(position)[:, np.newaxis]
    angle_rates = 1 / np.power(10000, (2 * (np.arange(hid_dim)[np.newaxis, :]//2)) / np.float32(hid_dim))
    angle_rads =  position * angle_rates

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

sample_pos_encoding = get_sinusoid_encoding_table(50, 128)

plt.pcolormesh(sample_pos_encoding.numpy()[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 128))
plt.ylabel('Position')
plt.colorbar()
plt.show()

"""
C02. Scaled dot product attention
"""
def ScaledDotProductAttention(query, key, value, mask):
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
    
    # 1. MatMul Q, K-transpose. Attention score matrix.
    matmul_qk = tf.matmul(query, key, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # 2. scale matmul_qk
    dk = tf.cast(tf.shape(key)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 3. add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # 4. softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    
    # 5. MatMul attn_prov, V
    output = tf.matmul(attention_weights, value)  # (..., seq_len_q, depth_v)

    return output, attention_weights

"""
C03. Multi head attention
"""
class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    
    def __init__(self, hid_dim, n_heads):
        super(MultiHeadAttentionLayer, self).__init__()
        self.n_heads = n_heads
        assert hid_dim % self.n_heads == 0
        self.hid_dim = hid_dim
        
        # hid_dim divided by n_heads.
        self.depth = int(hid_dim/self.n_heads)
        
        # Define dense layers corresponding to WQ, WK, and WV
        self.q_linear = tf.keras.layers.Dense(hid_dim)
        self.k_linear = tf.keras.layers.Dense(hid_dim)
        self.v_linear = tf.keras.layers.Dense(hid_dim)
        
        # Dense layer definition corresponding to WO
        self.out = tf.keras.layers.Dense(hid_dim)

    def split_heads(self, inputs, batch_size):
        """Split the last dimension into (n_heads, depth).
        Transpose the result such that the shape is (batch_size, n_heads, seq_len, depth)
        """
        inputs = tf.reshape(
            inputs, (batch_size, -1, self.n_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, value, key, query, mask):
        batch_size = tf.shape(query)[0]
        
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
        scaled_attention, attention_weights = ScaledDotProductAttention(
            query, key, value, mask)
        
        # (batch_size, sentence length of query, n_heads, hid_dim/n_heads)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        
        # 5. Concatenate the heads
        # (batch_size, sentence length of query, hid_dim)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.hid_dim))
        
        # 6. Pass through the dense layer corresponding to WO
        # (batch_size, sentence length of query, hid_dim)
        outputs = self.out(concat_attention)

        return outputs, attention_weights

"""
C04. Positionwise Feedforward Layer
"""
class PositionwiseFeedforwardLayer(tf.keras.layers.Layer):
    def __init__(self, hid_dim, pf_dim):
        super(PositionwiseFeedforwardLayer, self).__init__()
        self.linear_1 = tf.keras.layers.Dense(pf_dim, activation='relu')
        self.linear_2 = tf.keras.layers.Dense(hid_dim)

    def forward(self, attention):
        output = self.linear_1(attention)
        output = self.linear_2(output)
        return output

"""
C05. Encoder layer
"""
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, pf_dim, hid_dim, n_heads, dropout):
        super(EncoderLayer, self).__init__()
        
        self.attn = MultiHeadAttentionLayer(hid_dim, n_heads)
        self.ffn  = PositionwiseFeedforwardLayer(hid_dim, pf_dim)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training, padding_mask):
        
        # 1. Encoder mutihead attention is defined
        attention, _ = self.attn(inputs, inputs, inputs, padding_mask)  # (batch_size, input_seq_len, hid_dim)
        attention   = self.dropout1(attention, training=training)
        
        # 2. 1 st residual layer
        attention   = self.layernorm1(inputs + attention)  # (batch_size, input_seq_len, hid_dim)
        
        # 3. Feed Forward Network
        ffn_outputs = self.ffn(attention)  # (batch_size, input_seq_len, hid_dim)
        
        ffn_outputs = self.dropout2(ffn_outputs, training=training)
        
        # 4. 2 nd residual layer
        ffn_outputs = self.layernorm2(attention + ffn_outputs)  # (batch_size, input_seq_len, hid_dim)

        # 5. Encoder output of each encoder layer
        return ffn_outputs

"""
C06. Encoder
"""
class Encoder(tf.keras.layers.Layer):
    def __init__(self, n_enc_vocab, n_layers, pf_dim, hid_dim, n_heads,
                 maximum_position_encoding, dropout):
        super(Encoder, self).__init__()
        
        self.hid_dim  = hid_dim
        self.n_layers = n_layers
        
        self.embedding = tf.keras.layers.Embedding(n_enc_vocab, hid_dim)
        self.pos_encoding = get_sinusoid_encoding_table(maximum_position_encoding,
                                                self.hid_dim)

        self.enc_layers = [EncoderLayer(pf_dim, hid_dim, n_heads, dropout)
                           for _ in range(n_layers)]

        self.dropout1 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training, padding_mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        # 1. Token Embedding
        emb = self.embedding(x)  # (batch_size, input_seq_len, hid_dim)
        emb *= tf.math.sqrt(tf.cast(self.hid_dim, tf.float32))
        
        # 2. Sinusoidal positional Encoding
        emb += self.pos_encoding[:, :seq_len, :]

        output = self.dropout1(emb, training=training)
        
        # 3. Self padding Mask is created from encoder input

        # 4. Encoder layers are stacked
        for i in range(self.n_layers):
            output = self.enc_layers[i](output, training, padding_mask)
            
        # 5. Final layer's output is the encoder output

        return output  # (batch_size, input_seq_len, hid_dim)
    
"""
C07. Decoder layer
"""
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, pf_dim, hid_dim, n_heads, dropout):
        super(DecoderLayer, self).__init__()

        self.attn   = MultiHeadAttentionLayer(hid_dim, n_heads)
        self.attn_2 = MultiHeadAttentionLayer(hid_dim, n_heads)

        self.ffn = PositionwiseFeedforwardLayer(hid_dim, pf_dim)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.dropout3 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, hid_dim)
        
        # 1. 1st Encoder mutihead attention is defined. Q,K,V is same and it's comes from decoder input or previous decoder output
        attention1, attn_weights_block1 = self.attn(
            inputs, inputs, inputs, look_ahead_mask)  # (batch_size, target_seq_len, hid_dim)
        attention1 = self.dropout1(attention1, training=training)
        
        # 2. 1st residual layer
        attention1 = self.layernorm1(inputs + attention1)

        # 3. 2nd Encoder mutihead attention is defined. Q comes from Multi-Head attention. K,V are same and comes from encoder output
        attention2, attn_weights_block2 = self.attn_2(
            enc_output, enc_output, attention1, padding_mask)  # (batch_size, target_seq_len, hid_dim)
    
        attention2 = self.dropout2(attention2, training=training)

        # 4. 2nd residual layer
        attention2 = self.layernorm2(attention1 + attention2)  # (batch_size, target_seq_len, hid_dim)

        # 5. Feed Forward Network
        ffn_outputs = self.ffn(attention2)  # (batch_size, target_seq_len, hid_dim)
        ffn_outputs = self.dropout3(ffn_outputs, training=training)
        
        # 6. 3 rd residual layer
        ffn_outputs = self.layernorm3(attention2 + ffn_outputs)  # (batch_size, target_seq_len, hid_dim)

        # 7. Decoder output of each decoder layer
        return ffn_outputs, attn_weights_block1, attn_weights_block2

"""
C08. Decoder
"""
class Decoder(tf.keras.layers.Layer):
    def __init__(self, n_dec_vocab, n_layers, pf_dim, hid_dim, n_heads, 
                 maximum_position_encoding, dropout):
        super(Decoder, self).__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = tf.keras.layers.Embedding(n_dec_vocab, hid_dim)
        self.pos_encoding = get_sinusoid_encoding_table(maximum_position_encoding, hid_dim)

        self.dec_layers = [DecoderLayer(pf_dim, hid_dim, n_heads, dropout)
                           for _ in range(n_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, dec_input, enc_output, training,
             look_ahead_mask, padding_mask):

        seq_len = tf.shape(dec_input)[1]
        attention_weights = {}

        # 1. Decoder input Token Embedding
        emb = self.embedding(dec_input)
        emb *= tf.math.sqrt(tf.cast(self.hid_dim, tf.float32))
        
        # 2. Sinusoidal positional Encoding
        emb += self.pos_encoding[:, :seq_len, :]

        output = self.dropout(emb, training=training)

        # 3. Padding mask is created from **encoder inputs** in this implementation
        # 4. Look ahead Mask is created from **decoder inputs**
        # 5. Decoder layers are stacked
        for i in range(self.n_layers):
            output, block1, block2 = self.dec_layers[i](output, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
    
        # 6. Final layer's output is the decoder output
        
        return output, attention_weights
    
"""
C09. Attention pad mask
"""
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # (batch_size, 1, 1, key의 문장 길이)
    return seq[:, tf.newaxis, tf.newaxis, :]

""" 
C10. Attention decoder mask
"""
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

""" 
C11. Create masks
"""
def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  
    return enc_padding_mask, look_ahead_mask, dec_padding_mask

"""
C12. Transformer Class
"""
class Transformer(tf.keras.Model):
    
    def __init__(self, n_enc_vocab, n_dec_vocab,
                 n_layers, pf_dim, hid_dim, n_heads,
                 pe_input, pe_target, dropout):
        super(Transformer, self).__init__()
        self.encoder = Encoder(n_enc_vocab,
                               n_layers, pf_dim, hid_dim, n_heads,
                               pe_input, dropout)

        self.decoder = Decoder(n_dec_vocab,
                               n_layers, pf_dim, hid_dim, n_heads,
                               pe_target, dropout)

        self.fin_output = tf.keras.layers.Dense(n_dec_vocab)
    
    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        
        # 1. "inp" is the encoder input of tokenized datasets
        # 2. "enc_padding_mask" are the self padding mask from encoder inputs
        
        enc_output = self.encoder(inp, training, enc_padding_mask)
        
        # 3. Encoder outputs are created from encoder model and it is given to the "Key" and "Value" of Multi-Head Attention 2 of decoder layers 
        # 4. "tar" is the Decoder input of tokenized datasets. It is not a final output
        # 5. "look_ahead_mask" are created from decoder inputs, it is given to the "Key" and "Value" of Multi-Head Attention 1
        # 6. "dec_padding_mask" are the self padding mask from encoder inputs
        
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        # 7. Decoder output is creadted from decoder model
        
        final_output = self.fin_output(dec_output)
        
        # 8. Final outputs are created. Then it is used or Language Model. In the official tutorial "Softmax" was missed

        return final_output, attention_weights
    
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

'''
M05. Define Loss Function
'''

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

'''
M06. Learning Rate Scheduling
Not used in this implementation.
'''

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, hid_dim, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.hid_dim = hid_dim
        self.hid_dim = tf.cast(self.hid_dim, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.hid_dim) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(hid_dim)

temp_learning_rate_schedule = CustomSchedule(hid_dim)

plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")

'''
M07. Define Optimizer
'''
# Let's use the default parameters of Adam Optimizer

# optimizer = Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
optimizer = tf.keras.optimizers.Adam()

'''
M08. [Opt] Define Accuracy Metrics
'''

def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)
    
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

# tf.keras.utils.plot_model(
#     model, to_file='transformer.png', show_shapes=True)
'''
M09. [OPT] Define Checkpoints Manager
'''

checkpoint_path = "./checkpoints"

checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    checkpoint.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

'''
M10. Define train loop
'''

@tf.function
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    '''
    1. Clear Gradients
    '''
    with tf.GradientTape() as tape:
        '''
        2. Forward Pass
        '''
        predictions, _ = model(inp, tar_inp, 
            True, 
            enc_padding_mask, 
            combined_mask, 
            dec_padding_mask
        )
        '''
        3. Compute loss
        '''
        loss = loss_function(tar_real, predictions)
    
    '''
    4. Compute gradients / Backpropagation
    '''
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    
    '''
    5. Adjust learnable parameters
    '''
    optimizer.apply_gradients(zip(gradients, variables))
    
    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions))

'''
M11. Define Episode / each step process
'''

for epoch in range(N_EPOCHS):
    train_loss.reset_states()
    
    with tqdm_notebook(total=len(dataset), desc=f"Train {epoch+1}") as pbar:
        for (batch, (inp, tar)) in enumerate(dataset):
            train_step(inp, tar)
            
            pbar.update(1)
            pbar.set_postfix_str(f"Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}")
            
    '''
    Run Checkpoint manager
    '''
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))

'''
M12. Explore the training result with new raw sentence
'''

def evaluate(text):
    text = SRC_tokenizer.texts_to_sequences([text])
    text = pad_sequences(text, maxlen=ENCODER_LEN, padding='post', truncating='post')

    encoder_input = tf.expand_dims(text[0], 0)

    decoder_input = [TRG_tokenizer.word_index['<sos>']]
    output = tf.expand_dims(decoder_input, 0)
    
    # Decoder's prediction starts
    for i in range(DECODER_LEN):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

        predictions, attention_weights = model(
            encoder_input, 
            output,
            False,
            enc_padding_mask,
            combined_mask,
            dec_padding_mask
        )

        # Receives the predicted word at the current (last) point in time.
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # Stop prediction if last time prediction word is end token
        if predicted_id == TRG_tokenizer.word_index['<eos>']:
            return tf.squeeze(output, axis=0), attention_weights

        # 마지막 시점의 예측 단어를 출력에 연결한다.
        # 이는 for문을 통해서 디코더의 입력으로 사용될 예정이다.
        # Connect the last time prediction word to the output
        # This will be used as the input of the decoder through the for statement.
        output = tf.concat([output, predicted_id], axis=-1)
        
    return tf.squeeze(output, axis=0), attention_weights

def predict(text):
    prediction = evaluate(text=text)[0].numpy()
    prediction = np.expand_dims(prediction[1:], 0)  
    predicted_sentence = TRG_tokenizer.sequences_to_texts(prediction)[0]
    
    return predicted_sentence

for idx in (11, 21, 31, 41, 51):
    print("Input        :", raw_src_df[idx])
    print("Prediction   :", predict(raw_src_df[idx]))
    print("Ground Truth :", raw_trg_df[idx],"\n")

'''
M13. [PASS] Explore the training result with test dataset
'''
    

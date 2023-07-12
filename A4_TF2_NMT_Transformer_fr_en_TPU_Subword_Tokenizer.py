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

import tensorflow_datasets as tfds

print("Tensorflow version {}".format(tf.__version__))
import random
# Setup seeds
SEED = 1234
tf.random.set_seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE

'''
DX. TPU Define
'''

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU {}'.format(tpu.cluster_spec().as_dict()['worker']))
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: {}".format(strategy.num_replicas_in_sync))

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

# 텐서플로우 dataset을 이용하여 셔플(shuffle)을 수행하되, 배치 크기로 데이터를 묶는다.
# 또한 이 과정에서 교사 강요(teacher forcing)을 사용하기 위해서 디코더의 입력과 실제값 시퀀스를 구성한다.
# Perform shuffle using TensorFlow dataset, but group data by batch size
# Also, in this process, to use teacher forcing, the decoder's input and real-value sequences are constructed.
BATCH_SIZE   = 64 * strategy.num_replicas_in_sync
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
D9. [PASS] Add <SOS>, <EOS> for source and target
Subword Tokenizer have other method, See the below coding
'''
print('Translation Pair :',len(raw_src)) # 리뷰 개수 출력


'''
D10. Define tokenizer
'''
SRC_tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    raw_src, target_vocab_size=2**13)

TRG_tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    raw_trg, target_vocab_size=2**13)

# Assign integers to start and end tokens
START_TOKEN, END_TOKEN = [TRG_tokenizer.vocab_size], [TRG_tokenizer.vocab_size + 1]

# Size of word set by taking start and end tokens into account + 2
n_enc_vocab = SRC_tokenizer.vocab_size
n_dec_vocab = TRG_tokenizer.vocab_size + 2

print('ID of start Token        :',START_TOKEN)
print('ID of end Token          :',END_TOKEN)
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
    txt_2_ids = SRC_tokenizer.encode(line)
    ids_2_txt = SRC_tokenizer.decode(txt_2_ids)
    print("Input     :", line)
    print("txt_2_ids :", txt_2_ids)
    print("ids_2_txt :", ids_2_txt,"\n")

# Target Tokenizer
lines = [
  "C'est l'hiver et il fait très froid.",
  "Ce Noël sera-t-il un Noël blanc ?",
  "Attention à ne pas attraper froid en hiver et bonne année."
]
for line in lines:
    txt_2_ids = TRG_tokenizer.encode(line)
    ids_2_txt = TRG_tokenizer.decode(txt_2_ids)
    print("Input     :", line)
    print("txt_2_ids :", txt_2_ids)
    print("ids_2_txt :", ids_2_txt,"\n")
    
# 각 정수는 각 단어와 어떻게 mapping되는지 병렬로 출력
# 서브워드텍스트인코더는 의미있는 단위의 서브워드로 토크나이징한다. 띄어쓰기 단위 X 형태소 분석 단위 X
# Print in parallel how each integer maps to each word
# A subword text encoder tokenizes subwords in meaningful units. Spacing Unit X Morphological Analysis Unit X
for ts in txt_2_ids:
    print ('{} ----> {}'.format(ts, TRG_tokenizer.decode([ts])))

'''
D12. Tokenize
'''
# tokenize / encode integers / add start and end tokens / padding
tokenized_inputs, tokenized_outputs = [], []

for (sentence1, sentence2) in zip(raw_src, raw_trg):
    sentence1 = SRC_tokenizer.encode(sentence1)
    sentence2 = START_TOKEN + TRG_tokenizer.encode(sentence2) + END_TOKEN

    tokenized_inputs.append(sentence1)
    tokenized_outputs.append(sentence2)

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

# The start token must be removed from the actual value sequence of the decoder.
dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': tkn_sources,
        'dec_inputs': tkn_targets[:, :-1] # input of the decoder. The last padding token is removed.
    },
    {
        'outputs': tkn_targets[:, 1:]     # The first token is removed. In other words, the start token is removed.
    },
))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

# 임의의 샘플에 대해서 [:, :-1]과 [:, 1:]이 어떤 의미를 가지는지 테스트해본다.
# Test the meaning of [:, :-1] and [:, 1:] for a random sample.
print(tkn_targets[0]) # existing sample
print(tkn_targets[:1][:, :-1]) # By removing the last padding token, the length becomes 39(N-1).
print(tkn_targets[:1][:, 1:])  # The first token is removed. In other words, the start token is removed. The length is also 39(N-1).

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
N_EPOCHS  = 100

'''
M03. [PASS] Load datasets
'''

'''
M04. Build Transformer model
'''

""" 
C01. Sinusoid position encoding
"""
class get_sinusoid_encoding_table(tf.keras.layers.Layer):

    def __init__(self, position, hid_dim):
        super(get_sinusoid_encoding_table, self).__init__()
        self.pos_encoding = self.positional_encoding(position, hid_dim)

    def get_angles(self, position, i, hid_dim):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(hid_dim, tf.float32))
        return position * angles

    def positional_encoding(self, position, hid_dim):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(hid_dim, dtype=tf.float32)[tf.newaxis, :],
            hid_dim=hid_dim)

        # 배열의 짝수 인덱스(2i)에는 사인 함수 적용
        # Apply sine function to even index of array (2i)
        sines = tf.math.sin(angle_rads[:, 0::2])

        # 배열의 홀수 인덱스(2i+1)에는 코사인 함수 적용
        # Apply the cosine function to odd indices of an array (2i+1)
        cosines = tf.math.cos(angle_rads[:, 1::2])

        angle_rads = np.zeros(angle_rads.shape)
        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = cosines
        pos_encoding = tf.constant(angle_rads)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        print(pos_encoding.shape)
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

sample_pos_encoding = get_sinusoid_encoding_table(50, 128)

plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
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
    
    def __init__(self, hid_dim, n_heads, name="multi_head_attention"):
        super(MultiHeadAttentionLayer, self).__init__(name=name)
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

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
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

        return outputs

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
def encoder_layer(pf_dim, hid_dim, n_heads, dropout, name="encoder_layer"):
    
    inputs = tf.keras.Input(shape=(None, hid_dim), name="inputs")

    # Encoder uses padding mask
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    # 1. Encoder mutihead attention is defined
    attention = MultiHeadAttentionLayer(
        hid_dim, n_heads, name="attention")({
        'query': inputs, 'key': inputs, 'value': inputs, # Q = K = V
        'mask': padding_mask # Use a padding mask
    })

    # Dropout + Residual Connectivity and Layer Normalization
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    # 2. 1 st residual layer
    attention = tf.keras.layers.LayerNormalization( epsilon=1e-6)(inputs + attention)

    # 3. Feed Forward Network
    outputs = PositionwiseFeedforwardLayer(hid_dim, pf_dim)(attention)
    
    # Dropout + Residual Connectivity and Layer Normalization
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    # 4. 2 nd residual layer
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

    # 5. Encoder output of each encoder layer
    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)

"""
C06. Encoder
"""
def encoder(n_enc_vocab, n_layers, pf_dim, hid_dim, n_heads, dropout,
            name="encoder"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")

    # 인코더는 패딩 마스크 사용
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    # adding embedding and position encoding.
    # 1. Token Embedding
    emb = tf.keras.layers.Embedding(n_enc_vocab, hid_dim)(inputs)
    emb *= tf.math.sqrt(tf.cast(hid_dim, tf.float32))
    # 2. Sinusoidal positional Encoding
    emb = get_sinusoid_encoding_table(n_enc_vocab, hid_dim)(emb)
    outputs = tf.keras.layers.Dropout(rate=dropout)(emb)
    # 3. Self padding Mask

    # 4. Encoder layers are stacked
    for i in range(n_layers):
        outputs = encoder_layer(pf_dim=pf_dim, hid_dim=hid_dim, n_heads=n_heads,
            dropout=dropout, name="encoder_layer_{}".format(i),
        )([outputs, padding_mask])
    # 5. Final layer's output is the encoder output
    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)

    
"""
C07. Decoder layer
"""
def DecoderLayer(pf_dim, hid_dim, n_heads, dropout, name="DecoderLayer"):
    inputs = tf.keras.Input(shape=(None, hid_dim), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, hid_dim), name="encoder_outputs")

    # The decoder uses both a lookahead mask (first sublayer) and a padding mask (second sublayer).
    look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    # 1. 1st Encoder mutihead attention is defined. Q,K,V is same and it's comes from decoder input or previous decoder output
    attention1 = MultiHeadAttentionLayer(
        hid_dim, n_heads, name="attention_1")(inputs={
        'query': inputs, 'key': inputs, 'value': inputs, # Q = K = V
        'mask': look_ahead_mask # 룩어헤드 마스크
    })

    # 2. 1st residual layer
    attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1 + inputs)

    # 3. 2nd Encoder mutihead attention is defined. Q comes from Multi-Head attention. K,V are same and comes from encoder output
    attention2 = MultiHeadAttentionLayer(
        hid_dim, n_heads, name="attention_2")(inputs={
        'query': attention1, 'key': enc_outputs, 'value': enc_outputs, # Q != K = V
        'mask': padding_mask # 패딩 마스크
    })

    # 4. 2nd residual layer
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2 + attention1)

    # 5. Feed Forward Network
    outputs = PositionwiseFeedforwardLayer(hid_dim, pf_dim)(attention2)
    
    # 6. 3 rd residual layer
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs + attention2)

    # 7. Decoder output of each decoder layer
    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)

"""
C08. Decoder
"""
def decoder(n_dec_vocab, n_layers, pf_dim, hid_dim, n_heads, dropout,
            name='decoder'):
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, hid_dim), name='encoder_outputs')


    # 1. Decoder input Token Embedding
    emb = tf.keras.layers.Embedding(n_dec_vocab, hid_dim)(inputs)
    emb *= tf.math.sqrt(tf.cast(hid_dim, tf.float32))
    # 2. Sinusoidal positional Encoding
    emb = get_sinusoid_encoding_table(n_dec_vocab, hid_dim)(emb)

    outputs = tf.keras.layers.Dropout(rate=dropout)(emb)
    
    # 3. Padding mask is created from **encoder inputs** in this implementation
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
    
    # 4. Look ahead Mask is created from **decoder inputs**
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name='look_ahead_mask')
    
    # 5. Decoder layers are stacked
    for i in range(n_layers):
        outputs = DecoderLayer( pf_dim=pf_dim, hid_dim=hid_dim, n_heads=n_heads,
            dropout=dropout, name='DecoderLayer_{}'.format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])
		
    # 6. Final layer's output is the decoder output

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)
    
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
# A function to mask future tokens in the first sublayer of the decoder
def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)           # Also includes a padding mask
    return tf.maximum(look_ahead_mask, padding_mask)
""" 
C11. Create masks
"""
def create_masks(inp, tar):
    # Encoder's Padding Mask
    enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='enc_padding_mask')(inp)

    # The lookahead mask of the decoder (first sublayer)
    look_ahead_mask = tf.keras.layers.Lambda(
        create_look_ahead_mask, output_shape=(1, None, None),
        name='look_ahead_mask')(tar)

    # Padding mask of the decoder (second sublayer)
    dec_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='dec_padding_mask')(inp)
  
    return enc_padding_mask, look_ahead_mask, dec_padding_mask

"""
C12. Transformer Class
"""

def Transformer(n_enc_vocab, n_dec_vocab, n_layers, pf_dim, hid_dim, n_heads, dropout,
                name="Transformer"):

    # 1. Encoder input
    inputs = tf.keras.Input(shape=(None,), name="inputs")

    # 2. Decoder input
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")
    
    # 3. Create Masks
    enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inputs, dec_inputs)
    
    # 4. The output of the encoder is enc_outputs. passed to the decoder.
    enc_outputs = encoder(n_enc_vocab=n_enc_vocab, n_layers=n_layers, pf_dim=pf_dim,
                          hid_dim=hid_dim, n_heads=n_heads, dropout=dropout,
                         )(inputs=[inputs, enc_padding_mask]) # 인코더의 입력은 입력 문장과 패딩 마스크

    # 5. The output of the decoder is dec_outputs. passed to the output layer.
    dec_outputs = decoder(n_dec_vocab=n_dec_vocab, n_layers=n_layers, pf_dim=pf_dim,
                          hid_dim=hid_dim, n_heads=n_heads, dropout=dropout,
                         )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    # 6. Output layer for next word prediction
    outputs = tf.keras.layers.Dense(units=n_dec_vocab, name="outputs")(dec_outputs)
    
    # 7. Final outputs are created. Then it is used or Language Model. In the official tutorial "Softmax" was missed
    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)

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

tf.keras.backend.clear_session()

'''
M08. [Opt] Define Accuracy Metrics
'''

def accuracy(y_true, y_pred):
    # ensure labels have shape (batch_size, DECODER_LEN - 1)
    y_true = tf.reshape(y_true, shape=(-1, DECODER_LEN - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
    # return tf.keras.metrics.SparseCategoricalCrossentropy(y_true, y_pred)
    
'''
M09. strategy.scope()
'''
# initialize and compile model within strategy scope
with strategy.scope():
    '''
    M10. Model Define
    '''
    model = Transformer(
        n_enc_vocab = n_enc_vocab,
        n_dec_vocab = n_dec_vocab,
        n_layers    = n_layers,
        pf_dim      = pf_dim,
        hid_dim     = hid_dim,
        n_heads     = n_heads,
        dropout     = dropout)

    '''
    M11. Model Compilation
    '''
    model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

model.summary()

tf.keras.utils.plot_model(
    model, to_file='transformer.png', show_shapes=True)

'''
M12. Load from Checkpoints
'''
checkpoint_path = "./checkpoints/Transformer.h5"
if os.path.exists(checkpoint_path):
    model.load_weights(checkpoint_path)
    print('Latest checkpoint restored!!')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
# tf.keras.callbacks.EarlyStopping( monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

'''
M13. Callback function loss
'''
es = EarlyStopping(monitor='loss', min_delta=0.0001, patience = 20)

# mc = ModelCheckpoint(checkpoint_path, save_best_only=True)
# rlr = ReduceLROnPlateau(factor=0.1, patience=5)
# csvlogger = CSVLogger("your_path/file_name.log")
# Just put the above callback function in fit.
# model.fit(x_train, y_train, epochs=20, batch_size=128, callbacks=[es, mc, rlr, csvlogger])

'''
M14.  Train and Validation - model.fit
'''
model.fit(dataset, epochs=N_EPOCHS, callbacks=[es])
# model.fit(dataset, epochs=N_EPOCHS)


'''
M15.  Save at Checkpoints
'''
if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")
model.save_weights(checkpoint_path)

'''
M16. Explore the training result with new raw sentence
'''

def preprocess_sentence(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    return sentence

def evaluate(text):
    text = preprocess_sentence(text)

    encoder_input = tf.expand_dims(SRC_tokenizer.encode(text), axis=0)

    output = tf.expand_dims(START_TOKEN, 0)
    
    # Decoder's prediction starts
    for i in range(DECODER_LEN):

        predictions = model(inputs=[encoder_input, output], training=False)

        # Receives the predicted word at the current (last) point in time.
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # Stop prediction if last time prediction word is end token
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        # 마지막 시점의 예측 단어를 출력에 연결한다.
        # 이는 for문을 통해서 디코더의 입력으로 사용될 예정이다.
        # Connect the last time prediction word to the output
        # This will be used as the input of the decoder through the for statement.
        output = tf.concat([output, predicted_id], axis=-1)
        
    return tf.squeeze(output, axis=0)

def predict(text):
    prediction = evaluate(text)

    predicted_sentence = TRG_tokenizer.decode(
        [i for i in prediction if i < TRG_tokenizer.vocab_size])
    
    return predicted_sentence

for idx in (11, 21, 31, 41, 51):
    print("Input        :", raw_src[idx])
    print("Prediction   :", predict(raw_src[idx]))
    print("Ground Truth :", raw_trg[idx],"\n")

'''
M13. [PASS] Explore the training result with test dataset
'''
    

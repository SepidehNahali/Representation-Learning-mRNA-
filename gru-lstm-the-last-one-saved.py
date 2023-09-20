#!/usr/bin/env python
# coding: utf-8

# ****pd.DataFrame(train_measurements).describe().T****

# In[ ]:





# In[1]:


get_ipython().system('conda install -y -c bioconda viennarna')
get_ipython().system('pip install RNA')
get_ipython().system('pip install python-Levenshtein')

Structure_Interpretability_Test = True



# In[2]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd, numpy as np, seaborn as sns
import math, json, os, random
from matplotlib import pyplot as plt
from tqdm import tqdm

import tensorflow as tf
import tensorflow_addons as tfa
import keras.backend as K

from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.cluster import KMeans


# In[3]:


def seed_everything(seed = 34):
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
seed_everything()


# # Version Changes
# 
# **Version 10:**
# 
# * added competition metric, as inspired by [Xhlulu](https://www.kaggle.com/xhlulu)'s discussion post [here](https://www.kaggle.com/c/stanford-covid-vaccine/discussion/183211)
# * removed filtering (no `SN_filter == 1` constraint)
# * added kfold stratification by `SN_filter`
# 
# **Version 11 (and V12; V11 failed to commit):**
# 
# * changed repeats from 1 to 3
# * dropped all samples where `signal_to_noise < 1` as per [this discussion post](https://www.kaggle.com/c/stanford-covid-vaccine/discussion/183992)
# * cleaned up some code
# 
# **Version 13:**
# 
# * made models larger - `embed_dim = 200`, `hidden_dim = 256` and consequently lowered training epochs to 75
# 
# **Version 14:**
# * added feature engineering and augmentation from [Tito](https://www.kaggle.com/its7171)'s incredible kernel [here](https://www.kaggle.com/its7171/gru-lstm-with-feature-engineering-and-augmentation) (check it out, it is fantastic work!)
# * included all samples in training, but added sample weighting by `signal_to_noise`, as inspired (again) by Tito's notebook above
# * only validated against samples with `SN_filter == 1`
# 
# **Version 15/16/17/18:**
# * removed `bpps_nb` feature from training
# * added GroupKFold to put similar RNA into the same fold (another of Tito's ideas)
# * cleaned up some more code, updated some comments
# * accidentally trained two GRUs, thanks for spotting that @junkoda
# 
# 
# **Update 9/28/2020:**
# 
# As this competition is entering its last week, this will be the final version of this notebook. I wanted to clean up some more code and add some last minute improvements for those that perhaps reference this notebook during the next week. This notebook received far more eention than it deserved. It is nothing without [Xhlulu](https://www.kaggle.com/xhlulu)'s kernel [here](https://www.kaggle.com/xhlulu/openvaccine-simple-gru-model) and his contributions to the dicussion forums over the past few weeks. If you give this notebook an upvote, please give Xhlulu's one as well (and Tito's). Good luck to everyone over the next week.

# # Competition Overview
# 
# **In this [new competition](https://www.kaggle.com/c/stanford-covid-vaccine/overview) we are helping to fight against the worldwide pandemic COVID-19. mRNA vaccines are the fastest vaccine candidates to treat COVID-19 but they currently facing several limitations. In particular, it is a challenge to design stable messenger RNA molecules. Typical vaccines are packaged in syringes and shipped under refrigeration around the world, but that is not possible for mRNA vaccines (currently).**
# 
# **Researches have noticed that RNA molecules tend to spontaneously degrade, which is highly problematic because a single cut can render mRNA vaccines useless. Not much is known about which part of the backbone of a particular RNA is most susceptible to being damaged.**
# 
# **Without this knowledge, the current mRNA vaccines are shopped under intense refrigeration and are unlikely to reach enough humans unless they can be stabilized. This is our task as Kagglers: we must create a model to predict the most likely degradation rates at each base of an RNA molecule.**
# 
# **We are given a subset of an Eterna dataset comprised of over 3000 RNA molecules and their degradation rates at each position. Our models are then tested on the new generation of RNA sequences that were just created by Eterna players for COVID-19 mRNA vaccines**
# 
# **Before we get started, please check out [Xhlulu](https://www.kaggle.com/xhlulu)'s notebook [here](https://www.kaggle.com/xhlulu/openvaccine-simple-gru-model) as this one is based on it: I just added comments, made minor code changes, an LSTM, and fold training:**

# In[4]:


#get comp data
train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)
test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)
sample_sub = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')


# In[5]:


# for index,row in train.iterrows():
#     print(len(row['structure']))


# In[ ]:





# In[6]:


test


# # Brief EDA
# 
# **From the data [description tab](https://www.kaggle.com/c/stanford-covid-vaccine/data), we must predict multiple ground truths in this competition, 5 to be exact. While the submission requires all 5, only 3 are scored: `reactivity`, `deg_Mg_pH10` and `deg_Mg_50C`. It might be interesting to see how performance differs when training for all 5 predictors vs. just the 3 that are scored.**
# 
# **The training features we are given are as follows:**
# 
# * **id** - An arbitrary identifier for each sample.
# * **seq_scored** - (68 in Train and Public Test, 91 in Private Test) Integer value denoting the number of positions used in scoring with predicted values. This should match the length of `reactivity`, `deg_*` and `*_error_*` columns. Note that molecules used for the Private Test will be longer than those in the Train and Public Test data, so the size of this vector will be different.
# * **seq_length** - (107 in Train and Public Test, 130 in Private Test) Integer values, denotes the length of `sequence`. Note that molecules used for the Private Test will be longer than those in the Train and Public Test data, so the size of this vector will be different.
# * **sequence** - (1x107 string in Train and Public Test, 130 in Private Test) Describes the RNA sequence, a combination of `A`, `G`, `U`, and `C` for each sample. Should be 107 characters long, and the first 68 bases should correspond to the 68 positions specified in `seq_scored` (note: indexed starting at 0).
# * **structure** - (1x107 string in Train and Public Test, 130 in Private Test) An array of `(`, `)`, and `.` characters that describe whether a base is estimated to be paired or unpaired. Paired bases are denoted by opening and closing parentheses e.g. (....) means that base 0 is paired to base 5, and bases 1-4 are unpaired.
# * **reactivity** - (1x68 vector in Train and Public Test, 1x91 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likely secondary structure of the RNA sample.
# * **deg_pH10** - (1x68 vector in Train and Public Test, 1x91 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating without magnesium at high pH (pH 10).
# * **deg_Mg_pH10** - (1x68 vector in Train and Public Test, 1x91 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating with magnesium in high pH (pH 10).
# * **deg_50C** - (1x68 vector in Train and Public Test, 1x91 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating without magnesium at high temperature (50 degrees Celsius).
# * **deg_Mg_50C** - (1x68 vector in Train and Public Test, 1x91 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating with magnesium at high temperature (50 degrees Celsius).
# * **`*_error_*`** - An array of floating point numbers, should have the same length as the corresponding `reactivity` or `deg_*` columns, calculated errors in experimental values obtained in `reactivity` and `deg_*` columns.
# * **predicted_loop_type** - (1x107 string) Describes the structural context (also referred to as 'loop type')of each character in `sequence`. Loop types assigned by bpRNA from Vienna RNAfold 2 structure. From the bpRNA_documentation: S: paired "Stem" M: Multiloop I: Internal loop B: Bulge H: Hairpin loop E: dangling End X: eXternal loop

# **It seems we also have a `signal_to_noise` and a `SN_filter` column. These columns control the 'quality' of samples, and as such are important training hyperparameters. We will explore them shortly:**

# In[7]:


#sneak peak
print(train.shape)
if ~train.isnull().values.any(): print('No missing values')
train.head()


# In[8]:


#sneak peak
print(test.shape)
if ~test.isnull().values.any(): print('No missing values')
test.head()


# In[9]:


#sneak peak
print(sample_sub.shape)
if ~sample_sub.isnull().values.any(): print('No missing values')
sample_sub.head()


# **Now we explore `signal_to_noise` and `SN_filter` distributions. As per the data tab of this competition the samples in `test.json` have been filtered in the following way:**
# 
# 1. Minimum value across all 5 conditions must be greater than -0.5.
# 2. Mean signal/noise across all 5 conditions must be greater than 1.0. [Signal/noise is defined as mean( measurement value over 68 nts )/mean( statistical error in measurement value over 68 nts)]
# 3. To help ensure sequence diversity, the resulting sequences were clustered into clusters with less than 50% sequence similarity, and the 629 test set sequences were chosen from clusters with 3 or fewer members. That is, any sequence in the test set should be sequence similar to at most 2 other sequences.

# In[10]:


fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.kdeplot(train['signal_to_noise'], shade=True, ax=ax[0])
sns.countplot(train['SN_filter'], ax=ax[1])

ax[0].set_title('Signal/Noise Distribution')
ax[1].set_title('Signal/Noise Filter Distribution');


# In[ ]:





# In[11]:


print(f"Samples with signal_to_noise greater than 1: {len(train.loc[(train['signal_to_noise'] > 1 )])}")
print(f"Samples with SN_filter = 1: {len(train.loc[(train['SN_filter'] == 1 )])}")
print(f"Samples with signal_to_noise greater than 1, but SN_filter == 0: {len(train.loc[(train['signal_to_noise'] > 1) & (train['SN_filter'] == 0)])}")


# **Update: as per [this discussion post](https://www.kaggle.com/c/stanford-covid-vaccine/discussion/183992), both public *and* private test datasets are now filtered with the same 3 above conditions.**

# # Feature Engineering
# 
# **Check out [Tito](https://www.kaggle.com/its7171)'s kernel [here](https://www.kaggle.com/its7171/gru-lstm-with-feature-engineering-and-augmentation) for the feature engineering code below. The `bpps` folder contains Base Pairing Probabilities matrices for each sequence. These matrices give the probability that each pair of nucleotides in the RNA forms a base pair. Each matrix is a symmetric square matrix the same length as the sequence. For a complete EDA of the `bpps` folder, see this notebook [here](https://www.kaggle.com/hidehisaarai1213/openvaccine-checkout-bpps?scriptVersionId=42460013).**

# In[12]:


def read_bpps_sum(df):
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").sum(axis=1))
    return bpps_arr

def read_bpps_max(df):
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").max(axis=1))
    return bpps_arr

def read_bpps_nb(df):
    #mean and std from https://www.kaggle.com/symyksr/openvaccine-deepergcn 
    bpps_nb_mean = 0.077522
    bpps_nb_std = 0.08914
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps = np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy")
        bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]
        bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std
        bpps_arr.append(bpps_nb)
    return bpps_arr 

train['bpps_sum'] = read_bpps_sum(train)
test['bpps_sum'] = read_bpps_sum(test)
train['bpps_max'] = read_bpps_max(train)
test['bpps_max'] = read_bpps_max(test)
train['bpps_nb'] = read_bpps_nb(train)
test['bpps_nb'] = read_bpps_nb(test)

#sanity check
train.head()


# **Let's explore these newly engineered features to see if they can be trusted (i.e., are their distributions similar across the training set and the two testing sets?)**

# In[13]:


fig, ax = plt.subplots(3, figsize=(15, 10))
sns.kdeplot(np.array(train['bpps_max'].to_list()).reshape(-1),
            color="Blue", ax=ax[0], label='Train')
sns.kdeplot(np.array(test[test['seq_length'] == 107]['bpps_max'].to_list()).reshape(-1),
            color="Red", ax=ax[0], label='Public test')
sns.kdeplot(np.array(test[test['seq_length'] == 130]['bpps_max'].to_list()).reshape(-1),
            color="Green", ax=ax[0], label='Private test')
sns.kdeplot(np.array(train['bpps_sum'].to_list()).reshape(-1),
            color="Blue", ax=ax[1], label='Train')
sns.kdeplot(np.array(test[test['seq_length'] == 107]['bpps_sum'].to_list()).reshape(-1),
            color="Red", ax=ax[1], label='Public test')
sns.kdeplot(np.array(test[test['seq_length'] == 130]['bpps_sum'].to_list()).reshape(-1),
            color="Green", ax=ax[1], label='Private test')
sns.kdeplot(np.array(train['bpps_nb'].to_list()).reshape(-1),
            color="Blue", ax=ax[2], label='Train')
sns.kdeplot(np.array(test[test['seq_length'] == 107]['bpps_nb'].to_list()).reshape(-1),
            color="Red", ax=ax[2], label='Public test')
sns.kdeplot(np.array(test[test['seq_length'] == 130]['bpps_nb'].to_list()).reshape(-1),
            color="Green", ax=ax[2], label='Private test')

ax[0].set_title('Distribution of bpps_max')
ax[1].set_title('Distribution of bpps_sum')
ax[2].set_title('Distribution of bpps_nb')
plt.tight_layout();


# **Looks like `bpps_max` and `bpps_sum` are okay to use, but there is a large difference in the distribution of `bpps_nb` in public vs. private test sets. So even if it improves our LB (or local CV scores), we do not know if it will help with the private test score. For this reason, I will not include it in training.**

# # Augmentation
# 
# **Augmentation code can be found in [Tito](https://www.kaggle.com/its7171)'s notebook [here](https://www.kaggle.com/its7171/how-to-generate-augmentation-data). It can be used to generate augmented samples that you can use for training augmentation and test time augmentation (TTA). We are essentially generating new `structures` and `predicted_loop_types` for each `sequence` using the software that was originally used to create them (ARNIE, ViennaRNA, and bpRNA).**

# In[14]:


AUGMENT=False


# In[15]:


# aug_df = pd.read_csv('../input/openvaccineaugmented/aug_data_n2.csv')
# print(aug_df.shape)
# aug_df.head()


# In[16]:


def aug_data(df):
    target_df = df.copy()
    new_df = aug_df[aug_df['id'].isin(target_df['id'])]
                         
    del target_df['structure']
    del target_df['predicted_loop_type']
    new_df = new_df.merge(target_df, on=['id','sequence'], how='left')

    df['cnt'] = df['id'].map(new_df[['id','cnt']].set_index('id').to_dict()['cnt'])
    df['log_gamma'] = 100
    df['score'] = 1.0
    df = df.append(new_df[df.columns])
    return df


# In[17]:


print(f"Samples in train before augmentation: {len(train)}")
print(f"Samples in test before augmentation: {len(test)}")

if AUGMENT:
    train = aug_data(train)
    test = aug_data(test)

print(f"Samples in train after augmentation: {len(train)}")
print(f"Samples in test after augmentation: {len(test)}")

print(f"Unique sequences in train: {len(train['sequence'].unique())}")
print(f"Unique sequences in test: {len(test['sequence'].unique())}")


# # Processing

# In[18]:


DENOISE = True


# In[19]:


target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']


# In[20]:


token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}


# In[21]:


def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):

        base_fea = np.transpose(
            np.array(
                df[cols]
                .applymap(lambda seq: [token2int[x] for x in seq])
                .values
                .tolist()
            ),
            (0, 2, 1)
        )
        bpps_sum_fea = np.array(df['bpps_sum'].to_list())[:,:,np.newaxis]
        bpps_max_fea = np.array(df['bpps_max'].to_list())[:,:,np.newaxis]

        return np.concatenate([base_fea,bpps_sum_fea,bpps_max_fea], 2)


# In[22]:


if DENOISE:
    train = train[train['signal_to_noise'] > .25]


# # Model
# 
# **The below RNN architecture is adapted from the one and only [Xhlulu](https://www.kaggle.com/xhlulu)'s notebook [here](https://www.kaggle.com/xhlulu/openvaccine-simple-gru-model). For his explanation of the model/procedure, see his discussion post [here](https://www.kaggle.com/c/stanford-covid-vaccine/discussion/182303). I have made minor tweaks to some parameters and added an LSTM to experiment with blending.**
# 
# **Note that for submission, the output must be the same length as the input, which is 107 for `train.json` and `test.json` and 130 for the private test set. However, this is not true for training, so training prediction sequences only need to be 68 long**
# 
# **So we actually build 3 different models: one for training, one for predicting public test, and one for predicting private test set, each with different sequence lengths and prediction lengths. Luckily, we only need to train one model, save its weights, and load these weights into the other models.**
# 
# **The last thing to set is the size of the embedding layer. In the context of NLP, the input dimension size of an embedding layer is the size of the vocabulary, which in our case is `len(token2int)`. The output dimension is typically the length of the pre-trained vectors you are using, like the GloVe vectors or Word2Vec vectors, which we don't have in this case, so we are free to experiment with different sizes.**

# In[23]:


len(token2int)


# In[24]:


# https://www.kaggle.com/c/stanford-covid-vaccine/discussion/183211
def rmse(y_actual, y_pred):
    mse = tf.keras.losses.mean_squared_error(y_actual, y_pred)
    return K.sqrt(mse)

def mcrmse(y_actual, y_pred, num_scored=len(target_cols)):
    score = 0
    for i in range(num_scored):
        score += rmse(y_actual[:, :, i], y_pred[:, :, i]) / num_scored
    return score


# In[25]:


def gru_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
                                tf.keras.layers.GRU(hidden_dim,
                                dropout=dropout,
                                return_sequences=True,
                                kernel_initializer='orthogonal'))

def lstm_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
                                tf.keras.layers.LSTM(hidden_dim,
                                dropout=dropout,
                                return_sequences=True,
                                kernel_initializer='orthogonal'))


def attention(x_inner, x_outer, n_factor, dropout):
    x_Q =  tf.keras.layers.Conv1D(n_factor, 1, activation='linear', 
                  kernel_initializer='glorot_uniform',
                  bias_initializer='glorot_uniform',
                 )(x_inner)
    x_K =  tf.keras.layers.Conv1D(n_factor, 1, activation='linear', 
                  kernel_initializer='glorot_uniform',
                  bias_initializer='glorot_uniform',
                 )(x_outer)
    x_V =  tf.keras.layers.Conv1D(n_factor, 1, activation='linear', 
                  kernel_initializer='glorot_uniform',
                  bias_initializer='glorot_uniform',
                 )(x_outer)
    x_KT = tf.keras.layers.Permute((2, 1))(x_K)
    res = tf.keras.layers.Lambda(lambda c: K.batch_dot(c[0], c[1]) / np.sqrt(n_factor))([x_Q, x_KT])
    att = tf.keras.layers.Lambda(lambda c: K.softmax(c, axis=-1))(res)
    att = tf.keras.layers.Lambda(lambda c: K.batch_dot(c[0], c[1]))([att, x_V])
    return att

def multi_head_attention(x, y, n_factor, n_head, dropout):
    if n_head == 1:
        att = attention(x, y, n_factor, dropout)
    else:
        n_factor_head = n_factor // n_head
        heads = [attention(x, y, n_factor_head, dropout) for i in range(n_head)]
        att = L.Concatenate()(heads)
        att = L.Dense(n_factor, 
                      kernel_initializer='glorot_uniform',
                      bias_initializer='glorot_uniform',
                     )(att)
    x = L.Add()([x, att])
    x = L.LayerNormalization()(x)
    if dropout > 0:
        x = L.Dropout(dropout)(x)
    return x



def Attention_layer(hidden_dim):
    for unit in [64*2, 32*2]:
        xs = []
        multi_head_attention(hidden_dim, hidden_dim, unit, 4, 0.0)
        xs.append(x)
    hidden_dim = hidden_dim.Concatenate()(xs)
    return hidden_dim
            

          
            
#rnn = 'multi_head_attention' , 'VAE_layer', CNN_GCN_layer, 
def build_model(rnn='sadsad', convolve=False, conv_dim=512, 
                dropout=.4, sp_dropout=.2, embed_dim=200,
                hidden_dim=256, layers=3,
                seq_len=107, pred_len=68):
    
###############################################
#### Inputs
###############################################

    import tensorflow.keras.layers as L
    import tensorflow as tf       
    import keras.backend as K

#     if Structure_Interpretability_Test == True:
#         inputs = tf.keras.layers.Input(shape=(seq_len, 3))
#     else:
    inputs = tf.keras.layers.Input(shape=(seq_len, 5))

    categorical_feats = inputs[:, :, :3]
    numerical_feats = inputs[:, :, 3:]

    embed = tf.keras.layers.Embedding(input_dim=len(token2int),
                                      output_dim=embed_dim)(categorical_feats)
    reshaped = tf.reshape(
        embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))
    
    reshaped = tf.keras.layers.concatenate([reshaped, numerical_feats], axis=2)
    hidden = tf.keras.layers.SpatialDropout1D(sp_dropout)(reshaped)
#     print(" Hidden",type(hidden),hidden.shape,hidden)
    if convolve:
        hidden = tf.keras.layers.Conv1D(conv_dim, 5, padding='same', activation=tf.keras.activations.swish)(hidden)

###############################################
#### RNN Layers
###############################################

    if rnn is 'gru':
        for _ in range(layers):
            hidden = gru_layer(hidden_dim, dropout)(hidden)
        
    elif rnn is 'lstm':
        for _ in range(layers):
            hidden = lstm_layer(hidden_dim, dropout)(hidden)
            
#     elif rnn is 'CNN_GCN_layer':
#         for _ in range(layers):
#             hidden = CNN_GCN_layer(hidden_dim, dropout)(hidden)
            
#     elif rnn is 'multi_head_attention':
#         for _ in range(layers):
#             hidden = lstm_layer(hidden_dim, dropout)(hidden)
            

###############################################
#### Output
###############################################





    out = hidden[:, :pred_len]
    out = tf.keras.layers.Dense(5, activation='linear')(out)
    
    model = tf.keras.Model(inputs=inputs, outputs=out)
    adam = tf.optimizers.Adam()
    
#     High degree: If the coefficient value lies between ± 0.50 and ± 1, then it is said to be a strong correlation. 
#     Moderate degree: If the value lies between ± 0.30 and ± 0.49, then it is said to be a medium correlation.
#     Low degree: When the value lies below + . 29, then it is said to be a small correlation
    
    def tf_pearson(x, y):    
        mx = tf.math.reduce_mean(input_tensor=x)
        my = tf.math.reduce_mean(input_tensor=y)
        xm, ym = x-mx, y-my
        r_num = tf.math.reduce_mean(input_tensor=tf.multiply(xm,ym))        
        r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
        return  r_num / r_den
    model.compile(optimizer=adam, loss=mcrmse,  metrics=['mse', tf_pearson, 'mae'] )
   

    return model


# In[26]:


test_model = build_model(rnn='gru')
test_model.summary()


# # KFold Training and Inference
# 
# **In previous commits, I either filtered by `SN_filter == 1` or with `signal_to_noise > 1`. But it seems that these RNN models generalize better when exposed to the noisier samples in the dataset. If you review the `tf.keras` [documentation](https://www.tensorflow.org/api_docs/python/tf/keras/Model), you can see that you can pass a Numpy array of weights during training used to weight the loss function. So we can weight samples with higher `signal_to_noise` values more during training than the noisier samples. As inspired by [Tito](https://www.kaggle.com/its7171), we will pass the following array to `sample_weight`: `np.log1p(train.signal_to_noise + epsilon)/2` where epsilon is a small number to ensure we don't get `log(1)` for any weights.**
# 
# **But since the competition hosts have said [here](https://www.kaggle.com/c/stanford-covid-vaccine/discussion/183992), the public and private test sets only contain samples where `SN_filter == 1`, so we ought to validate against the such samples as well:**

# In[27]:


def train_and_infer(rnn, STRATIFY=True, FOLDS=4, EPOCHS=50, BATCH_SIZE=64,
                    REPEATS=1, SEED=34, VERBOSE=2):


    #get test now for OOF 
    public_df = test.query("seq_length == 107").copy()
    private_df = test.query("seq_length == 130").copy()
    
    
    if Structure_Interpretability_Test == True:
            s_107 = '.' * 107
            l_107 = 'E'* 107
            s_130 = '.' * 130
            l_130 = 'E'* 130
            train.loc[:, 'structure'] = s_107
            train.loc[:, 'predicted_loop_type'] = l_107
            public_df.loc[:, 'structure'] = s_107
            private_df.loc[:, 'structure'] = s_130
            public_df.loc[:, 'predicted_loop_type'] = l_107
            private_df.loc[:, 'predicted_loop_type'] = l_130


    print(private_df['structure'])
    print(public_df['structure'])
    print(private_df['predicted_loop_type'])
    print(public_df['predicted_loop_type'])


#     if Structure_Interpretability_Test == True:
#         private_preds = np.zeros((private_df.shape[0], 130, 3))
#         public_preds = np.zeros((public_df.shape[0], 107, 3))
#     else:
    private_preds = np.zeros((private_df.shape[0], 130, 5))
    public_preds = np.zeros((public_df.shape[0], 107, 5))        
        
    public_inputs = preprocess_inputs(public_df)
    private_inputs = preprocess_inputs(private_df)

    #to evaluate TTA effects/post processing
    holdouts = []
    holdout_preds = []
    
    #to view learning curves
    histories = []
    
    #put similar RNA in the same fold
    gkf = GroupKFold(n_splits=FOLDS)
#     kf=KFold(n_splits=FOLDS, random_state=SEED)
    kf=KFold(n_splits=FOLDS)
#     kmeans_model = KMeans(n_clusters=200, random_state=SEED).fit(preprocess_inputs(train)[:,:,0])




    kmeans_model = KMeans(n_clusters=200).fit(preprocess_inputs(train)[:,:,0])
    train['cluster_id'] = kmeans_model.labels_
    
    

    for _ in range(REPEATS):
        
        for f, (train_index, val_index) in enumerate((gkf if STRATIFY else kf).split(train,
                train['reactivity'], train['cluster_id'] if STRATIFY else None)):

            #define training callbacks
            lr_callback = tf.keras.callbacks.ReduceLROnPlateau(patience=8, 
                                                               factor=.1,
                                                               #min_lr=1e-5,
                                                               verbose=VERBOSE)
            save = tf.keras.callbacks.ModelCheckpoint(f'model-{f}.h5')

            #define sample weight function
            epsilon = .1
            sample_weighting = np.log1p(train.iloc[train_index]['signal_to_noise'] + epsilon) / 2

            #get train data
            trn = train.iloc[train_index]
            trn_ = preprocess_inputs(trn)
            trn_labs = np.array(trn[target_cols].values.tolist()).transpose((0, 2, 1))

            #get validation data
            val = train.iloc[val_index]
            val_all = preprocess_inputs(val)
            val = val[val.SN_filter == 1]
            val_ = preprocess_inputs(val)
            val_labs = np.array(val[target_cols].values.tolist()).transpose((0, 2, 1))

            #pre-build models for different sequence lengths
            model = build_model(rnn=rnn)
            model_short = build_model(rnn=rnn,seq_len=107, pred_len=107)
            model_long = build_model(rnn=rnn,seq_len=130, pred_len=130)

            #train model
            history = model.fit(
                trn_, trn_labs,
                validation_data = (val_, val_labs),
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                sample_weight=sample_weighting,
                callbacks=[save, lr_callback],
                verbose=VERBOSE
            )

            histories.append(history)

            #load best models
            model.load_weights(f'model-{f}.h5')
            model_short.load_weights(f'model-{f}.h5')
            model_long.load_weights(f'model-{f}.h5')

            holdouts.append(train.iloc[val_index])
            holdout_preds.append(model.predict(val_all))

            public_preds += model_short.predict(public_inputs) / (FOLDS * REPEATS)
            private_preds += model_long.predict(private_inputs) / (FOLDS * REPEATS)
        

            
            val_losses = []
            y_preds_best = None
            PRED_COLS = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C']
            y_valid= val_labs
            y_preds = model.predict(val_)
            
            
            mcloss = mcrmse(val_labs, y_preds)
            val_losses.append(mcloss)
            print('np.min(val_losses)',np.min(val_losses),'val_losses',np.mean(val_losses),'mcloss',np.mean(mcloss))

            if np.min(np.mean(val_losses)) == np.mean(mcloss):
                y_preds_best = y_preds

            print('y_preds_best shape, y_valid shape', y_valid.shape, y_preds_best.shape)
            fig, ax = plt.subplots(1, 3, figsize=(24, 8))
            for i, p in enumerate(PRED_COLS):
                ax[2].scatter(y_valid[:, :, i].flatten(), y_preds_best[:, :, i].flatten(), alpha=0.5)#(not correct but) works if y_preds_best= y_preds = model.predict(val_)

            ax[2].legend(PRED_COLS)
            ax[2].set_xlabel('y_true')
            ax[2].set_ylabel('y_predicted')
            plt.show()
        
        
        
        
        print(f'min val_loss: {np.min(val_losses):.4f} at {np.argmin(val_losses) + 1} epoch')

        
#         del model, model_short, model_long
        
    return holdouts, holdout_preds, public_df, public_preds, private_df, private_preds, histories,model, model_short, model_long


# ### GRU & LSTM

# In[28]:


import time
start_time = time.time()  # Record the start time


gru_holdouts, gru_holdout_preds, public_df, gru_public_preds, private_df, gru_private_preds, gru_histories,model, model_short, model_long  = train_and_infer(rnn='gru')


end_time = time.time()  # Record the start time
duration = end_time - start_time
print("duration ,end_time , start_time",duration ,end_time , start_time)


# In[29]:


import time
start_time = time.time()  # Record the start time



lstm_holdouts, lstm_holdout_preds, public_df, lstm_public_preds, private_df, lstm_private_preds, lstm_histories,model, model_short, model_long = train_and_infer(rnn='lstm')
end_time = time.time()  # Record the start time
duration = end_time - start_time
print("duration ,end_time , start_time",duration ,end_time , start_time)


# In[30]:


STRATIFY=True
FOLDS=4
EPOCHS=2
BATCH_SIZE=64
REPEATS=1
SEED=34
VERBOSE=2


#get test now for OOF 
public_df = test.query("seq_length == 107").copy()
private_df = test.query("seq_length == 130").copy()
private_preds = np.zeros((private_df.shape[0], 130, 5))
public_preds = np.zeros((public_df.shape[0], 107, 5))
public_inputs = preprocess_inputs(public_df)
private_inputs = preprocess_inputs(private_df)

#to evaluate TTA effects/post processing
holdouts = []
holdout_preds = []

#to view learning curves
histories = []

#put similar RNA in the same fold
gkf = GroupKFold(n_splits=FOLDS)
#     kf=KFold(n_splits=FOLDS, random_state=SEED)
kf=KFold(n_splits=FOLDS)
#     kmeans_model = KMeans(n_clusters=200, random_state=SEED).fit(preprocess_inputs(train)[:,:,0])

kmeans_model = KMeans(n_clusters=200).fit(preprocess_inputs(train)[:,:,0])
train['cluster_id'] = kmeans_model.labels_

    
for f, (train_index, val_index) in enumerate((gkf if STRATIFY else kf).split(train,
        train['reactivity'], train['cluster_id'] if STRATIFY else None)):

    #define training callbacks
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(patience=8, 
                                                       factor=.1,
                                                       #min_lr=1e-5,
                                                       verbose=VERBOSE)
    save = tf.keras.callbacks.ModelCheckpoint(f'model-{f}.h5')

    #define sample weight function
    epsilon = .1
    sample_weighting = np.log1p(train.iloc[train_index]['signal_to_noise'] + epsilon) / 2

    #get train data
    trn = train.iloc[train_index]
    trn_ = preprocess_inputs(trn)
    trn_labs = np.array(trn[target_cols].values.tolist()).transpose((0, 2, 1))
    #get validation data
    val = train.iloc[val_index]
    val_all = preprocess_inputs(val)
    val = val[val.SN_filter == 1]
    val_ = preprocess_inputs(val)
    val_labs = np.array(val[target_cols].values.tolist()).transpose((0, 2, 1))


# In[31]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
import tensorflow as tf

# Load the pre-trained LSTM model
# model = load_model('lstm_model.h5')

# Get the input layer shape
input_shape = model.input_shape[1:]

# Generate an input sample
input_sample = trn_[16].reshape(1, *input_shape)#np.random.rand(1, *input_shape)
input_sample = tf.convert_to_tensor(input_sample)

# Define a function to compute the saliency map
def compute_saliency_map(input_sample, model):
    with tf.GradientTape() as tape:
        # starts a TensorFlow GradientTape context. The GradientTape allows us to record the operations performed on the tensors during the forward pass, and then compute the gradients of the output with respect to the inputs during the backward pass.
        tape.watch(input_sample)
        # tells the GradientTape to watch the input_sample tensor, so that the gradients of the loss with respect to the input can be computed later.
        prediction = model(input_sample)
        #uses the model to make a prediction on the input_sample. The prediction is a tensor of shape (1, number of classes).
        loss = tf.reduce_mean(prediction[:, 0])
        #computes the loss as the mean of the first column of the prediction tensor. This assumes that the first column of the prediction corresponds to the prediction of the first class.
    grads = tape.gradient(loss, input_sample)
    #This line computes the gradients of the loss with respect to the input_sample using the GradientTape. The gradients are a tensor of the same shape as the input sample.
    grads = grads / (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
    #This line normalizes the gradients by dividing them by their standard deviation plus a small constant (1e-5). This is to ensure that the values of the gradients are between 0 and 1.
    return grads

# Compute the saliency map for the input sample
saliency_map = compute_saliency_map(input_sample, model)
#The function computes the saliency map, which is a visual representation of the influence of each input feature on the prediction.
# Plot the saliency map as a heatmap
sns.heatmap(saliency_map.numpy().reshape(*input_shape)[:20], cmap='Reds', annot=True, vmin=0, vmax=1)
sns.heatmap(saliency_map.numpy().reshape(*input_shape), cmap='Reds', annot=True, vmin=0, vmax=1)
plt.savefig('saliency_map_GRU.jpg', format='jpg')
plt.show()
# The saliency map is a tensor of the same shape as the input sample, and its values represent the influence of each input feature on the prediction.
# A high value indicates that a change in the corresponding input feature will have a significant impact on the prediction, while a low value indicates 
# that the feature has a small influence on the prediction


# In[32]:


input_shape


# In[33]:


saliency_map.numpy().reshape(*input_shape)[:10].shape


# If an index has the maximum repetition in the influential_instances list, it means that the predicted label for the equence at that index has a high residual(loss) and is considered an influential instance. This could indicate that the model has a particularly difficult time predicting the label for that sequence and it may be worth investigating the reasons for this difficulty. It could also suggest that the data point has some unique characteristics that are not well-represented in the training data, which could indicate a need for additional data or feature engineering:

# In[34]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# # Generate sample data
# data = np.random.normal(size=(100, 10))
# labels = np.random.normal(size=(100, 1))

# # Build LSTM model
# model = Sequential()
# model.add(LSTM(64, input_shape=(10, 1)))
# model.add(Dense(1, activation='linear'))
# model.compile(loss='mse', optimizer='adam')

# # Train the model
# model.fit(data.reshape((100, 10, 1)), labels, epochs=50, batch_size=32)

# # Predict the labels for the data
pred_labels =  model(val_)#model.predict(data.reshape((100, 10, 1)))
     

# Calculate the residuals
residuals = np.abs(val_labs - pred_labels)

# Calculate the mean and standard deviation of the residuals
mean_res = np.mean(residuals)
std_res = np.std(residuals)

# Calculate the z-score for each residual
z_scores = (residuals - mean_res) / std_res

# Identify the influential instances
threshold = 3  # Adjust this threshold as needed
influential_instances = np.where(z_scores > threshold)[0]


def count_frequency(numbers):
    frequency_dict = {}
    for num in numbers:
        if num in frequency_dict:
            frequency_dict[num] += 1
        else:
            frequency_dict[num] = 1
    return frequency_dict

# print(count_frequency(influential_instances))
max_value = max(count_frequency(influential_instances).values())

def find_key(my_dict, value):
    for key in my_dict:
        if my_dict[key] == value:
            return key
        
Instance = find_key(count_frequency(influential_instances), max_value)
print('The most influential instance is ',Instance, ' which has repeated ',max_value,' times') 


# In[35]:


from keras.models import Model
from keras.layers import Input, LSTM, Dense

# Define input shape
# input_shape = (timesteps, input_dim)

# Define input layer
input_shape = model.input_shape[1:]



# Get the activations for the LSTM layer
lstm_activations_model = Model(inputs=model.input, outputs=model.get_layer(index=1).output)
lstm_activations = lstm_activations_model.predict(public_inputs)

# Get the activations for the output layer
output_activations_model = Model(inputs=model.input, outputs=model.output)
output_activations = output_activations_model.predict(public_inputs)


import matplotlib.pyplot as plt


print(lstm_activations.shape)
print(output_activations.shape)

# Plot the LSTM activations
plt.figure(figsize=(10, 5))
for i in range(lstm_activations.shape[2]):
    plt.plot(lstm_activations[0, :, i], label='LSTM Activation {}'.format(i+1))
plt.legend()
plt.title('LSTM Activations')
plt.xlabel('RNA Sequence position')
plt.ylabel('Activation Value')
plt.savefig('LSTMActivation_GRU.jpg', format='jpg')
plt.show()

# Plot the output activations
plt.figure(figsize=(10, 5))
for i in range(output_activations.shape[2]):
    plt.plot(output_activations[0, :, i], label='Output Activation {}'.format(i+1))
plt.legend()
plt.title('Output Activations')
plt.xlabel('RNA Sequence position')
plt.ylabel('Activation Value')
plt.savefig('OutputActivation_GRU.jpg', format='jpg')
plt.show()


# In[36]:


import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load data
data = lstm_activations  # replace with your own data file path

# Reshape data into 2D array
n_samples = data.shape[0]
n_features = np.prod(data.shape[1:])
data_2d = data.reshape(n_samples, n_features)

# Apply t-SNE to map data to 2D space
tsne = TSNE(n_components=2, random_state=42)
data_2d_tsne = tsne.fit_transform(data_2d)

# Get cluster labels (optional)
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=100, random_state=42)
cluster_labels = kmeans.fit_predict(data_2d_tsne)

# Plot the results with cluster labels
plt.scatter(data_2d_tsne[:, 0], data_2d_tsne[:, 1], c=cluster_labels)
plt.show()


# In[37]:


import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load data
data = output_activations  # replace with your own data file path

# Reshape data into 2D array
n_samples = data.shape[0]
n_features = np.prod(data.shape[1:])
data_2d = data.reshape(n_samples, n_features)

# Apply t-SNE to map data to 2D space
tsne = TSNE(n_components=2, random_state=42)
data_2d_tsne = tsne.fit_transform(data_2d)

# Get cluster labels (optional)
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=100, random_state=42)
cluster_labels = kmeans.fit_predict(data_2d_tsne)

# Plot the results with cluster labels
plt.scatter(data_2d_tsne[:, 0], data_2d_tsne[:, 1], c=cluster_labels)
plt.show()


# The cosine similarity between two vectors ranges from -1 to 1, where a value of 1 indicates that the vectors are identical, a value of -1 indicates that the vectors are opposite to each other, and a value of 0 indicates that the vectors are orthogonal (perpendicular) to each other.

# In[38]:


# Next, we plot the cosine similarity as a heatmap using imshow function from matplotlib. 
# We set the axis labels to indicate the index of each array, and we add a colorbar to indicate 
# the similarity values. We rotate the tick labels for the x-axis to improve readability. 
# Finally, we loop over the similarity values and add text annotations to the heatmap.
# The resulting plot shows the cosine similarity between each pair of arrays as a heatmap.
# The brighter the color, the higher the cosine similarity between the corresponding arrays.

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Generate 629 random 68x5 arrays
arrays = data[:30]#np.random.rand(629, 68, 5)

# Compute the cosine similarity between all pairs of arrays
similarity = cosine_similarity(arrays.reshape(30, -1))

# Reshape the similarity array back to a 629x629 square matrix
similarity = similarity.reshape(30, 30)

# Plot the cosine similarity as a heatmap
fig, ax = plt.subplots()
im = ax.imshow(similarity, cmap='YlGnBu')

# Add a colorbar
cbar = ax.figure.colorbar(im, ax=ax)

# Set the axis labels
ax.set_xticks(np.arange(len(arrays)))
ax.set_yticks(np.arange(len(arrays)))
ax.set_xticklabels(np.arange(1, len(arrays)+1))
ax.set_yticklabels(np.arange(1, len(arrays)+1))
ax.set_xlabel('Array index')
ax.set_ylabel('Array index')

# Rotate the tick labels and set their alignment
plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
         rotation_mode='anchor')

# Loop over the data and add text annotations
for i in range(len(arrays)):
    for j in range(len(arrays)):
        text = ax.text(j, i, f'{similarity[i, j]:.2f}',
                       ha='center', va='center', color='black')

# Set the title
ax.set_title('Cosine similarity between 30 out of 629 arrays')
plt.savefig('cosinesimilarity_GRU.jpg', format='jpg')

# Show the plot
plt.show()


# When reporting the performance of your RNA embedding method on a dataset of 700 RNAs, you could consider the following steps:
# 
# Compute the cosine similarity, sequence similarity, and structural similarity for all pairs of RNAs in the dataset.
# For each RNA, calculate its average similarity score (cosine similarity, sequence similarity, and structural similarity) with all other RNAs in the dataset.
# Report the average similarity scores for the entire dataset as well as any relevant statistics such as the standard deviation or range of the scores.
# Compare the performance of your embedding method with other state-of-the-art methods for RNA similarity comparison, if applicable.
# It may also be useful to visualize the results using a heatmap or a scatter plot to show the relationship between the different similarity scores.

# The Below code is the computation for a single cell of the above heatmap:

# In[39]:


# def embedding_similarity(data):
#     arrays = data
#     arrays = arrays.reshape(data.shape[0], -1)#629 for public
#     vec1 = arrays[i]# embedding for the sequence in public_df.iloc[0]['sequence']
#     vec2 = arrays[j]#embedding for the sequence in public_df.iloc[9]['sequence']
#     vec1 = vec1.reshape(1, -1)
#     vec2 = vec2.reshape(1, -1)
#     similarity = cosine_similarity(data, data.T)
#     return similarity

# embedding_similarity(data)


# Vienna similarity between RNA sequences:

# The alignment score provides a measure of the structural similarity between the two RNA sequences. The score is calculated based on the number of matches, mismatches, gaps, and other alignment parameters.
# 
# The score is typically expressed as a numerical value, where higher scores indicate a greater degree of similarity between the two RNA structures. The specific range of scores can vary depending on the alignment method and parameters used.
# 
# In general, an alignment score of 0 indicates no similarity between the two sequences, while a score closer to 1 indicates a higher degree of similarity. However, it's important to keep in mind that the interpretation of the alignment score depends on the specific RNA sequences being compared and the biological context in which they are studied.
# 
# In addition to the alignment score, it can also be useful to examine the aligned sequences themselves to gain insights into the specific similarities and differences between the two RNAs. For example, examining the alignment can help identify conserved secondary structures, functional domains, or other biologically relevant features.
# 
# 
# 
# 

# In[40]:


from Bio import pairwise2
from Bio.Seq import Seq

def sequence_similarity(public_df, i, j):

    # Define the loop and dot bracket information for two RNAs
    rna1 = public_df.iloc[i]['sequence']
    rna2 = public_df.iloc[j]['sequence']


    # Calculate the structural similarity between the two RNAs using pairwise sequence alignment
    alignments = pairwise2.align.globalxx(rna1, rna2)

    # Print the alignment score and the aligned sequences
    best_alignment = alignments[0]

    alignment_score = best_alignment.score

    # Calculate similarity score
    aligned_seq1, aligned_seq2, _, _,_ = best_alignment
    similarity_score = alignment_score / len(aligned_seq1)

#     print("Alignment score:", alignment.score)
#     print("similarity_score:", similarity_score)
    return similarity_score


# Vienna similarity between RNA structures:

# In[41]:


import Levenshtein  # install using pip install python-Levenshtein
def structural_similarity(public_df, i, j):
    # define the dot-bracket notations for two RNA structures
    rna1_structure = public_df.iloc[i]['structure']
    rna2_structure = public_df.iloc[j]['structure']

    # calculate the tree edit distance between the two structures
    distance = Levenshtein.distance(rna1_structure, rna2_structure)

    # calculate the structural similarity between the two structures
    similarity = 1 - distance / len(rna1_structure)

    # print the structural similarity between the two structures
#     print("The structural similarity between the two RNA structures is:", similarity)
    lev_distance = RNA.hamming_distance(rna1_structure, rna2_structure)

    # Print the Levenshtein distance
#     print("The Levenshtein distance between the two secondary structures is:", lev_distance)
    return similarity


# In this code, compare loops. We also define the loop types using a dictionary. The code then calculates the structural similarity between the two RNA molecules based on their loop types by counting the number of matching loop types and dividing it by the length of the sequences. The loop types of each RNA molecule are also printed along with their positions if they differ between the two sequences.

# In[42]:


def loop_type_similarity(public_df, i , j):

    seq1 = public_df.iloc[i]['predicted_loop_type']
    seq2 =public_df.iloc[j]['predicted_loop_type']
    # Define loop types
    loop_types = {
        'E': 'external loop',
        'S': 'stem',
        'B': 'bulge',
        'H': 'hairpin loop',
        'X': 'interior loop',
        'I': 'internal loop',
        'M': 'multiloop'


    }

    # Calculate structural similarity based on loop types
    similarity = sum([1 for i in range(len(seq1)) if seq1[i] == seq2[i]]) / len(seq1)

    # Print the structural similarity of two RNA molecules
#     print("The structural similarity between the two RNA sequences is:", similarity)

    # Print the loop types of each RNA molecule
#     for i in range(len(seq1)):
#         if seq1[i] != seq2[i]:
#             print("Sequence 1 has", loop_types[seq1[i]], "at position", i+1, "and sequence 2 has", loop_types[seq2[i]], "at position", i+1)
    return similarity


#  you have the dot-bracket notation for the second structure of two RNA molecules, you can compare their structural similarity using the tree edit distance algorithm, which measures the minimum number of edit operations (insertions, deletions, and substitutions) required to transform one structure into the other. Here's an example code snippet:

# In[43]:


data[:10].shape


# Here public_inputs are the input sequences and data is output embeddings. so data[0] represents the embedding for public_df.iloc[0]['sequence'] sequence.

# Here's some sample Python code to compare the cosine similarities matrix with the sequence similarities, structural similarities, and loop similarities matrices:This code calculates the correlation coefficients between the cosine similarities matrix and the other three matrices, and then creates a heatmap to visualize the results. The diagonal elements of the heatmap are all 1, since each matrix is perfectly correlated with itself. The off-diagonal elements show the correlation coefficients between the different pairs of matrices. The cmap parameter of imshow() specifies the color scheme for the heatmap (in this case, blue for negative correlation, red for positive correlation).

# 

# In[44]:


import numpy as np
import Levenshtein
import RNA
def compute_similarity(dataset):
    """
    Compute cosine similarity, sequence similarity, and structural similarity for all pairs of RNAs in the dataset.
    
    Args:
        dataset: a list of K RNA sequences
    
    Returns:
        cosine_similarities: a K x K numpy array of cosine similarities between RNA sequences
        sequence_similarities: a K x K numpy array of sequence similarities between RNA sequences
        structural_similarities: a K x K numpy array of structural similarities between RNA sequences
        loop_similarities: a K x K numpy array of loop similarities between RNA sequences

    """
    K = len(dataset)
    
    n_samples, n_features, n_dims = dataset.shape
    
    # Reshape the dataset into a 2D array of shape (n_samples, n_features * n_dims)
    vectorized_dataset = np.reshape(dataset, (n_samples, n_features * n_dims))
    
    # Compute cosine similarity
    cosine_similarities = cosine_similarity(vectorized_dataset)
    
    # Compute sequence similarity
    sequence_similarities = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            sequence_similarities[i, j] = sequence_similarity(public_df, i, j) / max(len(dataset[i]), len(dataset[j]))
    
    # Compute structural similarity
    structural_similarities = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            structural_similarities[i, j] = structural_similarity(public_df, i, j) / max(len(dataset[i]), len(dataset[j]))
    
    
    loop_similarities = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            loop_similarities[i, j] = loop_type_similarity(public_df, i , j) / max(len(dataset[i]), len(dataset[j]))

    
    
    return cosine_similarities, sequence_similarities, structural_similarities, loop_similarities


cosine_similarities, sequence_similarities, structural_similarities, loop_similarities = compute_similarity(data[:30])
print(cosine_similarities.shape, sequence_similarities.shape, structural_similarities.shape, loop_similarities.shape)


# Caveat: The below  correlation plot is reasonable when we train the model only with sequences and then plot the embeddings correlation with sequence,loop and strucrures.

# In[45]:


import numpy as np
import matplotlib.pyplot as plt

# Load the similarity matrices
cosine_sim = cosine_similarities#np.load('cosine_similarities.npy')
seq_sim = sequence_similarities#np.load('sequence_similarities.npy')
struct_sim = structural_similarities#np.load('structural_similarities.npy')
loop_sim = loop_similarities#np.load('loop_similarities.npy')

# Normalize the similarity matrices
cosine_sim_norm = cosine_sim / np.max(cosine_sim)
seq_sim_norm = seq_sim / np.max(seq_sim)
struct_sim_norm = struct_sim / np.max(struct_sim)
loop_sim_norm = loop_sim / np.max(loop_sim)

# Calculate the correlation coefficients
corr_cos_seq = np.corrcoef(cosine_sim_norm.flatten(), seq_sim_norm.flatten())[0,1]
corr_cos_struct = np.corrcoef(cosine_sim_norm.flatten(), struct_sim_norm.flatten())[0,1]
corr_cos_loop = np.corrcoef(cosine_sim_norm.flatten(), loop_sim_norm.flatten())[0,1]

# Create a heatmap of the correlation coefficients
corr_matrix = np.array([[1, corr_cos_seq, corr_cos_struct, corr_cos_loop],
                       [corr_cos_seq, 1, 0, 0],
                       [corr_cos_struct, 0, 1, 0],
                       [corr_cos_loop, 0, 0, 1]])
fig, ax = plt.subplots()
im = ax.imshow(corr_matrix, cmap='coolwarm')
ax.set_xticks(np.arange(len(['Cosine', 'Sequence', 'Structural', 'Loop'])))
ax.set_yticks(np.arange(len(['Cosine', 'Sequence', 'Structural', 'Loop'])))
ax.set_xticklabels(['Cosine', 'Sequence', 'Structural', 'Loop'])
ax.set_yticklabels(['Cosine', 'Sequence', 'Structural', 'Loop'])
plt.colorbar(im)
plt.savefig('correlation_cosine_str_loop_seq_GRU.jpg', format='jpg')
plt.show()


# In[46]:


import tensorflow as tf
import matplotlib.pyplot as plt

# Get the attention weights
attention_layer = model.layers[0]
attention_model = tf.keras.models.Model(inputs=model.inputs, outputs=attention_layer.output)
attention_weights = attention_model.predict(val_)

# Reshape attention weights
attention_weights = attention_weights.reshape(384,5,107)

# Plot the attention weights
fig, ax = plt.subplots(figsize=(50, 10))
im = ax.imshow(attention_weights[5,:,:], cmap='YlGnBu', interpolation='nearest')
ax.set_xlabel('RNA Sequence position')
ax.set_ylabel('target')
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Attention weight', rotation=-90, va="bottom")

plt.savefig('attaaaaaa.jpg', format='jpeg')
plt.show()
# Find the most important input timesteps
max_attention_timesteps = np.argmax(attention_weights, axis=1)
print("Most important input position for each sample:", max_attention_timesteps)


# ### Learning Curves and Evaluation

# In[47]:


def plot_learning_curves(results):

    fig, ax = plt.subplots(1, len(results['histories']), figsize = (20, 10))
    fig2, ax2 = plt.subplots(1, len(results['histories']), figsize = (20, 10))
    fig3, ax3 = plt.subplots(1, len(results['histories']), figsize = (20, 10))
    fig4, ax4 = plt.subplots(1, len(results['histories']), figsize = (20, 10))

    for i, result in enumerate(results['histories']):
        for history in result:
            ############################################################## MCRMSE 

            ax[i].plot(history.history['loss'], color='C0')
            ax[i].plot(history.history['val_loss'], color='C1')
            ax[i].plot(history.history['mae'], color='C2')
            ax[i].plot(history.history['mse'], color='C3')
            ax[i].set_title(f"{results['models'][i]}")
            ax[i].set_ylabel('Loss')
            ax[i].set_xlabel('Epoch')
            ax[i].legend(['train', 'validation','MAE','MSE'], loc = 'upper right')
                    
            ############################################################## Pearson Correlation
            ax2[i].plot(history.history['tf_pearson'], color='C0')
            ax2[i].plot(history.history['val_tf_pearson'], color='C1')
            ax2[i].set_title(f"{results['models'][i]}")
            ax2[i].set_ylabel('Pearson correlation')
            ax2[i].set_xlabel('Epoch')
            ax2[i].legend(['train', 'validation'], loc = 'upper right')
            
            
results = {
            "models" : ['GRU', 'LSTM'],    
            "histories" : [gru_histories, lstm_histories],
            }


# In[ ]:


def plot_label_clusters(vae, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = vae.encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


# (x_train, y_train), _ = keras.datasets.mnist.load_data()
# x_train = np.expand_dims(x_train, -1).astype("float32") / 255

# plot_label_clusters(vae, x_train, y_train)


# In[ ]:


#https://www.kaggle.com/xhlulu/openvaccine-simple-gru-model
def format_predictions(test_df, test_preds, val=False):
    preds = []
    
    for df, preds_ in zip(test_df, test_preds):
        for i, uid in enumerate(df['id']):
            single_pred = preds_[i]

            single_df = pd.DataFrame(single_pred, columns=target_cols)
            single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]
            if val: single_df['SN_filter'] = df[df['id'] == uid].SN_filter.values[0]

            preds.append(single_df)
    return pd.concat(preds).groupby('id_seqpos').mean().reset_index() if AUGMENT else pd.concat(preds)


# In[ ]:


def get_error(preds):
    val = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)

    val_data = []
    for mol_id in val['id'].unique():
        sample_data = val.loc[val['id'] == mol_id]
        sample_seq_length = sample_data.seq_length.values[0]
        for i in range(68):
            sample_dict = {
                           'id_seqpos' : sample_data['id'].values[0] + '_' + str(i),
                           'reactivity_gt' : sample_data['reactivity'].values[0][i],
                           'deg_Mg_pH10_gt' : sample_data['deg_Mg_pH10'].values[0][i],
                           'deg_Mg_50C_gt' : sample_data['deg_Mg_50C'].values[0][i],
                           }
            
            val_data.append(sample_dict)
            
    val_data = pd.DataFrame(val_data)
    val_data = val_data.merge(preds, on='id_seqpos')

    rmses = []
    mses = []
    
    for col in ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']:
        rmse = ((val_data[col] - val_data[col+'_gt']) ** 2).mean() ** .5
        mse = ((val_data[col] - val_data[col+'_gt']) ** 2).mean()
        rmses.append(rmse)
        mses.append(mse)
        print(col, rmse, mse)
    print(np.mean(rmses), np.mean(mses))
    print('')


# In[ ]:


results


# In[50]:


import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves(results):
    # Number of models
    num_models = len(results['histories'])
    
    # Create average history placeholders
    avg_histories = []
    
    # Calculate the average of histories
    for i, result in enumerate(results['histories']):
        avg_history = {}
        
        # Assuming all histories for each model have the same length
        num_epochs = len(result[0].history['loss'])
        
        # Initialize average metrics
        for key in result[0].history:
            avg_history[key] = [0] * num_epochs
        
        # Sum up values
        for history in result:
            for key in avg_history:
                for epoch in range(num_epochs):
                    avg_history[key][epoch] += history.history[key][epoch]
        
        # Divide by number of folds to get the average
        for key in avg_history:
            for epoch in range(num_epochs):
                avg_history[key][epoch] /= len(result)
        
        avg_histories.append(avg_history)
    
    # Plot Losses
    fig, ax = plt.subplots(1, num_models, figsize=(20, 10))
    for i, avg_history in enumerate(avg_histories):
        # MCRMSE
        ax[i].plot(avg_history['loss'], color='C0')
        ax[i].plot(avg_history['val_loss'], color='C1')
        ax[i].plot(avg_history['mae'], color='C2')
        ax[i].plot(avg_history['mse'], color='C3')
        ax[i].set_title(f"{results['models'][i]}")
        ax[i].set_ylabel('Loss')
        ax[i].set_xlabel('Epoch')
        ax[i].legend(['train', 'validation', 'MAE', 'MSE'], loc='upper right')
    fig.savefig('losses_plot.jpg')
    
    # Plot Pearson Correlation
    fig2, ax2 = plt.subplots(1, num_models, figsize=(20, 10))
    for i, avg_history in enumerate(avg_histories):
        ax2[i].plot(avg_history['tf_pearson'], color='C0')
        ax2[i].plot(avg_history['val_tf_pearson'], color='C1')
        ax2[i].set_title(f"{results['models'][i]}")
        ax2[i].set_ylabel('Pearson correlation')
        ax2[i].set_xlabel('Epoch')
        ax2[i].legend(['train', 'validation'], loc='upper right')
    fig2.savefig('pearson_correlation_plot.jpg')

results = {
    "models": ['GRU', 'LSTM'],
    "histories": [gru_histories, lstm_histories],
}



plot_learning_curves(results)


# In[ ]:


gru_val_preds = format_predictions(gru_holdouts, gru_holdout_preds, val=True)
lstm_val_preds = format_predictions(lstm_holdouts, lstm_holdout_preds, val=True)

print('-'*25); print('Unfiltered training results'); print('-'*25)
print('GRU training results'); print('')
get_error(gru_val_preds)
print('LSTM training results'); print('')
get_error(lstm_val_preds)
print('-'*25); print('SN_filter == 1 training results'); print('-'*25)
print('GRU training results'); print('')
get_error(gru_val_preds[gru_val_preds['SN_filter'] == 1])
print('LSTM training results'); print('')
get_error(lstm_val_preds[lstm_val_preds['SN_filter'] == 1])


# # Submission

# In[ ]:


gru_preds = [gru_public_preds, gru_private_preds]
lstm_preds = [gru_public_preds, gru_private_preds]
test_df = [public_df, private_df]
gru_preds = format_predictions(test_df, gru_preds)
lstm_preds = format_predictions(test_df, lstm_preds)


# In[ ]:


gru_weight = .5
lstm_weight = .5


# In[ ]:


blended_preds = pd.DataFrame()
blended_preds['id_seqpos'] = gru_preds['id_seqpos']
blended_preds['reactivity'] = gru_weight*gru_preds['reactivity'] + lstm_weight*lstm_preds['reactivity']
blended_preds['deg_Mg_pH10'] = gru_weight*gru_preds['deg_Mg_pH10'] + lstm_weight*lstm_preds['deg_Mg_pH10']
blended_preds['deg_pH10'] = gru_weight*gru_preds['deg_pH10'] + lstm_weight*lstm_preds['deg_pH10']
blended_preds['deg_Mg_50C'] = gru_weight*gru_preds['deg_Mg_50C'] + lstm_weight*lstm_preds['deg_Mg_50C']
blended_preds['deg_50C'] = gru_weight*gru_preds['deg_50C'] + lstm_weight*lstm_preds['deg_50C']


# In[ ]:


submission = sample_sub[['id_seqpos']].merge(blended_preds, on=['id_seqpos'])
submission.head()


# In[ ]:


submission.to_csv(f'submission_new.csv', index=False)
print('Submission saved')


# In[ ]:


import matplotlib.pyplot as plt

# create data for x and y axis
model_scores = [90, 85, 80, 75, 70]
embeddings_size =[32,64,128,256,512]

# create data for each embedding method
isoGlove = [0.01,
0.39,
0.41,
0.49,
0.63]

GloVe = [0.003,
0.04,
0.27,
0.34,
0.44]
Node2vec = [0.003,
0.06,
0.27,
0.36,
0.45,
]
Hope = [0.003,
0.31,
0.018,
0.52,
0.6,
]
GF = [0.001,
0.01,
0.49,
0.21,
0.33
]

# plot line charts for each embedding method
plt.plot(  embeddings_size,isoGlove, label='IsoGlove')
plt.plot(  embeddings_size ,GloVe,label='GloVe')
plt.plot(  embeddings_size,Node2vec, label='Node2vec')
plt.plot(  embeddings_size,Hope, label='Hope')
plt.plot( embeddings_size,GF, label='GF')

# set chart title and axis labels
plt.title('Embeddings size by Model Scores')
plt.ylabel('Model Scores')
plt.xlabel('Embeddings Size')

# show legend
plt.legend()

# show chart
plt.savefig('embedding_size_vs_model_score.jpg', format='jpg')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

# create data for x and y axis
embeddings_size =[32,64,128,256,512]

# create data for each embedding method
isoGlove = [0.014,
0.0019,
0.006,
0.00006,
0.009
]

GloVe = [0.00044,
0.004,
0.001,
0.0034,
0.043
]
Node2vec = [0.0005,
0.005,
0.002,
0.0036,
0.045
]
Hope = [0.0008,
0.008,
0.01,
0.0052,
0.03
]
GF = [0.0002,
0.002,
0.004,
0.0021,
0.0033
]

# plot line charts for each embedding method
plt.plot(  embeddings_size,isoGlove, label='IsoGlove')
plt.plot(  embeddings_size ,GloVe,label='GloVe')
plt.plot(  embeddings_size,Node2vec, label='Node2vec')
plt.plot(  embeddings_size,Hope, label='Hope')
plt.plot( embeddings_size,GF, label='GF')

# set chart title and axis labels
plt.title('Embeddings size by Model Scores')
plt.ylabel('MAP')
plt.xlabel('Embeddings Size')

# show legend
plt.legend()

# show chart
plt.savefig('embedding_size_vs_MAP.jpg', format='jpg')
# plt.savefig("embeddings_size.jpg", dpi=300, bbox_inches='tight')
plt.show()


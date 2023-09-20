#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


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
# As this competition is entering its last week, this will be the final version of this notebook. I wanted to clean up some more code and add some last minute improvements for those that perhaps reference this notebook during the next week. This notebook received far more attention than it deserved. It is nothing without [Xhlulu](https://www.kaggle.com/xhlulu)'s kernel [here](https://www.kaggle.com/xhlulu/openvaccine-simple-gru-model) and his contributions to the dicussion forums over the past few weeks. If you give this notebook an upvote, please give Xhlulu's one as well (and Tito's). Good luck to everyone over the next week.

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

# In[ ]:


#get comp data
train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)
test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)
sample_sub = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')


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

# In[ ]:


print(train.columns)


# **It seems we also have a `signal_to_noise` and a `SN_filter` column. These columns control the 'quality' of samples, and as such are important training hyperparameters. We will explore them shortly:**

# In[ ]:


#sneak peak
print(train.shape)
if ~train.isnull().values.any(): print('No missing values')
train.head()


# In[ ]:


#sneak peak
print(test.shape)
if ~test.isnull().values.any(): print('No missing values')
test.head()


# In[ ]:


#sneak peak
print(sample_sub.shape)
if ~sample_sub.isnull().values.any(): print('No missing values')
sample_sub.head()


# **Now we explore `signal_to_noise` and `SN_filter` distributions. As per the data tab of this competition the samples in `test.json` have been filtered in the following way:**
# 
# 1. Minimum value across all 5 conditions must be greater than -0.5.
# 2. Mean signal/noise across all 5 conditions must be greater than 1.0. [Signal/noise is defined as mean( measurement value over 68 nts )/mean( statistical error in measurement value over 68 nts)]
# 3. To help ensure sequence diversity, the resulting sequences were clustered into clusters with less than 50% sequence similarity, and the 629 test set sequences were chosen from clusters with 3 or fewer members. That is, any sequence in the test set should be sequence similar to at most 2 other sequences.

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.kdeplot(train['signal_to_noise'], shade=True, ax=ax[0])
sns.countplot(train['SN_filter'], ax=ax[1])

ax[0].set_title('Signal/Noise Distribution')
ax[1].set_title('Signal/Noise Filter Distribution');


# In[ ]:


print(f"Samples with signal_to_noise greater than 1: {len(train.loc[(train['signal_to_noise'] > 1 )])}")
print(f"Samples with SN_filter = 1: {len(train.loc[(train['SN_filter'] == 1 )])}")
print(f"Samples with signal_to_noise greater than 1, but SN_filter == 0: {len(train.loc[(train['signal_to_noise'] > 1) & (train['SN_filter'] == 0)])}")


# **Update: as per [this discussion post](https://www.kaggle.com/c/stanford-covid-vaccine/discussion/183992), both public *and* private test datasets are now filtered with the same 3 above conditions.**

# # Feature Engineering
# 
# **Check out [Tito](https://www.kaggle.com/its7171)'s kernel [here](https://www.kaggle.com/its7171/gru-lstm-with-feature-engineering-and-augmentation) for the feature engineering code below. The `bpps` folder contains Base Pairing Probabilities matrices for each sequence. These matrices give the probability that each pair of nucleotides in the RNA forms a base pair. Each matrix is a symmetric square matrix the same length as the sequence. For a complete EDA of the `bpps` folder, see this notebook [here](https://www.kaggle.com/hidehisaarai1213/openvaccine-checkout-bpps?scriptVersionId=42460013).**

# In[ ]:


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

# In[ ]:


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

# In[ ]:


AUGMENT=False 


# In[ ]:


# aug_df = pd.read_csv('../input/openvaccineaugmented/aug_data_n2.csv')
# print(aug_df.shape)
# aug_df.head()


# In[ ]:


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


# In[ ]:


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

# In[ ]:


DENOISE = False


# In[ ]:


target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']


# In[ ]:


token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}


# In[ ]:


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


# In[ ]:


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

# In[ ]:


token2int


# In[ ]:


# https://www.kaggle.com/c/stanford-covid-vaccine/discussion/183211
def rmse(y_actual, y_pred):
    mse = tf.keras.losses.mean_squared_error(y_actual, y_pred)
    return K.sqrt(mse)

def mcrmse(y_actual, y_pred, num_scored=len(target_cols)):
    score = 0
    for i in range(num_scored):
        score += rmse(y_actual[:, :, i], y_pred[:, :, i]) / num_scored
    return score


# In[ ]:


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
def dense_layer(hidden_dim, dropout):
    return tf.keras.layers.Dense(
                                hidden_dim, activation='relu',
                                kernel_initializer='orthogonal')

def build_model(rnn='mlp', convolve=False, conv_dim=512, 
                dropout=.4, sp_dropout=.2, embed_dim=200,
                hidden_dim=256, layers=3,
                seq_len=107, pred_len=68):
    
###############################################
#### Inputs
###############################################

    inputs = tf.keras.layers.Input(shape=(seq_len, 5))
    categorical_feats = inputs[:, :, :3]
    numerical_feats = inputs[:, :, 3:]

    embed = tf.keras.layers.Embedding(input_dim=len(token2int),
                                      output_dim=embed_dim)(categorical_feats)#(None, 107, 3, 200)
    reshaped = tf.reshape(
        embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))
    
    reshaped = tf.keras.layers.concatenate([reshaped, numerical_feats], axis=2)
    hidden = tf.keras.layers.SpatialDropout1D(sp_dropout)(reshaped)
    
    if convolve:
        hidden = tf.keras.layers.Conv1D(conv_dim, 5, padding='same', activation=tf.keras.activations.swish)(hidden)

###############################################
#### hidden Layers
###############################################

    if rnn is 'gru':
        for _ in range(layers):
            hidden = gru_layer(hidden_dim, dropout)(hidden)
        
    elif rnn is 'lstm':
        for _ in range(layers):
            hidden = lstm_layer(hidden_dim, dropout)(hidden)
     
    elif rnn is 'mlp':
        for _ in range(layers):
            hidden = dense_layer(hidden_dim, dropout)(hidden)

###############################################
#### Output
###############################################

    out = hidden[:, :pred_len]
    out = tf.keras.layers.Dense(5, activation='linear')(out)
    
    model = tf.keras.Model(inputs=inputs, outputs=out)
    adam = tf.optimizers.Adam()
    model.compile(optimizer=adam, loss=mcrmse)

    return model


# In[ ]:


test_model = build_model(rnn='mlp')
test_model.summary()


# # KFold Training and Inference
# 
# **In previous commits, I either filtered by `SN_filter == 1` or with `signal_to_noise > 1`. But it seems that these RNN models generalize better when exposed to the noisier samples in the dataset. If you review the `tf.keras` [documentation](https://www.tensorflow.org/api_docs/python/tf/keras/Model), you can see that you can pass a Numpy array of weights during training used to weight the loss function. So we can weight samples with higher `signal_to_noise` values more during training than the noisier samples. As inspired by [Tito](https://www.kaggle.com/its7171), we will pass the following array to `sample_weight`: `np.log1p(train.signal_to_noise + epsilon)/2` where epsilon is a small number to ensure we don't get `log(1)` for any weights.**
# 
# **But since the competition hosts have said [here](https://www.kaggle.com/c/stanford-covid-vaccine/discussion/183992), the public and private test sets only contain samples where `SN_filter == 1`, so we ought to validate against the such samples as well:**

# W2V:(train set)

# In[ ]:


import pickle
import numpy
public_df = test.query("seq_length == 107").copy()
private_df = test.query("seq_length == 130").copy()
private_preds = np.zeros((private_df.shape[0], 130, 5))
public_preds = np.zeros((public_df.shape[0], 107, 5))
public_inputs = preprocess_inputs(public_df)
private_inputs = preprocess_inputs(private_df)

   
########################### attach w2v vectors  for sequences from file to test train dataset
w2vAVG_train = open('../input/avgw2v-openvaccine-train-set/train_Seq_Vec (1)','rb')
seqs_train = pickle.load(w2vAVG_train)
npseqs_train= numpy.array(seqs_train)+4.1  ### w2v vectors are in range[-1/..,1/..] but token2int in  3-6 for seqs
arr2D_private_test = preprocess_inputs(private_df)
arr2D_train = preprocess_inputs(train)
arr2D_train = np.delete(arr2D_train, 0, axis=2)
w2v_train = np.dstack((npseqs_train, arr2D_train))



w2vAVG_public_test = open('../input/avgw2vopenvaccinepublic-test/public_test','rb')
seqs_public_test = pickle.load(w2vAVG_public_test)
npseqs_public_test= numpy.array(seqs_public_test)+4.1 ### w2v vectors are in range[-1/..,1/..] but token2int in 3-6 for seqs
arr2D_private_test = preprocess_inputs(private_df)
arr2D_public_test = preprocess_inputs(public_df)
arr2D_public_test = np.delete(arr2D_public_test, 0, axis=2)
w2v_public_test = np.dstack((npseqs_public_test, arr2D_public_test))




w2vAVG_private_test = open('../input/avgw2v-openvaccine-privatetest/private_test (1)','rb')
seqs_private_test = pickle.load(w2vAVG_private_test)
npseqs_private_test= numpy.array(seqs_private_test)+4.1   ### w2v vectors are in range[-1/..,1/..] but token2int in 3-6 for seqs
arr2D_private_test = preprocess_inputs(private_df)
arr2D_private_test = np.delete(arr2D_private_test, 0, axis=2)
w2v_private_test = np.dstack((npseqs_private_test, arr2D_private_test))







# In[ ]:


print('shape:',npseqs_train.shape
,'min:',(npseqs_train).min(),'max:',(npseqs_train).max())

 


# In[ ]:


def train_and_infer(rnn, STRATIFY=True, FOLDS=4, EPOCHS=50, BATCH_SIZE=64,
                    REPEATS=3, SEED=34, VERBOSE=2):
#EPOCHS=50,REPEATS=3,FOLDS=4
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
    kf=KFold(n_splits=FOLDS, random_state=SEED)
    kmeans_model = KMeans(n_clusters=200, random_state=SEED).fit(w2v_train[:,:,0])
#     kmeans_model = KMeans(n_clusters=200, random_state=SEED).fit(preprocess_inputs(train)[:,:,0])
    train['cluster_id'] = kmeans_model.labels_

    for _ in range(REPEATS):
        
        for f, (train_index, val_index) in enumerate((gkf if STRATIFY else kf).split(train,
                train['reactivity'], train['cluster_id'] if STRATIFY else None)):
            print('train_index:',train_index)
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
            trn_ = preprocess_inputs(trn)#1200 x 107 x 5
            npseqs_train_train= npseqs_train[train_index]#1200 out of 2400 word2vec trainset
            trn_ = np.delete(trn_, 0, axis=2)
            trn_ = np.dstack((npseqs_train_train, trn_))
            print('trn_',len(trn_),'trn_[0]',len(trn_[0]),'trn_[0][0]',len(trn_[0][0]) )
            trn_labs = np.array(trn[target_cols].values.tolist()).transpose((0, 2, 1))

            #get validation data
            val = train.iloc[val_index]
            val_all = preprocess_inputs(val)
            npseqs_train_val= npseqs_train[val_index]#1200 out of 2400 word2vec trainset
            val_all = np.delete(val_all, 0, axis=2)
            val_all = np.dstack((npseqs_train_val, val_all))           
            print('val_all',len(val_all),'val_all[0]',len(val_all[0]),'val_all[0][0]',len(val_all[0][0]) )

            
            val = val[val.SN_filter == 1]
            val_ = preprocess_inputs(val)
            npseqs_train_valsn= npseqs_train[val.index]#830 out of 2400 word2vec trainset
            val_ = np.delete(val_, 0, axis=2)
            val_ = np.dstack((npseqs_train_valsn, val_))  
            print('val_',len(val_),'val_[0]',len(val_[0]),'val_[0][0]',len(val_[0][0]) )
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
#                 sample_weight=sample_weighting,
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

#             public_preds += model_short.predict(public_inputs) / (FOLDS * REPEATS)
#             private_preds += model_long.predict(private_inputs) / (FOLDS * REPEATS)
            public_preds += model_short.predict(w2v_public_test) / (FOLDS * REPEATS)
            private_preds += model_long.predict(w2v_private_test) / (FOLDS * REPEATS)

    
        del model, model_short, model_long
        
    return holdouts, holdout_preds, public_df, public_preds, private_df, private_preds, histories


# > len(trn_) 1200
# len(trn_[0]) 107
# len(trn_[0][0]) 5
# len(val_all) 1200
# len(val_all[0]) 107
# len(val_all[0][0]) 5
# len(val_) 813
# len(val_[0]) 107
# len(val_[0][0]) 5

# {'(': 0,
#  ')': 1,
#  '.': 2,
#  'A': 3,
#  'C': 4,
#  'G': 5,
#  'U': 6,
#  'B': 7,
#  'E': 8,
#  'H': 9,
#  'I': 10,
#  'M': 11,
#  'S': 12,
#  'X': 13}
#  
#  input columns :'sequence', 'structure', 'predicted_loop_type','bpps_sum','bpps_max',

# ### GRU & LSTM

# In[ ]:


gru_holdouts, gru_holdout_preds, public_df, gru_public_preds, private_df, gru_private_preds, gru_histories = train_and_infer(rnn='gru')


# In[ ]:


lstm_hold outs, lstm_holdout_preds, public_df, lstm_public_preds, private_df, lstm_private_preds, lstm_histories = train_and_infer(rnn='lstm')


# In[ ]:


train


# ### Learning Curves and Evaluation

# In[ ]:


def plot_learning_curves(results):

    fig, ax = plt.subplots(1, len(results['histories']), figsize = (20, 10))
    
    for i, result in enumerate(results['histories']):
        for history in result:
            ax[i].plot(history.history['loss'], color='C0')
            ax[i].plot(history.history['val_loss'], color='C1')
            ax[i].set_title(f"{results['models'][i]}")
            ax[i].set_ylabel('MCRMSE')
            ax[i].set_xlabel('Epoch')
            ax[i].legend(['train', 'validation'], loc = 'upper right')
            
results = {
            "models" : ['GRU', 'LSTM'],    
            "histories" : [gru_histories, lstm_histories],
            }


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


#!/usr/bin/env python
# coding: utf-8

# https://gitee.com/greitzmann/ELMo-keras/tree/master/elmo

# Use (tab) for autocompleteting:

# In[ ]:


get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# 
# __init__.py
# Initial Commit
# 
# dropout.py
# Fix token character encodings + Improve documentation
# 
# highway.py
# Initial Commit
# 
# masking.py
# Fix token character encodings + Improve documentation
# 
# sampled_softmax.py
# Add full Softmax option in projection layer

# In[ ]:


# !pip install tensorflow==1.15
# this is very vital as the baseline ELMo is just compatible with this V of Tensorflow
#     ValueError: The two structures don't have the same sequence length. Input structure has length 0, while shallow structure has length 2.
#Note: Run it just at the beggining of each session for saving time 
get_ipython().system('pip install bilm')
get_ipython().system('pip install data')
# -*- coding: utf-8 -*-


# In[ ]:


get_ipython().system("pip install 'h5py==2.10.0' --force-reinstall")


# Degradation Prediction:

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
from collections import Counter
import pandas as pd, numpy as np, seaborn as sns
import math, json, os, random
from matplotlib import pyplot as plt
from tqdm import tqdm

import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow.keras import backend as K

from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.cluster import KMeans

seed = 34
def seed_everything(seed= 34):
    os.environ['PYTHONHASHSEED']=str(seed)
#     tf.random.set_random_seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
seed_everything()


#get comp data
train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)
test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)
sample_sub = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')







#sneak peak
print(train.shape)
if ~train.isnull().values.any(): print('No missing values')
train.head()

#sneak peak
print(test.shape)
if ~test.isnull().values.any(): print('No missing values')
test.head()

#sneak peak
print(sample_sub.shape)
if ~sample_sub.isnull().values.any(): print('No missing values')
sample_sub.head()

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.kdeplot(train['signal_to_noise'], shade=True, ax=ax[0])
sns.countplot(train['SN_filter'], ax=ax[1])

ax[0].set_title('Signal/Noise Distribution')
ax[1].set_title('Signal/Noise Filter Distribution');

print(f"Samples with signal_to_noise greater than 1: {len(train.loc[(train['signal_to_noise'] > 1 )])}")
print(f"Samples with SN_filter = 1: {len(train.loc[(train['SN_filter'] == 1 )])}")
print(f"Samples with signal_to_noise greater than 1, but SN_filter == 0: {len(train.loc[(train['signal_to_noise'] > 1) & (train['SN_filter'] == 0)])}")

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

AUGMENT=False 
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


print(f"Samples in train before augmentation: {len(train)}")
print(f"Samples in test before augmentation: {len(test)}")

if AUGMENT:
    train = aug_data(train)
    test = aug_data(test)

print(f"Samples in train after augmentation: {len(train)}")
print(f"Samples in test after augmentation: {len(test)}")

print(f"Unique sequences in train: {len(train['sequence'].unique())}")
print(f"Unique sequences in test: {len(test['sequence'].unique())}")

DENOISE = True



target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']




token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}


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


if DENOISE:
    train = train[train['signal_to_noise'] > .25]
    
    
    # https://www.kaggle.com/c/stanford-covid-vaccine/discussion/183211
def rmse(y_actual, y_pred):
    mse = tf.keras.losses.mean_squared_error(y_actual, y_pred)
    return K.sqrt(mse)

def mcrmse(y_actual, y_pred, num_scored=len(target_cols)):
    score = 0
    for i in range(num_scored):
        score += rmse(y_actual[:, :, i], y_pred[:, :, i]) / num_scored
    return score


# **Model.py**

# ELMo Object initializer changed!
# 
# Input dimention changed!
# 
# self.compile_elmo() got exra parameters!
# 
# 

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
def tf_pearson(x, y):    
    mx = tf.math.reduce_mean(input_tensor=x)
    my = tf.math.reduce_mean(input_tensor=y)
    xm, ym = x-mx, y-my
    r_num = tf.math.reduce_mean(input_tensor=tf.multiply(xm,ym))        
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    return  r_num / r_den
    


# In[ ]:


get_ipython().system('conda install -y -c bioconda viennarna')
get_ipython().system('pip install RNA')
get_ipython().system('pip install python-Levenshtein')

Structure_Interpretability_Test = True



# In[ ]:





# In[ ]:


import os
import time
import plotly.express as px



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt


import numpy as np
# from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Input, SpatialDropout1D,BatchNormalization
from tensorflow.keras.layers import LSTM, Activation
from tensorflow.keras.layers import Lambda, Embedding, Conv2D, GlobalMaxPool1D
from tensorflow.keras.layers import add, concatenate
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout
from tensorflow.keras import optimizers
import time
from keras.layers import CuDNNLSTM
    
from Bio import pairwise2
from Bio.Seq import Seq

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# from data import MODELS_DIR
MODELS_DIR='./'

# from .custom_layers import TimestepDropout, Camouflage, Highway, SampledSoftmax


class ELMo(object):
    def __init__(self, parameters, 
                dropout=.4, sp_dropout=.2, embed_dim=200,
                hidden_dim=256, layers=3,
                seq_len=107, pred_len=68):
        #pre-build models for different sequence lengths
        self._model = None
        self._elmo_model = None
        self.parameters = parameters
        self.seq_len=seq_len
        self.pred_len=pred_len
        self.compile_elmo()

    def __del__(self):
        K.clear_session()
        del self._model

    def char_level_token_encoder(self):
        charset_size = self.parameters['charset_size']
        char_embedding_size = self.parameters['char_embedding_size']
        token_embedding_size = self.parameters['hidden_units_size']
        n_highway_layers = self.parameters['n_highway_layers']
        filters = self.parameters['cnn_filters']
        token_maxlen = self.parameters['token_maxlen']

        # Input Layer, word characters (samples, words, character_indices), Size= (None, None)
#         inputs = Input(shape=(None, token_maxlen,), dtype='int32')
        inputs = Input(shape=(self.seq_len, 5))
        # Embed characters (samples, words, characters, character embedding),Size= (None, None, 200)
        embeds = Embedding(input_dim=len(token2int), output_dim=embed_dim)(inputs)
        token_embeds = []
        # Apply multi-filter 2D convolutions + 1D MaxPooling + tanh
        for (window_size, filters_size) in filters:
            convs = Conv2D(filters=filters_size, kernel_size=[window_size, char_embedding_size], strides=(1, 1),
                           padding="same")(embeds)
            convs = TimeDistributed(GlobalMaxPool1D())(convs)
            convs = Activation('tanh')(convs)
            convs = Camouflage(mask_value=0)(inputs=[convs, inputs])
            token_embeds.append(convs)
        token_embeds = concatenate(token_embeds)
        # Apply highways networks
        for i in range(n_highway_layers):
            token_embeds = TimeDistributed(Highway())(token_embeds)
            token_embeds = Camouflage(mask_value=0)(inputs=[token_embeds, inputs])
        # Project to token embedding dimensionality
        token_embeds = TimeDistributed(Dense(units=token_embedding_size, activation='linear'))(token_embeds)
        token_embeds = Camouflage(mask_value=0)(inputs=[token_embeds, inputs])

        token_encoder = Model(inputs=inputs, outputs=token_embeds, name='token_encoding')
        return token_encoder

    
    
    
    




    
    def compile_elmo(self,embed_dim=200, print_summary=False):
        """
        Compiles a Language Model RNN based on the given parameters
        """
        import numpy as np
        import Levenshtein
        import RNA
        if self.parameters['token_encoding'] == 'word':
            # Train word embeddings from scratch
            word_inputs = Input(shape=(self.seq_len, 5), name='word_indices')
            #Size= [(None, None)]
            categorical_feats = word_inputs[:, :, :3]
            numerical_feats = word_inputs[:, :, 3:]
            embeddings =tf.keras.layers.Embedding(name='token_encoding',input_dim=len(token2int),
                                      output_dim=200)
#             embeddingsformalite = Embedding(len(token2int), 
#                                             603, trainable=True,
#                                             name='token_encodingformalite')
            inputs = embeddings(categorical_feats)
#           inputs: Tensor("token_encoding/embedding_lookup/Identity_1:0", shape=(?, 107, 3, 200)

            reshaped = tf.reshape(inputs, shape=(-1, inputs.shape[1],  inputs.shape[2] * inputs.shape[3]))
#             reshaped1: Tensor("Reshape:0", shape=(?, 107, 600), dtype=float32) 

            reshaped = tf.keras.layers.concatenate([reshaped, numerical_feats], axis=2)
            
#              reshaped2: Tensor("concatenate/concat:0", shape=(?, 107, 602), dtype=float32) 

            
            # Token embeddings for Input
            drop_inputs = SpatialDropout1D(self.parameters['dropout_rate'])(reshaped)
            drop_inputs=BatchNormalization()(drop_inputs)

            lstm_inputs = TimestepDropout(self.parameters['word_dropout_rate'])(drop_inputs)
            lstm_inputs= BatchNormalization()(lstm_inputs)

#              lstm_inputs: Tensor("timestep_dropout/cond/Merge:0", shape=(?, 107, 602), dtype=float32) 
#              drop_inputs: Tensor("timestep_dropout/cond/Merge:0", shape=(?, 107, 602), dtype=float32) 

            # Pass outputs as inputs to apply sampled softmax
            next_ids = Input(shape=(None, 1), name='next_ids')
#              next_ids: Tensor("next_ids:0", shape=(?, ?, 1), dtype=float32) 
            previous_ids = Input(shape=(None, 1), name='previous_ids')
#              previous_ids: Tensor("previous_ids:0", shape=(?, ?, 1), dtype=float32) 

        elif self.parameters['token_encoding'] == 'char':
            # Train character-level representation
            word_inputs = Input(shape=(None, self.parameters['token_maxlen'],), name='char_indices')
            inputs = self.char_level_token_encoder()(word_inputs)
            categorical_feats = word_inputs[:, :, :3]
            numerical_feats = word_inputs[:, :, 3:]
#             print(' inputs:', inputs,'inputs size',inputs.shape)
            
            reshaped = tf.reshape(inputs, shape=(-1, inputs.shape[1],  inputs.shape[2] * inputs.shape[3]))
            reshaped = tf.keras.layers.concatenate([reshaped, numerical_feats], axis=2)
#             print(' reshaped:', reshaped,'reshaped size',reshaped.shape)


            
            # Token embeddings for Input
            drop_inputs = SpatialDropout1D(self.parameters['dropout_rate'])(reshaped)
            lstm_inputs = TimestepDropout(self.parameters['word_dropout_rate'])(drop_inputs)

            # Pass outputs as inputs to apply sampled softmax
            next_ids = Input(shape=(None, 1), name='next_ids')
            previous_ids = Input(shape=(None, 1), name='previous_ids')

        # Reversed input for backward LSTMs
        re_lstm_inputs = Lambda(function=ELMo.reverse)(lstm_inputs)
        re_lstm_inputs=BatchNormalization()(re_lstm_inputs)
        mask = Lambda(function=ELMo.reverse)(drop_inputs)
        mask=BatchNormalization()(mask)

        # Forward LSTMs
        for i in range(self.parameters['n_lstm_layers']):
            if self.parameters['cuDNN']:
                lstm = CuDNNLSTM(units=self.parameters['lstm_units_size'], return_sequences=True,
                                 kernel_constraint=MinMaxNorm(-1*self.parameters['cell_clip'],
                                                              self.parameters['cell_clip']),
                                 recurrent_constraint=MinMaxNorm(-1*self.parameters['cell_clip'],
                                                                 self.parameters['cell_clip']))(lstm_inputs)
            else:
                lstm = LSTM(units=self.parameters['lstm_units_size'], return_sequences=True,
                            activation="tanh",
                            recurrent_activation='sigmoid',
                            kernel_constraint=MinMaxNorm(-1 * self.parameters['cell_clip'],
                                                         self.parameters['cell_clip']),
                            recurrent_constraint=MinMaxNorm(-1 * self.parameters['cell_clip'],
                                                            self.parameters['cell_clip'])
                            )(lstm_inputs)

            lstm=BatchNormalization()(lstm)
            lstm = Camouflage(mask_value=0)(inputs=[lstm, drop_inputs])
            lstm=BatchNormalization()(lstm) 
            # Projection to hidden_units_size
            proj = TimeDistributed(Dense(self.parameters['hidden_units_size'], activation='linear',
                                         kernel_constraint=MinMaxNorm(-1 * self.parameters['proj_clip'],
                                                                      self.parameters['proj_clip'])
                                         ))(lstm)
            proj=BatchNormalization()(proj)

            print(' lstm_inputs:', lstm_inputs,'lstm_inputs size',lstm_inputs.shape)
            print(' proj:', proj,'proj size',proj.shape)

            # Merge Bi-LSTMs feature vectors with the previous ones
            lstm_inputs = add([proj, lstm_inputs], name='f_block_{}'.format(i + 1))
            lstm_inputs=BatchNormalization()(lstm_inputs)

            # Apply variational drop-out between BI-LSTM layers
            lstm_inputs = SpatialDropout1D(self.parameters['dropout_rate'])(lstm_inputs)
            lstm_inputs=BatchNormalization()(lstm_inputs)

        # Backward LSTMs
        for i in range(self.parameters['n_lstm_layers']):
            if self.parameters['cuDNN']:
                re_lstm = CuDNNLSTM(units=self.parameters['lstm_units_size'], return_sequences=True,
                                    kernel_constraint=MinMaxNorm(-1*self.parameters['cell_clip'],
                                                                 self.parameters['cell_clip']),
                                    recurrent_constraint=MinMaxNorm(-1*self.parameters['cell_clip'],
                                                                    self.parameters['cell_clip']))(re_lstm_inputs)
            else:
                re_lstm = LSTM(units=self.parameters['lstm_units_size'], return_sequences=True, activation='tanh',
                               recurrent_activation='sigmoid',
                               kernel_constraint=MinMaxNorm(-1 * self.parameters['cell_clip'],
                                                            self.parameters['cell_clip']),
                               recurrent_constraint=MinMaxNorm(-1 * self.parameters['cell_clip'],
                                                               self.parameters['cell_clip'])
                               )(re_lstm_inputs)
            re_lstm = BatchNormalization()(re_lstm)
            re_lstm = Camouflage(mask_value=0)(inputs=[re_lstm, mask])
            re_lstm = BatchNormalization()(re_lstm)   
            # Projection to hidden_units_size
            re_proj = TimeDistributed(Dense(self.parameters['hidden_units_size'], activation='linear',
                                            kernel_constraint=MinMaxNorm(-1 * self.parameters['proj_clip'],
                                                                         self.parameters['proj_clip'])
                                            ))(re_lstm)
            re_proj = BatchNormalization()(re_proj)

            # Merge Bi-LSTMs feature vectors with the previous ones
            re_lstm_inputs = add([re_proj, re_lstm_inputs], name='b_block_{}'.format(i + 1))
            re_lstm_inputs = BatchNormalization()(re_lstm_inputs)

            # Apply variational drop-out between BI-LSTM layers
            re_lstm_inputs = SpatialDropout1D(self.parameters['dropout_rate'])(re_lstm_inputs)
            re_lstm_inputs = BatchNormalization()(re_lstm_inputs)

        # Reverse backward LSTMs' outputs = Make it forward again
        re_lstm_inputs = Lambda(function=ELMo.reverse, name="reverse")(re_lstm_inputs)
        re_lstm_inputs = BatchNormalization()(re_lstm_inputs)

        # Project to Vocabulary with Sampled Softmax
#         sampled_softmax = SampledSoftmax(num_classes=self.parameters['vocab_size'],
#                                          num_sampled=int(self.parameters['num_sampled']),
#                                          tied_to=reshaped if self.parameters['weight_tying']
#                                          and self.parameters['token_encoding'] == 'word' else None)
#         print(' next_ids:', next_ids,'next_ids size',next_ids.shape)
#         print(' lstm_inputs:', lstm_inputs,'lstm_inputs size',lstm_inputs.shape)

#         outputs = sampled_softmax([lstm_inputs, next_ids])
#         re_outputs = sampled_softmax([re_lstm_inputs, previous_ids])
#         self._model = Model(inputs=[word_inputs, next_ids, previous_ids],outputs=[outputs, re_outputs])


#         outputs = tf.keras.layers.Softmax([lstm_inputs])
#         re_outputs = tf.keras.layers.Softmax(re_lstm_inputs)
        merge = add([lstm_inputs,re_lstm_inputs])
        merge=BatchNormalization()(merge)
        conv_dim=32#128
        merge = tf.keras.layers.Conv1D(conv_dim, 5, padding='same', activation=tf.keras.activations.swish)(merge)
    
        out = merge[:, :self.pred_len]
        out = tf.keras.layers.Dense(5, activation='linear')(out)
        out = BatchNormalization()(out)

        self._model = tf.keras.Model(inputs=word_inputs, outputs=out)
        adam = tf.keras.optimizers.Adam()
        self._model.compile(optimizer=adam, loss=mcrmse, metrics=['mse', tf_pearson, 'mae'])

#         self._model.compile(optimizer=Adagrad(lr=self.parameters['lr'], clipvalue=self.parameters['clip_value']),loss=None)              
#         self._model.save(i'model-{i}.hp5')
        if print_summary:
            self._model.summary()

    def train(self):
        import numpy as np
        import time
        start_time = time.time()  # Record the start time


        ######################################################################
        STRATIFY=False
        FOLDS=4
        VERBOSE=2
        histories = []

        #get test now for OOF 
        public_df = test.query("seq_length == 107").copy()
        private_df = test.query("seq_length == 130").copy()
        private_preds = np.zeros((private_df.shape[0], 130, 5))
        public_preds = np.zeros((public_df.shape[0], 107, 5))
        public_inputs = preprocess_inputs(public_df)
        private_inputs = preprocess_inputs(private_df)
        pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
        train_inputs = preprocess_inputs(train)
        train_labels = np.array(train[pred_cols].values.tolist()).transpose((0, 2, 1))
        
        
        
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=FOLDS,shuffle=True,random_state=seed)
        # with tf.device('/gpu'):
        # Recreate the exact same model, including its weights and the optimizer
#         self._model.load_weights('../input/model0h5/model0.h5')
#         print('new summary:')
#         # Show the model architecture
#         self._model.summary()
        
        for fold,(idxT,idxV) in enumerate(kf.split(train_inputs)):
#                 start_time = time.time()

                self.seq_len=107
                self.pred_len=107
                self.compile_elmo()
                model_short=self._model

                self.seq_len=130
                self.pred_len=130 
                self.compile_elmo()
                model_long=self._model
                
                self.seq_len=107
                self.pred_len=68
                self.compile_elmo()
#                 print('(idxT,idxV)',idxT,idxV)
                
                #get train data
                trn_ = train_inputs[idxT,:,:]
                trn_labs = train_labels[idxT,:,:]

                #get validation data
                val_ = train_inputs[idxV,:,:]
                val_labs = train_labels[idxV,:,:]

                
                history = self._model.fit(
                    trn_, trn_labs,
                    validation_data = (val_, val_labs),
                    batch_size=self.parameters['batch_size'],
                    epochs=self.parameters['epochs'],
#                     validation_split=(train_inputs[idxV,:,:],train_labels[idxV,:,:]),
                        callbacks=[
                    tf.keras.callbacks.ReduceLROnPlateau(),
                    tf.keras.callbacks.ModelCheckpoint('model'+str(fold)+'.h5',save_weights_only=True,save_best_only=True)
                ],verbose=2
                )
#                 print(f"{time.time() - start_time:.2f}s")

                histories.append(history)
                

                # Caveat: The prediction format requires the output to be the same length as the input,
                # although it's not the case for the training data.

                #for evaluation you should make a universal pred length adnd seq length and here set dem and check out with ifs
                #in the comlile elmo method.
                
                model_short.load_weights('model'+str(fold)+'.h5')
                model_long.load_weights('model'+str(fold)+'.h5')

                if fold == 0:
                    public_preds =  model_short.predict([public_inputs])/1
                    private_preds = model_long.predict([private_inputs])/1
                else:
                    public_preds +=  model_short.predict([public_inputs])/FOLDS
                    private_preds +=  model_long.predict([private_inputs])/FOLDS
   


#                 input_shape = (107, 5)#model.input_shape[1:]
#                 # Generate an input sample
#                 input_sample = public_inputs[16].reshape(1, *input_shape)#np.random.rand(1, *input_shape)
#                 input_sample = tf.convert_to_tensor(input_sample)

#                 # Define a function to compute the saliency map
#                 def compute_saliency_map(input_sample, model):
#                     with tf.GradientTape() as tape:
#                         tape.watch(input_sample)
#                         prediction = model(input_sample, training=False)
#                         loss = tf.reduce_mean(prediction[:, 0])
#                     grads = tape.gradient(loss, input_sample)
#                     grads = grads / (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
#                     return grads

#                 # Compute the saliency map for the input sample
#                 saliency_map = compute_saliency_map(input_sample, model_short)
#                 import matplotlib.pyplot as plt

#                 sns.heatmap(saliency_map.numpy().reshape(*input_shape)[:20], cmap='Reds', annot=True, vmin=0, vmax=1)
#                 sns.heatmap(saliency_map.numpy().reshape(*input_shape), cmap='Reds', annot=True, vmin=0, vmax=1)
#                 plt.savefig('saliency_map_ELmo.jpg', format='jpg')
#                 plt.show()


                

                
                
                
                
#                 from keras.models import Model
#                 from keras.layers import Input, LSTM, Dense
#                 import matplotlib.pyplot as plt


#                 # Get the activations for the LSTM layer
#                 lstm_activations_model = Model(inputs=model_short.input, outputs=model_short.get_layer(index=1).output)
#                 lstm_activations = lstm_activations_model.predict(public_inputs)

#                 # Get the activations for the output layer
#                 output_activations_model = Model(inputs=model_short.input, outputs=model_short.output)
#                 output_activations = output_activations_model.predict(public_inputs)




#                 print(lstm_activations.shape)
#                 print(output_activations.shape)

#                 # Plot the LSTM activations
#                 plt.figure(figsize=(10, 5))
#                 for i in range(lstm_activations.shape[2]):
#                     plt.plot(lstm_activations[0, :, i], label='LSTM Activation {}'.format(i+1))
#                 plt.legend()
#                 plt.title('LSTM Activations')
#                 plt.xlabel('RNA Sequence position')
#                 plt.ylabel('Activation Value')
#                 plt.savefig('LSTMActivation_Elmo.jpg', format='jpg')
#                 plt.show()

#                 # Plot the output activations
#                 plt.figure(figsize=(10, 5))
#                 for i in range(output_activations.shape[2]):
#                     plt.plot(output_activations[0, :, i], label='Output Activation {}'.format(i+1))
#                 plt.legend()
#                 plt.title('Output Activations')
#                 plt.xlabel('RNA Sequence position')
#                 plt.ylabel('Activation Value')
#                 plt.savefig('OutputActivation_Elmo.jpg', format='jpg')
#                 plt.show()


                
#                 def sequence_similarity(public_df, i, j):

#                     # Define the loop and dot bracket information for two RNAs
#                     rna1 = public_df.iloc[i]['sequence']
#                     rna2 = public_df.iloc[j]['sequence']


#                     # Calculate the structural similarity between the two RNAs using pairwise sequence alignment
#                     alignments = pairwise2.align.globalxx(rna1, rna2)

#                     # Print the alignment score and the aligned sequences
#                     best_alignment = alignments[0]

#                     alignment_score = best_alignment.score

#                     # Calculate similarity score
#                     aligned_seq1, aligned_seq2, _, _,_ = best_alignment
#                     similarity_score = alignment_score / len(aligned_seq1)

#                 #     print("Alignment score:", alignment.score)
#                 #     print("similarity_score:", similarity_score)
#                     return similarity_score


#                 def structural_similarity(public_df, i, j):
#                     import RNA
#                     import Levenshtein  # install using pip install python-Levenshtein

#                     # define the dot-bracket notations for two RNA structures
#                     rna1_structure = public_df.iloc[i]['structure']
#                     rna2_structure = public_df.iloc[j]['structure']

#                     # calculate the tree edit distance between the two structures
#                     distance = Levenshtein.distance(rna1_structure, rna2_structure)

#                     # calculate the structural similarity between the two structures
#                     similarity = 1 - distance / len(rna1_structure)

#                     # print the structural similarity between the two structures
#                 #     print("The structural similarity between the two RNA structures is:", similarity)
#                     lev_distance = RNA.hamming_distance(rna1_structure, rna2_structure)

#                     # Print the Levenshtein distance
#                 #     print("The Levenshtein distance between the two secondary structures is:", lev_distance)
#                     return similarity

#                 def loop_type_similarity(public_df, i , j):

#                     seq1 = public_df.iloc[i]['predicted_loop_type']
#                     seq2 =public_df.iloc[j]['predicted_loop_type']
#                     # Define loop types
#                     loop_types = {
#                         'E': 'external loop',
#                         'S': 'stem',
#                         'B': 'bulge',
#                         'H': 'hairpin loop',
#                         'X': 'interior loop',
#                         'I': 'internal loop',
#                         'M': 'multiloop'


#                     }

#                     # Calculate structural similarity based on loop types
#                     similarity = sum([1 for i in range(len(seq1)) if seq1[i] == seq2[i]]) / len(seq1)

#                     # Print the structural similarity of two RNA molecules
#                 #     print("The structural similarity between the two RNA sequences is:", similarity)

#                     # Print the loop types of each RNA molecule
#                 #     for i in range(len(seq1)):
#                 #         if seq1[i] != seq2[i]:
#                 #             print("Sequence 1 has", loop_types[seq1[i]], "at position", i+1, "and sequence 2 has", loop_types[seq2[i]], "at position", i+1)
#                     return similarity

#                 data = output_activations  # replace with your own data file path
#                 def compute_similarity(dataset):
#                     """
#                     Compute cosine similarity, sequence similarity, and structural similarity for all pairs of RNAs in the dataset.

#                     Args:
#                         dataset: a list of K RNA sequences

#                     Returns:
#                         cosine_similarities: a K x K numpy array of cosine similarities between RNA sequences
#                         sequence_similarities: a K x K numpy array of sequence similarities between RNA sequences
#                         structural_similarities: a K x K numpy array of structural similarities between RNA sequences
#                         loop_similarities: a K x K numpy array of loop similarities between RNA sequences

#                     """
#                     K = len(dataset)

#                     n_samples, n_features, n_dims = dataset.shape

#                     # Reshape the dataset into a 2D array of shape (n_samples, n_features * n_dims)
#                     vectorized_dataset = np.reshape(dataset, (n_samples, n_features * n_dims))

#                     # Compute cosine similarity
#                     cosine_similarities = cosine_similarity(vectorized_dataset)

#                     # Compute sequence similarity
#                     sequence_similarities = np.zeros((K, K))
#                     for i in range(K):
#                         for j in range(K):
#                             sequence_similarities[i, j] = sequence_similarity(public_df, i, j) / max(len(dataset[i]), len(dataset[j]))

#                     # Compute structural similarity
#                     structural_similarities = np.zeros((K, K))
#                     for i in range(K):
#                         for j in range(K):
#                             structural_similarities[i, j] = structural_similarity(public_df, i, j) / max(len(dataset[i]), len(dataset[j]))


#                     loop_similarities = np.zeros((K, K))
#                     for i in range(K):
#                         for j in range(K):
#                             loop_similarities[i, j] = loop_type_similarity(public_df, i , j) / max(len(dataset[i]), len(dataset[j]))



#                     return cosine_similarities, sequence_similarities, structural_similarities, loop_similarities

#                 cosine_similarities, sequence_similarities, structural_similarities, loop_similarities = compute_similarity(data[:30])
# #                 print(cosine_similarities.shape, sequence_similarities.shape, structural_similarities.shape, loop_similarities.shape)
                
#                 # Next, we plot the cosine similarity as a heatmap using imshow function from matplotlib. 
#                 # We set the axis labels to indicate the index of each array, and we add a colorbar to indicate 
#                 # the similarity values. We rotate the tick labels for the x-axis to improve readability. 
#                 # Finally, we loop over the similarity values and add text annotations to the heatmap.
#                 # The resulting plot shows the cosine similarity between each pair of arrays as a heatmap.
#                 # The brighter the color, the higher the cosine similarity between the corresponding arrays.


#                 # Generate 629 random 68x5 arrays
#                 arrays = data[:30]#np.random.rand(629, 68, 5)

#                 # Compute the cosine similarity between all pairs of arrays
#                 similarity = cosine_similarity(arrays.reshape(30, -1))

#                 # Reshape the similarity array back to a 629x629 square matrix
#                 similarity = similarity.reshape(30, 30)

#                 # Plot the cosine similarity as a heatmap
#                 fig, ax = plt.subplots()
#                 im = ax.imshow(similarity, cmap='YlGnBu')

#                 # Add a colorbar
#                 cbar = ax.figure.colorbar(im, ax=ax)

#                 # Set the axis labels
#                 ax.set_xticks(np.arange(len(arrays)))
#                 ax.set_yticks(np.arange(len(arrays)))
#                 ax.set_xticklabels(np.arange(1, len(arrays)+1))
#                 ax.set_yticklabels(np.arange(1, len(arrays)+1))
#                 ax.set_xlabel('Array index')
#                 ax.set_ylabel('Array index')

#                 # Rotate the tick labels and set their alignment
#                 plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
#                          rotation_mode='anchor')

#                 # Loop over the data and add text annotations
#                 for i in range(len(arrays)):
#                     for j in range(len(arrays)):
#                         text = ax.text(j, i, f'{similarity[i, j]:.2f}',
#                                        ha='center', va='center', color='black')

#                 # Set the title
#                 ax.set_title('Cosine similarity between 30 out of 629 arrays')
#                 plt.savefig('cosinesimilarity_Elmo.jpg', format='jpg')

#                 # Show the plot
#                 plt.show()

                
                
                
                
                
                
#                 import numpy as np
#                 import matplotlib.pyplot as plt

#                 # Load the similarity matrices
#                 cosine_sim = cosine_similarities#np.load('cosine_similarities.npy')
#                 seq_sim = sequence_similarities#np.load('sequence_similarities.npy')
#                 struct_sim = structural_similarities#np.load('structural_similarities.npy')
#                 loop_sim = loop_similarities#np.load('loop_similarities.npy')

#                 # Normalize the similarity matrices
#                 cosine_sim_norm = cosine_sim / np.max(cosine_sim)
#                 seq_sim_norm = seq_sim / np.max(seq_sim)
#                 struct_sim_norm = struct_sim / np.max(struct_sim)
#                 loop_sim_norm = loop_sim / np.max(loop_sim)

#                 # Calculate the correlation coefficients
#                 corr_cos_seq = np.corrcoef(cosine_sim_norm.flatten(), seq_sim_norm.flatten())[0,1]
#                 corr_cos_struct = np.corrcoef(cosine_sim_norm.flatten(), struct_sim_norm.flatten())[0,1]
#                 corr_cos_loop = np.corrcoef(cosine_sim_norm.flatten(), loop_sim_norm.flatten())[0,1]

#                 # Create a heatmap of the correlation coefficients
#                 corr_matrix = np.array([[1, corr_cos_seq, corr_cos_struct, corr_cos_loop],
#                                        [corr_cos_seq, 1, 0, 0],
#                                        [corr_cos_struct, 0, 1, 0],
#                                        [corr_cos_loop, 0, 0, 1]])
#                 fig, ax = plt.subplots()
#                 im = ax.imshow(corr_matrix, cmap='coolwarm')
#                 ax.set_xticks(np.arange(len(['Cosine', 'Sequence', 'Structural', 'Loop'])))
#                 ax.set_yticks(np.arange(len(['Cosine', 'Sequence', 'Structural', 'Loop'])))
#                 ax.set_xticklabels(['Cosine', 'Sequence', 'Structural', 'Loop'])
#                 ax.set_yticklabels(['Cosine', 'Sequence', 'Structural', 'Loop'])
#                 plt.colorbar(im)
#                 plt.savefig('correlation_cosine_str_loop_seq_Elmo.jpg', format='jpg')
#                 plt.show()

                ################################## Pearson correlation

                val_losses = []
                y_preds_best = None
                PRED_COLS = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C']
                y_valid= val_labs
                y_preds = self._model.predict(val_)


                mcloss = mcrmse(val_labs, y_preds)
                val_losses.append(mcloss)
                print('np.min(val_losses)',np.min(val_losses),'val_losses',np.mean(val_losses),'mcloss',np.mean(mcloss))

                if np.min(np.mean(val_losses)) == np.mean(mcloss):
                    y_preds_best = y_preds
                import matplotlib.pyplot as plt

                print('y_preds_best shape, y_valid shape', y_valid.shape, y_preds_best.shape)
                fig, ax = plt.subplots(1, 3, figsize=(24, 8))
                for i, p in enumerate(PRED_COLS):
                    ax[2].scatter(y_valid[:, :, i].flatten(), y_preds_best[:, :, i].flatten(), alpha=0.5)#(not correct but) works if y_preds_best= y_preds = model.predict(val_)

                ax[2].legend(PRED_COLS)
                ax[2].set_xlabel('y_true')
                ax[2].set_ylabel('y_predicted')
                plt.show()
                
                
                

        end_time = time.time()  # Record the start time
        duration = end_time - start_time
        print("duration ,end_time , start_time",duration ,end_time , start_time)



        import numpy as np
        import matplotlib.pyplot as plt

        # Assuming 'histories' is a list of lists containing individual fold histories

        results = {"models": ['elmo', 'elmo'], "histories": [histories, histories]}
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


                
       
#         results = {"models" : ['elmo','elmo'],"histories" : [histories,histories]}
#         fig, ax = plt.subplots(1, len(results['histories']), figsize = (20, 10))
#         fig2, ax2 = plt.subplots(1, len(results['histories']), figsize = (20, 10))      

#         for i, result in enumerate(results['histories']):
#                 for history in result:
# #                     print('history.history.keys()',history.history.keys())
#                     ax[i].plot(history.history['loss'], color='C0')
#                     ax[i].plot(history.history['val_loss'], color='C1')
#                     ax[i].plot(history.history['mae'], color='C2')
#                     ax[i].plot(history.history['mse'], color='C3')
#                     ax[i].set_title(f"{results['models'][i]}")
#                     ax[i].set_ylabel('Loss')
#                     ax[i].set_xlabel('Epoch')
#                     ax[i].legend(['train', 'validation','MAE','MSE'], loc = 'upper right')
                    
#                 for history in result:
#                     ax2[i].plot(history.history['tf_pearson'], color='C1')
#                     ax2[i].plot(history.history['val_tf_pearson'], color='C2')
#                     ax2[i].set_title(f"{results['models'][i]}")
#                     ax2[i].set_ylabel('Pearson Correlation')
#                     ax2[i].set_xlabel('Epoch')
#                     ax2[i].legend(['train', 'validation'], loc = 'upper right')
                                      
                    
# # history.history.keys() dict_keys(['loss', 'mse', 'tf_pearson', 'mae', 'val_loss', 'val_mse', 'val_tf_pearson', 'val_mae', 'lr
                   
                
     
        

        preds_ls = []
        for df, preds in [(public_df, public_preds), (private_df, private_preds)]:
            for i, uid in enumerate(df.id):
                single_pred = preds[i]

                single_df = pd.DataFrame(single_pred, columns=pred_cols)
                single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

                preds_ls.append(single_df)
        sample_df = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')

        preds_df = pd.concat(preds_ls)
        submission = sample_df[['id_seqpos']].merge(preds_df, on=['id_seqpos'])
        submission.to_csv('submission.csv', index=False)
        print('Submission saved')
        return results

    def evaluate(self, test_data):

        def unpad(x, y_true, y_pred):
            y_true_unpad = []
            y_pred_unpad = []
            for i, x_i in enumerate(x):
                for j, x_ij in enumerate(x_i):
                    if x_ij == 0:
                        y_true_unpad.append(y_true[i][:j])
                        y_pred_unpad.append(y_pred[i][:j])
                        break
            return np.asarray(y_true_unpad), np.asarray(y_pred_unpad)

        # Generate samples
        x, y_true_forward, y_true_backward = [], [], []
        for i in range(len(test_data)):
            test_batch = test_data[i][0]
            x.extend(test_batch[0])
            y_true_forward.extend(test_batch[1])
            y_true_backward.extend(test_batch[2])
        x = np.asarray(x)
        y_true_forward = np.asarray(y_true_forward)
        y_true_backward = np.asarray(y_true_backward)

        # Predict outputs
        y_pred_forward, y_pred_backward = self._model.predict([x, y_true_forward, y_true_backward])

        # Unpad sequences
        y_true_forward, y_pred_forward = unpad(x, y_true_forward, y_pred_forward)
        y_true_backward, y_pred_backward = unpad(x, y_true_backward, y_pred_backward)

        # Compute and print perplexity
        print('Forward Langauge Model Perplexity: {}'.format(ELMo.perplexity(y_pred_forward, y_true_forward)))
        print('Backward Langauge Model Perplexity: {}'.format(ELMo.perplexity(y_pred_backward, y_true_backward)))
    def wrap_multi_elmo_encoder(self, print_summary=False, save=False):
        """
        Wrap ELMo meta-model encoder, which returns an array of the 3 intermediate ELMo outputs
        :param print_summary: print a summary of the new architecture
        :param save: persist model
        :return: None
        """

        elmo_embeddings = list()
        elmo_embeddings.append(concatenate([self._model.get_layer('token_encoding').output, self._model.get_layer('token_encoding').output],
                                           name='elmo_embeddings_level_0'))
        for i in range(self.parameters['n_lstm_layers']):
            elmo_embeddings.append(concatenate([self._model.get_layer('f_block_{}'.format(i + 1)).output,
                                                Lambda(function=ELMo.reverse)
                                                (self._model.get_layer('b_block_{}'.format(i + 1)).output)],
                                               name='elmo_embeddings_level_{}'.format(i + 1)))

        camos = list()
        for i, elmo_embedding in enumerate(elmo_embeddings):
            camos.append(Camouflage(mask_value=0.0, name='camo_elmo_embeddings_level_{}'.format(i + 1))([elmo_embedding,
                                                                                                         self._model.get_layer(
                                                                                                             'token_encoding').output]))

        self._elmo_model = Model(inputs=[self._model.get_layer('word_indices').input], outputs=camos)

        if print_summary:
            self._elmo_model.summary()

        if save:
            self._elmo_model.save(os.path.join(MODELS_DIR, 'ELMo_Encoder.hd5'))
            print('ELMo Encoder saved successfully')

    def save(self, sampled_softmax=False):
        """
        Persist model in disk
        :param sampled_softmax: reload model using the full softmax function
        :return: None
        """
        if not sampled_softmax:
            self.parameters['num_sampled'] = self.parameters['vocab_size']
#         self._model.load_weights(os.path.join(MODELS_DIR, 'elmo_best_weights.hdf5'))

        self._model.save(os.path.join(MODELS_DIR, 'ELMo_LM_EVAL.hd5'))
        print('ELMo Language Model saved successfully')

    def load(self):
        self._model = load_model(os.path.join(MODELS_DIR, 'ELMo_LM.h5'),custom_objects={'TimestepDropout': TimestepDropout,'Camouflage': Camouflage})

    def load_elmo_encoder(self):
        self._elmo_model = load_model(os.path.join(MODELS_DIR, 'ELMo_Encoder.hd5'),custom_objects={'TimestepDropout': TimestepDropout,'Camouflage': Camouflage})

    def get_outputs(self, test_data, output_type='word', state='last'):
        """
       Wrap ELMo meta-model encoder, which returns an array of the 3 intermediate ELMo outputs
       :param test_data: data generator
       :param output_type: "word" for word vectors or "sentence" for sentence vectors
       :param state: 'last' for 2nd LSTMs outputs or 'mean' for mean-pooling over inputs, 1st LSTMs and 2nd LSTMs
       :return: None
       """
        # Generate samples
        x = []
        for i in range(len(test_data)):
            test_batch = test_data[i][0]
            x.extend(test_batch[0])

        preds = np.asarray(self._elmo_model.predict(np.asarray(x)))
        if state == 'last':
            elmo_vectors = preds[-1]
        else:
            elmo_vectors = np.mean(preds, axis=0)

        if output_type == 'words':
            return elmo_vectors
        else:
            return np.mean(elmo_vectors, axis=1)

    @staticmethod
    def reverse(inputs, axes=1):
        return K.reverse(inputs, axes=axes)

    @staticmethod
    def perplexity(y_pred, y_true):

        cross_entropies = []
        for y_pred_seq, y_true_seq in zip(y_pred, y_true):
            # Reshape targets to one-hot vectors
            y_true_seq = to_categorical(y_true_seq, y_pred_seq.shape[-1])
            # Compute cross_entropy for sentence words
            cross_entropy = K.categorical_crossentropy(K.tf.convert_to_tensor(y_true_seq, dtype=K.tf.float32),K.tf.convert_to_tensor(y_pred_seq, dtype=K.tf.float32))
            cross_entropies.extend(cross_entropy.eval(session=K.get_session()))

        # Compute mean cross_entropy and perplexity
        cross_entropy = np.mean(np.asarray(cross_entropies), axis=-1)

        return pow(2.0, cross_entropy)


# The model should be sth like:
# 
# Layer (type)                    Output Shape         Param #     Connected to                     
# __________________________________________________________________________________________________
# input_2 (InputLayer)            [(None, 107, 5)]     0                                            
# __________________________________________________________________________________________________
# tf_op_layer_strided_slice_3 (Te [(None, 107, 3)]     0           input_2[0][0]                    
# __________________________________________________________________________________________________
# embedding_1 (Embedding)         (None, 107, 3, 200)  2800        tf_op_layer_strided_slice_3[0][0]
# __________________________________________________________________________________________________
# tf_op_layer_Reshape_1 (TensorFl [(None, 107, 600)]   0           embedding_1[0][0]          

# In[ ]:





# In[ ]:


from __future__ import absolute_import
from __future__ import division
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

from tensorflow.keras.layers import InputSpec
# from tensorflow.keras.layers import Dropout

class TimestepDropout(Dropout):
    """Word Dropout.

    This version performs the same function as Dropout, however it drops
    entire timesteps (e.g., words embeddings) instead of individual elements (features).

    # Arguments
        rate: float between 0 and 1. Fraction of the timesteps to drop.

    # Input shape
        3D tensor with shape:
        `(samples, timesteps, channels)`

    # Output shape
        Same as input

    # References
        - N/A
    """

    def __init__(self, rate, **kwargs):
        super(TimestepDropout, self).__init__(rate, **kwargs)
        self.input_spec = InputSpec(ndim=3)

    def _get_noise_shape(self, inputs):
        input_shape = K.shape(inputs)
        noise_shape = (input_shape[0], input_shape[1], 1)
        return noise_shape
    
from tensorflow.keras import initializers, regularizers, constraints, activations
from tensorflow.keras.initializers import Constant
from tensorflow.keras import backend as K


class Highway(Layer):
    """Highway network, a natural extension of LSTMs to feedforward networks.

    # Arguments
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        transform_activation: Activation function to use
            for the transform unit
            (see [activations](../activations.md)).
            Default: sigmoid (`sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).x
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        transform_initializer: Initializer for the `transform` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        transform_bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
            Default: -2 constant.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        transform_regularizer: Regularizer function applied to
            the `transform` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        transform_bias_regularizer: Regularizer function applied to the transform bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # References
        - [Highway Networks](http://arxiv.org/pdf/1505.00387v2.pdf)
    """

    def __init__(self,
                 activation='relu',
                 transform_activation='sigmoid',
                 kernel_initializer='glorot_uniform',
                 transform_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 transform_bias_initializer=-2,
                 kernel_regularizer=None,
                 transform_regularizer=None,
                 bias_regularizer=None,
                 transform_bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.activation = activations.get(activation)
        self.transform_activation = activations.get(transform_activation)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.transform_initializer = initializers.get(transform_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        if isinstance(transform_bias_initializer, int):
            self.transform_bias_initializer = Constant(value=transform_bias_initializer)
        else:
            self.transform_bias_initializer = initializers.get(transform_bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.transform_regularizer = regularizers.get(transform_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.transform_bias_regularizer = regularizers.get(transform_bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(Highway, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[-1]

        self.W = self.add_weight(shape=(input_dim, input_dim),
                                 name='{}_W'.format(self.name),
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.W_transform = self.add_weight(shape=(input_dim, input_dim),
                                           name='{}_W_transform'.format(self.name),
                                           initializer=self.transform_initializer,
                                           regularizer=self.transform_regularizer,
                                           constraint=self.kernel_constraint)

        self.bias = self.add_weight(shape=(input_dim,),
                                 name='{}_bias'.format(self.name),
                                 initializer=self.bias_initializer,
                                 regularizer=self.bias_regularizer,
                                 constraint=self.bias_constraint)

        self.bias_transform = self.add_weight(shape=(input_dim,),
                                           name='{}_bias_transform'.format(self.name),
                                           initializer=self.transform_bias_initializer,
                                           regularizer=self.transform_bias_regularizer)

        self.built = True

    def call(self, x, mask=None):
        x_h = self.activation(K.dot(x, self.W) + self.bias)
        x_trans = self.transform_activation(K.dot(x, self.W_transform) + self.bias_transform)
        output = x_h * x_trans + (1 - x_trans) * x
        return output

    def get_config(self):
        config = {'activation': activations.serialize(self.activation),
                  'transform_activation': activations.serialize(self.transform_activation),
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'transform_initializer': initializers.serialize(self.transform_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'transform_bias_initializer': initializers.serialize(self.transform_bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'transform_regularizer': regularizers.serialize(self.transform_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'transform_bias_regularizer': regularizers.serialize(self.transform_bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)
                  }
        base_config = super(Highway, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    # -*- coding: utf-8 -*-
"""Core Keras layers.
"""


class Camouflage(Layer):
    """Masks a sequence by using a mask value to skip timesteps based on another sequence.
       LSTM and Convolution layers may produce fake tensors for padding timesteps. We need
       to eliminate those tensors by replicating their initial values presented in the second input.

       inputs = Input()
       lstms = LSTM(units=100, return_sequences=True)(inputs)
       padded_lstms = Camouflage()([lstms, inputs])
       ...
    """

    def __init__(self, mask_value=0., **kwargs):
        super(Camouflage, self).__init__(**kwargs)
        self.mask_value = mask_value

    def call(self, inputs):
        boolean_mask = K.any(K.not_equal(inputs[1], self.mask_value),
                             axis=-1, keepdims=True)
        return inputs[0] * K.cast(boolean_mask, K.dtype(inputs[0]))

    def get_config(self):
        config = {'mask_value': self.mask_value}
        base_config = super(Camouflage, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape[0]
    


class SampledSoftmax(Layer):
    """Sampled Softmax, a faster way to train a softmax classifier over a huge number of classes.

        # Arguments
            num_classes: number of classes
            num_sampled: number of classes to be sampled at each batch
            tied_to: layer to be tied with (e.g., Embedding layer)
            kwargs:
        # Input shape
            2D tensor with shape: `(nb_samples, input_dim)`.
        # Output shape
            2D tensor with shape: `(nb_samples, input_dim)`.
        # References
            - [Tensorflow code](tf.nn.sampled_softmax_loss)
            - [Sampled SoftMax](https://www.tensorflow.org/extras/candidate_sampling.pdf)
        """
    def __init__(self, num_classes=50000, num_sampled=1000, tied_to=None, **kwargs):
            super(SampledSoftmax, self).__init__(**kwargs)
            self.num_sampled = num_sampled
            self.num_classes = num_classes
            self.tied_to = tied_to
            self.sampled = (self.num_classes != self.num_sampled)

    def build(self, input_shape):
            if self.tied_to is None:
                self.softmax_W = self.add_weight(shape=(self.num_classes, input_shape[0][-1]), name='W_soft', initializer='lecun_normal')
            self.softmax_b = self.add_weight(shape=(self.num_classes,), name='b_soft', initializer='zeros')
            self.built = True

    def call(self, x, mask=None):
        lstm_outputs, next_token_ids = x

    def sampled_softmax(x):
            lstm_outputs_batch, next_token_ids_batch = x
            batch_losses = tf.nn.sampled_softmax_loss(
                self.softmax_W if self.tied_to is None else self.tied_to.weights[0], self.softmax_b,
                next_token_ids_batch, lstm_outputs_batch,
                num_classes=self.num_classes,
                num_sampled=self.num_sampled
#                 ,partition_strategy='div'
            )
            batch_losses = tf.reduce_mean(batch_losses)
            return [batch_losses, batch_losses]

    def softmax(x):
            lstm_outputs_batch, next_token_ids_batch = x
            logits = tf.matmul(lstm_outputs_batch,
                                 tf.transpose(self.softmax_W) if self.tied_to is None else tf.transpose(self.tied_to.weights[0]))
            logits = tf.nn.bias_add(logits, self.softmax_b)
            batch_predictions = tf.nn.softmax(logits)
            labels_one_hot = tf.one_hot(tf.cast(next_token_ids_batch, dtype=tf.int32), self.num_classes)
            batch_losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=logits)
            return [batch_losses, batch_predictions]
        
            losses, predictions = tf.map_fn(sampled_softmax if self.sampled else softmax, [lstm_outputs, next_token_ids])
            self.add_loss(0.5 * tf.reduce_mean(losses[0]))
            return lstm_outputs if self.sampled else predictions

    def compute_output_shape(self, input_shape):
        return input_shape[0] if self.sampled else (input_shape[0][0], input_shape[0][1], self.num_classes)


# **LMDataGenerator.py**
# 
# **if token_encoding ==word : assigns each word in a sentence a pre-assigned integer value from vocab.token. processes all the corpus line by line. at the end encryptes each sentence to a 1 x 100 array of integers.
# 
# **
# 
# train_generator.indices#array([    1,     3,     5, ..., 36713, 36715, 36717]) even numbers in a row
# 
# len(train_generator.vocab)#28914
# 
# len(train_generator.indices)#18359
# 
# train_generator.__getitem__(18358)# last can be reached
# 
# train_generator.indices#array([    1,     3,     5, ..., 36713, 36715, 36717]) even numbers in a row
# 
# len(train_generator.vocab)#28914
# 
# len(train_generator.indices)#18359
# 
# train_generator.__getitem__(18358)# last can be reached
# 

# In[ ]:


import numpy as np
from tensorflow import keras 


# class LMDataGenerator(keras.utils.Sequence):
#     """Generates data for Keras"""

#     def __len__(self):
#         """Denotes the number of batches per epoch"""
#         return int(np.ceil(len(self.indices)/self.batch_size))

#     def __init__(self, corpus, vocab, sentence_maxlen=100, token_maxlen=50, batch_size=32, shuffle=True, token_encoding='word'):
#         """Compiles a Language Model RNN based on the given parameters
#         :param corpus: filename of corpus
#         :param vocab: filename of vocabulary
#         :param sentence_maxlen: max size of sentence
#         :param token_maxlen: max size of token in characters
#         :param batch_size: number of steps at each batch
#         :param shuffle: True if shuffle at the end of each epoch
#         :param token_encoding: Encoding of token, either 'word' index or 'char' indices
#         :return: Nothing
#         """

#         self.corpus = corpus
#         self.vocab = {line.split()[0]: int(line.split()[1]) for line in open(vocab).readlines()}
#         self.sent_ids = corpus
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.sentence_maxlen = sentence_maxlen
#         self.token_maxlen = token_maxlen
#         self.token_encoding = token_encoding
#         with open(self.corpus) as fp:
#             self.indices = np.arange(len(fp.readlines()))
#             newlines = [index for index in range(0, len(self.indices), 2)]
#             self.indices = np.delete(self.indices, newlines)

#     def __getitem__(self, index):
#         """Generate one batch of data"""
#         # Generate indexes of the batch
#         batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

#         # Read sample sequences
#         word_indices_batch = np.zeros((len(batch_indices), self.sentence_maxlen), dtype=np.int32)
#         if self.token_encoding == 'char':
#             word_char_indices_batch = np.full((len(batch_indices), self.sentence_maxlen, self.token_maxlen), 260, dtype=np.int32)

#         for i, batch_id in enumerate(batch_indices):
#             # Read sentence (sample)
#             word_indices_batch[i] = self.get_token_indices(sent_id=batch_id)
#             if self.token_encoding == 'char':
#                 word_char_indices_batch[i] = self.get_token_char_indices(sent_id=batch_id)

#         # Build forward targets
#         for_word_indices_batch = np.zeros((len(batch_indices), self.sentence_maxlen), dtype=np.int32)

#         padding = np.zeros((1,), dtype=np.int32)

#         for i, word_seq in enumerate(word_indices_batch ):
#             for_word_indices_batch[i] = np.concatenate((word_seq[1:], padding), axis=0)

#         for_word_indices_batch = for_word_indices_batch[:, :, np.newaxis]

#         # Build backward targets
#         back_word_indices_batch = np.zeros((len(batch_indices), self.sentence_maxlen), dtype=np.int32)

#         for i, word_seq in enumerate(word_indices_batch):
#             back_word_indices_batch[i] = np.concatenate((padding, word_seq[:-1]), axis=0)

#         back_word_indices_batch = back_word_indices_batch[:, :, np.newaxis]

#         return [word_indices_batch if self.token_encoding == 'word' else word_char_indices_batch, for_word_indices_batch, back_word_indices_batch], []

#     def on_epoch_end(self):
#         """Updates indexes after each epoch"""
#         if self.shuffle:
#             np.random.shuffle(self.indices)

#     def get_token_indices(self, sent_id: int):
#         with open(self.corpus) as fp:
#             for i, line in enumerate(fp):
#                 if i == sent_id:
#                     token_ids = np.zeros((self.sentence_maxlen,), dtype=np.int32)
#                     # Add begin of sentence index
#                     token_ids[0] = self.vocab['<bos>']
#                     for j, token in enumerate(line.split()[:self.sentence_maxlen - 2]):
#                         if token.lower() in self.vocab:
#                             token_ids[j + 1] = self.vocab[token.lower()]
#                         else:
#                             token_ids[j + 1] = self.vocab['<unk>']
#                     # Add end of sentence index
#                     if token_ids[1]:
#                         token_ids[j + 2] = self.vocab['<eos>']
#                     return token_ids

#     def get_token_char_indices(self, sent_id: int):
#         def convert_token_to_char_ids(token, token_maxlen):
#             bos_char = 256  # <begin sentence>
#             eos_char = 257  # <end sentence>
#             bow_char = 258  # <begin word>
#             eow_char = 259  # <end word>
#             pad_char = 260  # <pad char>
#             char_indices = np.full([token_maxlen], pad_char, dtype=np.int32)
#             # Encode word to UTF-8 encoding
#             word_encoded = token.encode('utf-8', 'ignore')[:(token_maxlen - 2)]
#             # Set characters encodings
#             # Add begin of word char index
#             char_indices[0] = bow_char
#             if token == '<bos>':
#                 char_indices[1] = bos_char
#                 k = 1
#             elif token == '<eos>':
#                 char_indices[1] = eos_char
#                 k = 1
#             else:
#                 # Add word char indices
#                 for k, chr_id in enumerate(word_encoded, start=1):
#                     char_indices[k] = chr_id + 1
#             # Add end of word char index
#             char_indices[k + 1] = eow_char
#             return char_indices

#         with open(self.corpus) as fp:
#             for i, line in enumerate(fp):
#                 if i == sent_id:
#                     token_ids = np.zeros((self.sentence_maxlen, self.token_maxlen), dtype=np.int32)
#                     # Add begin of sentence char indices
#                     token_ids[0] = convert_token_to_char_ids('<bos>', self.token_maxlen)
#                     # Add tokens' char indices
#                     for j, token in enumerate(line.split()[:self.sentence_maxlen - 2]):
#                         token_ids[j + 1] = convert_token_to_char_ids(token, self.token_maxlen)
#                     # Add end of sentence char indices
#                     if token_ids[1]:
#                         token_ids[j + 2] = convert_token_to_char_ids('<eos>', self.token_maxlen)
#         return token_ids


# In[ ]:


import os
import tensorflow.keras.backend as K

# from data import DATA_SET_DIR
DATA_SET_DIR='../input/wikielmo/data/datasets'
# from elmo.lm_generator import LMDataGenerator
# from elmo.model import ELMo

parameters = {
    'multi_jcessing': False,
    'n_threads': 4,
    'cuDNN': True,
    'train_dataset': 'wikitext-2/wiki.train.tokens',
    'valid_dataset': 'wikitext-2/wiki.valid.tokens',
    'test_dataset': 'wikitext-2/wiki.test.tokens',
    'vocab': 'wikitext-2/wiki.vocab',
    'vocab_size': 28914,
    'num_sampled': 1000,
    'charset_size': 262,
    'sentence_maxlen': 100,
    'token_maxlen': 50,
    'token_encoding': 'word',
    'epochs': 50,
    'patience': 2,
    'batch_size': 1,
    'clip_value': 1,
    'cell_clip': 5,
    'proj_clip': 5,
    'lr': 0.2,
    'shuffle': True,
    'n_lstm_layers': 1,
    'n_highway_layers': 1,
    'cnn_filters': [[1, 32],
                    [2, 32],
                    [3, 64],
                    [4, 128],
                    [5, 256],
                    [6, 512],
                    [7, 512]
                    ],
    'lstm_units_size': 400,
    'hidden_units_size': 602,# as proj should be same length as input lstm for adding
    'char_embedding_size': 16,
    'dropout_rate': 0.4,
    'word_dropout_rate': 0.1,
    'weight_tying': False,# I changed this to false as in sampled softmax with embeddings paramertee as "tied into" the dimension is not compatible with reshaped one (602 & 200) by changing
#     it manually youu'll get tensor has no weight_tying parameter 
}

# Compile ELMo
elmo_model = ELMo(parameters)
plots = elmo_model.compile_elmo(print_summary=True)

# Train ELMo
# elmo_model.train(train_data=train_generator, valid_data=val_generator)
import time
start_time = time.time()  # Record the start time


results= elmo_model.train()


end_time = time.time()  # Record the start time
duration = end_time - start_time
print("duration ,end_time , start_time",duration ,end_time , start_time)





# Persist ELMo Bidirectional Language Model in disk
elmo_model.save(sampled_softmax=False)

# Evaluate Bidirectional Language Model
# elmo_model.evaluate(test_generator)

     


# Build ELMo meta-model to deploy for production and persist in disk
# elmo_model.wrap_multi_elmo_encoder(print_summary=True, save=True)

# Load ELMo encoder
# elmo_model.load_elmo_encoder()

# Get ELMo embeddings to feed as inputs for downstream tasks
# elmo_embeddings = elmo_model.get_outputs(test_generator, output_type='word', state='mean')

# BUILD & TRAIN NEW KERAS MODEL FOR DOWNSTREAM TASK (E.G., TEXT CLASSIFICATION)


# In[ ]:


STRATIFY=True
FOLDS=4
EPOCHS=50
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


# In[ ]:


model = elmo_model


# In[ ]:


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


# In[ ]:


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




plot_learning_curves(results)


# 

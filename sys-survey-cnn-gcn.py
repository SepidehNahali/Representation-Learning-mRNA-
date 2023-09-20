#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
import os
import time
import traceback
from contextlib import contextmanager
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
get_ipython().system('pip install nyaggle')
from nyaggle.validation import StratifiedGroupKFold
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import tensorflow as tf
Structure_Interpretability_Test = True


# In[2]:


TOKEN2INT = {x: i for i, x in enumerate('().ACGUBEHIMSX')}
PRED_COLS_SCORED = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']
PRED_COLS = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C']

DATA_DIR = "../input/stanford-covid-vaccine/"
REPLACE_DATA_PATH = "../input/eternafold/eternafold_mfe.csv"
PRIMARY_BPPS_DIR = "../input/eternafold/bpps/"
SECONDARY_BPPS_DIR = "../input/bpps-by-viennat70/"
NFOLDS = 4
BATCH_SIZE = 64
TRAIN_EPOCHS = 50#140


# In[3]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device,'::')


# In[4]:


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class Conv(nn.Module):
    def __init__(self, d_in, d_out, kernel_size, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(d_in, d_out, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(d_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        return self.dropout(self.relu(self.bn(self.conv(src))))


class ResidualGraphAttention(nn.Module):
    def __init__(self, d_model, kernel_size, dropout):
        super().__init__()
        self.conv1 = Conv(d_model, d_model, kernel_size=kernel_size, dropout=dropout)
        self.conv2 = Conv(d_model, d_model, kernel_size=kernel_size, dropout=dropout)
        self.relu = nn.ReLU()

    def forward(self, src, attn):
        h = self.conv2(self.conv1(torch.bmm(src, attn)))
        return self.relu(src + h)
    

class SEResidual(nn.Module):
    def __init__(self, d_model, kernel_size, dropout):
        super().__init__()
        self.conv1 = Conv(d_model, d_model, kernel_size=kernel_size, dropout=dropout)
        self.conv2 = Conv(d_model, d_model, kernel_size=kernel_size, dropout=dropout)
        self.relu = nn.ReLU()
        self.se = SELayer(d_model)

    def forward(self, src):
        h = self.conv2(self.conv1(src))
        return self.se(self.relu(src + h))


# class FusedEmbedding(nn.Module):
#     def __init__(self, n_emb):
#         super().__init__()
#         self.emb = nn.Embedding(len(TOKEN2INT), n_emb)
#         self.n_emb = n_emb

#     def forward(self, src, se):
#         # src: [batch, seq, feature]
#         # se: [batch, seq]
#         embed = self.emb(src)
#         embed = embed.reshape((-1, embed.shape[1], embed.shape[2] * embed.shape[3]))
#         embed = torch.cat((embed, se), 2)

#         return embed

#     @property
#     def d_out(self):
#         d_emb = 3 * self.n_emb
#         d_feat = 2 * 5  # max, sum, 2nd, 3rd, nb_count
#         return d_emb + d_feat
class FusedEmbedding(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.emb = nn.Embedding(len(TOKEN2INT), n_emb)
        self.n_emb = n_emb

    def forward(self, src, se):
        # src: [batch, seq, feature]
        # se: [batch, seq]
        src = src.long()  # Convert the src tensor to long for the embedding layer
        embed = self.emb(src)
        embed = embed.reshape((-1, embed.shape[1], embed.shape[2] * embed.shape[3]))
        embed = embed.float()  # Convert the embed tensor back to float for gradient computation
        embed = torch.cat((embed, se), 2)

        return embed
    def d_out(self):
        d_emb = 3 * self.n_emb
        d_feat = 2 * 5  # max, sum, 2nd, 3rd, nb_count
        return d_emb + d_feat  

class ConvModel(nn.Module):
    def __init__(self, d_emb=50, d_model=256, dropout=0.6, dropout_res=0.4, dropout_emb=0.0,
                 kernel_size_conv=7, kernel_size_gc=7):
        super().__init__()

        self.embedding = FusedEmbedding(d_emb)
        self.dropout = nn.Dropout(dropout_emb)
        self.conv = Conv(self.embedding.d_out(), d_model, kernel_size=3, dropout=dropout)

        self.block1 = SEResidual(d_model, kernel_size=kernel_size_conv, dropout=dropout_res)
        self.block2 = SEResidual(d_model, kernel_size=kernel_size_conv, dropout=dropout_res)
        self.block3 = SEResidual(d_model, kernel_size=kernel_size_conv, dropout=dropout_res)
        self.block4 = SEResidual(d_model, kernel_size=kernel_size_conv, dropout=dropout_res)
        self.block5 = SEResidual(d_model, kernel_size=kernel_size_conv, dropout=dropout_res)

        self.attn1 = ResidualGraphAttention(d_model, kernel_size=kernel_size_gc, dropout=dropout_res)
        self.attn2 = ResidualGraphAttention(d_model, kernel_size=kernel_size_gc, dropout=dropout_res)
        self.attn3 = ResidualGraphAttention(d_model, kernel_size=kernel_size_gc, dropout=dropout_res)
        self.attn4 = ResidualGraphAttention(d_model, kernel_size=kernel_size_gc, dropout=dropout_res)

        self.linear = nn.Linear(d_model, len(PRED_COLS))

    def forward(self, 
                src: torch.Tensor, 
                features: torch.Tensor, 
                bpps: torch.Tensor, 
                adj: torch.Tensor):
        # src: [batch, seq, 3]
        # features: [batch, seq, 10]
        # bpps: [batch, seq, seq, 2]
        # adj: [batch, seq, seq]
        
        x = self.dropout(self.embedding(src, features))############################################ for Saliency map commented
        x = x.permute([0, 2, 1])  # [batch, d-emb, seq]
        
        x = self.conv(x)
        x = self.block1(x)
        x = self.attn1(x, adj)
        x = self.block2(x)
        x = self.attn2(x, adj)
        x = self.block3(x)
        x = self.attn3(x, bpps[:, :, :, 0])
        x = self.attn4(x, bpps[:, :, :, 1])
        x = self.block4(x)
        x = self.block5(x)

        x = x.permute([0, 2, 1])  # [batch, seq, features]
        out = self.linear(x)

        out = torch.clamp(out, -0.5, 1e8)

        return out
    


class WRMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yhat, y, sample_weight=None):
        l = (yhat - y) ** 2

        if sample_weight is not None:
            l = l * sample_weight.unsqueeze(dim=1)

        return torch.sqrt(torch.mean(l))


class ColWiseLoss(nn.Module):
    def __init__(self, base_loss):
        super().__init__()
        self.base_loss = base_loss
        self.len_scored = 68

    def forward(self, yhat, y, column_weight=None, sample_weight=None):
        score = 0
        for i in range(len(PRED_COLS)):
            s = self.base_loss(
                yhat[:, :self.len_scored, i], 
                y[:, :self.len_scored, i], 
                sample_weight
            ) / len(PRED_COLS)
            
            if column_weight is not None:
                s *= column_weight[i]
                
            score += s
        return score


class MCRMSELoss(ColWiseLoss):
    def __init__(self):
        super().__init__(WRMSELoss())


# In[5]:


@contextmanager
def timer(name):
    s = time.time()
    yield
    print(f"{name}: {time.time() - s:.3f}sec")


def pandas_list_to_array(df: pd.DataFrame) -> np.ndarray:
    return np.transpose(
        np.array(
            df.values
                .tolist()
        ),
        (0, 2, 1)
    )


def preprocess_inputs(df: pd.DataFrame) -> np.ndarray:
    return pandas_list_to_array(
        df[['sequence', 'structure', 'predicted_loop_type']]
            .applymap(lambda seq: [TOKEN2INT[x] for x in seq])
    )


def build_adj_matrix(src_df: pd.DataFrame, normalize: bool = True) -> np.ndarray:
    n = len(src_df['structure'].iloc[0])
    mat = np.zeros((len(src_df), n, n))
    start_token_indices = []

    for r, structure in tqdm(enumerate(src_df['structure'])):
        for i, token in enumerate(structure):
            if token == "(":
                start_token_indices.append(i)
            elif token == ")":
                j = start_token_indices.pop()
                mat[r, i, j] = 1
                mat[r, j, i] = 1

    assert len(start_token_indices) == 0

    if normalize:
        mat = mat / (mat.sum(axis=2, keepdims=True) + 1e-8)

    return mat


def replace_data(train_df: pd.DataFrame, test_df: pd.DataFrame, replace_data_dir: str):
    print(f"using data from {replace_data_dir}")

    aux = pd.read_csv(replace_data_dir)
    del train_df['structure']
    del train_df['predicted_loop_type']
    del test_df['structure']
    del test_df['predicted_loop_type']
    train_df = pd.merge(train_df, aux, on='id', how='left')
    test_df = pd.merge(test_df, aux, on='id', how='left')
    assert len(train_df) == 2400
    assert len(test_df) == 3634
    assert train_df['structure'].isnull().sum() == 0
    assert train_df['predicted_loop_type'].isnull().sum() == 0
    assert test_df['structure'].isnull().sum() == 0
    assert test_df['predicted_loop_type'].isnull().sum() == 0
    return train_df, test_df


def load_bpps(df: pd.DataFrame, data_dir: str) -> np.ndarray:
    return np.array([np.load(f'{data_dir}bpps/{did}.npy') for did in df.id])


def make_bpps_features(bpps_list: List[np.ndarray]) -> np.ndarray:
    ar = []

    for b in bpps_list:
        ar.append(b.sum(axis=2))

        # max, 2ndmax, 3rdmax
        bpps_sorted = np.sort(b, axis=2)[:, :, ::-1]
        ar.append(bpps_sorted[:, :, 0])
        ar.append(bpps_sorted[:, :, 1])
        ar.append(bpps_sorted[:, :, 2])

        # number of nonzero
        bpps_nb_mean = 0.077522  # mean of bpps_nb across all training data
        bpps_nb_std = 0.08914  # std of bpps_nb across all training data
        nb = (b > 0).sum(axis=2)
        nb = (nb - bpps_nb_mean) / bpps_nb_std
        ar.append(nb)

    return np.transpose(np.array(ar), (1, 2, 0))


def make_dataset(device, x: np.ndarray, y: np.ndarray,
                 bpps_primary: np.ndarray,
                 bpps_secondary: np.ndarray,
                 adj_matrix: np.ndarray,
                 prediction_mask: np.ndarray,
                 signal_to_noise=None):
    x = copy.deepcopy(x)
    if y is not None:
        y = copy.deepcopy(y)
    bpps_primary = copy.deepcopy(bpps_primary)
    bpps_secondary = copy.deepcopy(bpps_secondary)
    bpps = np.concatenate([
        bpps_primary[:, :, :, np.newaxis],
        bpps_secondary[:, :, :, np.newaxis]
    ], axis=-1)

    adj_matrix = copy.deepcopy(adj_matrix)
    prediction_mask = copy.deepcopy(prediction_mask)

    if y is not None:
        y = np.clip(y, -0.5, 10)
        mask = np.abs(y).max(axis=(1, 2)) < 10
    else:
        mask = [True] * len(x)

    tensors = [
        torch.LongTensor(x[mask]),
        torch.Tensor(make_bpps_features([bpps_primary[mask], bpps_secondary[mask]])),
        torch.Tensor(bpps[mask]),
        torch.Tensor(adj_matrix[mask]),
        torch.Tensor(prediction_mask[mask])
    ]

    if y is not None:
        tensors.append(torch.Tensor(y[mask]))
        
        sample_weight = np.clip(np.log(signal_to_noise[mask] + 1.1) / 2, 0, 100)
        tensors.append(torch.Tensor(sample_weight))

    return torch.utils.data.TensorDataset(*[t.to(device) for t in tensors])


def make_dataset_from_df(device, df: pd.DataFrame, bpps_dir: str, secondary_bpps_dir: str):
    assert df['seq_scored'].nunique() == 1

    inputs = preprocess_inputs(df)
    bpps = load_bpps(df, bpps_dir)
    adj = build_adj_matrix(df)
    secondary_bpps = load_bpps(df, secondary_bpps_dir)

    mask = np.zeros((len(df), len(df['sequence'].iloc[0]), len(PRED_COLS)))
    mask[:, :df['seq_scored'].iloc[0], :] = 1

    return make_dataset(device, inputs, None, bpps, secondary_bpps, adj, mask)


def dist(s1: str, s2: str) -> int:
    return sum([c1 != c2 for c1, c2 in zip(s1, s2)])


def get_distance_matrix(s: pd.Series) -> np.ndarray:
    mat = np.zeros((len(s), len(s)))

    for i in tqdm(range(len(s))):
        for j in range(i + 1, len(s)):
            mat[i, j] = mat[j, i] = dist(s[i], s[j])
    return mat


def batch_predict(model: nn.Module, loader: DataLoader) -> np.ndarray:
    y_preds = np.zeros((len(loader.dataset), loader.dataset[0][0].shape[0], len(PRED_COLS)))

    for i, (x_batch, x_se, x_bpps, x_adj, y_mask) in enumerate(loader):
        y_pred = model(x_batch, x_se, x_bpps, x_adj).detach() * y_mask
        y_preds[i * loader.batch_size:(i + 1) * loader.batch_size, :, :] = y_pred.cpu().numpy()
        
    return y_preds


def calc_loss(y_true: np.ndarray, y_pred: np.ndarray):
    err_w_valid = [1 if s in PRED_COLS_SCORED else 0 for s in PRED_COLS]
    ############################################################################# pearson correlation
    mx = tf.math.reduce_mean(input_tensor=torch.Tensor(y_pred))
    my = tf.math.reduce_mean(input_tensor=torch.Tensor(y_true))
    xm, ym = torch.Tensor(y_pred)-mx, torch.Tensor(y_true)-my
    r_num = tf.math.reduce_mean(input_tensor=tf.multiply(xm,ym))        
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    raw1 = r_num / r_den
    ############################################################################# pearson correlation

    
    ############################################################################# MCRMSELoss
    raw2 = MCRMSELoss()(torch.Tensor(y_pred), torch.Tensor(y_true), err_w_valid).item()
    ############################################################################# MCRMSELoss
    
    
    return raw1 * len(PRED_COLS) / len(PRED_COLS_SCORED),raw2 * len(PRED_COLS) / len(PRED_COLS_SCORED)


# In[6]:


def train_model(model, train_loader, valid_loader, y_valid,
                train_epochs, train_loss, verbose=True,
                model_path='model'):
    params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f'number of params: {params}')

    err_w_train_1 = [1 if s in PRED_COLS_SCORED else 1.0 for s in PRED_COLS]
    err_w_train_2 = [1 if s in PRED_COLS_SCORED else 0.01 for s in PRED_COLS]

    criterion_train = train_loss
    criterion_train2 = train_loss
    criterion_train3 = train_loss

    optimizer = torch.optim.Adam(model.parameters())

    losses = []
    val_losses = []
    MAEerrs = []
    MSEerrs = []
    val_Pearson = []
    tr_Pearson = []
    y_preds_best = None

    for epoch in range(train_epochs):
        start_time = time.time()

        model.train()
        avg_loss = 0.
        avg_MSEloss = 0.
        avg_MAEloss = 0.
        avg_pearson = 0.


        for x_batch, x_se, x_bpps, x_adj, y_mask, y_batch, sample_weight in tqdm(train_loader, disable=True):
            y_pred = model(x_batch, x_se, x_bpps, x_adj) * y_mask
            
            # use 5 columns for the first 30 epoch
            w = err_w_train_1 if epoch < 30 else err_w_train_2
            
            loss1 = criterion_train(y_pred, y_batch, w, sample_weight)
            loss2 = criterion_train2(y_pred, y_batch, w, sample_weight)
            loss3 = criterion_train3(y_pred, y_batch, w, sample_weight)
################################################################################################### Pearson Corr
            x = y_pred.cpu().detach().numpy()
            y = y_batch.cpu().detach().numpy() 
            mx = tf.math.reduce_mean(input_tensor=x)
            my = tf.math.reduce_mean(input_tensor=y)
            xm, ym = x-mx, y-my
            r_num = tf.math.reduce_mean(input_tensor=tf.multiply(xm,ym))        
            r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
            trn_pearson = r_num / r_den
################################################################################################### Pearson Corr
            optimizer.zero_grad()
            loss = loss1 + loss2 + loss3+ trn_pearson.cpu().numpy()
#             print('loss2 , loss3 ',loss2, loss3)
            loss.backward()
#             trn_pearson.backward()
#             loss1.backward()
#             loss2.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            avg_loss += loss1.item() / len(train_loader)
            avg_MSEloss += loss2.item() / len(train_loader)
            avg_MAEloss += loss3.item() / len(train_loader)
            avg_pearson += trn_pearson.cpu().numpy().item() / len(train_loader)

#             avg_loss1 += loss1.item() / len(train_loader)
#             avg_loss2 += loss2.item() / len(train_loader)
        model.eval()
        
        y_preds = batch_predict(model, valid_loader)
        Pearson,mcloss = calc_loss(y_valid, y_preds)
#         print('calc_loss(y_valid, y_preds)',mcloss)

        val_losses.append(mcloss) # Validation MCRMSE
        val_Pearson.append(Pearson)# Validation Pearson
        s = f"{epoch:03d}: trn:{avg_loss:.4f}, clean={mcloss:.4f}, {time.time() - start_time:.2f}s"
#         print('pearson_Corr:',Pearson)


        losses.append(avg_loss) # Train MCRMSE
#         print('avg_loss',avg_loss)
        tr_Pearson.append(trn_pearson.cpu().numpy().item())# Train Pearson
        MAEerrs.append(loss3)# Train MAE
        MSEerrs.append(loss2)# Train MSE
#         print('avg_MSEloss',avg_MSEloss)
#         print('avg_MAEloss',avg_MAEloss)

        if np.min(val_losses) == mcloss:
#             print('np.min(val_losses)',np.min(val_losses),'val_losses',val_losses,'mcloss',mcloss)
            y_preds_best = y_preds
            torch.save(model.state_dict(), model_path)

        if (isinstance(verbose, bool) and verbose) or (verbose > 0 and (epoch % verbose == 0)):
            print(s)

    print(f'min val_loss: {np.min(val_losses):.4f} at {np.argmin(val_losses) + 1} epoch')

    # recover best weight
    model.load_state_dict(torch.load(model_path))

    if not verbose:
        return np.min(val_losses)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig2, ax2 = plt.subplots(1, 2, figsize = (20, 10))
#     print('np.arange(1, len(losses) + 1), losses',np.arange(1, len(losses) + 1) , losses)
#     print('np.arange(1, len(val_losses) + 1), val_losses',np.arange(1, len(val_losses) + 1), val_losses )
#     print('np.arange(1, len(MAEerrs) + 1), MAEerrs )',np.arange(1, len(MAEerrs) + 1), MAEerrs )
#     print('np.arange(1, len(MSEerrs) + 1), MSEerrs',np.arange(1, len(MSEerrs) + 1), MSEerrs)
    ax[0].plot(np.arange(1, len(losses) + 1), losses , color='C0')
    ax[0].plot(np.arange(1, len(val_losses) + 1), val_losses, color='C1')
    ax[0].plot(np.arange(1, len(MAEerrs) + 1), MAEerrs , color='C2')
    ax[0].plot(np.arange(1, len(MSEerrs) + 1), MSEerrs, color='C3', linestyle="dashed")
    
#     print('y_valid',type(y_valid),y_valid.shape,y_valid)
#     print('y_preds_best',type(y_preds_best),y_preds_best.shape,y_preds_best)

    for i, p in enumerate(PRED_COLS):
        ax[1].scatter(y_valid[:, :, i].flatten(), y_preds_best[:, :, i].flatten(), alpha=0.5)

    ax[0].legend(['train', 'validation','MAE','MSE'])
#     ax[1].legend(['valid(clean)'])
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('loss')
#     ax[1].set_xlabel('epoch')
#     ax[1].set_ylabel('loss')
    
    
    ax[1].legend(PRED_COLS)
    ax[1].set_xlabel('y_true')
    ax[1].set_ylabel('y_predicted')
    ax[1].set_xlabel('y_true(clean)')
    ax[1].set_ylabel('y_predicted(clean)')
    
    
    ax2[0].plot(np.arange(1, len(tr_Pearson) + 1),tr_Pearson, color='C0')
    ax2[0].plot(np.arange(1, len(val_Pearson) + 1),val_Pearson, color='C1')
    ax2[0].set_ylabel('Pearson correlation')
    ax2[0].set_xlabel('Epoch')
    ax2[0].legend(['train', 'validation'], loc = 'upper right')

    
    
    plt.show()

    return np.min(val_losses)


# ## Load data

# In[7]:


with timer("load data"):
    train_df = pd.read_json(DATA_DIR + 'train.json', lines=True)
    test_df = pd.read_json(DATA_DIR + 'test.json', lines=True)
    sample_df = pd.read_csv(DATA_DIR + 'sample_submission.csv')

    train_df, test_df = replace_data(train_df, test_df, REPLACE_DATA_PATH)

with timer("clustering"):
    # use clustering based on edit distance
    seq_dist = get_distance_matrix(train_df['sequence'])
    clf = AgglomerativeClustering(n_clusters=None, 
                                  distance_threshold=20, 
                                  affinity='precomputed',
                                  linkage='average')
    group_index = clf.fit_predict(seq_dist)


# In[8]:


with timer("preprocess"):
    public_df = test_df.query("seq_length != 130")
    private_df = test_df.query("seq_length == 130")
    
    if Structure_Interpretability_Test == True:
        s_107 = '.' * 107
        l_107 = 'E'* 107
        s_130 = '.' * 130
        l_130 = 'E'* 130
        train_df.loc[:, 'structure'] = s_107
        train_df.loc[:, 'predicted_loop_type'] = l_107
        public_df.loc[:, 'structure'] = s_107
        private_df.loc[:, 'structure'] = s_130
        public_df.loc[:, 'predicted_loop_type'] = l_107
        private_df.loc[:, 'predicted_loop_type'] = l_130
        
    
    print(private_df['structure'])
    print(public_df['structure'])
    print(private_df['predicted_loop_type'])
    print(public_df['predicted_loop_type'])


    x = preprocess_inputs(train_df)
    y = pandas_list_to_array(train_df[PRED_COLS])

    label_mask = np.ones_like(y)
    pad = np.zeros((y.shape[0], x.shape[1] - y.shape[1], y.shape[2]))
    y = np.concatenate((y, pad), axis=1)
    label_mask = np.concatenate((label_mask, pad), axis=1)

    assert x.shape[1] == y.shape[1]

    train_adj = build_adj_matrix(train_df)
    primary_bpps = load_bpps(train_df, PRIMARY_BPPS_DIR)    
#     secondary_bpps = load_bpps(train_df, SECONDARY_BPPS_DIR)
    secondary_bpps = load_bpps(train_df, PRIMARY_BPPS_DIR)

    public_data = make_dataset_from_df(device, public_df, PRIMARY_BPPS_DIR, SECONDARY_BPPS_DIR)
    private_data = make_dataset_from_df(device, private_df, PRIMARY_BPPS_DIR, SECONDARY_BPPS_DIR)


# ## Training

# In[9]:


start_time = time.time()

kf = StratifiedGroupKFold(NFOLDS, random_state=42, shuffle=True)

pred_oof = np.zeros_like(y)
pred_public = np.zeros((len(public_data), len(public_df['sequence'].iloc[0]), len(PRED_COLS)))
pred_private = np.zeros((len(private_data), len(private_df['sequence'].iloc[0]), len(PRED_COLS)))

public_loader = DataLoader(public_data, batch_size=32, shuffle=False)
private_loader = DataLoader(private_data, batch_size=32, shuffle=False)

clean_idx = [i for i in range(len(train_df)) if train_df['SN_filter'].iloc[i]]
sn_mask = train_df['SN_filter'] == 1

criterion_train = MCRMSELoss()
criterion_train2 = nn.MSELoss() 
criterion_train3 = nn.L1Loss() 

model_path = "model_fold{}"

losses = []

for i, (train_index, valid_index) in enumerate(kf.split(x, train_df['SN_filter'], groups=group_index)):
    print(f'fold {i}')
    model = ConvModel().to(device)
    s = time.time()

    train_data = make_dataset(device, x[train_index], y[train_index],
                              primary_bpps[train_index], secondary_bpps[train_index],
                              train_adj[train_index],
                              label_mask[train_index],
                              signal_to_noise=train_df['signal_to_noise'][train_index].values)

    valid_index_c = [v for v in valid_index if v in clean_idx]
    valid_data_clean = make_dataset(device, x[valid_index_c], None,
                                    primary_bpps[valid_index_c],
                                    secondary_bpps[valid_index_c],
                                    train_adj[valid_index_c],
                                    label_mask[valid_index_c])
    valid_data_noisy = make_dataset(device, x[valid_index], None,
                                  primary_bpps[valid_index],
                                  secondary_bpps[valid_index],
                                  train_adj[valid_index],
                                  label_mask[valid_index])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader_clean = DataLoader(valid_data_clean, batch_size=32, shuffle=False)
    valid_loader_noisy = DataLoader(valid_data_noisy, batch_size=32, shuffle=False)
    print(criterion_train)

    loss = train_model(model, 
                       train_loader, 
                       valid_loader_clean, 
                       y[valid_index_c], 
                       TRAIN_EPOCHS, 
                       criterion_train,
                       verbose=5, 
                       model_path=model_path.format(i))

#     losses.append(loss) # Validation losses

    # predict
    pred_oof[valid_index] = batch_predict(model, valid_loader_noisy)
    pred_public += batch_predict(model, public_loader) / NFOLDS
    pred_private += batch_predict(model, private_loader) / NFOLDS

    print(f'elapsed: {time.time() - s:.1f}sec')

    
    
# oof_score = calc_loss(y, pred_oof)
# print(f'oof(all): {oof_score: .4f}')

# oof_score = calc_loss(y[sn_mask], pred_oof[sn_mask])
# print(f'oof(clean): {oof_score: .4f}')

oof_score1,oof_score2 = calc_loss(y, pred_oof)
print(f'oof(all): {oof_score1: .4f}')
print(f'oof(all): {oof_score2: .4f}')

oof_score1,oof_score2 = calc_loss(y[sn_mask], pred_oof[sn_mask])
print(f'oof(clean): {oof_score1: .4f}')
print(f'oof(clean): {oof_score2: .4f}')

# make submission and oof
preds_ls = []

for df, preds in [(public_df, pred_public), (private_df, pred_private)]:
    for i, uid in enumerate(df.id):
        single_pred = preds[i]
        single_df = pd.DataFrame(single_pred, columns=PRED_COLS)
        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]
        preds_ls.append(single_df)

preds_df = pd.concat(preds_ls)
preds_df.head()

submission = sample_df[['id_seqpos']].merge(preds_df, on=['id_seqpos'])
submission.to_csv('submission.csv', index=False)

np.save('oof', pred_oof)
np.save('public', pred_public)
np.save('private', pred_private)

print(losses)

end_time = time.time()

print(f"Total training time: {end_time - start_time:.2f} seconds")



# In[10]:


import seaborn as sns
# Load a sample input
seq_len = 107
model = ConvModel().to(device)
state_dict = torch.load('model_fold1', map_location=device)
model.load_state_dict(state_dict)

input_sample = public_data[16]
x_batch, x_se, x_bpps, x_adj, y_mask = input_sample

# If your tensors are already on the correct device, there is no need to move them
# ...
# ...

# ...

x_batch = x_batch.float().unsqueeze(0)
x_batch.requires_grad = True
x_batch.retain_grad()
x_se = x_se.float().unsqueeze(0)
x_se.requires_grad = True
x_se.retain_grad()
x_bpps = x_bpps.float().unsqueeze(0)
x_bpps.requires_grad = True
x_bpps.retain_grad()
x_adj = x_adj.float().unsqueeze(0)
x_adj.requires_grad = True
x_adj.retain_grad()

# Pack them into a tuple
input_sample = (x_batch, x_se, x_bpps, x_adj)

# Continue with the rest of the code...

# Continue with the rest of the code...

# Continue with the rest of the code...


# Continue with the rest of the code...

# Your function
def compute_saliency_map(input_sample, model):
    prediction = model(*input_sample)
#     print(prediction.shape)
    loss = torch.mean(prediction[:, 0])
    model.zero_grad()
    loss.backward()
    
    grads = []
    for input_tensor in input_sample:
        if input_tensor.grad is not None:
            grad = input_tensor.grad
            grad = grad / (torch.sqrt(torch.mean(torch.square(grad))) + 1e-5)
            grads.append(grad)
    return grads

# Compute the saliency map for the input sample
saliency_maps = compute_saliency_map(input_sample, model)


inputshape = (107,5)

Vec = saliency_maps[0][:5].mean(dim=0)
print(Vec[:,:5].shape)
# If saliency_maps is not empty
if saliency_maps:
    # Plot the saliency map as a heatmap for x_batch only (example)
    sns.heatmap(Vec[:,:5].detach().cpu().numpy(), cmap='Reds',annot=True, vmin=0, vmax=1)
    plt.savefig('CNNsaliency_map.jpg', format='jpg')
    plt.show()
else:
    print("All gradients were None.")







# In[11]:


a = len(saliency_maps)
for i in range (a):
    print(saliency_maps[i].shape)


# In[12]:


# Normalize the gradients
saliency_maps_np = saliency_maps
saliency_maps_np = saliency_maps_np / np.max(np.abs(saliency_maps_np))

# Clip values to a certain percentile for better color balance
min_val = np.percentile(saliency_maps_np, 1)
max_val = np.percentile(saliency_maps_np, 99)
saliency_maps_np = np.clip(saliency_maps_np, min_val, max_val)

# Now plot using a diverging colormap
sns.heatmap(saliency_maps_np, cmap='RdBu')

plt.show()


# 

# In[ ]:


# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from keras.models import load_model
# import tensorflow as tf

# # Load the model and its state dict
# model = ConvModel().to(device)
# state_dict = torch.load('model_fold1', map_location=device)
# model.load_state_dict(state_dict)


# # Generate an input sample

# public_data = make_dataset_from_df(device, public_df, PRIMARY_BPPS_DIR, SECONDARY_BPPS_DIR)
# input_sample = public_data
# x_batch, x_se, x_bpps, x_adj, y_mask = input_sample[0]

# # input_sample = trn_[16].reshape(1, *input_shape)#np.random.rand(1, *input_shape)
# # input_sample = tf.convert_to_tensor((x_batch.unsqueeze(0), x_se.unsqueeze(0), x_bpps.unsqueeze(0), x_adj.unsqueeze(0)))

# # Define a function to compute the saliency map
# def compute_saliency_map(input_sample, model):
#     with tf.GradientTape() as tape:
#         # starts a TensorFlow GradientTape context. The GradientTape allows us to record the operations performed on the tensors during the forward pass, and then compute the gradients of the output with respect to the inputs during the backward pass.
#         tape.watch(input_sample)
#         # tells the GradientTape to watch the input_sample tensor, so that the gradients of the loss with respect to the input can be computed later.
#         prediction =  batch_predict(model, valid_loader_clean)
#         #uses the model to make a prediction on the input_sample. The prediction is a tensor of shape (1, number of classes).
#         loss = tf.reduce_mean(prediction[:, 0])
#         #computes the loss as the mean of the first column of the prediction tensor. This assumes that the first column of the prediction corresponds to the prediction of the first class.
#     grads = tape.gradient(loss, input_sample)
#     #This line computes the gradients of the loss with respect to the input_sample using the GradientTape. The gradients are a tensor of the same shape as the input sample.
#     grads = grads / (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
#     #This line normalizes the gradients by dividing them by their standard deviation plus a small constant (1e-5). This is to ensure that the values of the gradients are between 0 and 1.
#     return grads




# # Compute the saliency map for the input sample
# saliency_map = compute_saliency_map(input_sample, model)
# #The function computes the saliency map, which is a visual representation of the influence of each input feature on the prediction.
# # Plot the saliency map as a heatmap
# sns.heatmap(saliency_map.numpy().reshape(*input_shape)[:20], cmap='Reds', annot=True, vmin=0, vmax=1)
# sns.heatmap(saliency_map.numpy().reshape(*input_shape), cmap='Reds', annot=True, vmin=0, vmax=1)
# plt.savefig('saliency_map_GRU.jpg', format='jpg')
# plt.show()
# # The saliency map is a tensor of the same shape as the input sample, and its values represent the influence of each input feature on the prediction.
# # A high value indicates that a change in the corresponding input feature will have a significant impact on the prediction, while a low value indicates 
# # that the feature has a small influence on the prediction


# In[ ]:


import matplotlib.pyplot as plt

def plot_activations(module, inputt, output): 

#     print(len(inputt[0].shape)
#     print(output.shape)

    inputt= inputt[0].detach().cpu().numpy()
    output = output[:68].detach().cpu().numpy()
#     print('module, input, output:',inputt[0].shape,len(inputt), output[:68].shape)
    inputt = inputt[:,:,:3]
    print('input, output:',inputt.shape, output.shape)
#     print('input, output:',inputt[106].shape, output[67].shape)

#     if isinstance(module, nn.Linear):
        # Input layer activations
    plt.figure(figsize=(10, 5))
    plt.title('CNN input Activations')
    plt.xlabel('RNA Sequence position')
    plt.ylabel('Activation Value')
    for i in range(inputt.shape[2]):
        plt.plot(inputt[0, :, i], label='Input Activation {}'.format(i+1))
    plt.legend()
    plt.savefig('InputActivation_CNN.jpg', format='jpg')
    plt.show()

#     if isinstance(module, nn.Linear):

    # Output layer activations
    plt.figure(figsize=(10, 5))
    plt.title('CNN output Activations')

    for i in range(output.shape[2]):
        plt.plot(output[0, :, i][:68], label=' Output Activation {}'.format(i+1))
    plt.legend()
    plt.xlabel('RNA Sequence position')
    plt.ylabel('Activation Value')
    plt.savefig('OutputActivation_CNN.jpg', format='jpg')
    plt.show()

# Load the model and register the hook to its input and output layers
model = ConvModel().to(device)
model.eval()
state_dict = torch.load('model_fold3', map_location=device)
model.load_state_dict(state_dict)




# CAVEAT:: Should be run twice: each time comment input_layer_hook or output_layer_hook, since we need(64, 107, 3) and (64, 107, 5)
# input_layer_hook = model.embedding.register_forward_hook(plot_activations)# inputt[0].shape :torch.Size([64, 107, 3]) ,inputt[1] :torch.Size([64, 107, 10])
                                                                         #output.shape : torch.Size([64, 107, 160])
output_layer_hook = model.linear.register_forward_hook(plot_activations)#inputt[0].shape :torch.Size([64, 107, 256]),output.shape:torch.Size([64, 107, 5])

# Run the model and the hooks will plot the activations
loader  = public_loader
y_preds = np.zeros((len(loader.dataset), loader.dataset[0][0].shape[0], len(PRED_COLS)))
for i, (src, features, bpps, adj, y_mask) in enumerate(loader):
    if i == 0:# just for one sample
        y_pred = model(src, features, bpps, adj).detach() * y_mask
        y_preds[i * loader.batch_size:(i + 1) * loader.batch_size, :, :] = y_pred.cpu().numpy()
# Remove the hooks
input_layer_hook.remove()
output_layer_hook.remove()


# In[ ]:


# Next, we plot the cosine similarity as a heatmap using imshow function from matplotlib. 
# We set the axis labels to indicate the index of each array, and we add a colorbar to indicate 
# the similarity values. We rotate the tick labels for the x-axis to improve readability. 
# Finally, we loop over the similarity values and add text annotations to the heatmap.
# The resulting plot shows the cosine similarity between each pair of arrays as a heatmap.
# The brighter the color, the higher the cosine similarity between the corresponding arrays.

data = y_preds
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
plt.savefig('cosinesimilarity_CNN.jpg', format='jpg')

# Show the plot
plt.show()


# In[ ]:


get_ipython().system('conda install -y -c bioconda viennarna')
get_ipython().system('pip install RNA')
get_ipython().system('pip install python-Levenshtein')


# In[ ]:


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
plt.savefig('correlation_cosine_str_loop_seq_CNN.jpg', format='jpg')
plt.show()


# In[ ]:





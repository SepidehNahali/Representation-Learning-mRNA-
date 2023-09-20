#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# source: https://github.com/xypan1232/iDeepV/blob/master/RNA2Vec.py

# In[ ]:


'''
This script performs learning the distributed representation for 6-mers using the continuous skip-gram model with 5 sample negative sampling
'''
import sys
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
import pickle
import pdb

min_count = 5
dims = [50,]
windows = [5,]
allWeights = []

def get_6_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**6
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n//base
        ch1=chars[n%base]
        n=n//base
        ch2=chars[n%base]
        n=n//base
        ch3=chars[n%base]
        n=n//base
        ch4=chars[n%base]
        n=n//base
        ch5=chars[n%base]
        nucle_com.append(ch0 + ch1 + ch2 + ch3 + ch4 + ch5)
    return  nucle_com   

def get_7_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**7
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n//base
        ch1=chars[n%base]
        n=n//base
        ch2=chars[n%base]
        n=n//base
        ch3=chars[n%base]
        n=n//base
        ch4=chars[n%base]
        n=n//base
        ch5=chars[n%base]
        n=n//base
        ch6=chars[n%base]
        nucle_com.append(ch0 + ch1 + ch2 + ch3 + ch4 + ch5 + ch6)
    return  nucle_com   

def get_4_nucleotide_composition(tris, seq):
    seq_len = len(seq)
    tri_feature = []
    k = len(tris[0])
    #tmp_fea = [0] * len(tris)
    for x in range(len(seq) + 1- k):
        kmer = seq[x:x+k]
        if kmer in tris:
            ind = tris.index(kmer)
            tri_feature.append(str(ind))
    #tri_feature = [float(val)/seq_len for val in tmp_fea]
        #pdb.set_trace()        
    return tri_feature

def seq2words(sequ):
    tris = get_6_trids()
    seq = sequ
    seq = seq.replace('T', 'U')
#     pdb.set_trace()
    trvec = get_4_nucleotide_composition(tris, seq)
    return trvec


def train_rnas(seq_file ,dimenion):
    min_count = 1
    dim = dimenion
    window = 5

    print('dim: ' + str(dim) + ', window: ' + str(window))
    train = pd.read_json(seq_file, lines=True)
    df_Seq = train['sequence']
    seq_dict = df_Seq.to_dict()
    #text = seq_dict.values()
    tris = get_6_trids()# all poibl compoitions
    sentences = []
    for seq in seq_dict.values():
        seq = seq.replace('T', 'U')
        trvec = get_4_nucleotide_composition(tris, seq)
        #for aa in range(len(text)):
        sentences.append(trvec)

#     print('sentences',sentences)

    model = None

    model = Word2Vec(sentences, min_count=min_count, vector_size =dim, window=window, sg=1, batch_words=100)
    

    vocab = list(model.wv.index_to_key)
    fw = open('rna_dict', 'w')
    for val in vocab:
        fw.write(val + '\n')
    fw.close()

    embeddingWeights = np.empty([len(vocab), dim])

    for i in range(len(vocab)):
        embeddingWeights[i,:] = model.wv[vocab[i]]  

    allWeights.append(embeddingWeights)


    return model
    
def testseq(model,sequence_):  
    ##############################################vector of Sequence
    word_vectors = model.wv
    word_vectors.save("word2vec.wordvectors")
    wv = KeyedVectors.load("word2vec.wordvectors", mmap='r')
    sentencevector=[]
    wordslist=seq2words(sequence_)
    print('wordslist:',wordslist,'length=',len(wordslist))
    print('#')

    for i in wordslist:
            print(' wv[i]', wv[i],'word:',i)

            vector = wv[i]  # Get numpy vector of a word (Size= 25)
            sentencevector.append(vector)
    ##############################################vector of Sequence
    return sentencevector




# In[ ]:


# TEST_TRAIN_W2V='train'
# TEST_TRAIN_W2V='public_test'
TEST_TRAIN_W2V='private_test'

if TEST_TRAIN_W2V=='train':
    print('train vectors are training')
    dimenion=107
    input_file = '../input/stanford-covid-vaccine/train.json'
    model=train_rnas(input_file,dimenion)
    train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)
    df_Seq = train['sequence']
    
elif TEST_TRAIN_W2V=='public_test':
    print('public_test vectors are training')
    dimenion=107
    input_file = '../input/stanford-covid-vaccine/test.json'
    model=train_rnas(input_file,dimenion)
    test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)
    public_df = test.query("seq_length == 107").copy()
    df_Seq = public_df['sequence']

else:
    print('private_test')
    dimenion=130
    input_file = '../input/stanford-covid-vaccine/test.json'
    model=train_rnas(input_file,dimenion)
    test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)
    private_df = test.query("seq_length == 130").copy()
    df_Seq = private_df['sequence']  
    
allseqvecs={}
for i,s in df_Seq.items():
    vec=testseq(model,s)
    allseqvecs[i]=vec
    
#allseqvecs[sequence s]=vector of s


# In[ ]:


len(vec)


# In[ ]:


len(allseqvecs[1])


# In[ ]:


# allseqvecs


# In[ ]:


# allseqvecs[1]


# In[ ]:


import numpy as np
train_Seq_Vec=[]

for i,_ in df_Seq.items():
    Seq_each_Word_vec = np.array(allseqvecs[i])
    Seq_vec=np.average(Seq_each_Word_vec, axis=0)
    train_Seq_Vec.append(Seq_vec)


# In[ ]:


with open(TEST_TRAIN_W2V, 'wb') as f:
        pickle.dump(train_Seq_Vec, f)


# In[ ]:


w2vAVG = open(TEST_TRAIN_W2V,'rb')
new_dict = pickle.load(w2vAVG)



# In[ ]:





# In[ ]:


len(train_Seq_Vec[0])


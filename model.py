

import csv
import numpy as np
import sys
import pandas as pd
import itertools
import math
import time

from sklearn import svm, linear_model, neighbors
from sklearn import tree, ensemble
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

import networkx as nx
import random
import numbers

from sklearn.model_selection import StratifiedKFold
from src import ml
from sklearn.model_selection import train_test_split

import random
import numbers

from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from keras import optimizers


from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import TimeDistributed
from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from keras.utils import np_utils

from keras import backend as K
K.image_dim_ordering='tf'
import matplotlib.pyplot as plt
import itertools
from keras.regularizers import L1L2


from time import time
import numpy as np
import keras.backend as K
from keras.layers import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adagrad
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import accuracy_score
from sklearn import manifold
import keras.layers.normalization as bn

from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import os
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
import keras
import sys
from keras.layers import concatenate
from keras import layers

from keras.layers import Embedding

from keras.layers import Input, Dense, LSTM, MaxPooling1D, Conv1D, RepeatVector
from keras.models import Model
#from autoencoder import *
from keras.layers import Attention

import matplotlib.pyplot as plt
from keras.regularizers import L1L2
import tensorflow as tf

from sklearn.model_selection import KFold

train = np.load('train.npy')
Y=np.load('y.npy')
train_X,test_X,train_Y,test_Y = train_test_split(train, Y, test_size=0.1,shuffle=True)
train1 = train_X[:,:,0:200]
train2=train_X[:,:,200:400]
test1 = test_X[:,:,0:200]
test2=test_X[:,:,200:400]
train_X= train_X.reshape((train_X.shape[0], -1))
train_X.shape
test_y= test_Y.reshape((test_Y.shape[0], -1))
train1 = train1.reshape((train1.shape[0], -1))
train2 = train2.reshape((train2.shape[0], -1))
test1 = test1.reshape((test1.shape[0], -1))
test2 = test2.reshape((test1.shape[0], -1))

event_num=2
droprate=0.4

def AttentionKGDDI():
    train_input1 = Input(shape=(200), name='Inputlayer1')
    train_input2 = Input(shape=(200), name='Inputlayer2')



    # Attention Neural Network
    train_input = keras.layers.Concatenate()([train_input1, train_input2])


    attention_probs = Dense(200,activation='relu',use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros', name='attention1')(train_input)

    att = Dense(200,activation='softmax', kernel_initializer='random_uniform', name='attention')(attention_probs)

    vec = keras.layers.Multiply()([train_input1, train_input2])
    attention_mul = keras.layers.Multiply()([vec, att])

    # Deep neural network classifier
    train_in = Dense(256, activation='relu', name='FullyConnectLayer1')(attention_mul)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)

    train_in = Dense(128, activation='relu', name="FullyConnectLayer2")(train_in)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)

    train_in = Dense(64, activation='relu', name="FullyConnectLayer3")(train_in)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)


    train_in = Dense(1, name="SoftmaxLayer")(train_in)

    out = Activation('sigmoid', name="OutputLayer")(train_in)

    model = Model(inputs=[train_input1,train_input2],outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

model=AttentionKGDDI()
model.summary()

# Compile the model
model.compile(loss='BinaryCrossentropy',
                optimizer=optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0),
                metrics=['accuracy'])
history=model.fit([train1,train2],train_y,validation_split=0.1,batch_size=32, epochs=10, shuffle=True)

# Visualize history
# Plot history: Loss
plt.plot(history.history['val_loss'])
plt.title('Validation loss history')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.show()

# Plot history: Accuracy
plt.plot(history.history['accuracy'])
plt.title('Validation accuracy history')
plt.ylabel('Accuracy value (%)')
plt.xlabel('No. epoch')
plt.show()

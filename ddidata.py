

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

from sklearn.model_selection import StratifiedKFold

import networkx as nx
import random
import numbers

import findspark
findspark.init()

from pyspark import SparkConf, SparkContext

# Imports
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

# Create a SparkConf object
conf = SparkConf().setAppName("MyApp")\
            .setMaster("local[10]")\
            .set("spark.executor.memory", "70g")\
            .set('spark.driver.memory', '90g')\
            .set("spark.memory.offHeap.enabled",True)\
            .set("spark.memory.offHeap.size","50g")\
            .set("spark.network.timeout","100000s")

# Create a SparkSession object
spark = SparkSession.builder.config(conf=conf).getOrCreate()

ddi_df = pd.read_csv("data/input/ddi_v4.txt", sep='\t')
ddi_df.head()

featureFilename = "vectors/DB/RDF2Vec_sg_200_5_5_15_2_500_d5_uniform.txt"
embedding_df = pd.read_csv(featureFilename, delimiter='\t')
embedding_df.Entity =embedding_df.Entity.str[-8:-1]
embedding_df.rename(columns={'Entity':'Drug'},inplace=True)

pairs, classes = ml.generatePairs(ddi_df, embedding_df)
classes = np.array(classes)
pairs = np.array(pairs)
pair_df = pd.DataFrame(list(zip(pairs[:,0],pairs[:,1],classes)), columns=['Drug1','Drug2','Class'])

Y=pair_df['Class'].values

pair_df.to_csv('pair.csv', index=False)
train_df= spark.read.csv("pair.csv", header=True, inferSchema=True)
embedding_df= spark.createDataFrame(embedding_df)

# Create aliases for the DataFrames
train_df_alias = train_df.alias('train')
embedding_df_alias1 = embedding_df.alias('embed1')
embedding_df_alias2 = embedding_df.alias('embed2')
#merge dataframes
merged_df = train_df_alias.\
    join(embedding_df_alias1, train_df_alias['Drug1'] == embedding_df_alias1['Drug'], how='inner').\
    join(embedding_df_alias2, train_df_alias['Drug2'] == embedding_df_alias2['Drug'], how='inner').\
    drop(embedding_df_alias1['Drug']).\
    drop(embedding_df_alias2['Drug'])

pandas_df = merged_df.toPandas()
df = pandas_df.to_numpy()
train_X=df.reshape(-1,1,400)
np.save('train.npy', train_X)
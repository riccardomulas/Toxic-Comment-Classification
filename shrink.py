import pandas as pd
import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
import pyspark.sql.functions as F
import pyspark.sql.types as T 
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
import os
import nltk

conf= SparkConf().set("spark.eventLog.enabled", "false")
conf.setAppName("Proof")
conf.set('spark.driver.memory','45g')
conf.set('spark.executor.memory','12g')
conf.set('spark.cores.max',156)
conf.set('spark.driver.maxResultSize', '10G')

sc = SparkContext(conf= conf) # SparkContext
sqlc = SQLContext(sc) # SqlContext

train = pd.read_csv('train.csv', encoding='utf-8')
train.fillna("", inplace=True)
train.dropna()
trainDF = pd.DataFrame(train)
trainDF['words'] = trainDF.apply(lambda row: nltk.word_tokenize(row['comment_text']), axis=1)
listOfVocabulary = trainDF['words'].values.tolist()
flat_vocabulary = [word.lower() for sublist in listOfVocabulary for word in sublist]

test = pd.read_csv("test.csv")
test.fillna("", inplace=True)
test.dropna()
testDF = pd.DataFrame(test)
testDF['words'] = testDF.apply(lambda row: nltk.word_tokenize(row['comment_text']), axis=1)
listOfTestVocabulary = testDF['words'].values.tolist()
flat_test_vocabulary = [word.lower() for sublist in listOfTestVocabulary for word in sublist]

mergedVocabulary = flat_vocabulary + flat_test_vocabulary
vocabulary = set(mergedVocabulary)
sparkDF = sqlc.createDataFrame(trainDF)
sparkTestDF = sqlc.createDataFrame(testDF)

def load(path, vocabulary=None):
    print('Loading word embeddings at', path)
    embeddings = {}
    counter = 0
    with open(path) as f:
        for line in f:
            counter += 1
            try:
                values = line.split()
                word = values[0]
                if (vocabulary is None) or (word in vocabulary):
                    vector = np.asarray(values[1:], dtype='float32')
                    embeddings[word] = vector
            except IndexError:
                print('Index error at line ', counter)
            except:
                print('Unexpected error at line:', counter)
    return embeddings

print("Loading embeddings")
def shrink_to_vocabulary(embeddings_input_path, vocabulary):
    embeddings = load(embeddings_input_path, vocabulary)
    dirname, filename = os.path.split(embeddings_input_path)
    outputdir = os.path.join(dirname, 'shrunk')
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    embeddings_output_path = os.path.join(outputdir, filename)
    save(embeddings, embeddings_output_path)


def save(embeddings, path):
    with open(path, 'a') as f:
        for word in embeddings.keys():
            vector = embeddings[word]
            values = ' '.join(map(str, vector))
            f.write(word + ' ' + values + '\n')

shrink_to_vocabulary(embeddings_input_path, vocabulary)
print("End loading")
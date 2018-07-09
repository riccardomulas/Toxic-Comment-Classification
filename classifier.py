import pandas as pd
import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
import pyspark.sql.functions as F
import pyspark.sql.types as T 
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors, VectorUDT
import os
import nltk
import shrink

conf= SparkConf().set("spark.eventLog.enabled", "false")
conf.setAppName("Proof")
conf.set('spark.driver.memory','45g')
conf.set('spark.executor.memory','12g')
conf.set('spark.cores.max',156)
conf.set('spark.driver.maxResultSize', '10G')

sc = SparkContext(conf= conf) 
sqlc = SQLContext(sc) 

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

#shrink_to_vocabulary(embeddings_input_path, vocabulary)

sizeEmbeddings = 256
print("loading")
embedding_model = load('embeddings256.txt', None)
print("end")

embedding_broadcast = sc.broadcast(embedding_model)

def getEmbeddings(words):
    default = np.zeros(sizeEmbeddings, dtype='float32')
    lista = [embedding_broadcast.value.get(word, default) for word in words]
    sumEmbeddings = np.zeros(sizeEmbeddings, dtype='float32')
    for vector in lista:
        if len(vector) != 256:
            continue
        sumEmbeddings = sumEmbeddings + vector
    length = len(sumEmbeddings)
    avgNP = sumEmbeddings / length
    avg = avgNP.tolist()
    return avg

udfEmbeddings = F.udf(lambda x: getEmbeddings(x), T.ArrayType(T.FloatType()))

newDF = sparkDF.withColumn("vectors", udfEmbeddings(sparkDF["words"]))
newSparkDF = sparkTestDF.withColumn("vectors", udfEmbeddings(sparkTestDF["words"]))

list_to_vector_udf = F.udf(lambda l: Vectors.dense(l), VectorUDT())
newDFwithVectors = newDF.select(
    newDF["id"], 
    newDF["comment_text"],
    newDF["toxic"],
    newDF["severe_toxic"],
    newDF["obscene"], 
    newDF["threat"],
    newDF["insult"],
    newDF["identity_hate"],
    newDF["words"],
    list_to_vector_udf(newDF["vectors"]).alias("vectors")
)

newSparkDFwithVectors = newSparkDF.select(
    newSparkDF["id"],
    newSparkDF["comment_text"],
    newSparkDF["words"],
    list_to_vector_udf(newSparkDF["vectors"]).alias("vectors")
)

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

REG = 0.3

for label in labels:
    print(label)
    lr = LogisticRegression(featuresCol="vectors", labelCol=label, regParam=REG)
    print("...fitting")
    lrModel = lr.fit(newDFwithVectors)
    trainingSummary = lrModel.summary
    print("...predicting")
    res = lrModel.transform(newSparkDFwithVectors)
    print("...appending result")
    accuracy = trainingSummary.accuracy
    print("Accuracy: %s\n"
      % (accuracy))
    print("areaUnderROC: " + str(trainingSummary.areaUnderROC))



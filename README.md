# Toxic Comment Classification

This project was developed in the course of Architettura degli Elaboratori 2, at University of Cagliari. The aim was to provide an alternative solution to the Kaggle Challenge at https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

### Prerequisites

The project runs on Apache Spark, so you need to install the version 2.3.1 and it is developed in Python. 

To run the project, you need to execute (on a Mac): 
```
brew install python 
easy_install pip
pip install pandas numpy scipy 
brew install scala
brew install apache-spark
```
To run the project, you need to check the paths of the files and then execute:
```
spark-submit classifier.py
```
To improve the accuracy of the model, word embeddings have been used. They can be downloaded at http://www.maurodragoni.com/research/opinionmining/dranziera/embeddings-evaluation.php
## Authors

* **Riccardo Mulas** 


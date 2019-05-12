import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from preprocessing import TextPreparation
from methods.my_logistic_regression import LogisticRegressionMethod
from methods.random_forest import RandomForestMethod
from methods.svm import SVMMethod
from methods.knn import KNNMethod
from methods.naive_bayes import NaiveBayesMethod
from methods.reccurent_neural import ReccurentNeuralMethod
from methods.sklearn_neural import SKlearnNeural
from methods.voting_classifier import VoitingClassifierMethod
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import datetime
import numpy as np
from my_new_approach import *

# Word cloud visualization
def word_cloud_visualization():
    X = df_train['text']
    wordcloud1 = WordCloud().generate(X[0])  # for EAP
    wordcloud2 = WordCloud().generate(X[1])  # for HPL
    wordcloud3 = WordCloud().generate(X[3])  # for MWS
    print(X[0])
    print(df_train['author'][0])
    plt.imshow(wordcloud1, interpolation='bilinear')
    plt.show()
    print(X[1])
    print(df_train['author'][1])
    plt.imshow(wordcloud2, interpolation='bilinear')
    plt.show()
    print(X[3])
    print(df_train['author'][3])
    plt.imshow(wordcloud3, interpolation='bilinear')
    plt.show()

# make_dataset()

def text_process(tex):
    # 1. Removal of Punctuation Marks
    lemmatiser = WordNetLemmatizer()
    nopunct = [char for char in tex if char not in string.punctuation]
    nopunct = ''.join(nopunct)
    # 2. Lemmatisation
    a = ''
    i = 0
    for i in range(len(nopunct.split())):
        b = lemmatiser.lemmatize(nopunct.split()[i], pos="v")
        a = a + b + ' '
    # 3. Removal of Stopwords
    return [word for word in a.split() if word.lower() not
            in stopwords.words('russian')]

df_train = pd.read_csv('my_train.csv')
df_test = pd.read_csv('my_test.csv')

X_tr = df_train['text']
y_train = df_train['author']
X_te = df_test['text']
y_test = df_test['author']
labelencoder = LabelEncoder()
y_train = labelencoder.fit_transform(y_train)
y_test = labelencoder.transform(y_test)

# 80-20 splitting the dataset (80%->Training and 20%->Validation)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
# defining the bag-of-words transformer on the text-processed corpus # i.e., text_process() declared in II is executed...
bow_transformer = CountVectorizer(analyzer=text_process, max_features=10000).fit(X_tr)
# transforming into Bag-of-Words and hence textual data to numeric..
X_train = bow_transformer.transform(X_tr).toarray() # ONLY TRAINING DATA
# transforming into Bag-of-Words and hence textual data to numeric..
X_test = bow_transformer.transform(X_te).toarray()  # TEST DATA


def logistic_regression(X_train, X_test, y_train, y_test):
    lr_solvers = ["newton-cg", "sag", "saga"]
    # solver = "newton-cg"
    for solver in lr_solvers:
        print("-----------------LOGISTIC REGRESSION: {}------------------------".format(solver))
        # lr = LogisticRegressionMethod(np.array(X_train).astype("float32"), np.array(X_test).astype("float32"),
        #                               np.array(y_train).astype("float32"), np.array(y_test).astype("float32"),
        #                               solver=solver)
        lr = LogisticRegressionMethod(X_train, X_test, y_train, y_test)
        lr.do_LR()

def random_forests(X_train, X_test, y_train, y_test):
    print("-----------------RANDOM FORESTS------------------------")
    rf = RandomForestMethod(X_train, X_test, y_train, y_test)
    rf.do_RF()

def svm(X_train, X_test, y_train, y_test):
    print("-----------------SVM METHOD------------------------")
    svm = SVMMethod(np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test))
    svm.do_SVM()

def knn(X_train, X_test, y_train, y_test):
    print("-----------------KNN------------------------")
    knn = KNNMethod(X_train, X_test, y_train, y_test)
    knn.do_knn()

def naive_bayes(X_train, X_test, y_train, y_test):
    print("-----------------NAIVE BAYES------------------------")
    bayes = NaiveBayesMethod(X_train, X_test, y_train, y_test)
    bayes.do_bayes()

def skleran_neural(X_train, X_test, y_train, y_test):

    print("-----------------SKLEARN NEURAL------------------------")
    bayes = SKlearnNeural(X_train, X_test, y_train, y_test)
    bayes.do_SVM()


def reccurent_neural(X_train, X_test, y_train, y_test):
    # print("-----------------RECURRENT NEURAL------------------------")
    recurent_neural = ReccurentNeuralMethod(np.array(X_train).astype("float32"), np.array(X_test).astype("float32"),
                                            np.array(y_train).astype("float32"), np.array(y_test).astype("float32"))
    # recurent_neural = ReccurentNeuralMethod()
    recurent_neural.do_rc()

def voiting_classifier(X_train, X_test, y_train, y_test):
    # print("-----------------VOITING CLASSIFIER------------------------")
    voiting = VoitingClassifierMethod(np.array(X_train).astype("float32"), np.array(X_test).astype("float32"),
                                            np.array(y_train).astype("float32"), np.array(y_test).astype("float32"))
    # recurent_neural = ReccurentNeuralMethod()
    voiting.do_voiting()

# make_dataset()
# X_train, X_test, y_train, y_test = get_dataset()

logistic_regression(X_train, X_test, y_train, y_test)
random_forests(X_train, X_test, y_train, y_test)
svm(X_train, X_test, y_train, y_test)
knn(X_train, X_test, y_train, y_test)
naive_bayes(X_train, X_test, y_train, y_test)
# skleran_neural(X_train, X_test, y_train, y_test)
# reccurent_neural(X_train, X_test, y_train, y_test)
voiting_classifier(X_train, X_test, y_train, y_test)

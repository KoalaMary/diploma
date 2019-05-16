import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
import nltk.data
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
import io
import os
import csv


def get_all_authors():
    path = os.path.join(os.getcwd(), "data")
    return os.listdir(path)


def get_all_authors_files(author):
    path = os.path.join(os.getcwd(), "data", author)
    return os.listdir(path)


def tokenize_text(text, blocks_size=100):
    # tokenizer = nltk.data.load('tokenizers/punkt/russian.pickle')
    # text = text.replace("\n", "").replace("-", "")
    # return tokenizer.tokenize(text)
    text = text.split()
    text2 = [text[x:x + blocks_size] for x in range(0, len(text), blocks_size)]
    return text2


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


def word_cloud_visualization(df_train):
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


def prepare_text(author=None, data=None):
    sentences = tokenize_text(data)
    train_sentences = sentences[:int(len(sentences) * 0.75)]
    test_sentences = sentences[int(len(sentences) * 0.75):]
    train_res = [author] * len(train_sentences)
    test_res = [author] * len(test_sentences)

    return train_sentences, test_sentences, train_res, test_res


def make_data():
    authors = get_all_authors()
    data_path = os.path.join(os.getcwd(), "data")

    text_column = "text"
    author_column = "author"
    train_file = os.path.join(os.getcwd(), "my_train.csv")
    test_file = os.path.join(os.getcwd(), "my_test.csv")

    train_sentences = []
    test_sentences = []
    train_res = []
    test_res = []

    for author in authors[:2]:
        files = get_all_authors_files(author=author)
        for file in files[:1]:
            print("Started file: {}".format(file))
            with io.open(data_path + "/" + author + "/" + file, 'r', encoding='utf8') as infile:
                data = infile.read()
            train_sentences_au, test_sentence_au, train_res_au, test_res_au = prepare_text(author=author, data=data)
            train_sentences += train_sentences_au
            test_sentences += test_sentence_au
            train_res += train_res_au
            test_res += test_res_au
            print("Finished file: {}".format(file))

    train_content = {text_column: train_sentences, author_column: train_res}
    test_content = {text_column: test_sentences, author_column: test_res}
    train_df = pd.DataFrame(data=train_content)
    test_df = pd.DataFrame(data=test_content)
    train_df.to_csv(train_file, mode="a")
    test_df.to_csv(test_file, mode="a")


def make_dataset(max_features=1000):
    train_file = os.path.join(os.getcwd(), "train_dataset.csv")
    test_file = os.path.join(os.getcwd(), "test_dataset.csv")
    train_res_file = os.path.join(os.getcwd(), "train_res.csv")
    test_res_file = os.path.join(os.getcwd(), "test_res.csv")

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
    y_tr = df_train['author']
    X_te = df_test['text']
    y_te = df_test['author']
    labelencoder = LabelEncoder()
    y_train = labelencoder.fit_transform(y_tr)
    y_test = labelencoder.transform(y_te)

    bow_transformer = CountVectorizer(analyzer=text_process, max_features=max_features).fit(X_tr)
    a = bow_transformer.vocabulary
    X_train = bow_transformer.transform(X_tr).toarray()
    X_test = bow_transformer.transform(X_te).toarray()

    with open("train_dataset.csv", "a") as file:
        wr = csv.writer(file)
        wr.writerows(X_train)
    file.close()

    with open("test_dataset.csv", "a") as file:
        wr = csv.writer(file)
        wr.writerows(X_test)
    file.close()

    with open("train_res.csv", "a") as file:
        wr = csv.writer(file)
        wr.writerows(y_train)
    file.close()

    with open("test_res.csv", "a") as file:
        wr = csv.writer(file)
        wr.writerows(y_test)
    file.close()

    # train_df = pd.DataFrame(data=X_train)
    # test_df = pd.DataFrame(data=X_test)
    # train_res_df = pd.DataFrame(data=y_train)
    # test_res_df = pd.DataFrame(data=y_test)
    # train_df.to_csv(train_file, mode="a")
    # test_df.to_csv(test_file, mode="a")
    # train_res_df.to_csv(train_res_file, mode="a")
    # test_res_df.to_csv(test_res_file, mode="a")


def get_dataset():
    train_file = os.path.join(os.getcwd(), "train_dataset.csv")
    test_file = os.path.join(os.getcwd(), "test_dataset.csv")
    train_res_file = os.path.join(os.getcwd(), "train_res.csv")
    test_res_file = os.path.join(os.getcwd(), "test_res.csv")

    X_train = pd.read_csv(train_file).values.tolist()
    X_test = pd.read_csv(test_file).values.tolist()
    y_train = pd.read_csv(train_res_file)
    y_test = pd.read_csv(test_res_file).values.tolist()

    # X_train = df_train['text']
    # y_train = df_train_res['author']
    # X_test = df_test['text']
    # y_test = df_test_res['author']

    return X_train, X_test, y_train, y_test
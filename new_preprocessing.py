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
    # text_column = "text"
    # author_column = "author"
    # train_file = os.path.join(os.getcwd(), "my_train.csv")
    # test_file = os.path.join(os.getcwd(), "my_test.csv")

    sentences = tokenize_text(data)
    train_sentences = sentences[:int(len(sentences) * 0.75)]
    test_sentences = sentences[int(len(sentences) * 0.75):]
    train_res = [author] * len(train_sentences)
    test_res = [author] * len(test_sentences)

    return train_sentences, test_sentences, train_res, test_res

    # train_content = {text_column: train_sentences, author_column: [author] * len(train_sentences)}
    # test_content = {text_column: test_sentences, author_column: [author] * len(test_sentences)}
    # train_df = pd.DataFrame(data=train_content)
    # test_df = pd.DataFrame(data=test_content)
    # train_df.to_csv(train_file, mode="a")
    # test_df.to_csv(test_file, mode="a")


def make_dataset():
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

    for author in authors:
        files = get_all_authors_files(author=author)
        for file in files:
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


def get_dataset():
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
    bow_transformer = CountVectorizer(analyzer=text_process).fit(X_tr)
    # transforming into Bag-of-Words and hence textual data to numeric..
    X_train = bow_transformer.transform(X_tr)  # ONLY TRAINING DATA
    # transforming into Bag-of-Words and hence textual data to numeric..
    X_test = bow_transformer.transform(X_te)  # TEST DATA

    return X_train, X_test, y_train, y_test

import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
# from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import io
import os
import csv
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


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


# def word_cloud_visualization(df_train):
#     X = df_train['text']
#     wordcloud1 = WordCloud().generate(X[0])  # for EAP
#     wordcloud2 = WordCloud().generate(X[1])  # for HPL
#     wordcloud3 = WordCloud().generate(X[3])  # for MWS
#     print(X[0])
#     print(df_train['author'][0])
#     plt.imshow(wordcloud1, interpolation='bilinear')
#     plt.show()
#     print(X[1])
#     print(df_train['author'][1])
#     plt.imshow(wordcloud2, interpolation='bilinear')
#     plt.show()
#     print(X[3])
#     print(df_train['author'][3])
#     plt.imshow(wordcloud3, interpolation='bilinear')
#     plt.show()


def prepare_text(author=None, data=None):
    # sentences = tokenize_text(data)
    # train_sentences = sentences[:int(len(sentences) * 0.75)]
    # test_sentences = sentences[int(len(sentences) * 0.75):]
    # train_res = [author] * len(train_sentences)
    # test_res = [author] * len(test_sentences)
    #
    # return train_sentences, test_sentences, train_res, test_res

    text = data.replace("\n", " ")
    text = re.sub("[^а-яА-Я\-\s]", "", text)
    text = re.sub("[\-]", " ", text.lower())

    stop_words = set(stopwords.words("russian"))
    words = word_tokenize(text=text, language="russian", preserve_line=True)
    words_filtered = []

    for w in words:
        if w not in stop_words:
            words_filtered.append(w)
    text = ' '.join(words_filtered)

    m = Mystem()
    m.start()
    lemmas = m.lemmatize(str(text))
    text = ''.join(lemmas)
    #
    # ps = PorterStemmer()
    # # for word in text:
    # text2 = ps.stem(text)


    return ''.join(text)


def prepare_all_data():
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
        for file in files[:3]:
            print("Started train file: {}".format(file))
            with io.open(data_path + "/" + author + "/" + file, 'r', encoding='utf8') as infile:
                data = infile.read()
            # train_sentences_au, test_sentence_au, train_res_au, test_res_au = prepare_text(author=author, data=data)
            text = prepare_text(data=data)
            train_sentences.append(text)
            # test_sentences += test_sentence_au
            train_res.append(author)
            # test_res += test_res_au
            print("Finished train file: {}".format(file))
        for file in files[3:]:
            print("Started test file: {}".format(file))
            with io.open(data_path + "/" + author + "/" + file, 'r', encoding='utf8') as infile:
                data = infile.read()
            # train_sentences_au, test_sentence_au, train_res_au, test_res_au = prepare_text(author=author, data=data)
            data = prepare_text(data=data)
            test_sentences.append(data)
            # test_sentences += test_sentence_au
            test_res.append(author)
            # test_res += test_res_au
            print("Finished test file: {}".format(file))

    train_content = {text_column: train_sentences, author_column: train_res}
    test_content = {text_column: test_sentences, author_column: test_res}
    train_df = pd.DataFrame(data=train_content)
    test_df = pd.DataFrame(data=test_content)
    train_df.to_csv(train_file, mode="a")
    test_df.to_csv(test_file, mode="a")


def make_dataset(max_features=500, folder="text_process"):
    # nltk.download('wordnet')

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

    bow_transformer = CountVectorizer(analyzer="word", max_features=max_features).fit(X_tr)

    X_train = bow_transformer.transform(X_tr).toarray()
    X_test = bow_transformer.transform(X_te).toarray()

    path = os.path.join(os.getcwd(), "datasets", folder)

    with open(path + "/" + "train_dataset.csv", "a") as file:
        wr = csv.writer(file)
        wr.writerows(X_train)
    file.close()

    with open(path + "/" + "test_dataset.csv", "a") as file:
        wr = csv.writer(file)
        wr.writerows(X_test)
    file.close()

    with open(path + "/" + "train_res.csv", "a") as file:
        wr = csv.writer(file)
        wr.writerow(y_train)
    file.close()

    with open(path + "/" + "test_res.csv", "a") as file:
        wr = csv.writer(file)
        wr.writerow(y_test)
    file.close()

    return X_train, X_test, y_train, y_test


def get_dataset(folder="text_process"):
    path = os.path.join(os.getcwd(), "datasets", folder) + "/"
    X_train = pd.read_csv(path + "train_dataset.csv", header=None).values
    X_test = pd.read_csv(path + "test_dataset.csv", header=None).values
    y_train = pd.read_csv(path + "train_res.csv", header=None).values.tolist()[0]
    y_test = pd.read_csv(path + "test_res.csv", header=None).values.tolist()[0]

    return X_train, X_test, y_train, y_test

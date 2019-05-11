import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
from sklearn.feature_extraction.text import CountVectorizer
import csv
import io
import os
import pandas as pd
from random import shuffle
from keras.preprocessing.text import Tokenizer
import numpy as np


class TextPreparation:

    def __init__(self, max_features=500):
        self.vectorizer = CountVectorizer(max_features=max_features)
        # self.vectorizer = CountVectorizer(ngram_range=(2, 2), analyzer='char', max_features=max_features)
        # self.tokenizer = Tokenizer()

    @staticmethod
    def delete_punctuation(text):
        return re.sub("[^а-яА-Я\-\s]", "", text)

    @staticmethod
    def to_lower_case(text):
        return re.sub("[\-]", " ", text.lower())

    @staticmethod
    def delete_stop_words(text):
        stop_words = set(stopwords.words("russian"))
        words = word_tokenize(text=text, language="russian", preserve_line=True)
        words_filtered = []

        for w in words:
            if w not in stop_words:
                words_filtered.append(w)
        return ' '.join(words_filtered)

    @staticmethod
    def make_lemmantisation(text):
        m = Mystem()
        m.start()
        lemmas = m.lemmatize(str(text))
        return lemmas

    def prepare_text(self, text):
        text1 = self.delete_punctuation(text)
        text2 = self.to_lower_case(text1)
        text3 = self.delete_stop_words(text2)
        text4 = self.make_lemmantisation(text3)
        return ''.join(text4).replace("\n", " ")

    @staticmethod
    def create_blocks(text, blocks_size):
        text = text.split()
        text2 = [text[x:x + blocks_size] for x in range(0, len(text), blocks_size)]
        return text2

    @staticmethod
    def get_all_authors():
        path = os.path.join(os.getcwd(), "data")
        return os.listdir(path)

    @staticmethod
    def get_all_authors_files(author):
        path = os.path.join(os.getcwd(), "data", author)
        return os.listdir(path)

    ################# NEW PREPARATION#########################

    def prepare_all_files(self, authors_number=6, files_number=4, file_length=15000):
        all_res_path = os.path.join(os.getcwd(), "all_data.txt")
        res_path = os.path.join(os.getcwd(), "prepared_texts")

        authors = self.get_all_authors()
        data_path = os.path.join(os.getcwd(), "data")

        with io.open(all_res_path, 'w', encoding='utf8') as outfile_all:
            for author in authors[:authors_number]:
                files = self.get_all_authors_files(author=author)
                for file in files[:files_number]:
                    print("Started file: {}".format(file))
                    prepared_text = None
                    with io.open(data_path + "/" + author + "/" + file, 'r', encoding='utf8') as infile:
                        data = infile.read()
                        prepared_text = self.prepare_text(data[:file_length])
                    with io.open(res_path + "/" + file, 'w', encoding='utf8') as outfile:
                        outfile.write(prepared_text)
                    outfile_all.write(prepared_text)
                    print("Finished file: {}".format(file))

    def add_file_to_train(self, file_name=None, author_mark=None, blocks_size=None):
        data_path = os.path.join(os.getcwd(), "prepared_texts", file_name)

        with io.open(data_path, 'r', encoding='utf8') as f:
            data = f.read()

        result = []

        if blocks_size is not None:
            blocks = self.create_blocks(data, blocks_size)

            for block in blocks:
                vector = self.vectorizer.transform(block)
                temp = vector.toarray()[0]
                temp = temp.tolist()
                temp.append(author_mark)

                result.append(temp)
        else:
            vector = self.vectorizer.transform([data]).toarray()[0].tolist()
            vector.append(author_mark)

            result.append(vector)

        with open("train.csv", "a") as file:
            wr = csv.writer(file)
            wr.writerows(result)
        file.close()

    def create_train(self, blocks_size=None, authors_number=6, files_number=4,):
        all_data_path = os.path.join(os.getcwd(), "all_data.txt")
        with io.open(all_data_path, 'r', encoding='utf8') as f:
            all_data = f.read().splitlines()

        self.vectorizer.fit_transform(raw_documents=all_data)
        print(self.vectorizer.get_feature_names())

        authors = self.get_all_authors()
        author_mark = 0
        for author in authors[:authors_number]:
            files = self.get_all_authors_files(author=author)
            for file in files[:files_number]:
                # self.add_file_to_train(file_name=files[1], author_mark=author_mark, blocks_size=blocks_size)
                self.add_file_to_train(file_name=file, author_mark=author_mark, blocks_size=blocks_size)

            author_mark = author_mark + 1

    @staticmethod
    def get_dataset_new():
        train_path = os.path.join(os.getcwd(), "train.csv")
        data = pd.read_csv(train_path)
        dataset = data.values.tolist()
        shuffle(dataset)
        x = []
        y = []
        for row in dataset:
            x.append(row[:-1])
            y.append(row[-1])

        print("*****************************")
        print("LEN X: {}".format(len(x)))
        print("*****************************")

        return x, y

    #################### KERAS TOKENIZER########################

    def add_file_to_train_keras(self, file_name, author_mark, blocks_size):
        data_path = os.path.join(os.getcwd(), "prepared_texts", file_name)

        with io.open(data_path, 'r', encoding='utf8') as f:
            data = f.read()

        blocks = self.create_blocks(data, blocks_size)

        x = []
        y = []
        result = []

        result = self.tokenizer.texts_to_matrix(blocks)
        y = [[author_mark]] * len(blocks)
        # temp.append(author_mark)

        # result.append(temp)

        with open("train.csv", "a") as file:
            wr = csv.writer(file)
            wr.writerows(result)
        file.close()

        with open("res.csv", "a") as file2:
            wr = csv.writer(file2)
            wr.writerows(y)
        file2.close()

    def create_train_keras(self, blocks_size=100):
        all_data_path = os.path.join(os.getcwd(), "all_data.txt")
        with io.open(all_data_path, 'r', encoding='utf8') as f:
            all_data = f.read().splitlines()

        self.tokenizer.fit_on_texts(all_data)

        authors = self.get_all_authors()
        author_mark = 0
        for author in authors[:2]:
            files = self.get_all_authors_files(author=author)
            for file in files:
                # self.add_file_to_train(file_name=files[1], author_mark=author_mark, blocks_size=blocks_size)
                self.add_file_to_train_keras(file_name=file, author_mark=author_mark, blocks_size=blocks_size)

            author_mark = author_mark + 1

    @staticmethod
    def get_dataset_keras():
        train_path = os.path.join(os.getcwd(), "train.csv")
        res_path = os.path.join(os.getcwd(), "res.csv")

        data = pd.read_csv(train_path)
        res = pd.read_csv(res_path)
        dataset = data.values.tolist()
        res = data.values.tolist()
        # shuffle(dataset)
        # x = []
        # y = []
        # for row in dataset:
        #     x.append(row[:-1])
        #     y.append(row[-1])

        print("*****************************")
        print("DATASET: {}".format(len(dataset)))
        # print("RES: {}".format(len(res)))
        print("***************************")

        return dataset, res

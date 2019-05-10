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

    def __init__(self):
        self.vectorizer = CountVectorizer(max_features=600)
        self.tokenizer = Tokenizer()
        # self.define_dictionary()

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

    def merge_all_data_files(self):
        res_path = os.path.join(os.getcwd(), "all_data.txt")
        file_exists = os.path.isfile(res_path)

        if not file_exists:
            authors = self.get_all_authors()
            data_path = os.path.join(os.getcwd(), "data")

            with io.open(res_path, 'w', encoding='utf8') as outfile:
                for author in authors:
                    files = self.get_all_authors_files(author=author)
                    for file in files:
                        with io.open(data_path + "/" + author + "/" + file, 'r', encoding='utf8') as infile:
                            for line in infile:
                                outfile.write(self.delete_stop_words(line))

        return res_path

    def define_dictionary(self):
        res_path = self.merge_all_data_files()
        with io.open(res_path, 'r', encoding='utf8') as f:
            data = f.read().splitlines()

        clear_text = self.prepare_text(''.join(data))
        self.vectorizer.fit(raw_documents=clear_text)

    def create_dictionary(self, file_name, author, author_mark, blocks_size):
        data_path = os.path.join(os.getcwd(), "data", author, file_name)
        # train_path = os.path.join(os.getcwd(), "train.csv")

        with io.open(data_path, 'r', encoding='utf8') as f:
            data = f.read().splitlines()

        clear_text = self.prepare_text(''.join(data))
        blocks = self.create_blocks(clear_text, blocks_size)

        result = []

        for block in blocks:
            # self.vectorizer.fit_transform(block)
            vector = self.vectorizer.transform(block)
            temp = vector.toarray()[0]
            temp = temp.tolist()
            temp.append(author_mark)

            result.append(temp)

        with open("train.csv", "a") as file:
            wr = csv.writer(file)
            wr.writerows(result)

    def create_all_dicts(self, blocks_size=100):
        authors = self.get_all_authors()
        author_mark = 0
        for author in authors:
            files = self.get_all_authors_files(author=author)
            for file in files:
                # self.create_dictionary(file_name=files[1], author=author, author_mark=author_mark, blocks_size=blocks_size)
                self.create_dictionary(file_name=file, author=author, author_mark=author_mark, blocks_size=blocks_size)

            author_mark = author_mark + 1

    # @staticmethod
    # def get_dataset():
    #     train_path = os.path.join(os.getcwd(), "train.csv")
    #     data = pd.read_csv(train_path)
    #     dataset = data.values.tolist()
    #     shuffle(dataset)
    #     x = []
    #     y = []
    #     for row in dataset:
    #         x.append(row[:-1])
    #         y.append(row[-1])
    #
    #     print("*****************************")
    #     print("LEN X: {}".format(len(x)))
    #     print("*****************************")
    #
    #     return x, y

    # def get_dataset_for_neural(self, blocks_size=100):
    #     res_path = self.merge_all_data_files()
    #     with io.open(res_path, 'r', encoding='utf8') as f:
    #         data = f.read().splitlines()
    #
    #     tokenizer = Tokenizer(num_words=None,
    #                           filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    #                           lower=True,
    #                           split=' ',
    #                           char_level=False,
    #                           oov_token=None,
    #                           document_count=0)
    #
    #     tokenizer.fit_on_texts(texts=data)
    #
    #     authors = self.get_all_authors()
    #     author_mark = 0
    #     result = []
    #     authors_list = []
    #     for author in authors:
    #         files = self.get_all_authors_files(author=author)
    #         for file in files:
    #             data_path = os.path.join(os.getcwd(), "data", author, file)
    #
    #             with io.open(data_path, 'r', encoding='utf8') as f:
    #                 data = f.read().splitlines()
    #
    #             clear_text = self.delete_stop_words(''.join(data))
    #             blocks = self.create_blocks(clear_text, blocks_size)
    #
    #             result = tokenizer.texts_to_sequences(blocks)
    #             authors_list.append(author_mark)
    #
    #             with open("train.csv", "a") as file:
    #                 wr = csv.writer(file)
    #                 wr.writerows(result)
    #
    #         author_mark = author_mark + 1
    #
    #     # # dataset = result.tolist()
    #     # x = []
    #     # y = []
    #     # for row in dataset:
    #     #     x.append(row[:-1])
    #     #     y.append(row[-1])
    #     #
    #     # print("*****************************")
    #     # print("LEN X: {}".format(len(x)))
    #     # print("*****************************")
    #
    #     return result, authors_list

    ################# NEW PREPARATION#########################

    def prepare_all_files(self):
        all_res_path = os.path.join(os.getcwd(), "all_data.txt")
        res_path = os.path.join(os.getcwd(), "prepared_texts")

        authors = self.get_all_authors()
        data_path = os.path.join(os.getcwd(), "data")

        with io.open(all_res_path, 'w', encoding='utf8') as outfile_all:
            for author in authors:
                files = self.get_all_authors_files(author=author)
                for file in files:
                    print("Started file: {}".format(file))
                    prepared_text = None
                    with io.open(data_path + "/" + author + "/" + file, 'r', encoding='utf8') as infile:
                        data = infile.read()
                        prepared_text = self.prepare_text(data)
                    with io.open(res_path + "/" + file, 'w', encoding='utf8') as outfile:
                        outfile.write(prepared_text)
                    outfile_all.write(prepared_text)
                    print("Finished file: {}".format(file))

    def add_file_to_train(self, file_name, author_mark, blocks_size):
        data_path = os.path.join(os.getcwd(), "prepared_texts", file_name)

        with io.open(data_path, 'r', encoding='utf8') as f:
            data = f.read()

        blocks = self.create_blocks(data, blocks_size)

        x = []
        y = []
        result = []

        for block in blocks:
            vector = self.vectorizer.transform(block)
            temp = vector.toarray()[0]
            temp = temp.tolist()
            temp.append(author_mark)

            result.append(temp)

        with open("train.csv", "a") as file:
            wr = csv.writer(file)
            wr.writerows(result)
        file.close()

    def create_train(self, blocks_size=100):
        all_data_path = os.path.join(os.getcwd(), "all_data.txt")
        with io.open(all_data_path, 'r', encoding='utf8') as f:
            all_data = f.read().splitlines()

        self.vectorizer.fit_transform(raw_documents=all_data)

        authors = self.get_all_authors()
        author_mark = 0
        for author in authors[:2]:
            files = self.get_all_authors_files(author=author)
            for file in files:
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
        for author in authors:
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

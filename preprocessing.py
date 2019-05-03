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


class TextPreparation:

    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.define_dictionary()

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
        return text4

    @staticmethod
    def create_blocks(text):
        return [text[x:x + 100] for x in range(0, len(text), 100)]

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
                                outfile.write(line)

        return res_path

    def define_dictionary(self):
        res_path = self.merge_all_data_files()
        with io.open(res_path, 'r', encoding='utf8') as f:
            data = f.read().splitlines()

        clear_text = self.prepare_text(''.join(data))
        self.vectorizer.fit(raw_documents=clear_text)

    def create_dictionary(self, file_name, author, author_mark):
        data_path = os.path.join(os.getcwd(), "data", author, file_name)
        # train_path = os.path.join(os.getcwd(), "train.csv")

        with io.open(data_path, 'r', encoding='utf8') as f:
            data = f.read().splitlines()

        clear_text = self.prepare_text(''.join(data))
        blocks = self.create_blocks(clear_text)

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

    def create_all_dicts(self):
        authors = self.get_all_authors()
        author_mark = 0
        for author in authors:
            files = self.get_all_authors_files(author=author)
            for file in files:
                self.create_dictionary(file_name=file, author=author, author_mark=int(bin(author_mark)[2:]))

            author_mark = author_mark + 1

    @staticmethod
    def get_dataset():
        train_path = os.path.join(os.getcwd(), "train.csv")
        data = pd.read_csv(train_path)
        dataset = data.values.tolist()
        shuffle(dataset)
        x = []
        y = []
        for row in dataset:
            x.append(row[:-1])
            y.append(row[-1])

        return x, y

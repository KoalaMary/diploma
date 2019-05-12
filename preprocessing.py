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
        self.vectorizer = CountVectorizer(max_features=500)
        # self.vectorizer = CountVectorizer(ngram_range=(3, 3), analyzer='char', max_features=max_features)
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

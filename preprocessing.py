import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
from sklearn.feature_extraction.text import CountVectorizer
import csv


# text = "Недоброе таится в мужчинах, избегающих вина, игр, общества прелестных женщин, застольной беседы. Такие люди или тяжко больны, или втайне ненавидят окружающих. "
# text2 = "О, гляньте, гляньте на меня, я погибаю! Вьюга в подворотне ревет мне отходную, и я вою с нею"


class TextPreparation:

    def delete_punctuation(self, text):
        return re.sub("[^а-яА-Я\-\s]", "", text)

    def to_lower_case(self, text):
        return re.sub("[\-]", " ", text.lower())

    def delete_stop_words(self, text):
        stop_words = set(stopwords.words("russian"))
        words = word_tokenize(text=text, language="russian", preserve_line=True)
        words_filtered = []

        for w in words:
            if w not in stop_words:
                words_filtered.append(w)
        return ' '.join(words_filtered)

    def make_lemmantisation(self, text):
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

    def create_blocks(self, text):
        return [text[x:x + 100] for x in range(0, len(text), 100)]

    def create_dictionary(self, file_name):
        vectorizer = CountVectorizer()
        with open(file_name) as file:
            data = file.read()

        clear_text = self.prepare_text(data)
        blocks = self.create_blocks(clear_text)

        result = []

        for block in blocks:
            vector = vectorizer.transform([block])
            temp = vector.toarray()[0]
            temp = temp.tolist()
            temp.append(0)

            result.append(temp)

        with open("train.csv", "a") as file:
            wr = csv.writer(file)
            wr.writerows(result)

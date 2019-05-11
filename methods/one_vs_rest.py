# from sklearn import datasets
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.svm import LinearSVC
# from sklearn.metrics import accuracy_score
#
#
# iris = datasets.load_iris()
# X, y = iris.data, iris.target
# y_pred = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)
# y_pred2 = LinearSVC(random_state=0).fit(X, y).predict(X)
# res2 = accuracy_score(y, y_pred)
# res3 = accuracy_score(y, y_pred2)
# print(res2)
# print(res3)
# print(y)


from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer

sentence = "повесть профессор персик апрель год вечер профессор зоология государственный университет директор "
sentence2 = "привет меня зовут коала и я очень люблю панду"
sentence3 = sentence + sentence2
n = 3
ngrams = ngrams(sentence, n)

# vectorizer = CountVectorizer(vocabulary=ngrams)
# ex = vectorizer.fit_transform(sentence)

# print(ex)


v = CountVectorizer(ngram_range=(3, 3), analyzer='char', max_features=10)
v.fit([sentence3])
print(v.fit([sentence3]).vocabulary_)
print(v.transform(sentence.split()))
# print(v.transform(sentence2.split()).toarray())

print("#######################")
print(v.get_feature_names())

# for grams in ngrams:
#   print(grams)
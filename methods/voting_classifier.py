import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score


class VoitingClassifierMethod:
    solver = None
    X_train = None
    X_test = None
    y_train = None
    y_test = None

    def __init__(self, X_train, X_test, y_train, y_test, solver="newton-cg"):
        self.solver = solver
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def do_voiting(self):

        # clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1)
        #
        # clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
        #
        # clf3 = GaussianNB()
        #
        clf1 = LinearSVC(multi_class="ovr")

        clf2 = RandomForestClassifier(n_estimators=50, random_state=1)

        clf3 = MLPClassifier(random_state=0)

        eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

        eclf1 = eclf1.fit(self.X_train, self.y_train)
        y_pred1 = eclf1.predict(self.X_test)
        print("ACCURANCY 1: {}".format(accuracy_score(self.y_test, y_pred1)))
        #
        # np.array_equal(eclf1.named_estimators_.lr.predict(self.X_test),
        #                eclf1.named_estimators_['lr'].predict(self.X_test))
        #
        # eclf2 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
        # eclf2 = eclf2.fit(self.X_test, self.y_test)
        # y_pred2 = eclf2.predict(self.X_test)
        # print("ACCURANCY 2: {}".format(accuracy_score(self.y_test, y_pred2)))

        eclf3 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], flatten_transform=True)
        eclf3 = eclf3.fit(self.X_train, self.y_train)
        y_pred3 = eclf3.predict(self.X_test)
        # cross_validation = cross_val_score(eclf3, self.X_train, self.y_train, cv=10)
        print("ACCURANCY 3: {}".format(accuracy_score(self.y_test, y_pred3)))
        # print("Cross validation: {}".format(cross_validation))

        return y_pred3

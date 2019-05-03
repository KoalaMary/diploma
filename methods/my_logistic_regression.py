import pandas as pd
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os


class LogisticRegressionMethod:

    @staticmethod
    def get_dataset():
        train_path = os.path.join(os.getcwd(), "..", "train.csv")
        data = pd.read_csv(train_path)
        dataset = data.values.tolist()
        shuffle(dataset)
        x = []
        y = []
        for row in dataset:
            x.append(row[:-1])
            y.append(row[-1])

        return x, y

    def do_LR(self):
        # Training
        x, y = self.get_dataset()
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
        model = clf.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)
        res = model.score(X_test, y_test)

        print("Y PREd: {}".format(y_pred))
        print("Y TEST: {}".format(y_test))
        print("RES: {}".format(res))

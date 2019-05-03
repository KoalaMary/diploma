from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from preprocessing import TextPreparation


class RandomForestMethod:

    def do_RF(self):
        # Training
        x, y = TextPreparation.get_dataset()
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        n_est = 100
        max_d = 5
        rfc = RandomForestClassifier(n_jobs=-1, n_estimators=n_est, max_depth=max_d)
        rfc.fit(X_train, y_train)

        # Prediction
        accuracy_forest = rfc.score(X_test, y_test)
        y_pred = rfc.predict(X_test)

        print("Y PREd: {}".format(y_pred))
        print("Y TEST: {}".format(y_test))
        print("accuracy_forest: {}".format(accuracy_forest))
        # print("RES: {}".format(res))

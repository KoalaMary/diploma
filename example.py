from preprocessing import TextPreparation
from methods.my_logistic_regression import LogisticRegressionMethod
from methods.random_forest import RandomForestMethod
from methods.svm import SVMMethod
from methods.knn import KNNMethod


# ex = TextPreparation()
# ex.create_all_dicts()

print("-----------------LINEAR REGRESSION------------------------")
lr = LogisticRegressionMethod()
lr.do_LR()


print("-----------------RANDOM FORESTS------------------------")
rf = RandomForestMethod()
rf.do_RF()


print("-----------------SVM METHOD------------------------")
svm = SVMMethod()
svm.do_SVM()


print("-----------------KNN------------------------")
knn = KNNMethod()
knn.do_knn()
from methods.do_methods import *
from reports.plots import *


# prepare_all_data()
# make_dataset(folder="char3_700", max_features=700)
X_train, X_test, y_train, y_test = get_dataset(folder="char3_700")
#
# y_pred_lr = logistic_regression(X_train, X_test, y_train, y_test)
# fig_lr = plot_confusion_matrix(y_true=y_test, y_pred=y_pred_lr, normalize=True, title=f'LOGISTIC REGRESSION')
# fig_lr.show()
#
# y_pred_forests = random_forests(X_train, X_test, y_train, y_test)
# fig_forests = plot_confusion_matrix(y_true=y_test, y_pred=y_pred_forests, normalize=True, title=f'RANDOM FORESTS')
# fig_forests.show()
#
y_pred_svm = svm(X_train, X_test, y_train, y_test)
# fig_svm = plot_confusion_matrix(y_true=y_test, y_pred=y_pred_svm, normalize=True, title=f'SVM')
# fig_svm.show()
#
# y_pred_knn = knn(X_train, X_test, y_train, y_test)
# fig_knn = plot_confusion_matrix(y_true=y_test, y_pred=y_pred_knn, normalize=True, title=f'KNN')
# fig_knn.show()
#
# y_pred_bayes = naive_bayes(X_train, X_test, y_train, y_test)
# fig_bayes = plot_confusion_matrix(y_true=y_test, y_pred=y_pred_bayes, normalize=True, title=f'NAIVE BAYES')
# fig_bayes.show()
#
y_pred_neural = skleran_neural(X_train, X_test, y_train, y_test)
# fig_neural = plot_confusion_matrix(y_true=y_test, y_pred=y_pred_neural, normalize=True, title=f'NEURAL')
# fig_neural.show()
#
#
# # reccurent_neural(X_train, X_test, y_train, y_test)
# y_pred_voiting = voiting_classifier(X_train, X_test, y_train, y_test)
# fig_voiting = plot_confusion_matrix(y_true=y_test, y_pred=y_pred_voiting, normalize=True, title=f'VOITING')
# fig_voiting.show()

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def load_dataset(filename):
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pd.read_csv(filename,names=names)
    return dataset

def get_models():
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('SVM', SVC(gamma='auto')))
    return models

def get_result(models, X_train, Y_train):
    results = []
    names = []
    print("Models    Accuracy  (stdev.)")
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('{:6}: {:10.4f} ({:.4f})'.format(name, cv_results.mean(), cv_results.std()))
    return (results, names)

def vis_result(results, names):
    pyplot.boxplot(results, labels=names)
    pyplot.title('Algorithm Comparison')
    pyplot.show()

def make_prediction(X_train, Y_train, X_validation, Y_validation):
    model = SVC(gamma='auto')
    model.fit(X_train,Y_train)
    prediction = model.predict(X_validation)

    print(accuracy_score(Y_validation, prediction))
    print(confusion_matrix(Y_validation, prediction))
    print(classification_report(Y_validation, prediction))

    disp = plot_confusion_matrix(model, X_validation,Y_validation,cmap=plt.cm.Blues, normalize='true')

    plt.show()


"""
dataset = load_dataset('iris.csv')
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,test_size = 0.2, random_state =1)
models = get_models()
results, names = get_result(models, X_train, Y_train)
vis_result(results, names)
make_prediction(X_train,Y_train,X_validation,Y_validation)
"""
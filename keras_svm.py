from __future__ import print_function

import os

import sklearn
from sklearn.multiclass import OneVsOneClassifier
from sklearn import cross_validation, grid_search, model_selection
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.externals import joblib
import numpy as np

import model
from model import Model


class Svm(Model):

    def prepare_data(self, dataset):
        """Takes data loaded by spio.loadmat and prepares training and test sets."""

        # load training dataset
        x_train = dataset["dataset"][0][0][0][0][0][0]
        x_train = x_train.astype(np.float32)

        # load training labels
        y_train = dataset["dataset"][0][0][0][0][0][1]

        # load test dataset
        x_test = dataset["dataset"][0][0][1][0][0][0]
        x_test = x_test.astype(np.float32)

        # load test labels
        y_test = dataset["dataset"][0][0][1][0][0][1]

        # store labels for visualization
        #train_labels = y_train
        #test_labels = y_test

        x_train /= 255
        x_test /= 255

        y_min = y_train.min()
        y_max = y_train.max()

        num_classes = y_max - y_min + 1
        y_train -= y_min
        y_test -= y_min

        y_train = np.ravel(y_train)
        y_test = np.ravel(y_test)

        return (x_train, y_train), (x_test, y_test), num_classes

    def run_model_selection(self, dataset, model_name, dataset_name):
        """Runs training with cross-validation and saves the best classifier into a file"""

        (x_train, y_train), (x_test, y_test), num_classes = self.prepare_data(dataset)

        x_train = x_train
        y_train = y_train

        param_grid = [
            {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
            {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
        ]

        # request probability estimation (takes much longer when set to true)
        svm = SVC(probability=False)

        # 10-fold cross validation, use 4 thread as each fold and each parameter set can be trained in parallel
        clf = model_selection.GridSearchCV(svm, param_grid, cv=5, n_jobs=4, verbose=3)

        clf.fit(x_train, y_train)

        self.save_estimator(clf.best_estimator_, model_name, dataset_name)
        self.save_cv_results(clf.cv_results_, model_name, dataset_name)

        print("\nBest parameters set:")
        print(clf.best_params_)

    def run_training(self, dataset, model_name, dataset_name):
        """Runs training on a given classifier"""

        (x_train, y_train), (x_test, y_test), num_classes = self.prepare_data(dataset)

        x_train = x_train
        y_train = y_train

        clf = SVC(probability=False, C=1000, gamma=0.001, kernel='rbf', verbose=True)

        clf.fit(x_train, y_train)

        self.save_estimator(clf, model_name, dataset_name)

    def run_parallel_training(self, dataset, model_name, dataset_name, n_jobs):
        """Runs training with cross-validation and saves the best classifier into a file"""

        (x_train, y_train), (x_test, y_test), num_classes = self.prepare_data(dataset)

        x_train = x_train
        y_train = y_train

        clf = OneVsOneClassifier(SVC(probability=False, C=1000, gamma=0.001, kernel='rbf', verbose=True), n_jobs=n_jobs)

        clf.fit(x_train, y_train)

        self.save_estimator(clf, model_name, dataset_name, append='_parallel')

    def run_test(self, clf, dataset, model_name, dataset_name):
        """Runs test from EMNIST dataset on a given classifier"""

        (x_train, y_train), (x_test, y_test), num_classes = self.prepare_data(dataset)

        y_predict = clf.predict(x_test)

        labels = [y for y in y_train]
        labels = sorted(list(set(labels)))
        print("\nConfusion matrix:")
        print("Labels: {0}\n".format(",".join([str(y) for y in labels])))
        print(confusion_matrix(y_test, y_predict, labels=labels))

        print("\nClassification report:")
        print(classification_report(y_test, y_predict))

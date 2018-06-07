import matplotlib.pyplot as plt
import numpy as np
import os
import json
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import sklearn
from sklearn import cross_validation, grid_search, model_selection
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.externals import joblib
from keras.callbacks import EarlyStopping

models_directory = "output/Models/"
plots_directory = "output/Plots/"

if not os.path.exists(models_directory):
    os.makedirs(models_directory)
if not os.path.exists(plots_directory):
    os.makedirs(plots_directory)

class Model:

    def save_test_results(self, confusion_matrix, clf_report, model_name, dataset_name):

        test_result_filename = self.change_names_into_filename(model_name, dataset_name) + '_test_result'
        test_result = {'conf_matrix': confusion_matrix, 'clf_report': clf_report}

        import io
        try:
            to_unicode = unicode
        except NameError:
            to_unicode = str

        with io.open(test_result_filename + '.json', 'w', encoding='utf8') as outfile:
            str_ = json.dumps(test_result,
                              indent=4, sort_keys=True,
                              separators=(',', ': '), ensure_ascii=False)
            outfile.write(to_unicode(str_))

        return None

    def expand_dataset(self, x_train, y_train):

        expanded_training_pairs = []
        j = 0  # counter
        for x, y in zip(x_train, y_train):
            expanded_training_pairs.append((x, y))
            image = np.reshape(x, (-1, 28))
            j += 1
            if j % 1000 == 0: print("Expanding image number", j)
            # iterate over data telling us the details of how to
            # do the displacement
            for d, axis, index_position, index in [
                (1, 0, "first", 0),
                (-1, 0, "first", 27),
                (1, 1, "last", 0),
                (-1, 1, "last", 27)]:
                new_img = np.roll(image, d, axis)
                if index_position == "first":
                    new_img[index, :] = np.zeros(28)
                else:
                    new_img[:, index] = np.zeros(28)
                expanded_training_pairs.append((np.reshape(new_img, 784), y))
        random.shuffle(expanded_training_pairs)
        #expanded_training_data = [list(d) for d in zip(*expanded_training_pairs)]

        expanded_x_train = [x for (x, y) in expanded_training_pairs]
        expanded_y_train = [y for (x, y) in expanded_training_pairs]

        ex_x_train = np.array(expanded_x_train)
        ex_y_train = np.array(expanded_y_train)

        return ex_x_train, ex_y_train

    def change_names_into_filename(self, model_name, dataset_name):
        """Returns filename corresponding with model and dataset names"""

        return models_directory + model_name + "_" + dataset_name

    def save_estimator(self, estimator, model_name, dataset_name, append=''):
        filename = self.change_names_into_filename(model_name, dataset_name) + append

        if not os.path.exists(filename):
            os.mknod(filename)
        if os.path.exists(filename):
            joblib.dump(estimator, filename)
        else:
            print("Cannot save trained svm model to {0}.".format(filename))

    def load_estimator(self, model_name, dataset_name, append=''):
        filename = self.change_names_into_filename(model_name, dataset_name) + append
        print("Loading estimator " + filename)
        estimator = joblib.load(filename)
        return estimator

    def save_cv_results(self, cv_results, model_name, dataset_name):
        filename = self.change_names_into_filename(model_name, dataset_name) + "_cv_results"

        if not os.path.exists(filename):
            os.mknod(filename)
        if os.path.exists(filename):
            joblib.dump(cv_results, filename)
        else:
            print("Cannot save cv_results to {0}.".format(filename))

    def load_cv_results(self, model_name, dataset_name):
        cv_results = joblib.load(self.change_names_into_filename(model_name, dataset_name) + "_cv_results")
        return cv_results

class NeuralNet(Model):

    def save_training_results(self, history, model_name, dataset_name):

        import io
        try:
            to_unicode = unicode
        except NameError:
            to_unicode = str

        filename = self.change_names_into_filename(model_name, dataset_name) + '_train_results'
        history.history['epoch'] = history.epoch
        with io.open(filename + '.json', 'w', encoding='utf8') as outfile:
            str_ = json.dumps(history.history,
                              indent=4, sort_keys=True,
                              separators=(',', ': '), ensure_ascii=False)
            outfile.write(to_unicode(str_))

        return None

    def plot_training_results(self, model_name, dataset_name):
        filename = self.change_names_into_filename(model_name, dataset_name) + "_train_results"

        with open(filename + '.json') as data_file:
            data_loaded = json.load(data_file)

        fig = plt.figure(1, figsize=(9, 3))
        plt.plot(data_loaded['epoch'], data_loaded['acc'], 'r', label='acc')
        plt.plot(data_loaded['epoch'], data_loaded['val_acc'], 'b', label='val_acc')
        plt.xticks(data_loaded['epoch'])
        plt.xlabel('epoch')
        plt.ylabel('accuracy %')
        plt.legend(('Training accuracy', 'Validation accuracy'))
        plt.title('Training & validation set accuracy')
        plt.savefig(filename + '_acc_plot.png')
        plt.close(fig)

        fig = plt.figure(2, figsize=(12, 4))
        plt.plot(data_loaded['epoch'], data_loaded['loss'], 'r', label='loss')
        plt.plot(data_loaded['epoch'], data_loaded['val_loss'], 'b', label='val_loss')
        plt.xticks(data_loaded['epoch'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(('Training loss', 'Validation loss'))
        plt.title('Training & validation set loss')
        plt.savefig(filename + '_loss_plot.png')
        plt.close(fig)

    def visualize_model(self, model, model_name):

        from keras.utils import plot_model
        plot_model(model, to_file=models_directory + model_name + '_visual.png')

    def save_model(self, model, model_name, dataset_name):

        filename = self.change_names_into_filename(model_name, dataset_name) + '.h5'
        model.save(filename)

    def load_model(self, model_name, dataset_name):

        filename = self.change_names_into_filename(model_name, dataset_name) + '.h5'
        from keras.models import load_model
        model = load_model(filename)
        return model

    def run_model_selection(self, dataset, batch_size, epochs, model_name, dataset_name):

        (x_train, y_train), (x_test, y_test), num_classes, input_shape = self.prepare_data(dataset)

        x_train = x_train
        y_train = y_train
        self.print_dataset_stats(x_train, y_train, x_test, y_test)

        model = KerasClassifier(build_fn=self.create_net, num_classes=num_classes, input_shape=input_shape)
        #activation = ['relu', 'tanh']
        #momentum = [0.0, 0.2]
        #epochs = [1, 2]
        #batch_size = [128]
        param_grid = dict(epochs=epochs, batch_size=batch_size)

        clf = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=1)
        clf.fit(x_train, y_train)

        #self.save_estimator(clf.best_estimator_, model_name, dataset_name)
        self.save_model(clf.best_estimator_.model, model_name, dataset_name)
        self.save_cv_results(clf.cv_results_, model_name, dataset_name)

        print("\nBest parameters set:")
        print(clf.best_params_)

        #model = self.create_net(num_classes, input_shape)

        '''
        history = model.fit(
                            x=x_train,
                            y=y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_split=0.1)
        '''

        #self.save_model(history, model_name, dataset_name)
        #self.save_training_results(history, model_name, dataset_name)
        # self.plot_training_results('MLP1_mnist')

    def run_training(self, dataset, batch_size, max_epochs, model_name, dataset_name, nr_of_validation_samples):

        (x_train, y_train), (x_test, y_test), num_classes, input_shape = self.prepare_data(dataset)

        x_train = x_train[:-nr_of_validation_samples]
        y_train = y_train[:-nr_of_validation_samples]
        x_val = x_train[-nr_of_validation_samples:]
        y_val = y_train[-nr_of_validation_samples:]

        self.print_dataset_stats(x_train, y_train, x_test, y_test)

        clf = self.create_net(num_classes, input_shape)

        callbacks = [
            EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1)
            ]

        history = clf.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=max_epochs,
            verbose=1,
            #validation_split=0.166,
            validation_data=(x_val, y_val),
            callbacks=callbacks)

        self.save_model(clf, model_name, dataset_name)
        self.save_training_results(history, model_name, dataset_name)

    def run_test(self, dataset, clf, model_name, dataset_name):
        """Runs test from EMNIST dataset using given model"""

        (x_train, y_train), (x_test, y_test), num_classes, input_shape = self.prepare_data(dataset)

        original_x_test = x_test
        original_y_test = y_test

        y_predict = clf.predict(x_test)

        temp_train = []
        temp_predict = []
        temp_test = []
        for i, vect in enumerate(y_train):
            temp_train.append(np.argmax(y_train[i]))

        for i, vect in enumerate(y_predict):
            temp_predict.append(np.argmax(y_predict[i]))

        for i, vect in enumerate(y_test):
            temp_test.append(np.argmax(y_test[i]))

        y_train = temp_train
        y_predict = temp_predict
        y_test = temp_test

        labels = [y for y in y_train]

        labels = sorted(list(set(labels)))
        print("\nConfusion matrix:")
        print("Labels: {0}\n".format(",".join([str(y) for y in labels])))
        print(confusion_matrix(y_test, y_predict, labels=labels))

        print("\nClassification report:")
        print(classification_report(y_test, y_predict))

        score = clf.evaluate(original_x_test, original_y_test, verbose=1)

        print('Test loss:', score[0])
        print('Test accuracy:', score[1])



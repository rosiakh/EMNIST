from keras_cnn import Cnn
from keras_mlp import Mlp
from keras_svm import Svm

from scipy import io as spio

all_datasets = ["mnist", "balanced", "letters", "digits", "bymerge", "byclass"]  # sorted by size

datasets_directory = "data/Data/EMNIST/emnist_matlab_format/matlab/"

mlp = Mlp()
cnn = Cnn()
svm = Svm()

f_data = 0
l_data = 6

# Test Neural Network models
for dataset in all_datasets[f_data:l_data]:
    loaded_dataset = spio.loadmat("{0}emnist-{1}.mat".format(datasets_directory, dataset))

    mlp_model = mlp.load_model(model_name='MLP', dataset_name=dataset)
    mlp.run_test(dataset=loaded_dataset, clf=mlp_model, model_name='MLP', dataset_name=dataset)

    cnn_model = cnn.load_model(model_name='CNN', dataset_name=dataset)
    cnn.run_test(dataset=loaded_dataset, clf=cnn_model, model_name='CNN', dataset_name=dataset)


# Test SVC models
for dataset in all_datasets[f_data:l_data]:
    loaded_dataset = spio.loadmat("{0}emnist-{1}.mat".format(datasets_directory, dataset))

    clf = svm.load_estimator(model_name='SVC', dataset_name=dataset, append='')
    svm.run_test(clf=clf, dataset=loaded_dataset, model_name='SVC', dataset_name=dataset)

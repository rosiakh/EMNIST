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

    mlp.plot_training_results(model_name='MLP', dataset_name=dataset)
    cnn.plot_training_results(model_name='CNN', dataset_name=dataset)



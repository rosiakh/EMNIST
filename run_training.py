from keras_cnn import Cnn
from keras_mlp import Mlp
from keras_svm import Svm

from scipy import io as spio

all_datasets = ["mnist", "balanced", "letters", "digits", "bymerge", "byclass"]  # sorted by size
nr_of_validation_samples = [10000, 18800, 14800, 40000, 110000, 110000]

datasets_directory = "data/Data/EMNIST/emnist_matlab_format/matlab/"

mlp = Mlp()
cnn = Cnn()
svm = Svm()

f_data = 0
l_data = 6


# Train Neural Networks
for dataset, nr_of_val in zip(all_datasets[f_data:l_data], nr_of_validation_samples[f_data:l_data]):
    loaded_dataset = spio.loadmat("{0}emnist-{1}.mat".format(datasets_directory, dataset))

    
    mlp.run_training(
        dataset=loaded_dataset,
        batch_size=128,
        max_epochs=200,
        model_name='MLP',
        dataset_name=dataset,
        nr_of_validation_samples=nr_of_val)
    

    cnn.run_training(
        dataset=loaded_dataset, 
        batch_size=128, 
        max_epochs=200,
        model_name='CNN', 
        dataset_name=dataset, 
        nr_of_validation_samples=nr_of_val)
    


# Train SVC
for dataset, nr_of_val in zip(all_datasets[f_data:l_data], nr_of_validation_samples[f_data:l_data]):
    loaded_dataset = spio.loadmat("{0}emnist-{1}.mat".format(datasets_directory, dataset))

    svm.run_training(dataset=loaded_dataset, model_name='SVC', dataset_name=dataset)

    # Does the same but in parallel
    # Resultant model is bigger
    #svm.run_parallel_training(dataset=loaded_dataset, model_name='SVC', dataset_name=dataset, n_jobs=3)

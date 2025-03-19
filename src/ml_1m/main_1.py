import numpy as np
import random
from Interactions import Interactions, InstanceSpliter
from Models import NextItNet_DistributedWeights
import Models
import tensorflow as tf


epistemic_resutls = 'results/NextItNet/epistemic_uncertainty/'
aleatoric_results = 'results/NextItNet/aleatoric_uncertainty/'

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def init(seed=2022, sequence_length=10, target_length=1, k_fold=5):
    data_root = 'data'
    set_seed(seed)

    # load dataset
    instances = Interactions(data_root, sequence_length=sequence_length, target_length=target_length)
    instances.data_preprocess()
    instances.create_test_train_instances()

    # get_target_samples(instances, num_negative_samples=1)

    instance_spliter = InstanceSpliter(data=instances, k_fold=k_fold)
    instance_spliter.generate_folds()

    return instances, instance_spliter

data, spliter = init(sequence_length=100, target_length=1, seed=2022, k_fold=5)
between_users = {
    'nicest': {
        'threshold': 7.50,
        'isMost': True,
        'split_array': data.x_avg_ratings,
        'isModelTrained': False,
        'isEvaluationMetricsCalculated': False,
        'isGraphsPloted': False,
        'x_label': 'Average Value of Ratings',
    },

    'harshest': {
        'threshold': 7.50,
        'isMost': False,
        'split_array': data.x_avg_ratings,
        'isModelTrained': False,
        'isEvaluationMetricsCalculated': False,
        'isGraphsPloted': False,
        'x_label': 'Average Value of Ratings',
    },

    'most_active': {
        'threshold': 100.00,
        'isMost': True,
        'split_array': data.x_n_ratings,
        'isModelTrained': False,
        'isEvaluationMetricsCalculated': False,
        'isGraphsPloted': False,
        'x_label': 'Number of Rated Movies',
    },

    'least_active': {
        'threshold': 100.00,
        'isMost': False,
        'split_array': data.x_n_ratings,
        'isModelTrained': False,
        'isEvaluationMetricsCalculated': False,
        'isGraphsPloted': False,
        'x_label': 'Number of Rated Movies',
    },

    'most_typical': {
        'threshold': 0.13,
        'isMost': True,
        'split_array': data.x_similarity,
        'isModelTrained': False,
        'isEvaluationMetricsCalculated': False,
        'isGraphsPloted': False,
        'x_label': 'Average User Similarities to All Others',
    },

    'least_typical': {
        'threshold': 0.13,
        'isMost': False,
        'split_array': data.x_similarity,
        'isModelTrained': False,
        'isEvaluationMetricsCalculated': False,
        'isGraphsPloted': False,
        'x_label': 'Average User Similarities to All Others',
    },

    'most_consistent': {
        'threshold': 4.00,
        'isMost': True,
        'split_array': data.x_var_ratings,
        'isModelTrained': False,
        'isEvaluationMetricsCalculated': False,
        'isGraphsPloted': False,
        'x_label': 'None',
    },

    'least_consistent': {
        'threshold': 4.00,
        'isMost': False,
        'split_array': data.x_var_ratings,
        'isModelTrained': False,
        'isEvaluationMetricsCalculated': False,
        'isGraphsPloted': False,
        'x_label': 'None',
    },

    'most_open_minded': {
        'threshold': 16,
        'isMost': True,
        'split_array': data.x_n_genres,
        'isModelTrained': False,
        'isEvaluationMetricsCalculated': False,
        'isGraphsPloted': False,
        'x_label': 'None',
    },

    'least_open_minded': {
        'threshold': 16,
        'isMost': False,
        'split_array': data.x_n_genres,
        'isModelTrained': False,
        'isEvaluationMetricsCalculated': False,
        'isGraphsPloted': False,
        'x_label': 'None',
    },

}

between_user_data_reduction_techniques = [
#    'most_open_minded',  # [Done]
#    'least_open_minded',  # [Done]
#    'most_consistent',  # [Done]
#    'least_consistent',  # [Done]
#    'most_active',  # [Done]
#    'least_active',  # [Done]
    'nicest',  # [Done]
#    'harshest',  # [Done]
    # 'most_typical',
    # 'least_typical',
]

retained_user_parcentages = [
    1.00,
    0.75,
    0.50,
    0.25,
    0.00
]


def train_model(model, train, validation, batch_size=512, max_epochs=1, verbose=1):
    model.fit(
        x=train.sequences,
        y=train.targets,
        verbose=verbose,
        batch_size=batch_size,
        validation_data=(validation.sequences, validation.targets),
        validation_batch_size=batch_size,
        callbacks=tf.keras.callbacks.TerminateOnNaN(),
    )


def evaluate_model(model, test, T=10, requireAleatoricUncertainty=False):  # returns HR@20 and aleatoric uncertainty
    m = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=20)
    hrT = np.zeros(T)
    varience = np.zeros((T, test.sequences.shape[0]))
    for t in range(T):
        y_pred = model.predict(test.sequences, verbose=0)
        varience[t] = y_pred[:, -1]
        m.update_state(test.targets, tf.nn.softmax(y_pred[:, :-1]))
        hrT[t] = m.result().numpy()

    if requireAleatoricUncertainty:
        return np.mean(hrT), np.mean(varience)
    else:
        return np.mean(hrT)

def test_model(nnRecommenderModel, train, val, test):
    max_epochs = 10
    verbose = 1
    batch_size = 1024
    for iteration in range(max_epochs):
        history = nnRecommenderModel.fit(train.sequences, train.targets,
            verbose=verbose,
            batch_size=batch_size,
            validation_data=(val.sequences, val.targets),
            validation_batch_size=batch_size,
        )
        y_pred = nnRecommenderModel.predict(test.sequences)
        print('Test Accuracy: ', Models.CustomSparseTopKCategoricalAccuracy(test.targets, y_pred))

def epistemic_uncertainty_model_training(data, spliter):
    max_epochs = 10
    batch_size = 512

    for between_user_technique in between_user_data_reduction_techniques:
        for retained_parcent in retained_user_parcentages:
            print('---------------------------------------------------------------------------------------')
            print('Technique:', between_user_technique, '--> Retained Parcentage:', retained_parcent)
            print('---------------------------------------------------------------------------------------')
            for fold in range(spliter.k_fold):
                print('Fold Id:', fold)
                train, validation, test = spliter.split_between_user_strategy(fold_id=fold, retained_parcent=retained_parcent, name=between_user_technique, between_users=between_users)

                nnRecModel = NextItNet_DistributedWeights(data, T=5)
                model = nnRecModel.get_model(drop_prob=0.1, isDropoutTraining=True)

                max_HR = 0
                for epochs in range(max_epochs):

                    train_model(model, train, validation, batch_size=batch_size)

                    hr = evaluate_model(model, test, T=10)
                    if hr > max_HR:
                        max_HR = hr
                        model.save_weights(nnRecModel.saved_dir + 'epistemic_uncertainty/' + between_user_technique + '/%.2f/best_models/'%(retained_parcent) + 'fold_%i_best_model'%(fold))
                        print('best_model_saved  with HR@20 =', hr)

                    print('Fold:', fold, ' Epochs:', epochs, ' Evaluation: HR@20: ', hr)
                    model.save_weights(nnRecModel.saved_dir + 'epistemic_uncertainty/' + between_user_technique + '/%.2f/'%(retained_parcent) + 'fold_%i_epoch_%i'%(fold, epochs))

if __name__ == '__main__':
    print('Hello Ruhani')
    data, spliter = init(sequence_length=100, target_length=1, seed=2022, k_fold=5)
    print(NextItNet_DistributedWeights(data, T=5).get_model(drop_prob=0.1, isDropoutTraining=True).summary())
    epistemic_uncertainty_model_training(data, spliter)


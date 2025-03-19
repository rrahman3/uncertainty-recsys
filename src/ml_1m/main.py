import numpy as np
import random
from Interactions import Interactions, InstanceSpliter
from Models import NextItNet_DistributedWeights
import Models
import tensorflow as tf
import pickle


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
    'nicest',  # [Done]
    'harshest',  # [Done]
    'most_active',  # [Done]
    'least_active',  # [Done]
    'most_consistent',  # [Done]
    'least_consistent',  # [Done]
    'most_open_minded',  # [Done]
    'least_open_minded',  # [Done]
    # 'most_typical',
    # 'least_typical',
]

retained_user_parcentages = [
    0.00,
    0.25,
    0.50,
    0.75,
    # 1.00
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


def aleatoric_uncertainty_full_model_training(data, spliter):
    max_epochs = 10
    batch_size = 512
    print('---------------------------------------------------------------------------------------')
    print('Aleatoric Uncertainty Model Training')
    print('---------------------------------------------------------------------------------------')
    for fold in range(spliter.k_fold):
        print('Fold Id:', fold)
        train, validation, test = spliter.split_data(fold_id=fold)
        nnRecModel = NextItNet_DistributedWeights(data, T=10)
        model = nnRecModel.get_model(drop_prob=0.1, isDropoutTraining=True)

        max_HR = 0
        for epochs in range(max_epochs):
            train_model(model, train, validation, batch_size=batch_size)

            hr = evaluate_model(model, test, T=10, requireAleatoricUncertainty=False)
            if hr > max_HR:
                max_HR = hr
                model.save_weights(nnRecModel.saved_dir + 'aleatoric_uncertainty/full_model/best_models/fold_%i_best_model' % (fold))
                print('best_model_saved  with HR@20 =', hr)

            print('Fold:', fold, ' Epochs:', epochs, ' Evaluation: HR@20: ', hr)
            model.save_weights(nnRecModel.saved_dir + 'aleatoric_uncertainty/full_model/fold_%i_epoch_%i' % (fold, epochs))

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

                    hr = evaluate_model(model, test, T=10, requireAleatoricUncertainty=False)
                    if hr > max_HR:
                        max_HR = hr
                        model.save_weights(nnRecModel.saved_dir + 'epistemic_uncertainty/' + between_user_technique + '/%.2f/best_models/'%(retained_parcent) + 'fold_%i_best_model'%(fold))
                        print('best_model_saved  with HR@20 =', hr)

                    print('Fold:', fold, ' Epochs:', epochs, ' Evaluation: HR@20: ', hr)
                    model.save_weights(nnRecModel.saved_dir + 'epistemic_uncertainty/' + between_user_technique + '/%.2f/'%(retained_parcent) + 'fold_%i_epoch_%i'%(fold, epochs))

def epistemic_uncertainty_model_evaluation(data, spliter):
    for reduction_name in between_user_data_reduction_techniques:
        metrics = np.zeros((spliter.k_fold, len(retained_user_parcentages) + 1, 2))
        user_epistemic_uncertainty = np.zeros((len(retained_user_parcentages) + 1, data.num_users))
        for fold in range(spliter.k_fold):
            print('Fold Id:', fold)
            train, validation, test = spliter.split_between_user_strategy(fold_id=fold, retained_parcent=1.00,
                                                                          name=reduction_name,
                                                                          between_users=between_users)
            nnRecModel = NextItNet_DistributedWeights(data, T=10)
            model = nnRecModel.get_model(drop_prob=0.1, isDropoutTraining=True)

            model_location = nnRecModel.saved_dir + 'epistemic_uncertainty/' + reduction_name + '/%.2f/best_models/fold_%i_best_model' % (1.00, fold)
            model.load_weights(model_location)

            hr_full, ep_un_full, per_user_ep_un_full = Models.montecarlo_prediction(model, test, T=10)
            user_epistemic_uncertainty[-1, test.users.squeeze()] = per_user_ep_un_full
            print('Full Model:', hr_full, ep_un_full)
            for ind, retained_parcent in enumerate(retained_user_parcentages):
                # train, validation, test = spliter.split_between_user_strategy(fold_id=fold, retained_parcent=retained_parcent, name=reduction_name, between_users=between_users)
                nnRecModel_Reduced = NextItNet_DistributedWeights(data, T=100)
                model = nnRecModel_Reduced.get_model(drop_prob=0.1, isDropoutTraining=True)

                model_location = nnRecModel_Reduced.saved_dir + 'epistemic_uncertainty/' + reduction_name + '/%.2f/best_models/fold_%i_best_model' % (retained_parcent, fold)
                model.load_weights(model_location)

                hr_reduced, ep_un_reduced, per_user_ep_un_reduced = Models.montecarlo_prediction(model, test, T=100)
                user_epistemic_uncertainty[ind, test.users.squeeze()] = per_user_ep_un_reduced
                print('Reduced Data Points:', retained_parcent, 'HR@20 and Epistemic Uncertainty:', hr_reduced, ep_un_reduced)

                hr, eu = hr_reduced / hr_full, np.log(ep_un_reduced / ep_un_full)

                metrics[fold, ind] = [hr, eu]
                print('p=%f percent data kept:: HR@20: %f, Varience: %f' % (retained_parcent, hr, eu))
            metrics[fold, -1] = [1, np.log(1)]  # 0==np.log(1)==np.log(var_full/var_full)

        for j, p in enumerate(retained_user_parcentages):
            location = nnRecModel_Reduced.saved_dir + 'epistemic_uncertainty/' + reduction_name + '_user_ep_un_kept_%.2f' % (p)
            with open(location, 'wb') as f:
                pickle.dump(user_epistemic_uncertainty[j], f)
        with open(nnRecModel_Reduced.saved_dir + 'epistemic_uncertainty/' + reduction_name + '_user_ep_un_kept_full', 'wb') as f:
            pickle.dump(user_epistemic_uncertainty[-1], f)

        metrics_dump = nnRecModel_Reduced.saved_dir + reduction_name + '_metrics.pkl'
        with open(metrics_dump, 'wb') as f:
            pickle.dump(metrics, f)


if __name__ == '__main__':
    print('Hello Ruhani')
    data, spliter = init(sequence_length=100, target_length=1, seed=2022, k_fold=5)
    print(NextItNet_DistributedWeights(data, T=5).get_model(drop_prob=0.1, isDropoutTraining=True).summary())
    # epistemic_uncertainty_model_training(data, spliter)
    epistemic_uncertainty_model_evaluation(data, spliter)


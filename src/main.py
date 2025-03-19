import numpy as np
import random
from codes.Interactions import Interactions, InstanceSpliter, ItemSequence, DataReductionStategis
from codes.Models import NextItNet_DistributedWeights, NextItNet_Dropout, NextItNet2D_DistributedWeights, GRU_Dropout
import codes.Models as Models
import codes.utils as utils
import tensorflow as tf
import pickle
import argparse

epistemic_resutls = 'results/NextItNet/epistemic_uncertainty/'
aleatoric_results = 'results/NextItNet/aleatoric_uncertainty/'
#################### Arguments ####################



# data, spliter = init(sequence_length=100, target_length=1, seed=2022, k_fold=5)


def train_model(model, train, validation, batch_size=512, max_epochs=1, verbose=1):
    train = ItemSequence(train.sequences, train.targets, batch_size)

    return model.fit(train.x,train.y,
        # x=train.sequences,
        # y=train.targets,
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


def aleatoric_model_training(args, data, spliter, nnRecSys):
    print('---------------------------------------------------------------------------------------')
    print('Aleatoric Uncertainty Model Training')
    print('---------------------------------------------------------------------------------------')
    for fold in range(spliter.k_fold):
        print('Fold Id:', fold)
        train, validation, test = spliter.split_data(fold_id=fold)
        # nnRecSys = NextItNet_DistributedWeights(data=data, args=args, T=10)
        nnRecSysModel = nnRecSys.get_model(drop_prob=0.25, isDropoutTraining=True)

        max_HR = 0
        for epochs in range(args.max_epochs):
            train_model(nnRecSysModel, train, validation, batch_size=args.batch_size)

            hr = evaluate_model(nnRecSysModel, test, T=10, requireAleatoricUncertainty=False)
            if hr > max_HR:
                max_HR = hr
                nnRecSysModel.save_weights(nnRecSys.saved_dir + 'alea/full/best_models/fold_%i_best_model' % (fold))
                print('best_model_saved  with HR@20 =', hr)

            print('Fold:', fold, ' Epochs:', epochs, ' Evaluation: HR@20: ', hr)
            # model.save_weights(nnRecModel.saved_dir + 'aleatoric_uncertainty/full_model/fold_%i_epoch_%i' % (fold, epochs))

def epistemic_uncertainty_model_training(data, spliter, reduction_strategy):
    max_epochs = 10
    batch_size = 512
    for between_user_technique in reduction_strategy.between_user_data_reduction_techniques:
        for retained_parcent in reduction_strategy.retained_user_parcentages:
            print('---------------------------------------------------------------------------------------')
            print('Technique:', between_user_technique, '--> Retained Parcentage:', retained_parcent)
            print('---------------------------------------------------------------------------------------')
            for fold in range(spliter.k_fold):
                print('Fold Id:', fold)
                train, validation, test = spliter.split_between_user_strategy(fold_id=fold, retained_parcent=retained_parcent, name=between_user_technique, between_users=reduction_strategy.between_users)

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

def epistemic_uncertainty_model_evaluation(data, spliter, reduction_strategy):
    for reduction_name in reduction_strategy.between_user_data_reduction_techniques:
        metrics = np.zeros((spliter.k_fold, len(reduction_strategy.retained_user_parcentages) + 1, 2))
        user_epistemic_uncertainty = np.zeros((len(reduction_strategy.retained_user_parcentages) + 1, data.num_users))
        for fold in range(spliter.k_fold):
            print('Fold Id:', fold)
            train, validation, test = spliter.split_between_user_strategy(fold_id=fold, retained_parcent=1.00,
                                                                          name=reduction_name,
                                                                          between_users=reduction_strategy.between_users)
            nnRecModel = NextItNet_DistributedWeights(data, T=10)
            model = nnRecModel.get_model(drop_prob=0.1, isDropoutTraining=True)

            model_location = nnRecModel.saved_dir + 'epistemic_uncertainty/' + reduction_name + '/%.2f/best_models/fold_%i_best_model' % (1.00, fold)
            model.load_weights(model_location)

            hr_full, ep_un_full, per_user_ep_un_full = Models.montecarlo_prediction(model, test, T=10)
            user_epistemic_uncertainty[-1, test.users.squeeze()] = per_user_ep_un_full
            print('Full Model:', hr_full, ep_un_full)
            for ind, retained_parcent in enumerate(reduction_strategy.retained_user_parcentages):
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

        for j, p in enumerate(reduction_strategy.retained_user_parcentages):
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
    args = utils.parse_args()
    args.dataset = 'reviews_Books_300K_sequences'
    # args.dataset = 'amazon-cloth'
    args.seqlen = 20
    args.embed_dims = 64

    # args.residual_channels = 256
    # args.dataset = 'ml-1m'
    dataset = args.path + args.dataset

    print(args)
    # data = Interactions(args).process_dataset()
    data, spliter = utils.init(args, k_fold=5)
    reduction_strategy = DataReductionStategis(data=data)
    train, validation, test = spliter.split_data(fold_id=0)
    if np.min(train.targets.squeeze()) < 0:
        print('Fuck You')
    nnRecSys = NextItNet_DistributedWeights(args, data, T=10)
    nnRecSysModel = nnRecSys.get_model(drop_prob=0.25, isDropoutTraining=True)
    print(nnRecSysModel.summary())
    # aleatoric_model_training(args, data, spliter, nnRecSys)
    # for iter in range(args.max_epochs):
    #     train_model(nnRecSysModel, train, validation, batch_size=args.batch_size)
    # for reduction_name in reduction_strategy.between_user_data_reduction_techniques:
    #     for reduction_strategy.between_user_data_reduction_techniques
    epistemic_uncertainty_model_training(data, spliter, reduction_strategy)
    # epistemic_uncertainty_model_evaluation(data, spliter)


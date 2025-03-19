# import pandas as pd
import numpy as np
import random
import pickle
import logging

import collections
from collections import defaultdict, Counter

# import scipy.sparse as sp
# from sklearn.metrics import ndcg_score, dcg_score

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.autograd import Variable

import argparse
from time import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# from google.colab import drive

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers

from tensorflow.keras import backend as K
from tensorflow import keras
# used to fix the tf error "ValueError: tf.function-decorated function tried to create variables on non-first call."

tf.config.run_functions_eagerly(True)
import logging
tf.get_logger().setLevel(logging.ERROR)

tf.__version__, tfp.__version__

import matplotlib.pyplot as plt
# %matplotlib inline

import time

import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


# utils.py

def set_seed(seed, cuda=False):

    np.random.seed(seed)
    random.seed(seed)
    # if cuda:
    #     torch.cuda.manual_seed(seed)
    # else:
    #     torch.manual_seed(seed)

set_seed(2022)

# connet to Drive
data_root = ''


class DriveConnection:
    def __init__(self):
        # from google.colab import drive
        # drive.mount('/content/drive')
        pass

    def get_root(self, data_location='Riyi'):
        # change working directory
        if data_location == 'Riyi':
            # %cd /content/drive/My Drive/ColabData/UncertaintyRecommenderSystem/RiyiQiu_Uncertainty_RecSys/data&script/
            # %ls
            data_root = '/content/drive/My Drive/ColabData/UncertaintyRecommenderSystem/RiyiQiu_Uncertainty_RecSys/data&script/'

        elif data_location == 'CosRec':
            # %cd /content/drive/My Drive/RuhaniRahman/UncertaintyRecommenderSystem/data
            # data_root = '/content/drive/My Drive/RuhaniRahman/UncertaintyRecommenderSystem/data'
            data_root = 'data'

        else:
            # %cd /content/drive/My Drive/RuhaniRahman/UncertaintyRecommenderSystem/data
            data_root = '/content/drive/My Drive/RuhaniRahman/UncertaintyRecommenderSystem/data'
            # data_root = '/content/drive/My Drive/Colab Notebooks/RuhaniRahman/UncertaintyRecommenderSystem/data'
            data_root = 'data'

        return data_root


class Instances:
    def __init__(self):
        self.sequences = None
        self.users = None
        self.targets = None  # movie ids
        self.item_samples = None
        self.y_true = None

    def shuffle(self):

        shuffle_indices = np.arange(len(self.sequences))
        np.random.shuffle(shuffle_indices)

        self.sequences = self.sequences[shuffle_indices]
        self.users = self.users[shuffle_indices]
        self.targets = self.targets[shuffle_indices]

        if self.item_samples is not None:
            self.item_samples = self.item_samples[shuffle_indices]
        if self.y_true is not None:
            self.y_true = self.y_true[shuffle_indices]


class Interactions:
    def __init__(self, data_root, sequence_length=5, target_length=1):
        self.user_interactions = []
        self.user_ratings = []
        self.train = Instances()
        self.test = Instances()
        self.data_root = data_root
        self.num_users = 0
        self.num_items = 0
        self.num_sequences = 0

        self.sequence_length = sequence_length
        self.target_length = target_length

    def data_preprocess(self):
        print('Root', self.data_root)
        self._read_file(self.data_root + '/ml-1m/train_set_sequences')
        self._read_file(self.data_root + '/ml-1m/test_set_sequences')
        self._read_file(self.data_root + '/ml-1m/val_set_sequences')

        self.user_interactions = np.array(self.user_interactions)
        self.user_ratings = np.array(self.user_ratings)

        self.num_users = self.user_interactions.shape[0]
        self.num_items = np.max(np.concatenate(self.user_interactions[:]))

        print('num_users = ', self.num_users)
        print('num_items = ', self.num_items)

        self.x_n_ratings = np.array([self.user_ratings[user_id].shape[0] for user_id in range(self.num_users)])
        self.x_var_ratings = np.array([np.var(self.user_ratings[user_id]) for user_id in range(self.num_users)])
        self.x_avg_ratings = np.array([np.mean(self.user_ratings[user_id]) for user_id in range(self.num_users)])

        with open(self.data_root + '/ml-1m/user_similarity.pkl', 'rb') as f:
            self.x_similarity = pickle.load(f)

        movie_mapping = {}
        with open(self.data_root + '/ml-1m/item_id_mapping', 'rb') as f:
            f.readline()
            for l in f:
                origin, new = l.strip().split()
                movie_mapping[int(new) + 1] = int(origin)

        # print(movie_mapping)
        movie_genres = {}
        with open(self.data_root + '/ml-1m/movies.dat', 'rb') as f:
            for l in f:
                # print(l)
                mid, _, genre = l.strip().decode('latin-1').split('::')
                # print(mid, genre)
                movie_genres[int(mid)] = genre
        # movie_genres

        self.x_n_genres = np.zeros(self.num_users)
        for user_id in range(self.num_users):
            genres = []
            for movie_id in self.user_interactions[user_id]:
                genres += movie_genres[movie_mapping[movie_id]].split('|')
            self.x_n_genres[user_id] = len(set(genres))

    def _read_file(self, file_path):
        with open(file_path, 'r') as fin:
            for line in fin:
                item_ratings = line.strip().split()[1:]
                item_ratings_np = np.array(item_ratings).astype(np.int64).reshape(-1, 2)

                item_ratings_np[:, 0] += 1  ### Adding 1 to the item_id, 0 reads as padding
                item_ratings_np[:, 1] *= 2  ### Rating is multiplied by 2, now counts as 10

                ### users must have rated at least 15 items
                if item_ratings_np.shape[0] >= 15:
                    self.user_interactions.append(item_ratings_np[:, 0])
                    self.user_ratings.append(item_ratings_np[:, 1])

    def create_test_train_instances(self):
        self._create_test_instances()
        self._create_train_instances()

    def _create_test_instances(self):

        ### Creating last item as prediction item
        self.test.targets = []

        ### Targets compaible with softmax output
        # for interactions in self.user_interactions:
        #     T = np.zeros(self.num_items)
        #     T[interactions[-self.target_length:]-1] = 1
        #     self.test.targets.append(T)
        # self.test.targets = np.array(self.test.targets).reshape(-1, self.num_items)
        # print('test.targets.shape:', len(self.test.targets))

        ### Test targets are compaible with sigmoid output and sparse softmax out
        for interactions in self.user_interactions:
            # self.test.targets.append(interactions[-self.target_length:])
            self.test.targets.append(interactions[-self.target_length:] - 1)
        self.test.targets = np.array(self.test.targets).reshape(-1, self.target_length)
        print('test.targets.shape:', self.test.targets.shape)

        ### users
        self.test.users = np.arange(self.num_users).reshape(-1, 1)
        print('test.users.shape:', self.test.users.shape)

        ###
        self.test.sequences = []
        for interaction in self.user_interactions:
            # interaction = interaction[:-self.target_length]
            if len(interaction) < self.sequence_length:
                num_paddings = self.sequence_length - len(interaction)
                interaction = np.pad(interaction, (num_paddings, 0), 'constant')
            else:
                interaction = interaction[-self.sequence_length:]

            self.test.sequences.append(interaction)
        self.test.sequences = np.array(self.test.sequences)
        print('test.sequences.shape:', self.test.sequences.shape)

    def _create_train_instances(self):
        max_sequence_length = self.sequence_length + self.target_length
        self.train.sequences = []
        self.train.users = []
        self.train.targets = []

        self.user_sequence_indexes = [list() for i in range(self.num_users)]

        sequence_index = 0
        for user_id, interaction in enumerate(self.user_interactions):
            # interaction = interaction[:-self.target_length]

            if len(interaction) < max_sequence_length:
                num_paddings = max_sequence_length - len(interaction)
                interaction = np.pad(interaction, (num_paddings, 0), 'constant')

                self._add_sequence(interaction[:self.sequence_length],
                                   user_id,
                                   interaction[-self.target_length:],
                                   sequence_index)
                sequence_index += 1

            else:
                for ind in range(len(interaction), max_sequence_length - 1, -1):
                    temp_sequence = interaction[ind - max_sequence_length: ind]
                    self._add_sequence(temp_sequence[:self.sequence_length],
                                       user_id,
                                       temp_sequence[-self.target_length:],
                                       sequence_index)
                    sequence_index += 1

        self.train.sequences = np.array(self.train.sequences)
        self.train.users = np.array(self.train.users).reshape(-1, 1)
        self.train.targets = np.array(self.train.targets)

        for user_id in range(self.num_users):
            self.user_sequence_indexes[user_id] = np.array(self.user_sequence_indexes[user_id])
        self.user_sequence_indexes = np.array(self.user_sequence_indexes)

        self.num_sequences = sequence_index

    def _add_sequence(self, seq, usr, target, sequence_index):
        self.train.sequences.append(seq)
        self.train.users.append(usr)

        # target_vector = np.zeros(self.num_items)
        # target_vector[target-1] = 1
        # self.train.targets.append(target_vector)

        # self.train.targets.append(target)
        self.train.targets.append(target - 1)
        self.user_sequence_indexes[usr].append(sequence_index)


def _generate_negative_samples(data, n):
    """
    Sample negative from a candidate set of each user. The
    candidate set of each user is defined by:
    {All Items} \ {Items Rated by User}

    Parameters
    ----------

    users: array of np.int64
        sequence users
    interactions: :class:`spotlight.interactions.Interactions`
        training instances, used for generate candidates
    n: int
        total number of negatives to sample for each sequence
    """
    negative_candidate_item = dict()
    users_ = data.train.users.squeeze()
    negative_samples = np.zeros((users_.shape[0], n), np.int64)

    ### set candidate items for negetive samples for each user
    all_items = np.arange(data.num_items - 1) + 1  # 0 for padding
    for user_id in range(data.num_users):
        negative_candidate_item[user_id] = list(set(all_items) - set(data.user_interactions[user_id]))

    for i, u in enumerate(users_):
        for j in range(n):
            negative_item = negative_candidate_item[u]
            negative_samples[i, j] = negative_item[np.random.randint(len(negative_item))]

    return negative_samples


# returns both positive and negative targets with their labels
def get_target_samples(data, num_negative_samples=1):
    negatives_np = _generate_negative_samples(data, num_negative_samples)
    data.train.item_samples = np.concatenate((data.train.targets, negatives_np), axis=1)

    y_pos = np.ones(data.train.targets.shape[0]).reshape(-1, 1)
    y_neg = np.zeros((negatives_np.shape[0], num_negative_samples)).reshape(-1, num_negative_samples)

    data.train.y_true = np.concatenate((y_pos, y_neg), axis=1)

    # return item_samples, y_true


class InstanceSpliter:

    def __init__(self, data, k_fold=5):
        self.data = data
        self.k_fold = k_fold

    def generate_folds(self):
        n_user_fold = self.data.num_users // self.k_fold
        user_ids = np.arange(self.data.num_users)
        np.random.shuffle(user_ids)

        ### Find folds for users
        self.user_fold_indexes = []
        for fold in range(self.k_fold):
            self.user_fold_indexes.append(user_ids[fold * n_user_fold: (fold + 1) * n_user_fold])
        self.user_fold_indexes = np.array(self.user_fold_indexes)

    def get_sequence_indexes(self, fold_indexes):
        # print(fold_indexes)
        return np.concatenate(self.data.user_sequence_indexes[fold_indexes])

    def get_sequence(self, sequence_indexes, isTest=False):
        _instances = Instances()
        if isTest:
            _split_instances = self.data.test
        else:
            _split_instances = self.data.train

        _instances.users = _split_instances.users[sequence_indexes]
        _instances.sequences = _split_instances.sequences[sequence_indexes]
        _instances.targets = _split_instances.targets[sequence_indexes]
        if not isTest:
            if _instances.item_samples is not None:
                _instances.item_samples = _split_instances.item_samples[sequence_indexes]
            if _instances.y_true is not None:
                _instances.y_true = _split_instances.y_true[sequence_indexes]

        return _instances

    def split_data(self, fold_id=0):
        if fold_id >= self.k_fold:
            raise Exception('Fold Id must be less than the k_fold')

        test_user_indexes = self.user_fold_indexes[fold_id]
        train_user_indexes = np.array(list(set(np.arange(self.data.num_users)) - set(self.user_fold_indexes[fold_id])))

        return self.create_split(train_user_indexes, test_user_indexes)

    def create_split(self, train_user_indexes, test_user_indexes):
        train_sequence_indexes = self.get_sequence_indexes(train_user_indexes)
        validation_sequence_indexes = self.get_sequence_indexes(test_user_indexes)

        train_set = self.get_sequence(train_sequence_indexes)
        validation_set = self.get_sequence(validation_sequence_indexes)
        test_set = self.get_sequence(test_user_indexes, isTest=True)

        # if len(set(train_set.users.squeeze()).intersection(set(test_set.users.squeeze())))==0:
        # if len(set(train_user_indexes).intersection(set(test_user_indexes))) == 0:
        #     print('Well Done')
        # else:
        #     print('Error Splitting')
        print(train_set.users.shape, validation_set.users.shape, np.unique(train_set.users.squeeze()).shape,
              test_set.users.shape)

        train_set.shuffle()
        validation_set.shuffle()

        return train_set, validation_set, test_set

    def split_within_user_strategy(self, fold_id=0, data_point_kept=1, name='most_recent'):
        pass

    '''
    isMost: True will return the users that meet the criteria higher than the threshold, False will do the oppostie
    For example, if isMost=True, and threshold=7.5 and the user wants the nicest user,
    The method will choose the data between user data reduction technique in such a way that it will take the 'p' percent of users whose average ratings is higher than the threshold 7.5
    and take all the users whose average ratings is less than the threshold.
    For the training set, it will only take the users whose given average ratings are higher than the threshold 7.5

    Riyi Qui set the threshold for different user data type:
        Nicest User --> Average Ratings > 7.5 (x_avg_ratings > 7.5) 
            #threshold=7.50, split_array=data.x_avg_ratings, isMost=True, name='Nicest User'
        Harshest User --> Average Ratings <= 7.5 (x_avg_ratings <= 7.5) 
            #threshold=7.50, split_array=data.x_avg_ratings, isMost=False, name='Harshest User'

        Most Active User --> Number of Ratings > 100 (x_n_ratings > 100) 
            #threshold=100.00, split_array=data.x_n_ratings, isMost=True, name='Most Active User'
        Least Active User --> Number of Ratings <= 100 (x_n_ratings <= 100) 
            #threshold=100.00, split_array=data.x_n_ratings, isMost=False, name='Least Active User'

        Most Typical User --> Similarity Score > 0.13 (x_similarity > 0.13) 
            #threshold=0.13, split_array=data.x_similarity, isMost=True, name='Most Typical User'
        Least Typical User --> Similarity Score <= 0.13 (x_similarity <= 0.13) 
            #threshold=0.13, split_array=data.x_similarity, isMost=False, name='Least Typical User'

        Most Consistent User --> Varience Ratings > 4 (x_var_ratings > 4) 
            #threshold=4.00, split_array=data.x_var_ratings, isMost=True, name='Most Consistent User'
        Least Consistent User --> Varience Ratings <= 4 (x_var_ratings <= 4) 
            #threshold=4.00, split_array=data.x_var_ratings, isMost=False, name='Least Consistent User'

        Most Open Minded User --> Number of Genres > 16 (x_n_genres > 4) 
            #threshold=16.00, split_array=data.x_n_genres, isMost=True, name='Most Open Minded User'
        Least Open Minded User --> Number of Genres <= 16 (x_n_genres <= 4) 
            #threshold=16.00, split_array=data.x_n_genres, isMost=False, name='Least Open Minded User'

    '''

    # def split(self, data, split_array, fold_id=0, threshold=7.5, retained_parcent=0.50, isMost=True, name='nicest user'):
    def split_between_user_strategy(self, between_users, fold_id=0, retained_parcent=0.50, name='nicest'):

        threshold = between_users[name]['threshold']
        isMost = between_users[name]['isMost']
        split_array = between_users[name]['split_array']
        print(name, threshold, isMost, split_array)

        test_user_indexes = self.user_fold_indexes[fold_id]
        if isMost:
            test_user_indexes = test_user_indexes[np.where(split_array[test_user_indexes] > threshold)[0]]
        else:
            test_user_indexes = test_user_indexes[np.where(split_array[test_user_indexes] <= threshold)[0]]
        # print(test_user_indexes)
        test_fold_indexes = self.user_fold_indexes[fold_id]
        if isMost:
            idx_non_target_user = np.where(split_array <= threshold)[0]
            idx_target_user = np.where(split_array > threshold)[0]
        else:
            idx_non_target_user = np.where(split_array > threshold)[0]
            idx_target_user = np.where(split_array <= threshold)[0]

        idx_non_target_user = np.array(list(set(idx_non_target_user) - set(test_fold_indexes)))
        idx_target_user = np.array(list(set(idx_target_user) - set(test_user_indexes)))
        idx_target_user = np.random.permutation(idx_target_user)[:int(len(idx_target_user) * retained_parcent)]
        train_user_indexes = np.concatenate([idx_non_target_user, idx_target_user])

        return self.create_split(train_user_indexes, test_user_indexes)


def init(seed=2022, sequence_length=10, target_length=1, k_fold=5):
    data_root = DriveConnection().get_root('CosRec')  # ['Riyi', 'CosRec']
    set_seed(seed)

    # load dataset
    instances = Interactions(data_root, sequence_length=sequence_length, target_length=target_length)
    instances.data_preprocess()
    instances.create_test_train_instances()

    # get_target_samples(instances, num_negative_samples=1)

    instance_spliter = InstanceSpliter(data=instances, k_fold=k_fold)
    instance_spliter.generate_folds()

    return instances, instance_spliter

num_items = 3416
def CustomSparseCategoricalCrossentropyLoss(y_true, y_pred):
    T = 10
    varience = y_pred[:, -1]
    before_softmax = y_pred[:, :-1]

    std = tf.sqrt(varience)
    dist = tfd.Normal(loc=K.zeros_like(std), scale=std)

    iterable = K.variable(np.ones(T))
    monte_carlo_results = K.map_fn(crossentropy_loss(y_true, before_softmax, dist, num_items), elems=iterable, name='monte_carlo_results')
    variance_loss = K.mean(monte_carlo_results, axis=0)

    return K.log(variance_loss)

def crossentropy_loss(y_true, y_pred, dist, n_category):
    def map_fn(i):
        std_samples = K.transpose(dist.sample(n_category))
        # std_samples = K.reshape(std_samples, shape=(tf.shape(y_pred)[0], n_category))
        distorted_loss = K.sparse_categorical_crossentropy(y_true, y_pred + std_samples, from_logits=True) #from_logis=True --> applies softmax first
        return K.exp(distorted_loss)
    return map_fn

def CustomSparseTopKCategoricalAccuracy(y_true, y_pred):

    varience = tf.reshape(y_pred[:,-1], (-1,1))
    y_pred = y_pred[:,:-1]
    y_pred = tf.nn.softmax(y_pred)

    m = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=20)
    m.update_state(y_true, y_pred)
    return m.result().numpy()

epsilon = 0.000001

def montecarlo_prediction(model, test_instances, T=100):
    predictions = np.array([model.predict(test_instances.sequences, verbose=0) for _ in range(T)]) #(T, batch_size, (num_items+softplus))
    prediction_probabilities = np.mean(predictions, axis=0) #(T, batch_size, (num_items+softplus))

    predictions = prediction_probabilities[:, :-1] #(batch_size, num_items)

    prediction_variances = np.apply_along_axis(predictive_entropy, axis=1, arr=prediction_probabilities) #(batch_size, 1), varience of each user

    return (prediction_probabilities, prediction_variances)

def predictive_entropy(prob):
    return -1 * np.sum(np.log(prob+epsilon) * (prob))


# divergence_fn = lambda q, p, _: tfd.kl_divergence(q, p) / train_instances.sequences.shape[0]


class NextItNet:
    def __init__(self, data=None, sequence_length=10, T=10):
        self.T = T
        if not data:
            self.data = None
            self.sequence_length = sequence_length
            self.num_users = 6040
            self.num_items = 3416  # tf.keras.layers.InputLayer((28, 28, 1)),
        else:
            self.data = data
            self.sequence_length = data.sequence_length
            self.num_users = data.num_users
            self.num_items = data.num_items  # Consider padding 0 a prediction

    def get_model(self, isDropoutTraining=False):
        divergence_fn = lambda q, p, _: tfd.kl_divergence(q, p) / self.data.train.sequences.shape[0]
        embed_dim = 64  # 256
        embed_size = 64  # 256
        dilations = [1, 2, 4]  # [1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4]
        residual_channels = 32  # 256
        kernel_size = 3  # 3

        seq_input = tf.keras.layers.Input(shape=(self.sequence_length,), name='seq_input')

        # (batch_size, sequence_length, embed_dim) --> (None, 100, 64)
        seq_embedding = tf.keras.layers.Embedding(
            input_dim=self.num_items+1,
            output_dim=embed_size,
            name='seq_embedding',
            embeddings_initializer='uniform',
            # embeding.weight.data.uniform_(-stdv, stdv) # important initializer, stdv = np.sqrt(1. / self.item_size)
            embeddings_regularizer=tf.keras.regularizers.L2(0)
        )(seq_input)  # (batch_size, seq_len, embed_dim) = (None, 100, 64)

        conv_layer = seq_embedding
        # conv_layer = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(seq_embedding) #(batch_size, seq_len, embed_dim, 1) (None, 100, 64, 1)

        for dilation in dilations:  # [1, 2, 4]

            conv_layer = tfp.layers.Convolution1DReparameterization(
                # conv_layer = tf.keras.layers.Conv1D(
                residual_channels,
                kernel_size=kernel_size,
                dilation_rate=dilation,
                padding='valid',
                activation='relu',
                kernel_prior_fn=tfpl.default_multivariate_normal_fn,
                kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                kernel_divergence_fn=divergence_fn,
                bias_prior_fn=tfpl.default_multivariate_normal_fn,
                bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                bias_divergence_fn=divergence_fn,
            )(
                conv_layer)  # (batch_size, seq_len, embed_dim, residual_channels) = (None, 100, 62, 32), (None, 100, 54, 32), (None, 100, 38, 32)

            # conv_layer = tfp.layers.Convolution1DReparameterization(
            # # conv_layer = tf.keras.layers.Conv1D(
            #         residual_channels,
            #         kernel_size=kernel_size,
            #         dilation_rate=dilation * 2,
            #         padding='valid',
            #         activation='relu',
            #         kernel_prior_fn = tfpl.default_multivariate_normal_fn,
            #         kernel_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular=False),
            #         kernel_divergence_fn = divergence_fn,
            #         bias_prior_fn = tfpl.default_multivariate_normal_fn,
            #         bias_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular=False),
            #         bias_divergence_fn = divergence_fn,
            #     )(conv_layer) # (batch_size, seq_len, embed_dim, residual_channels) = (None, 100, 58, 32), (None, 100, 46, 32), (None, 100, 22, 32)

        flatten_layer = tf.reshape(conv_layer[:, -1, ], (-1, residual_channels))  # (batch_size, embed_dim) = (None, 32)
        # fully_connected_layer = tf.keras.layers.Dense(residual_channels)(flatten_layer)
        variance_layer = tf.keras.layers.Dense(1, activation='softplus', name='variance')(flatten_layer)

        fully_connected_layer = tf.keras.layers.Dense(self.num_items)(
            flatten_layer)  # (batch_size, num_items) = (None, 3417)

        # softmax_output = tf.keras.layers.Activation(activation=tf.nn.softmax,name='softmax_output')(fully_connected_layer)
        # logits_variance = tf.keras.layers.concatenate([softmax_output, variance_layer], name='logits_variance')
        logits_variance = tf.keras.layers.concatenate([fully_connected_layer, variance_layer], name='logits_variance')

        model = tf.keras.Model(
            inputs=seq_input,
            outputs=logits_variance,
            # outputs=logits_variance,
        )
        ### model compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3, decay=1e-4),
            loss=CustomSparseCategoricalCrossentropyLoss,
            metrics=[CustomSparseTopKCategoricalAccuracy],
        )

        return model

# model = NextItNet(data, T=100).get_model()
# model.summary()

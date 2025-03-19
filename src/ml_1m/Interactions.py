import numpy as np
import pickle
import random

data_root = 'data'

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
            interaction = interaction[:-self.target_length]  # Most important line
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
            interaction = interaction[:-self.target_length]  # Most important line

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

import main
if __name__ == '__main__':
    d, s = main.init()
    print(s.user_fold_indexes)
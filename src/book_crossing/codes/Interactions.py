import numpy as np
import pickle
from collections import Counter

data_root = 'data'

class Instances:
    def __init__(self, name=''):
        self.name=name
        self.sequences = []
        self.users = []
        self.targets = []  # movie ids

    def shuffle(self):
        shuffle_indices = np.arange(len(self.sequences))
        np.random.shuffle(shuffle_indices)

        self.sequences = self.sequences[shuffle_indices]
        self.users = self.users[shuffle_indices]
        self.targets = self.targets[shuffle_indices]

    def pprint(self):
        print('Data Instances Shape:', self.name, end='\n')
        print('Sequences:', self.sequences.shape, 'Users:', self.users.shape, 'Targets:', self.targets.shape)

class Interactions:
    def __init__(self, args):
        self.args = args
        self.sequence_length = args.seqlen
        self.target_length = args.tarlen
        self.data_root = args.path + args.dataset

        self.user_interactions = []
        self.user_ratings = []
        self.train = Instances(name='Train Instances')
        self.test = Instances(name='Test Instances')
        self.num_users = 0
        self.num_items = 0
        self.num_sequences = 0

    def process_dataset(self):
        print('Root', self.data_root)
        if self.args.dataset == 'ml-1m':
            self._process_ml_1m_data()
        elif self.args.dataset == 'book-crossing':
            # self._process_book_crossing_data()
            self._read_file(self.data_root + '/train_set_sequences')
        else:
            print('Reading Files from', self.args.path + 'reviews_Clothing_Shoes_and_Jewelry_5_core_sequences')
            self._read_file(self.args.path + 'reviews_Clothing_Shoes_and_Jewelry_5_core_sequences')

        self.user_interactions = np.array(self.user_interactions, dtype=object)
        self.user_ratings = np.array(self.user_ratings, dtype=object)

        self.num_users = self.user_interactions.shape[0]
        self.num_items = int(np.max(np.concatenate(self.user_interactions[:])))

        print('num_users = ', self.num_users)
        print('num_items = ', self.num_items)

        self.x_n_ratings = np.array([self.user_ratings[user_id].shape[0] for user_id in range(self.num_users)])
        xn = Counter(self.x_n_ratings)
        print(xn)


        # self.generate_characteristics()

    def _process_book_crossing_data(self):
        file_path = self.data_root + '/BX-Book-Ratings.csv'
        users_dict = dict()
        items_dict = dict()
        user_count = 0
        item_count = 1
        rating_count = 0
        temp_interactions = []

        with open(file_path, mode='r', encoding='latin-1') as fin:
            for line in fin:
                rating_count += 1
                user, item, rating = line.strip().split(';')
                if user not in users_dict:
                    users_dict[user] = user_count
                    user_count += 1
                if item not in items_dict:
                    items_dict[item] = item_count
                    item_count += 1
                rating = int(rating[1:-1])


                temp_interactions.append((users_dict[user], items_dict[item], rating))

        self.user_interactions = [np.array([], dtype=int) for _ in range(user_count-1)]
        self.user_ratings = [np.array([], dtype=int) for _ in range(user_count-1)]

        for user, item, rating in temp_interactions:
            # print(user, item, rating)
            self.user_interactions[user-1] = np.append(self.user_interactions[user-1], int(item))
            self.user_ratings[user-1] = np.append(self.user_ratings[user-1], int(rating))


        filtered_user_dict = dict()
        filtered_item_dict = dict()
        new_user_count = 0
        new_item_count = 1
        all_temp = []
        for users, item_ratings in enumerate(zip(self.user_interactions, self.user_ratings)):
            if len(item_ratings[0]) < 5:
                continue
            filtered_user_dict[users] = new_user_count
            new_user_count += 1
            for item in item_ratings[0]:
                if item not in filtered_item_dict:
                    filtered_item_dict[item] = new_item_count
                    new_item_count += 1

            items = [filtered_item_dict[item] for item in item_ratings[0]]
            item_ratings = np.concatenate([np.array(items).reshape(-1, 1), np.array(item_ratings[1]).reshape(-1, 1)], axis=1)
            # item_ratings = np.array(item_ratings).reshape(-1, 2)
            temp = np.array([filtered_user_dict[users]])
            temp = np.append(temp, item_ratings)
            all_temp.append(temp)


        with open(self.data_root + '/train_set_sequences', 'w') as f:
            for tt in all_temp:
                f.write(" ".join(str(x) for x in tt))
                f.write("\n")


        print(item_ratings)



    def _process_ml_1m_data(self):
        self._read_file(self.data_root + '/train_set_sequences')
        self._read_file(self.data_root + '/test_set_sequences')
        self._read_file(self.data_root + '/val_set_sequences')

    def generate_characteristics(self):
        self.x_n_ratings = np.array([self.user_ratings[user_id].shape[0] for user_id in range(self.num_users)])
        self.x_var_ratings = np.array([np.var(self.user_ratings[user_id]) for user_id in range(self.num_users)])
        self.x_avg_ratings = np.array([np.mean(self.user_ratings[user_id]) for user_id in range(self.num_users)])

        if self.args.dataset == 'ml-1m':
            with open(self.data_root + '/user_similarity.pkl', 'rb') as f:
                self.x_similarity = pickle.load(f)

            movie_mapping = {}
            with open(self.data_root + '/item_id_mapping', 'rb') as f:
                f.readline()
                for l in f:
                    origin, new = l.strip().split()
                    movie_mapping[int(new) + 1] = int(origin)

            # print(movie_mapping)
            movie_genres = {}
            with open(self.data_root + '/movies.dat', 'rb') as f:
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

                if self.args.dataset=='ml-1m':
                    item_ratings_np[:, 0] += 1  ### Adding 1 to the item_id, 0 reads as padding
                    item_ratings_np[:, 1] *= 2  ### Rating is multiplied by 2, now counts as 10
                # elif self.args.dataset=='book-crossing':


                ### users must have rated at least 15 items
                # if item_ratings_np.shape[0] >= 15:
                self.user_interactions.append(item_ratings_np[:, 0])
                self.user_ratings.append(item_ratings_np[:, 1])

    def create_instances(self):
        self._create_test_instances()
        self._create_train_instances()

    def _create_test_instances(self):

        ### Creating last item as prediction item
        ### Test targets are compaible with sigmoid output and sparse softmax out
        for user_id, interaction in enumerate(self.user_interactions):
            if len(interaction) < 3:
                continue
            ### Users
            self.test.users.append(user_id)

            ### Targets
            self.test.targets.append(interaction[-self.target_length:] - 1)

            ### Sequences
            interaction = interaction[:-self.target_length]  # Most important line

            if len(interaction) < self.sequence_length:
                num_paddings = self.sequence_length - len(interaction)
                interaction = np.pad(interaction, (num_paddings, 0), 'constant')
            else:
                interaction = interaction[-self.sequence_length:]

            self.test.sequences.append(interaction)

        self.test.targets = np.array(self.test.targets).reshape(-1, self.target_length)
        ### users
        self.test.users = np.array(self.test.users).reshape(-1, 1)
        self.test.sequences = np.array(self.test.sequences)

        self.test.pprint()

    def _create_train_instances(self):
        max_sequence_length = self.sequence_length + self.target_length

        self.user_sequence_indexes = [list() for i in range(self.num_users)]

        sequence_index = 0
        for user_id, interaction in enumerate(self.user_interactions):
            if len(interaction) < 3:
                continue
            interaction = interaction[:-self.target_length]  # Most important line

            if len(interaction) < max_sequence_length:
                num_paddings = max_sequence_length - len(interaction)
                interaction = np.pad(interaction, (num_paddings, 0), 'constant')

                self._add_sequence(seq=interaction[:self.sequence_length], usr=user_id, target=interaction[-self.target_length:], sequence_index=sequence_index)
                sequence_index += 1

            else:
                for ind in range(len(interaction), max_sequence_length - 1, -1):
                    temp_sequence = interaction[ind - max_sequence_length: ind]
                    self._add_sequence(seq=temp_sequence[:self.sequence_length], usr=user_id, target=temp_sequence[-self.target_length:], sequence_index=sequence_index)
                    sequence_index += 1

        self.train.sequences = np.array(self.train.sequences)
        self.train.users = np.array(self.train.users).reshape(-1, 1)
        self.train.targets = np.array(self.train.targets)
        self.train.pprint()

        for user_id in range(self.num_users):
            self.user_sequence_indexes[user_id] = np.array(self.user_sequence_indexes[user_id])
        self.user_sequence_indexes = np.array(self.user_sequence_indexes, dtype=object)

        self.num_sequences = sequence_index

    def _add_sequence(self, seq, usr, target, sequence_index):
        self.train.sequences.append(seq)
        self.train.users.append(usr)
        if target == 0:
            self.train.targets.append(target)
        else:
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

    def get_sequence(self, sequence_indexes, isTest=False, name=''):
        _instances = Instances(name=name)
        if isTest:
            _split_instances = self.data.test
        else:
            _split_instances = self.data.train

        _instances.users = _split_instances.users[sequence_indexes]
        _instances.sequences = _split_instances.sequences[sequence_indexes]
        _instances.targets = _split_instances.targets[sequence_indexes]

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

        train_set = self.get_sequence(train_sequence_indexes, name='train')
        validation_set = self.get_sequence(validation_sequence_indexes, name='validation')
        test_set = self.get_sequence(test_user_indexes, isTest=True, name='test')

        # if len(set(train_set.users.squeeze()).intersection(set(test_set.users.squeeze())))==0:
        # if len(set(train_user_indexes).intersection(set(test_user_indexes))) == 0:
        #     print('Well Done')
        # else:
        #     print('Error Splitting')

        print(train_set.users.shape, validation_set.users.shape, np.unique(train_set.users.squeeze()).shape, test_set.users.shape)
        print('\n')
        train_set.pprint()
        validation_set.pprint()
        test_set.pprint()
        print('\n')

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


class DataReductionStategis:
    def __init__(self, data):
        self.data = data

    def set_between_users_reduction(self):
        self.between_users = {
            'nicest': {
                'threshold': 7.50,
                'isMost': True,
                'split_array': self.data.x_avg_ratings,
                'isModelTrained': False,
                'isEvaluationMetricsCalculated': False,
                'isGraphsPloted': False,
                'x_label': 'Average Value of Ratings',
            },

            'harshest': {
                'threshold': 7.50,
                'isMost': False,
                'split_array': self.data.x_avg_ratings,
                'isModelTrained': False,
                'isEvaluationMetricsCalculated': False,
                'isGraphsPloted': False,
                'x_label': 'Average Value of Ratings',
            },

            'most_active': {
                'threshold': 100.00,
                'isMost': True,
                'split_array': self.data.x_n_ratings,
                'isModelTrained': False,
                'isEvaluationMetricsCalculated': False,
                'isGraphsPloted': False,
                'x_label': 'Number of Rated Movies',
            },

            'least_active': {
                'threshold': 100.00,
                'isMost': False,
                'split_array': self.data.x_n_ratings,
                'isModelTrained': False,
                'isEvaluationMetricsCalculated': False,
                'isGraphsPloted': False,
                'x_label': 'Number of Rated Movies',
            },

            'most_typical': {
                'threshold': 0.13,
                'isMost': True,
                'split_array': self.data.x_similarity,
                'isModelTrained': False,
                'isEvaluationMetricsCalculated': False,
                'isGraphsPloted': False,
                'x_label': 'Average User Similarities to All Others',
            },

            'least_typical': {
                'threshold': 0.13,
                'isMost': False,
                'split_array': self.data.x_similarity,
                'isModelTrained': False,
                'isEvaluationMetricsCalculated': False,
                'isGraphsPloted': False,
                'x_label': 'Average User Similarities to All Others',
            },

            'most_consistent': {
                'threshold': 4.00,
                'isMost': True,
                'split_array': self.data.x_var_ratings,
                'isModelTrained': False,
                'isEvaluationMetricsCalculated': False,
                'isGraphsPloted': False,
                'x_label': 'None',
            },

            'least_consistent': {
                'threshold': 4.00,
                'isMost': False,
                'split_array': self.data.x_var_ratings,
                'isModelTrained': False,
                'isEvaluationMetricsCalculated': False,
                'isGraphsPloted': False,
                'x_label': 'None',
            },

            'most_open_minded': {
                'threshold': 16,
                'isMost': True,
                'split_array': self.data.x_n_genres,
                'isModelTrained': False,
                'isEvaluationMetricsCalculated': False,
                'isGraphsPloted': False,
                'x_label': 'None',
            },

            'least_open_minded': {
                'threshold': 16,
                'isMost': False,
                'split_array': self.data.x_n_genres,
                'isModelTrained': False,
                'isEvaluationMetricsCalculated': False,
                'isGraphsPloted': False,
                'x_label': 'None',
            },

        }

        self.between_user_data_reduction_techniques = [
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

        self.retained_user_parcentages = [
            0.00,
            0.25,
            0.50,
            0.75,
            # 1.00
        ]

    def set_within_users_reduction(self):
        self.within_users = {

            'random': {
                'split_array': self.data.x_n_ratings,
                'isModelTrained': False,
                'isEvaluationMetricsCalculated': False,
                'isGraphsPloted': False,
            },

            'most_recent': {
                'split_array': self.data.x_n_ratings,
                'isModelTrained': False,
                'isEvaluationMetricsCalculated': False,
                'isGraphsPloted': False,
            },

            'least_recent': {
                'split_array': self.data.x_n_ratings,
                'isModelTrained': False,
                'isEvaluationMetricsCalculated': False,
                'isGraphsPloted': False,
            },

            'most_favorite': {
                'split_array': self.data.x_n_ratings,
                'isModelTrained': False,
                'isEvaluationMetricsCalculated': False,
                'isGraphsPloted': False,
            },

            'least_favorite': {
                'split_array': self.data.x_n_ratings,
                'isModelTrained': False,
                'isEvaluationMetricsCalculated': False,
                'isGraphsPloted': False,
            },

            'most_rated': {
                'split_array': self.data.x_n_ratings,
                'isModelTrained': False,
                'isEvaluationMetricsCalculated': False,
                'isGraphsPloted': False,
            },

            'least_rated': {
                'split_array': self.data.x_n_ratings,
                'isModelTrained': False,
                'isEvaluationMetricsCalculated': False,
                'isGraphsPloted': False,
            },
        }

        self.characteristices = {
            'varience': {
                'x_val': self.data.x_var_ratings,
                'x_label': 'Varience of Ratings'
            },

            'activeness': {
                'x_val': self.data.x_n_ratings,
                'x_label': 'Number of Ratings'
            },

            'diverse': {
                'x_val': self.data.x_n_genres,
                'x_label': 'Average Value of Ratings'
            },

            'similarity': {
                'x_val': self.data.x_similarity,
                'x_label': 'Average user similarity of all other users'
            },

            'average': {
                'x_val': self.data.x_avg_ratings,
                'x_label': 'Number of Genres'
            },
        }

        self.within_user_data_reduction_techniques = [
            'random',
            'most_recent',
            'least_recent',
            'most_favorite',
            'least_favorite',
            'most_rated',
            'least_rated',
        ]

        self.n_kept = [
            1,
            5,
            10,
            25,
            50,
            75,
        ]


import main
if __name__ == '__main__':
    d, s = main.init()
    print(s.user_fold_indexes)


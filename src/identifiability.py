import numpy as np
import random
import pickle
from collections import defaultdict, Counter
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run GMF.")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--strategy', type=int, nargs='+',
                        help='Choose a within-user reduction strategy.')
    parser.add_argument('--n_kept', type=int, nargs='+',
                        help='Choose n_kep for within-user reduction strategy.')
    parser.add_argument('--nnModel', nargs='?', default='NextItNet_DistributedWeights', #[NextItNet_DistributedWeights, Bert4Rec]
                        help='Choose a recommender system model.')
    parser.add_argument('--mode', type=int, default=1,
                        help='Enter which uncertainty model is training')
    parser.add_argument('--seqlen', type=int, default=100,
                        help='Enter Sequence Length.')
    parser.add_argument('--T', type=int, default=10,
                        help='Enter T.')
    parser.add_argument('--tarlen', type=int, default=1,
                        help='Enter Target Length.')
    parser.add_argument('--setseed', type=int, default=2022,
                        help='Set a seed for random generator.')
    parser.add_argument('--max_epochs', type=int, default=10,
                        help='Number of epochs.')
    parser.add_argument('--embed_dims', type=int, default=32,
                        help='Embedding Dimension')
    parser.add_argument('--residual_channels', type=int, default=32,
                        help='CNN Residual Channels')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='CNN Kernel Size')
    parser.add_argument('--dilations', type=int, default=[1, 2, 4],
                        help='CNN Dilations')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[0,0]',
                        help="Regularization for user and item embeddings.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_known_args()

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def init(args, k_fold=5):
    set_seed(args.setseed)

    # load dataset
    instances = Interactions(args)
    instances.process_dataset()
    instances.create_instances()

    # get_target_samples(instances, num_negative_samples=1)

    instance_spliter = InstanceSpliter(data=instances, k_fold=k_fold)
    instance_spliter.generate_folds()

    return instances, instance_spliter


class Instances:
    def __init__(self, name=''):
        self.name = name
        self.sequences = []
        self.ratings = []
        self.users = []
        self.targets = []  # movie ids

    def shuffle(self):
        shuffle_indices = np.arange(len(self.sequences))
        np.random.shuffle(shuffle_indices)

        self.sequences = self.sequences[shuffle_indices]
        self.ratings = self.ratings[shuffle_indices]
        self.users = self.users[shuffle_indices]
        self.targets = self.targets[shuffle_indices]

    def pprint(self):
        print('Data Instances Shape:', self.name, end='\n')
        print('Sequences:', self.sequences.shape, 'Users:', self.users.shape, 'Targets:', self.targets.shape,
              'Ratings:', self.ratings.shape)


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
        elif self.args.dataset == 'ml-20m':
            print('Reading Files from', self.args.path + self.args.dataset)
            self._read_file(self.data_root + '/ml_20m_sequences')
        elif self.args.dataset == 'amazon-books':
            print('Reading Files from', self.args.path + self.args.dataset)
            self._read_file(self.data_root + '/amazon_books_sequences')
        elif self.args.dataset == 'book-crossing':
            # self._process_book_crossing_data()
            self._read_file(self.data_root + '/train_set_sequences')
        elif self.args.dataset == 'ab-1m':
            print('Reading Files from', self.args.path + self.args.dataset)
            self._read_file(self.data_root + '/reviews_Books_948K_sequences')
            # self._process_book_crossing_data(self.args.dataset+'.csv')
            # self._read_file(self.args.path + self.args.dataset)

            # print('Reading Files from', self.args.path + 'reviews_Clothing_Shoes_and_Jewelry_5_core_sequences')
            # self._read_file(self.args.path + 'reviews_Clothing_Shoes_and_Jewelry_5_core_sequences')

        self.user_interactions = np.array(self.user_interactions, dtype=object)
        self.user_ratings = np.array(self.user_ratings, dtype=object)

        self.num_users = self.user_interactions.shape[0]
        self.num_items = int(np.max(np.concatenate(self.user_interactions[:])))

        print('num_users = ', self.num_users)
        print('num_items = ', self.num_items)
        #
        # self.x_n_ratings = np.array([self.user_ratings[user_id].shape[0] for user_id in range(self.num_users)])
        # xn = Counter(self.x_n_ratings)
        # print(xn)

        self.generate_characteristics()

    def _process_book_crossing_data(self, file_name):
        # file_path = self.data_root + '/BX-Book-Ratings.csv'
        file_path = self.data_root + '/' + file_name
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

        self.user_interactions = [np.array([], dtype=int) for _ in range(user_count - 1)]
        self.user_ratings = [np.array([], dtype=int) for _ in range(user_count - 1)]

        for user, item, rating in temp_interactions:
            # print(user, item, rating)
            self.user_interactions[user - 1] = np.append(self.user_interactions[user - 1], int(item))
            self.user_ratings[user - 1] = np.append(self.user_ratings[user - 1], int(rating))

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
            item_ratings = np.concatenate([np.array(items).reshape(-1, 1), np.array(item_ratings[1]).reshape(-1, 1)],
                                          axis=1)
            # item_ratings = np.array(item_ratings).reshape(-1, 2)
            temp = np.array([filtered_user_dict[users]])
            temp = np.append(temp, item_ratings)
            all_temp.append(temp)

        with open(self.data_root + '/amazon_book_sequences', 'w') as f:
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

        with open(self.data_root + '/user_similarity.pkl', 'rb') as f:
            self.x_similarity = pickle.load(f)

        if self.args.dataset == 'ml-1m':
            # with open(self.data_root + '/user_similarity.pkl', 'rb') as f:
            #     self.x_similarity = pickle.load(f)

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

                if self.args.dataset == 'ml-1m':
                    item_ratings_np[:, 0] += 1  ### Adding 1 to the item_id, 0 reads as padding
                    item_ratings_np[:, 1] *= 2  ### Rating is multiplied by 2, now counts as 10
                # # elif self.args.dataset=='book-crossing':

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
        for user_id, interaction in enumerate(zip(self.user_interactions, self.user_ratings)):
            # interaction[0] --> user sequenes
            # interaction[1] --> user ratings
            sequence_interation = interaction[0]
            sequnece_rating = interaction[1]

            if len(sequence_interation) < 3:
                continue
            ### Users
            self.test.users.append(user_id)

            ### Targets
            self.test.targets.append(sequence_interation[-self.target_length:] - 1)

            ### Sequences
            sequence_interation = sequence_interation[:-self.target_length]  # Most important line
            sequnece_rating = sequnece_rating[:-self.target_length]  # Most important line

            if len(sequence_interation) < self.sequence_length:
                num_paddings = self.sequence_length - len(sequence_interation)
                sequence_interation = np.pad(sequence_interation, (num_paddings, 0), 'constant')
                sequnece_rating = np.pad(sequnece_rating, (num_paddings, 0), 'constant')
            else:
                sequence_interation = sequence_interation[-self.sequence_length:]
                sequnece_rating = sequnece_rating[-self.sequence_length:]

            self.test.sequences.append(sequence_interation)
            self.test.ratings.append(sequnece_rating)

        self.test.targets = np.array(self.test.targets).reshape(-1, self.target_length)
        ### users
        self.test.users = np.array(self.test.users).reshape(-1, 1)
        self.test.sequences = np.array(self.test.sequences)
        self.test.ratings = np.array(self.test.ratings)

        self.test.pprint()

    def _create_train_instances(self):
        max_sequence_length = self.sequence_length + self.target_length

        self.user_sequence_indexes = [list() for i in range(self.num_users)]

        sequence_index = 0
        for user_id, interaction in enumerate(zip(self.user_interactions, self.user_ratings)):
            # interaction[0] --> user sequenes
            # interaction[1] --> user ratings
            sequence_interation = interaction[0]
            sequnece_rating = interaction[1]

            if len(sequence_interation) < 3:
                continue
            sequence_interation = sequence_interation[:-self.target_length]  # Most important line
            sequnece_rating = sequnece_rating[:-self.target_length]  # Most important line

            if len(sequence_interation) < max_sequence_length:
                num_paddings = max_sequence_length - len(sequence_interation)
                sequence_interation = np.pad(sequence_interation, (num_paddings, 0), 'constant')
                sequnece_rating = np.pad(sequnece_rating, (num_paddings, 0), 'constant')

                self._add_sequence(
                    seq=sequence_interation[:self.sequence_length],
                    usr=user_id,
                    rating=sequnece_rating[:self.sequence_length],
                    target=sequence_interation[-self.target_length:],
                    sequence_index=sequence_index
                )
                sequence_index += 1

            else:
                for ind in range(len(sequence_interation), max_sequence_length - 1, -1):
                    temp_sequence = sequence_interation[ind - max_sequence_length: ind]
                    temp_rating = sequnece_rating[ind - max_sequence_length: ind]
                    self._add_sequence(
                        seq=temp_sequence[:self.sequence_length],
                        usr=user_id,
                        rating=temp_rating[:self.sequence_length],
                        target=temp_sequence[-self.target_length:],
                        sequence_index=sequence_index
                    )
                    sequence_index += 1

        self.train.sequences = np.array(self.train.sequences)
        self.train.ratings = np.array(self.train.ratings)
        self.train.users = np.array(self.train.users).reshape(-1, 1)
        self.train.targets = np.array(self.train.targets)
        self.train.pprint()

        for user_id in range(self.num_users):
            self.user_sequence_indexes[user_id] = np.array(self.user_sequence_indexes[user_id])
        self.user_sequence_indexes = np.array(self.user_sequence_indexes, dtype=object)

        self.num_sequences = sequence_index

    def _add_sequence(self, seq, usr, rating, target, sequence_index):
        self.train.sequences.append(seq)
        self.train.ratings.append(rating)
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

    def get_sequence(self, sequence_indexes, isTest=False):
        _instances = Instances()
        if isTest:
            _split_instances = self.data.test
        else:
            _split_instances = self.data.train

        print(type(sequence_indexes))
        _instances.users = _split_instances.users[sequence_indexes]
        _instances.sequences = _split_instances.sequences[sequence_indexes]
        _instances.ratings = _split_instances.ratings[sequence_indexes]
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

        train_set = self.get_sequence(train_sequence_indexes)
        validation_set = self.get_sequence(validation_sequence_indexes)
        test_set = self.get_sequence(test_user_indexes, isTest=True)

        # if len(set(train_set.users.squeeze()).intersection(set(test_set.users.squeeze())))==0:
        # if len(set(train_user_indexes).intersection(set(test_user_indexes))) == 0:
        #     print('Well Done')
        # else:
        #     print('Error Splitting')
        print(train_set.users.shape, validation_set.users.shape, np.unique(train_set.users.squeeze()).shape,
              validation_set.users.shape)

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

def within_user_reduction(reduction_name, test, n):
    print(reduction_name, type(reduction_name))
    reduced_seq = np.zeros(test.sequences.shape)
    reduced_rating_seq = np.zeros(test.sequences.shape)
    movie_counts = Counter(np.concatenate(data.user_interactions)) #For movie popularity
    for ind in range(test.sequences.shape[0]):
        if reduction_name == 'random':
            rand_permute = np.random.permutation(np.arange(data.sequence_length))[:n]
            reduced_seq[ind, rand_permute] = test.sequences[ind, rand_permute]
            reduced_rating_seq[ind, rand_permute] = test.ratings[ind, rand_permute]

        elif reduction_name == 'most_recent':
            reduced_seq[ind, max(data.sequence_length-n,0):data.sequence_length] = test.sequences[ind, max(data.sequence_length-n,0):data.sequence_length]
            reduced_rating_seq[ind, max(data.sequence_length-n,0):data.sequence_length] = test.ratings[ind, max(data.sequence_length-n,0):data.sequence_length]

        elif reduction_name == 'least_recent':
            if len(np.where(test.sequences[ind]==0)[0])>0:
                first_split = np.where(test.sequences[ind]==0)[0][-1]+1
            else:
                first_split = 0
            # reduced_seq[ind, first_split:min(first_split+n, data.sequence_length)] = test.sequences[ind, first_split:min(first_split+n, data.sequence_length)]
            # reduced_seq[ind, :n] = test.sequences[ind, first_split:min(first_split+n, data.sequence_length)]
            reduced_seq[ind, :test.sequences[ind, first_split:min(first_split+n, data.sequence_length)].shape[0]] = test.sequences[ind, first_split:min(first_split+n, data.sequence_length)]
            reduced_rating_seq[ind, :test.sequences[ind, first_split:min(first_split+n, data.sequence_length)].shape[0]] = test.ratings[ind, first_split:min(first_split+n, data.sequence_length)]

        elif reduction_name == 'most_favorite':
            # indexes = np.argsort(x_rating_pop[m,:x_size[m]])[::-1][:n]
            # x_test_par[m, indexes] = x_test[m, indexes]
            indexes = np.argsort(test.ratings[ind])[::-1][:n]
            reduced_seq[ind, indexes] = test.sequences[ind, indexes]
            reduced_rating_seq[ind, indexes] = test.ratings[ind, indexes]

        elif reduction_name == 'least_favorite':
            indexes = np.argsort(test.ratings[ind])[:n]
            reduced_seq[ind, indexes] = test.sequences[ind, indexes]
            reduced_rating_seq[ind, indexes] = test.ratings[ind, indexes]

        elif reduction_name == 'most_rated':
            # indexes = np.argsort(x_rating[ind,:x_size[m]])[::-1][:n]
            # x_test_par[m, indexes] = x_test[m, indexes]
            movie_popularity = np.array([])

            for movie_id in test.sequences[ind]:
                if movie_id==0:
                    movie_popularity = np.append(movie_popularity, 0)
                else:
                    movie_popularity = np.append(movie_popularity, movie_counts[movie_id])

            indexes = np.argsort(movie_popularity)[::-1][:n]
            reduced_seq[ind, indexes] = test.sequences[ind, indexes]
            reduced_rating_seq[ind, indexes] = test.ratings[ind, indexes]

        elif reduction_name == 'least_rated':
            # indexes = np.argsort(test.ratings[ind])[:n]
            # reduced_seq[ind, indexes] = test.sequences[ind, indexes]
            movie_popularity = np.array([])

            for movie_id in test.sequences[ind]:
                if movie_id==0:
                    movie_popularity = np.append(movie_popularity, 0)
                else:
                    movie_popularity = np.append(movie_popularity, movie_counts[movie_id])

            indexes = np.argsort(movie_popularity)[:n]
            reduced_seq[ind, indexes] = test.sequences[ind, indexes]
            reduced_rating_seq[ind, indexes] = test.ratings[ind, indexes]

        else:
            raise Exception("No within user reduction found!!!")

    test_reduced = Instances()
    test_reduced.sequences = reduced_seq
    test_reduced.ratings = reduced_rating_seq
    test_reduced.targets = test.targets
    test_reduced.users = test.users
    # print(test_reduced.sequences[125], '\n', test.sequences[125])
    return test_reduced

def stats(x):
    zeros = np.where(x == 0)[0].shape[0]
    print('Not Identifiable: ', zeros)
    print('Uniquely Identifiable: ', x.shape[0] - zeros)
    print('Sum: ', np.sum(x))
    print('Minimum # of movies to uniquely identify', np.sum(x)/(x.shape[0] - zeros))


within_user_data_reduction_techniques = [
    'random',
    'most_recent',
    'least_recent',
    'most_favorite',
    'least_favorite',
    'most_rated',
    'least_rated',
]
def calculate_identifiablity(strategy, n, data, test):
    if n == 20:
        test_reduced = test
    else:
        test_reduced = within_user_reduction(strategy, test, n=n)
    item_rating_pair = [set() for i in range(data.num_users)]
    min_unique_pairs = np.zeros(data.num_users)
    for index, user_id in enumerate(test_reduced.users.squeeze()):
        for i_r_pair in zip(test_reduced.sequences[index], test_reduced.ratings[index]):
            if i_r_pair == (0, 0):
                continue
            item_rating_pair[user_id].add(i_r_pair)

    u_minus_all = [set() for i in range(data.num_users)]
    u_minus_all_len = []
    for u_user_id in range(data.num_users):
        # print(u_user_id, end=',')
        all_except_u = set()
        for v_user_id in range(data.num_users):
            if u_user_id == v_user_id:
                continue
            if len(item_rating_pair[v_user_id]) == 0:
                continue
            all_except_u = all_except_u.union(item_rating_pair[v_user_id])

        u_minus_all[u_user_id] = item_rating_pair[u_user_id].difference(all_except_u)
        u_minus_all_len.append(len(u_minus_all[u_user_id]))

    stats(np.array(u_minus_all_len))

if __name__ == '__main__':

    args, _ = parse_args()
    print(args.path, args.dataset, args.seqlen)
    set_seed(2022)
    data, spliter = init(args)

    print('strategy:', args.strategy)
    print('n_kept:',args.n_kept)

    n_kept = args.n_kept
    for fold in range(5):
        _, _, test = spliter.split_data(fold_id=fold)
        print('Fold:', fold)
        for st in args.strategy:
            print(st)
            strategy = within_user_data_reduction_techniques[st]
            for n in args.n_kept:
                print(n)
                print(strategy, n)
                calculate_identifiablity(strategy, n, data, test)

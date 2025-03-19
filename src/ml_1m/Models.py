import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
from tensorflow.keras import backend as K
from tensorflow import keras
tf.config.run_functions_eagerly(True)

def CustomSparseCategoricalCrossentropyLoss(y_true, y_pred):
    T = 100
    varience = y_pred[:, -1]
    before_softmax = y_pred[:, :-1]

    std = tf.sqrt(varience)
    dist = tfd.Normal(loc=K.zeros_like(std), scale=std)

    iterable = K.variable(np.ones(T))
    monte_carlo_results = K.map_fn(crossentropy_loss(y_true, before_softmax, dist), elems=iterable, name='monte_carlo_results')
    variance_loss = K.mean(monte_carlo_results, axis=0)

    return K.log(variance_loss)

def crossentropy_loss(y_true, y_pred, dist):
    def map_fn(i):
        std_samples = K.transpose(dist.sample(y_pred.shape[1]))
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

def montecarlo_prediction(model, test, T=100):
    y_pred_T = np.array([model.predict(test.sequences, verbose=0) for _ in range(T)]) #(T, batch_size, (num_items+softplus))
    # print(predictions.shape)
    y_pred = np.mean(y_pred_T, axis=0) #(T, batch_size, (num_items+softplus))
    # print(prediction_probabilities.shape)
    y_pred = y_pred[:, :-1] #(batch_size, num_items)
    # print(predictions.shape)
    softmax_y_pred = tf.nn.softmax(y_pred)
    epistemic_uncertainty = np.apply_along_axis(predictive_entropy, axis=1, arr=softmax_y_pred) #(batch_size, 1), varience of each user
    # print(prediction_variances.shape)
    m = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=20)
    m.update_state(test.targets, softmax_y_pred)

    #return HR@20, Average epistemic Uncertainty, array of epistemic uncertainty for each user
    #Similat to evaluate_model() method
    return m.result().numpy(), np.mean(epistemic_uncertainty), epistemic_uncertainty

def predictive_entropy(prob):
    return -1 * np.sum(np.log(prob+epsilon) * (prob))



class NextItNet_DistributedWeights:
    def __init__(self, data=None, sequence_length=10, T=10):
        self.T = T
        self.data = data
        self.saved_dir = 'results/NextItNet_DistributedWeights/'
        if not data:
            self.sequence_length = sequence_length
            self.num_users = 6040
            self.num_items = 3416  # tf.keras.layers.InputLayer((28, 28, 1)),
        else:
            self.sequence_length = data.sequence_length
            self.num_users = data.num_users
            self.num_items = data.num_items  # Consider padding 0 a prediction

    def get_model(self, drop_prob=0.25, isDropoutTraining=True):
        divergence_fn = lambda q, p, _: tfd.kl_divergence(q, p) / self.data.train.sequences.shape[0]
        embed_size = 64  # 256
        dilations = [1, 2, 4]  # [1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4]
        residual_channels = 64  # 256
        kernel_size = 3  # 3

        seq_input = tf.keras.layers.Input(shape=(self.sequence_length,), name='seq_input')

        # (batch_size, sequence_length, embed_dim) --> (None, 100, 64)
        seq_embedding = tf.keras.layers.Embedding(
            input_dim=self.num_items + 1,
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
            )(conv_layer)  # (batch_size, seq_len, embed_dim, residual_channels) = (None, 100, 62, 32), (None, 100, 54, 32), (None, 100, 38, 32)
        conv_layer = tf.keras.layers.Dropout(drop_prob)(conv_layer, training=isDropoutTraining)

        flatten_layer = tf.reshape(conv_layer[:, -1, ], (-1, residual_channels))  # (batch_size, embed_dim) = (None, 32)
        # fully_connected_layer = tf.keras.layers.Dense(residual_channels)(flatten_layer)
        variance_layer = tf.keras.layers.Dense(1, activation='softplus', name='variance')(flatten_layer)

        fully_connected_layer = tf.keras.layers.Dense(self.num_items)(
            flatten_layer)  # (batch_size, num_items) = (None, 3417)
        fully_connected_layer = tf.keras.layers.Dropout(drop_prob)(fully_connected_layer, training=isDropoutTraining)
        # fully_connected_layer = tfp.layers.DenseVariational(self.num_items + 1, posterior_mean_field, prior_trainable, kl_weight=1/data.train.sequences.shape[0])(flatten_layer)

        # lambda_layer = tfp.layers.DistributionLambda(
        #                 lambda t: tfd.Normal(
        #                                         loc=t[..., :self.num_items],
        #                                         scale=1e-3 + tf.math.softplus(0.01 * t[..., self.num_items:]))
        #                 )(fully_connected_layer)
        # print("lambda_layer.shape", lambda_layer.shape)
        # softmax_output = tf.keras.layers.Activation(activation=tf.nn.softmax,name='softmax_output')(fully_connected_layer)
        # logits_variance = tf.keras.layers.concatenate([softmax_output, variance_layer], name='logits_variance')
        logits_variance = tf.keras.layers.concatenate([fully_connected_layer, variance_layer], name='logits_variance')

        model = tf.keras.Model(
            inputs=seq_input,
            # outputs=lambda_layer,
            outputs=logits_variance,
        )
        ### model compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3, decay=1e-4),
            loss=CustomSparseCategoricalCrossentropyLoss,
            metrics=[CustomSparseTopKCategoricalAccuracy],
        )

        return model


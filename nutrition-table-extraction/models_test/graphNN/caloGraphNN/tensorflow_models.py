import tensorflow
from caloGraphNN import *



def get_GravNet_model_for_clustering(input, training, momentum):
    feats = []
    x = input
    for i in range(4):
        x = layer_global_exchange(x)
        x = high_dim_dense(x, 64, activation=tf.nn.tanh)
        x = high_dim_dense(x, 64, activation=tf.nn.tanh)
        x = high_dim_dense(x, 64, activation=tf.nn.tanh)

        x = layer_GravNet(x,
                          n_neighbours=40,
                          n_dimensions=4,
                          n_filters=42,
                          n_propagate=18)
        x = tf.layers.batch_normalization(x, momentum=momentum, training=training)
        feats.append(x)

    x = tf.concat(feats, axis=-1)
    x = high_dim_dense(x, 128, activation=tf.nn.relu)
    x = high_dim_dense(x, 3, activation=tf.nn.relu)
    return x

def get_GarNet_model_for_clustering(input, training, momentum):
    aggregators = 11 * [4]
    filters = 11 * [32]
    propagate = 11 * [20]

    feat = layer_global_exchange(input)
    feat = tf.layers.batch_normalization(feat, training=training, momentum=momentum)
    feat = high_dim_dense(feat, 32, activation=tf.nn.tanh)
    feat_list = []
    for i in range(len(filters)):
        feat = layer_GarNet(feat,
                            aggregators[i],
                            n_filters=filters[i],
                            n_propagate=propagate[i]
                            )
        feat = tf.layers.batch_normalization(feat, training=training, momentum=momentum)
        feat_list.append(feat)
        # feat = tf.layers.dropout(feat, rate=0.0005, training=self.is_train)

    feat = tf.concat(feat_list, axis=-1)
    feat = tf.layers.dense(feat, 48, activation=tf.nn.relu)
    feat = tf.layers.dense(feat, 3, activation=tf.nn.relu)

    return feat

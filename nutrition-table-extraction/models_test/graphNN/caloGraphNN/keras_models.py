import sys
import traceback
try:
    import tensorflow.keras as keras
except ImportError:
    import keras

from caloGraphNN_keras import *

class GravNetClusteringModel(keras.Model):
    def __init__(self, n_neighbours=40, n_dimensions=4, n_filters=42, n_propagate=18, **kwargs):
        super(GravNetClusteringModel, self).__init__(**kwargs)

        self.blocks = []

        momentum = kwargs.get('momentum', 0.99)

        for i in range(4):
            gex = self.add_layer(GlobalExchange, name='gex_%d' % i)

            dense0 = self.add_layer(keras.layers.Dense, 64, activation='tanh', name='dense_%d-0' % i)
            dense1 = self.add_layer(keras.layers.Dense, 64, activation='tanh', name='dense_%d-1' % i)
            dense2 = self.add_layer(keras.layers.Dense, 64, activation='tanh', name='dense_%d-2' % i)

            gravnet = self.add_layer(GravNet, n_neighbours, n_dimensions, n_filters, n_propagate, name='gravnet_%d' % i)

            batchnorm = self.add_layer(keras.layers.BatchNormalization, momentum=momentum, name='batchnorm_%d' % i)

            self.blocks.append((gex, dense0, dense1, dense2, gravnet, batchnorm))

        self.output_dense_0 = self.add_layer(keras.layers.Dense, 128, activation='relu', name='output_0')
        self.output_dense_1 = self.add_layer(keras.layers.Dense, 3, activation='relu', name='output_1')

    def call(self, inputs):
        feats = []

        x = inputs

        for block in self.blocks:
            for layer in block:
                x = layer(x)

            feats.append(x)

        x = tf.concat(feats, axis=-1)

        x = self.output_dense_0(x)
        x = self.output_dense_1(x)

        return x

    def add_layer(self, cls, *args, **kwargs):
        layer = cls(*args, **kwargs)
        self._layers.append(layer)
        return layer


class GarNetClusteringModel(keras.Model):
    def __init__(self, aggregators=([4] * 11), filters=([32] * 11), propagate=([20] * 11), **kwargs):
        super(GarNetClusteringModel, self).__init__(**kwargs)
        
        self.blocks = []

        block_params = zip(aggregators, filters, propagate)

        momentum = kwargs.get('momentum', 0.99)

        self.input_gex = self.add_layer(GlobalExchange, name='input_gex')
        self.input_batchnorm = self.add_layer(keras.layers.BatchNormalization, momentum=momentum, name='input_batchnorm')
        self.input_dense = self.add_layer(keras.layers.Dense, 32, activation='tanh', name='input_dense')

        for i, (n_aggregators, n_filters, n_propagate) in enumerate(block_params):
            garnet = self.add_layer(GarNet, n_aggregators, n_filters, n_propagate, name='garnet_%d' % i)
            batchnorm = self.add_layer(keras.layers.BatchNormalization, momentum=momentum, name='batchnorm_%d' % i)

            self.blocks.append((garnet, batchnorm))

        self.output_dense_0 = self.add_layer(keras.layers.Dense, 48, activation='relu', name='output_0')
        self.output_dense_1 = self.add_layer(keras.layers.Dense, 3, activation='relu', name='output_1')

    def call(self, inputs):
        feats = []

        x = inputs

        x = self.input_gex(x)
        x = self.input_batchnorm(x)
        x = self.input_dense(x)

        for block in self.blocks:
            for layer in block:
                x = layer(x)

            feats.append(x)

        x = tf.concat(feats, axis=-1)

        x = self.output_dense_0(x)
        x = self.output_dense_1(x)

        return x

    def add_layer(self, cls, *args, **kwargs):
        layer = cls(*args, **kwargs)
        self._layers.append(layer)
        return layer

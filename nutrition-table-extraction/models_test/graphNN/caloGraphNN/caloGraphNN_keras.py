try:
    import tensorflow.keras as keras
except ImportError:
    import keras

K = keras.backend

try:
    from qkeras import QDense, ternary

    class NamedQDense(QDense):
        def add_weight(self, name=None, **kwargs):
            return super(NamedQDense, self).add_weight(name='%s_%s' % (self.name, name), **kwargs)

    def ternary_1_05():
        return ternary(alpha=1., threshold=0.5)

except ImportError:
    pass

# Hack keras Dense to propagate the layer name into saved weights
class NamedDense(keras.layers.Dense):
    def add_weight(self, name=None, **kwargs):
        return super(NamedDense, self).add_weight(name='%s_%s' % (self.name, name), **kwargs)

class CreateZeroMask(keras.layers.Layer):
    '''
    Creates a mask based on the n-th feature of the vertex
    To apply, use keras.Layers.Multiply
    '''
    def __init__(self, idx, **kwargs):
        super(CreateZeroMask, self).__init__(**kwargs)

        self.idx = idx
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 1)
    
    def call(self, inputs):
        mask = K.cast(K.not_equal(inputs[..., self.idx:self.idx + 1], 0.), 'float32')
        return mask
    

class GlobalExchange(keras.layers.Layer):
    def __init__(self, vertex_mask=None, **kwargs):
        super(GlobalExchange, self).__init__(**kwargs)

        self.vertex_mask = vertex_mask

    def build(self, input_shape):
        # tf.ragged FIXME?
        self.num_vertices = input_shape[1]
        super(GlobalExchange, self).build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=1, keepdims=True)
        # tf.ragged FIXME?
        # maybe just use tf.shape(x)[1] instead?
        mean = K.tile(mean, [1, self.num_vertices, 1])
        if self.vertex_mask is not None:
            mean = self.vertex_mask * mean

        return K.concatenate([x, mean], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[:2] + (input_shape[2] * 2,)


class GravNet(keras.layers.Layer):
    def __init__(self, n_neighbours, n_dimensions, n_filters, n_propagate, name, 
                 also_coordinates=False, feature_dropout=-1, 
                 coordinate_kernel_initializer=keras.initializers.Orthogonal(),
                 other_kernel_initializer='glorot_uniform',
                 fix_coordinate_space=False, 
                 coordinate_activation=None,
                 masked_coordinate_offset=None,
                 **kwargs):
        super(GravNet, self).__init__(**kwargs)

        self.n_neighbours = n_neighbours
        self.n_dimensions = n_dimensions
        self.n_filters = n_filters
        self.n_propagate = n_propagate
        self.name = name
        self.also_coordinates = also_coordinates
        self.feature_dropout = feature_dropout
        self.masked_coordinate_offset = masked_coordinate_offset
        
        self.input_feature_transform = NamedDense(n_propagate, name = name+'_FLR', kernel_initializer=other_kernel_initializer)
        self.input_spatial_transform = NamedDense(n_dimensions, name = name+'_S', kernel_initializer=coordinate_kernel_initializer, activation=coordinate_activation)
        self.output_feature_transform = NamedDense(n_filters, activation='tanh', name = name+'_Fout', kernel_initializer=other_kernel_initializer)

        self._sublayers = [self.input_feature_transform, self.input_spatial_transform, self.output_feature_transform]
        if fix_coordinate_space:
            self.input_spatial_transform = None
            self._sublayers = [self.input_feature_transform, self.output_feature_transform]

    def build(self, input_shape):
        if self.masked_coordinate_offset is not None:
            input_shape = input_shape[0]
            
        self.input_feature_transform.build(input_shape)
        if self.input_spatial_transform is not None:
            self.input_spatial_transform.build(input_shape)
        
        # tf.ragged FIXME?
        self.output_feature_transform.build((input_shape[0], input_shape[1], input_shape[2] + self.input_feature_transform.units * 2))

        for layer in self._sublayers:
            self._trainable_weights.extend(layer.trainable_weights)
            self._non_trainable_weights.extend(layer.non_trainable_weights)
        
        super(GravNet, self).build(input_shape)

    def call(self, x):
        
        if self.masked_coordinate_offset is not None:
            if not isinstance(x, list):
                raise Exception('GravNet: in mask mode, input must be list of input,mask')
            mask = x[1]
            x = x[0]
            
        features = self.input_feature_transform(x)
        
        if self.feature_dropout>0 and self.feature_dropout < 1:
            features = keras.layers.Dropout(self.feature_dropout)(features)
        
        if self.input_spatial_transform is not None:
            coordinates = self.input_spatial_transform(x)
        else:
            coordinates = x[:,:,0:self.n_dimensions]
            
        if self.masked_coordinate_offset is not None:
            sel_mask = K.tile(mask, [1, 1, K.shape(coordinates)[2]])
            coordinates = K.switch(K.greater(sel_mask, 0.), coordinates, K.zeros_like(coordinates) - self.masked_coordinate_offset)

        collected_neighbours = self.collect_neighbours(coordinates, features)

        updated_features = K.concatenate([x, collected_neighbours], axis=-1)
        output = self.output_feature_transform(updated_features)
        
        if self.masked_coordinate_offset is not None:
            output *= mask

        if self.also_coordinates:
            return [output, coordinates]
        return output
        
    def compute_output_shape(self, input_shape):
        if self.masked_coordinate_offset is not None:
            input_shape = input_shape[0]
        if self.also_coordinates:
            return [(input_shape[0], input_shape[1], self.output_feature_transform.units),
                    (input_shape[0], input_shape[1], self.n_dimensions)]
        
        # tf.ragged FIXME? tf.shape() might do the trick already
        return (input_shape[0], input_shape[1], self.output_feature_transform.units)

    def collect_neighbours(self, coordinates, features):
        import tensorflow as tf
        from caloGraphNN import euclidean_squared, gauss_of_lin
        
        # tf.ragged FIXME?
        # for euclidean_squared see caloGraphNN.py
        distance_matrix = euclidean_squared(coordinates, coordinates)

        ranked_distances, ranked_indices = tf.nn.top_k(-distance_matrix, self.n_neighbours)

        neighbour_indices = ranked_indices[:, :, 1:]

        n_batches = tf.shape(features)[0]
        
        # tf.ragged FIXME? or could that work?
        n_vertices = K.shape(features)[1]
        n_features = K.shape(features)[2]

        batch_range = K.arange(n_batches)
        batch_range = K.expand_dims(batch_range, axis=1)
        batch_range = K.expand_dims(batch_range, axis=1)
        batch_range = K.expand_dims(batch_range, axis=1) # (B, 1, 1, 1)

        # tf.ragged FIXME? n_vertices
        batch_indices = K.tile(batch_range, [1, n_vertices, self.n_neighbours - 1, 1]) # (B, V, N-1, 1)
        vertex_indices = K.expand_dims(neighbour_indices, axis=3) # (B, V, N-1, 1)
        indices = K.concatenate([batch_indices, vertex_indices], axis=-1)
    
        neighbour_features = tf.gather_nd(features, indices) # (B, V, N-1, F)
    
        distance = -ranked_distances[:, :, 1:]
    
        weights = gauss_of_lin(distance * 10.)
        weights = K.expand_dims(weights, axis=-1)
    
        # weight the neighbour_features
        neighbour_features *= weights
    
        neighbours_max = K.max(neighbour_features, axis=2)
        neighbours_mean = K.mean(neighbour_features, axis=2)
    
        return K.concatenate([neighbours_max, neighbours_mean], axis=-1)

    def get_config(self):
        config = {'n_neighbours': self.n_neighbours, 
                  'n_dimensions': self.n_dimensions, 
                  'n_filters': self.n_filters, 
                  'n_propagate': self.n_propagate,
                  'name':self.name,
                  'also_coordinates': self.also_coordinates,
                  'feature_dropout' : self.feature_dropout,
                  'masked_coordinate_offset'       : self.masked_coordinate_offset}
        base_config = super(GravNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GarNet(keras.layers.Layer):
    def __init__(self, n_aggregators, n_filters, n_propagate,
                 simplified=False,
                 collapse=None,
                 input_format='xn',
                 output_activation='tanh',
                 mean_by_nvert=False,
                 quantize_transforms=False,
                 **kwargs):
        super(GarNet, self).__init__(**kwargs)

        self._simplified = simplified
        self._output_activation = output_activation
        self._quantize_transforms = quantize_transforms

        self._setup_aux_params(collapse, input_format, mean_by_nvert)
        self._setup_transforms(n_aggregators, n_filters, n_propagate)

    def _setup_aux_params(self, collapse, input_format, mean_by_nvert):
        if collapse is None:
            self._collapse = None
        elif collapse in ['mean', 'sum', 'max']:
            self._collapse = collapse
        else:
            raise NotImplementedError('Unsupported collapse operation')

        self._input_format = input_format
        self._mean_by_nvert = mean_by_nvert

    def _setup_transforms(self, n_aggregators, n_filters, n_propagate):
        if self._quantize_transforms:
            self._input_feature_transform = NamedQDense(n_propagate, kernel_quantizer=ternary_1_05(), bias_quantizer=ternary_1_05(), name='FLR')
            self._output_feature_transform = NamedQDense(n_filters, activation=self._output_activation, kernel_quantizer=ternary_1_05(), name='Fout')
        else:
            self._input_feature_transform = NamedDense(n_propagate, name='FLR')
            self._output_feature_transform = NamedDense(n_filters, activation=self._output_activation, name='Fout')

        self._aggregator_distance = NamedDense(n_aggregators, name='S')

        self._sublayers = [self._input_feature_transform, self._aggregator_distance, self._output_feature_transform]

    def build(self, input_shape):
        super(GarNet, self).build(input_shape)

        if self._input_format == 'x':
            data_shape = input_shape
        elif self._input_format == 'xn':
            data_shape, _ = input_shape
        elif self._input_format == 'xen':
            data_shape, _, _ = input_shape
            data_shape = data_shape[:2] + (data_shape[2] + 1,)

        self._build_transforms(data_shape)

        for layer in self._sublayers:
            self._trainable_weights.extend(layer.trainable_weights)
            self._non_trainable_weights.extend(layer.non_trainable_weights)

    def _build_transforms(self, data_shape):
        self._input_feature_transform.build(data_shape)
        self._aggregator_distance.build(data_shape)
        if self._simplified:
            self._output_feature_transform.build(data_shape[:2] + (self._aggregator_distance.units * self._input_feature_transform.units,))
        else:
            self._output_feature_transform.build(data_shape[:2] + (data_shape[2] + self._aggregator_distance.units * self._input_feature_transform.units + self._aggregator_distance.units,))

    def call(self, x):
        data, num_vertex, vertex_mask = self._unpack_input(x)

        output = self._garnet(data, num_vertex, vertex_mask,
                              self._input_feature_transform,
                              self._aggregator_distance,
                              self._output_feature_transform)

        output = self._collapse_output(output)

        return output

    def _unpack_input(self, x):
        if self._input_format == 'x':
            data = x

            vertex_mask = K.cast(K.not_equal(data[..., 3:4], 0.), 'float32')
            num_vertex = K.sum(vertex_mask)

        elif self._input_format in ['xn', 'xen']:
            if self._input_format == 'xn':
                data, num_vertex = x
            else:
                data_x, data_e, num_vertex = x
                data = K.concatenate((data_x, K.reshape(data_e, (-1, data_e.shape[1], 1))), axis=-1)
    
            data_shape = K.shape(data)
            B = data_shape[0]
            V = data_shape[1]
            vertex_indices = K.tile(K.expand_dims(K.arange(0, V), axis=0), (B, 1)) # (B, [0..V-1])
            vertex_mask = K.expand_dims(K.cast(K.less(vertex_indices, K.cast(num_vertex, 'int32')), 'float32'), axis=-1) # (B, V, 1)
            num_vertex = K.cast(num_vertex, 'float32')

        return data, num_vertex, vertex_mask

    def _garnet(self, data, num_vertex, vertex_mask, in_transform, d_compute, out_transform):
        features = in_transform(data) # (B, V, F)
        distance = d_compute(data) # (B, V, S)

        edge_weights = vertex_mask * K.exp(-K.square(distance)) # (B, V, S)

        if not self._simplified:
            features = K.concatenate([vertex_mask * features, edge_weights], axis=-1)
        
        if self._mean_by_nvert:
            def graph_mean(out, axis):
                s = K.sum(out, axis=axis)
                # reshape just to enable broadcasting
                s = K.reshape(s, (-1, d_compute.units * in_transform.units)) / num_vertex
                s = K.reshape(s, (-1, d_compute.units, in_transform.units))
                return s
        else:
            graph_mean = K.mean

        # vertices -> aggregators
        edge_weights_trans = K.permute_dimensions(edge_weights, (0, 2, 1)) # (B, S, V)

        aggregated_mean = self._apply_edge_weights(features, edge_weights_trans, aggregation=graph_mean) # (B, S, F)

        if self._simplified:
            aggregated = aggregated_mean
        else:
            aggregated_max = self._apply_edge_weights(features, edge_weights_trans, aggregation=K.max)
            aggregated = K.concatenate([aggregated_max, aggregated_mean], axis=-1)

        # aggregators -> vertices
        updated_features = self._apply_edge_weights(aggregated, edge_weights) # (B, V, S*F)

        if not self._simplified:
            updated_features = K.concatenate([data, updated_features, edge_weights], axis=-1)

        return vertex_mask * out_transform(updated_features)

    def _collapse_output(self, output):
        if self._collapse == 'mean':
            if self._mean_by_nvert:
                output = K.sum(output, axis=1) / num_vertex
            else:
                output = K.mean(output, axis=1)
        elif self._collapse == 'sum': 
           output = K.sum(output, axis=1)
        elif self._collapse == 'max':
            output = K.max(output, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return self._get_output_shape(input_shape, self._output_feature_transform)

    def _get_output_shape(self, input_shape, out_transform):
        if self._input_format == 'x':
            data_shape = input_shape
        elif self._input_format == 'xn':
            data_shape, _ = input_shape
        elif self._input_format == 'xen':
            data_shape, _, _ = input_shape

        if self._collapse is None:
            return data_shape[:2] + (out_transform.units,)
        else:
            return (data_shape[0], out_transform.units)

    def get_config(self):
        config = super(GarNet, self).get_config()

        config.update({
            'simplified': self._simplified,
            'collapse': self._collapse,
            'input_format': self._input_format,
            'output_activation': self._output_activation,
            'quantize_transforms': self._quantize_transforms,
            'mean_by_nvert': self._mean_by_nvert
        })

        self._add_transform_config(config)

        return config

    def _add_transform_config(self, config):
        config.update({
            'n_aggregators': self._aggregator_distance.units,
            'n_filters': self._output_feature_transform.units,
            'n_propagate': self._input_feature_transform.units
        })

    @staticmethod
    def _apply_edge_weights(features, edge_weights, aggregation=None):
        features = K.expand_dims(features, axis=1) # (B, 1, v, f)
        edge_weights = K.expand_dims(edge_weights, axis=3) # (B, u, v, 1)

        out = edge_weights * features # (B, u, v, f)

        if aggregation:
            out = aggregation(out, axis=2) # (B, u, f)
        else:
            try:
                out = K.reshape(out, (-1, edge_weights.shape[1].value, features.shape[-1].value * features.shape[-2].value))
            except AttributeError: # TF 2
                out = K.reshape(out, (-1, edge_weights.shape[1], features.shape[-1] * features.shape[-2]))
        
        return out

    
class GarNetStack(GarNet):
    """
    Stacked version of GarNet. First three arguments to the constructor must be lists of integers.
    Basically offers no performance advantage, but the configuration is consolidated (and is useful
    when e.g. converting the layer to HLS)
    """
    
    def _setup_transforms(self, n_aggregators, n_filters, n_propagate):
        self._transform_layers = []
        # inputs are lists
        for it, (p, a, f) in enumerate(zip(n_propagate, n_aggregators, n_filters)):
            if self._quantize_transforms:
                input_feature_transform = NamedQDense(p, kernel_quantizer=ternary_1_05(), bias_quantizer=ternary_1_05(), name=('FLR%d' % it))
                output_feature_transform = NamedQDense(f, activation=self._output_activation, kernel_quantizer=ternary_1_05(), name=('Fout%d' % it))
            else:
                input_feature_transform = NamedDense(p, name=('FLR%d' % it))
                output_feature_transform = NamedDense(f, activation=self._output_activation, name=('Fout%d' % it))

            aggregator_distance = NamedDense(a, name=('S%d' % it))

            self._transform_layers.append((input_feature_transform, aggregator_distance, output_feature_transform))

        self._sublayers = sum((list(layers) for layers in self._transform_layers), [])

    def _build_transforms(self, data_shape):
        for in_transform, d_compute, out_transform in self._transform_layers:
            in_transform.build(data_shape)
            d_compute.build(data_shape)
            if self._simplified:
                out_transform.build(data_shape[:2] + (d_compute.units * in_transform.units,))
            else:
                out_transform.build(data_shape[:2] + (data_shape[2] + d_compute.units * in_transform.units + d_compute.units,))

            data_shape = data_shape[:2] + (out_transform.units,)

    def call(self, x):
        data, num_vertex, vertex_mask = self._unpack_input(x)

        for in_transform, d_compute, out_transform in self._transform_layers:
            data = self._garnet(data, num_vertex, vertex_mask, in_transform, d_compute, out_transform)
    
        output = self._collapse_output(data)

        return output

    def compute_output_shape(self, input_shape):
        return self._get_output_shape(input_shape, self._transform_layers[-1][2])

    def _add_transform_config(self, config):
        config.update({
            'n_propagate': list(ll[0].units for ll in self._transform_layers),
            'n_aggregators': list(ll[1].units for ll in self._transform_layers),
            'n_filters': list(ll[2].units for ll in self._transform_layers),
            'n_sublayers': len(self._transform_layers)
        })

    
# tf.ragged FIXME? the last one should be no problem
class weighted_sum_layer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(weighted_sum_layer, self).__init__(**kwargs)
        
    def get_config(self):
        base_config = super(weighted_sum_layer, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        assert input_shape[2] > 1
        inshape=list(input_shape)
        return tuple((inshape[0],input_shape[2]-1))
    
    def call(self, inputs):
        # input #B x E x F
        weights = inputs[:,:,0:1] #B x E x 1
        tosum   = inputs[:,:,1:]
        weighted = weights * tosum #broadcast to B x E x F-1
        return K.sum(weighted, axis=1)    

import tensorflow as tf

### helpers here


def gauss(x):
    return tf.exp(-1* x*x)

def gauss_of_lin(x):
    return tf.exp(-1*(tf.abs(x)))

def euclidean_squared(A, B):
    """
    Returns euclidean distance between two batches of shape [B,N,F] and [B,M,F] where B is batch size, N is number of
    examples in the batch of first set, M is number of examples in the batch of second set, F is number of spatial
    features.

    Returns:
    A matrix of size [B, N, M] where each element [i,j] denotes euclidean distance between ith entry in first set and
    jth in second set.

    """

    shape_A = A.get_shape().as_list()
    shape_B = B.get_shape().as_list()
    
    assert (A.dtype == tf.float32 or A.dtype == tf.float64) and (B.dtype == tf.float32 or B.dtype == tf.float64)
    assert len(shape_A) == 3 and len(shape_B) == 3
    assert shape_A[0] == shape_B[0]# and shape_A[1] == shape_B[1]

    # Finds euclidean distance using property (a-b)^2 = a^2 + b^2 - 2ab
    sub_factor = -2 * tf.matmul(A, tf.transpose(B, perm=[0, 2, 1]))  # -2ab term
    dotA = tf.expand_dims(tf.reduce_sum(A * A, axis=2), axis=2)  # a^2 term
    dotB = tf.expand_dims(tf.reduce_sum(B * B, axis=2), axis=1)  # b^2 term
    return tf.abs(sub_factor + dotA + dotB)


def nearest_neighbor_matrix(spatial_features, k=10):
    """
    Nearest neighbors matrix given spatial features.

    :param spatial_features: Spatial features of shape [B, N, S] where B = batch size, N = max examples in batch,
                             S = spatial features
    :param k: Max neighbors
    :return:
    """

    shape = spatial_features.get_shape().as_list()

    assert spatial_features.dtype == tf.float32 or spatial_features.dtype == tf.float64
    assert len(shape) == 3

    D = euclidean_squared(spatial_features, spatial_features)
    D, N = tf.nn.top_k(-D, k)
    return N, -D




def indexing_tensor(spatial_features, k=10, n_batch=-1):

    shape_spatial_features = spatial_features.get_shape().as_list()
    n_batch = shape_spatial_features[0]
    n_max_entries = shape_spatial_features[1]

    # All of these tensors should be 3-dimensional
    assert len(shape_spatial_features) == 3

    # Neighbor matrix should be int as it should be used for indexing
    assert spatial_features.dtype == tf.float64 or spatial_features.dtype == tf.float32

    neighbor_matrix, distance_matrix = nearest_neighbor_matrix(spatial_features, k)

    batch_range = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.range(0, n_batch), axis=1),axis=1), axis=1)
    batch_range = tf.tile(batch_range, [1,n_max_entries,k,1])
    expanded_neighbor_matrix = tf.expand_dims(neighbor_matrix, axis=3)
    
    
    indexing_tensor = tf.concat([batch_range, expanded_neighbor_matrix], axis=3)

    return tf.cast(indexing_tensor, tf.int64), distance_matrix




#not really needed, maybe some performance advantage
def high_dim_dense(inputs,nodes,**kwargs):
    if len(inputs.shape) == 3:
        return tf.layers.conv1d(inputs, nodes, kernel_size=(1), strides=(1), padding='valid', 
                                **kwargs)
        
    if len(inputs.shape) == 4:
        return tf.layers.conv2d(inputs, nodes, kernel_size=(1,1), strides=(1,1), padding='valid', 
                                **kwargs)
        
    if len(inputs.shape) == 5:
        return tf.layers.conv3d(inputs, nodes, kernel_size=(1,1,1), strides=(1,1,1), padding='valid', 
                                **kwargs)




def apply_edges(vertices, edges, reduce_sum=True, flatten=True,expand_first_vertex_dim=True, aggregation_function=tf.reduce_max): 
    '''
    edges are naturally BxVxV'xF
    vertices are BxVxF'  or BxV'xF'
    This function returns BxVxF'' if flattened and summed
    '''
    edges = tf.expand_dims(edges,axis=3)
    if expand_first_vertex_dim:
        vertices = tf.expand_dims(vertices,axis=1)
    vertices = tf.expand_dims(vertices,axis=4)

    out = edges*vertices # [BxVxV'x1xF] x [Bx1xV'xF'x1] = [BxVxV'xFxF']

    if reduce_sum:
        out = aggregation_function(out,axis=2)
    if flatten:
        out = tf.reshape(out,shape=[out.shape[0],out.shape[1],-1])
    
    return out



### 
### 
### 
### 
### 
### 
### actual layers
### 
### 
### 
### 
### 
### 

def layer_GarNet(vertices_in,
                 n_aggregators,
                 n_filters,
                 n_propagate,
                 plus_mean=True
                 ):
    
    vertices_in_orig = vertices_in
    vertices_in = tf.layers.dense(vertices_in,n_propagate,activation=None)
    
    agg_nodes = tf.layers.dense(vertices_in_orig,n_aggregators,activation=None) #BxVxNA, vertices_in: BxVxF
    agg_nodes = gauss(agg_nodes)
    vertices_in = tf.concat([vertices_in,agg_nodes], axis=-1)
    
    edges = tf.expand_dims(agg_nodes,axis=3) # BxVxNAx1
    edges = tf.transpose(edges, perm=[0,2, 1,3]) # [BxVxV'xF]
    
    
    vertices_in_collapsed = apply_edges(vertices_in, edges, reduce_sum=True, flatten=True)#,aggregation_function=tf.reduce_mean)# [BxNAxF]
    vertices_in_mean_collapsed = apply_edges(vertices_in, edges, reduce_sum=True, flatten=True ,aggregation_function=tf.reduce_mean)# [BxNAxF]
    
    vertices_in_collapsed= tf.concat([vertices_in_collapsed,vertices_in_mean_collapsed],axis=-1 )
    
    edges = tf.transpose(edges, perm=[0,2, 1,3]) # [BxVxV'xF]
    
    expanded_collapsed = apply_edges(vertices_in_collapsed, edges, reduce_sum=False, flatten=True)# [BxVxF]
    
    expanded_collapsed = tf.concat([vertices_in_orig,expanded_collapsed,agg_nodes], axis=-1)
    
    merged_out = high_dim_dense(expanded_collapsed,n_filters,activation=tf.nn.tanh)
    
    return merged_out
    
    
def layer_GravNet(vertices_in,
                  n_neighbours,
                  n_dimensions,
                  n_filters,
                  n_propagate):
    
    
    
    vertices_prop = high_dim_dense(vertices_in,n_propagate,activation=None)    
    neighb_dimensions = high_dim_dense(vertices_in,n_dimensions,activation=None) #BxVxND, 
    
    def collapse_to_vertex(indexing,distance,vertices):
        neighbours = tf.gather_nd(vertices, indexing)  #BxVxNxF
        distance = tf.expand_dims(distance,axis=3)
        distance = distance*10. # input is tanh activated or batch normed, allow for some more spread
        edges = gauss_of_lin(distance)[:,:,1:,:]
        neighbours = neighbours[:,:,1:,:]
        scaled_feat = edges*neighbours
        collapsed = tf.reduce_max(scaled_feat, axis=2)
        collapsed_mean = tf.reduce_mean(scaled_feat,axis=2)
        collapsed = tf.concat([collapsed,collapsed_mean],axis=-1)
        return collapsed
    
    indexing, distance = indexing_tensor(neighb_dimensions, n_neighbours)
    collapsed = collapse_to_vertex(indexing,distance,vertices_prop)
    updated_vertices = tf.concat([vertices_in,collapsed],axis=-1)

    return high_dim_dense(updated_vertices,n_filters,activation=tf.nn.tanh)


def layer_global_exchange(vertices_in):
    trans_vertices_in = vertices_in

    global_summed = tf.reduce_mean(trans_vertices_in, axis=1, keepdims=True)

    global_summed = tf.tile(global_summed, [1, vertices_in.shape[1], 1])
    vertices_out = tf.concat([vertices_in, global_summed], axis=-1)

    return vertices_out













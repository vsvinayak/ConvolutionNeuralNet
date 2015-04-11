
import cv2
import numpy as np

def pool(features, pool_dim, mode='max'):
    """
    Implements max/mean pooling 

    :param features: single or set of features
    :type images: nupy nd array

    :param pool_dim: dimension of pooling window
    :type pool_dim: tuple

    :param mode: type of pooling to be used. Available types are
                 maxp ooling and mean pooling
    :type mode: string

    :returns: pooled nd array 
    """

    n_features = features.shape[0] # number of input features
    feat_rows = features.shape[1] # rows in the input features
    feat_cols = features.shape[2] # cols in the input features
    
    # get the pooling window params
    window_rows, window_cols = pool_dim

    # dimensions of the pooled image
    pool_rows = feat_rows/window_rows
    pool_cols = feat_cols/window_cols

    # create resultant list of pooled features
    pooled_features = []

    # pooling mode
    pool_func = np.max if mode == 'max' else np.mean

    # iterate through each feature and pool 
    for feat in xrange(n_features):
        
        # process individual feature arrays
        feature = features[feat]
        
        # initialize an empty pooled array
        pool_feat = np.zeros((pool_rows, pool_cols))

        itm_size = feature.itemsize # size of a single element
        shape = (pool_rows, pool_cols, window_rows, 
                 window_cols)

        # calculate numpy strides
        strides = itm_size*np.array([feat_cols*window_rows, window_cols, 
                                     feat_cols, 1])
        
        # get the non overlapping windows of dimension pool_dim
        blocks = np.lib.stride_tricks.as_strided(feature, shape=shape,
                                               strides=strides)
        
        # get the pooled values from the window 
        for i in xrange(pool_rows):
            for j in xrange(pool_cols):
                pool_feat[i][j] = pool_func(blocks[i][j])
        
        # append the pooled feature array
        pooled_features.append(pool_feat)
    
    return np.array(pooled_features, dtype=features.dtype)

def activate(X, method='tanh'):
    """
    Returns the activation of an array
    
    :param X: array whose activation is to be calculated
    :type X : numpy array

    :param method: activation method
    :type method : string

    :returns: numpy nd array, with activation applied
    """
    
    if method == 'tanh': # hyperbolic tangent
        return np.tanh(X)
    if method == 'relu': # rectified linear
        return np.maximum(X, [0])
    else:                # sigmoid  
        return (1 / (1 + np.exp(-X)))



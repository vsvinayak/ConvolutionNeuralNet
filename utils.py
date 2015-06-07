
import numpy as np
import cv2

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


def draw_rectangle(image, rect, color=(0,255,255)):
    """
    Draws a rectange on an image
    """
    x,y,w,h = rect
    cv2.rectangle(image, (x,y), (x+w,y+h), color, 3)
    return image

def is_existing_face(image, trackers, face):
    """
    Check if the face is already an existing one among the one
    being tracked
    """

    x1, y1, w1, h1 = face
    face_mask = np.zeros_like(image)
    face_mask[y1:y1+h1, x1:x1+w1] = 1
    for t in trackers:
        try:
            x,y,w,h = t.bounding_box
            t_mask = np.zeros_like(image)
            t_mask[y:y+h, x:x+w] = 1

            union = np.sum(np.bitwise_or(face_mask, t_mask))
            intersection = np.bitwise_and(face_mask, t_mask)
            if float(np.sum(intersection))/union > 0.3 or float(np.sum(intersection))/np.sum(t_mask+1) > 0.7:
                return (t, True)
        except Exception:
            pass
    
    return (None, False)

def in_rect(keypoints, tl, br):
    """
    Check if the keypoints are within
    the rectangle
    """
    x = keypoints[:, 0]
    y = keypoints[:, 1]

    C1 = x > tl[0]
    C2 = y > tl[1]
    C3 = x < br[0]
    C4 = y < br[1]

    result = C1 & C2 & C3 & C4

    return result

def is_redundant(t, t_objects):
    """
    Checking if there is any face being tracked is redundant/duplicate
    """

    x,y,w,h = t.bounding_box

    for tracker in t_objects:
        if t.face_id == tracker.face_id:
            continue
        x_t, y_t, w_t, h_t = tracker.bounding_box
        result =  in_rect(np.array([[x,y],[x+w,y], [x,y+h], [x+w,y+h]]),
                          (x_t, y_t), (x_t+w_t, y_t+h_t))

        if sum(result) > 1:
            return True
    return False


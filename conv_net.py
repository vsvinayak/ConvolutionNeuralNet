import numpy as np
import json

from utils import pool, activate
from scipy.signal import convolve2d

class ConvNet(object):
    """
    Implements convolution neural net
    """
    
    class ConvLayer:
        """
        Construct a layer object for convolution layer
        """

        def __init__(self, conv_obj, conv_dim, pool_dim, num_feat_in, num_feat_out):
            """
            Constructor
            
            :param conv_obj : object of the ConvNet class

            :param conv_dim : dimension of the convolution operation
            :type conv_dim : tuple

            :param pool_dim : dimension of the pooling operation
            :type pool_dim : tuple

            :param num_feat : no. of output feature arrays
            :type num_feat : int
            """
            
            self.layer_type = 'convolution'

            self.conv_dim = conv_dim # convolution window dims
            self.pool_dim = pool_dim # pooling dimension
            self.conv_obj = conv_obj # calling object
            self.num_feat_in = num_feat_in # input features 
            self.num_feat_out = num_feat_out # output features

            # initialize the output feature nd array
            self.kernels = np.zeros((num_feat_out, num_feat_in, conv_dim[0], 
                                     conv_dim[1]), dtype=np.float32)

            # initialize the output biases
            self.biases = np.zeros(num_feat_out, dtype=np.float32)
            
        def execute(self, features):
            """
            Executes the layer in convolution->pooling order

            :param features : input features
            :type features : numpy nd array
            """
            
            # convolution
            convolved_features = self.conv_obj.convolution(features,
                                                           self.kernels, 
                                                           self.biases) 
            # pooling
            self.out = pool(convolved_features, self.pool_dim)

    
    class MLPLayer:
        """
        Multi layer perceprton
        """

        def __init__(self, conv_obj, num_feat_in, num_feat_out):
            """
            Constructor
            
            :param conv_obj : object of the calling ConvNet class

            :param num_feat_in : number of input features
            :type num_feat_in : int

            :param num_feat_out : number of output features
            :type num_feat_out = int
            """

            self.layer_type = 'mlp'

            self.conv_obj = conv_obj # calling object
            self.num_feat_in = num_feat_in # number of input features
            self.num_feat_out = num_feat_out # number of output features
            
            # initialize weights
            self.weights = np.zeros((num_feat_in, num_feat_out), 
                                    dtype=np.float32)

            # initialize biases
            self.biases = np.zeros(num_feat_out)

        def execute(self, features):
            """
            Executes the layer, ie, constructs the output feature array 
            as activations of the layer absed on the stored weights
            
            :params features: input features
            :type features : numpy 1d array
            """
            
            # perform dot product operation on input against the weights
            # to get the effective input to each neuron in the layer
            weighted_input = np.dot(features, self.weights) + self.biases
            
            # the output is the activation performed on weighted_input
            self.out = activate(weighted_input, method='sigmoid')


    def __init__(self, conv_params, nnet_params, input_dim):
        """
        Constructor

        :param conv_params : list of dicts convolution/subsampling config
                             eg - [{'c1' : (5,5),
                                   'p1' : (2,2),
                                   'f_in1' : 1,
                                   'f_out1' : 3}, 
                                   {'c2' : (5,5),
                                   'p2' : (2,2),
                                   'f_in2' : 3,
                                   'f_out2: 10}]

                                   cx - convolution kernel size
                                   px - pooling window size
                                   f_inx - no. of input feature arrays
                                   f_outx - np. of output feature arrays
        :type conv_params : dict

        :param nnet_params : list of tuples of mlp config
                            eg - [(1024,512), (512,10)]
        :type nnet_params: tuple

        :param input_dim : dimension of the input image, eg (45,45)
        :type input_dim : tuple
        """
        
        self.input_dim = input_dim

        # params for convolution net and mlp
        self.conv_params = conv_params
        self.nnet_params = nnet_params

        self.num_conv_layers = len(self.conv_params)
        self.num_nnet_layers = len(self.nnet_params)
        
        self.layers = []
        
        # create a layer object for each layer
        for l in range(self.num_conv_layers):
            conv_layer = self.conv_params[l]
            conv_dim = conv_layer['c'+str(l+1)]
            pool_dim = conv_layer['p'+str(l+1)]
            num_feat_in = conv_layer['f_in'+str(l+1)]
            num_feat_out = conv_layer['f_out'+str(l+1)]
            
            # create a layer object for each layer
            l_obj = self.ConvLayer(self, conv_dim, pool_dim, num_feat_in, 
                                   num_feat_out)
            
            self.layers.append(l_obj)
        
        # create the final mlp layer
        for l in range(self.num_nnet_layers):
            mlp_layer = self.nnet_params[l]
            num_feat_in = mlp_layer[0]
            num_feat_out = mlp_layer[1]

            l_obj = self.MLPLayer(self, num_feat_in, num_feat_out)

            self.layers.append(l_obj)

    
    def load_params(self, params_file):
        """
        Loads the params from a file. The params should be in json
        format

        :param params_file: the params file name
        :type params_file : string
        """

        # load the params to a dict
        params = json.load(open(params_file))

        # load weights to convolution layers
        for idx in xrange(self.num_conv_layers):
            
            layer = self.layers[idx]
            # initialize a kernel set
            kernel_set = []

            # iterate through all the weights in the params file
            # weights is of the form `Wl i j` where
            # l - layer index
            # i - ith output feature
            # j - jth input feature
            for feat_out in xrange(layer.num_feat_out):

                ker_per_feature = []
                for feat_in in xrange(layer.num_feat_in):
                    
                    index = 'W'+str(idx)+' '+str(feat_out)+' '+str(feat_in)
                    ker = [l.split(' ') for l in params[index].strip('\n').split('\n')]

                    ker_per_feature.append(ker)
                
                kernel_set.append(ker_per_feature)
                
            # load the biases
            layer.biases = np.array(params['b'+str(idx)].strip('\n').split('\n'),
                                    dtype= np.float32)
            
            # assign the weight params as part of the layer object
            layer.kernels = np.array(kernel_set, dtype=np.float32)

        # mlp layer starting index
        mlp_idx = idx

        # load the mlp weights and biases
        for idx in xrange(self.num_nnet_layers):
            
            mlp_idx += 1
            layer = self.layers[mlp_idx]

            # load the weights to the layer's weight
            index = 'W'+str(mlp_idx)
            layer.weights = np.array([l.split(' ') for l in params[index].strip('\n').split('\n')],
                                   dtype=np.float32)
            layer.biases = np.array(params['b'+str(mlp_idx)].strip('\n').split('\n'),
                                    dtype=np.float32)


    def convolution(self, features, kernels, biases):
        """
        Implements a single convolution layer

        :param features: single or a list of features on which convolution
                         needs to be performed
        :type features: numpy nd array

        :param kernels: an array of filter kernels. Only square kernels 
                        are supported as of now
        :type kernels: numpy nd array

        :param biases: an array of biases, one for each kernel
        :type biases : numpy 1d array

        :returns: numpy nd array, output features
        """
        
        n_features = features.shape[0] # number of input feature arrays
        feat_rows = features.shape[1] # rows in the input features
        feat_cols = features.shape[2] # cols in the input features

        n_kernel_sets = kernels.shape[0] # number of kernel sets, one set 
        n_kernels = kernels.shape[1] # no. of kernels per set
        kernel_dim = kernels.shape[2] # dimension of a kernel 
        
        # dimensions of a single convolved feature array
        conv_feat_rows = feat_rows - kernel_dim + 1
        conv_feat_cols = feat_cols - kernel_dim + 1

        convolved_features = [] # resultant array
        
        # iterate through all sets of filter/kernels. 
        for ker_set_idx in xrange(n_kernel_sets):
           
            # initialize a single output feature
            out_feature = np.zeros((conv_feat_rows, conv_feat_cols))
            
            for ker_idx in xrange(n_kernels):
                
            # an output feature is the sum of all convolved individual
            # input features with a particular kernel
                out_feature += convolve2d(features[ker_idx], 
                                              kernels[ker_set_idx][ker_idx],
                                              mode='valid')
            
            # add bias to the resultant feature
            out_feature += biases[ker_set_idx]
            
            # append the resultant feature array
            convolved_features.append(activate(out_feature, method='sigmoid'))
        
        return np.array(convolved_features, dtype=np.float32)

    def predict(self, test_vector):
        """
        Gives the outermost activation array for a test array

        :param test_vector : input vector
        :type test_vector : numpy nd array

        :returns: outermost acivation numpy array
        """
        
        #assert test_vector.shape == self.input_dims
        
        layer_input = test_vector
        
        # iterate through all convolution layers
        for layer in self.layers[:self.num_conv_layers]:
            
            # perfor convolution-pooling operations
            layer.execute(layer_input)
            
            # input for the next stage is output of the present stage
            layer_input = layer.out
        
        # flatten the output of the last convolution stage
        layer_input = layer_input.flatten()
        
        #iterate through mlp layers
        for layer in self.layers[self.num_conv_layers:]:
            
            # execute the layer
            layer.execute(layer_input)

            # input for next stage
            layer_input = layer.out

        # final output is the output from the last layer
        return layer.out


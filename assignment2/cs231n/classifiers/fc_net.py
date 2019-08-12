from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params['W1'] = np.random.normal(scale=weight_scale, size=(input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)                              
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass
        out_l1, cache_l1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        scores, cache_l2 = affine_forward(out_l1, self.params['W2'], self.params['b2'])
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg *(np.sum(self.params['W2']**2) + np.sum(self.params['W1']**2)) 
        dl2, grads['W2'], grads['b2'] = affine_backward(dscores, cache_l2)
        grads['W2'] += self.reg * self.params['W2']        
        
        dl1, grads['W1'], grads['b1'] = affine_relu_backward(dl2, cache_l1)
        grads['W1'] += self.reg * self.params['W1']
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

class Layer(object):
    
    def __init__(self,
                 index,
                 params_storage,
                 input_dim,
                 output_dim,
                 weight_scale,
                 forward_func,
                 backward_func):
        
        layer_suffix = str(index + 1)
        self.w_key = "W" + layer_suffix
        self.b_key = "b" + layer_suffix
        self.gamma_key = 'gamma' + layer_suffix
        self.beta_key = 'beta' + layer_suffix
        self.params_storage = params_storage
        self.size = (input_dim, output_dim)
        self.params_storage[self.w_key] = np.random.normal(scale=weight_scale, size=self.size)
        self.params_storage[self.b_key] = np.zeros(output_dim)
        
        self._forward_func = forward_func
        self._backward_func = backward_func
            
    def get_w(self):
            return self.params_storage[self.w_key]
            
    def get_b(self):
            return self.params_storage[self.b_key]
        
    def get_gamma(self):
            return self.params_storage[self.gamma_key]
        
    def get_beta(self):
            return self.params_storage[self.beta_key]
        
    W = property(get_w)
    b = property(get_b)
    gamma = property(get_gamma)
    beta = property(get_beta)
    
    def forward(self, layer_input):
        out, self.cache = self._forward_func(layer_input, self.W, self.b)
        return out
    
    def backward(self, dprevious):
        if not self.cache:
            raise "The forward method was not called beforehand!"
            
        return self._backward_func(dprevious, self.cache)
    
    def affine_batchnorm_relu_forward(self, layer_input):
        a, fc_cache = affine_forward(layer_input, self.W, self.b)
        bn, bn_cache = batchnorm_forward(a, self.gamma, self.beta, self.bn_param)
        out, relu_cache = relu_forward(bn)
        self.cache = (fc_cache, bn_cache, relu_cache)
        return out

    def affine_batchnorm_relu_backward(self, dprevious):
        if not self.cache:
            raise "The forward method was not called beforehand!"
        fc_cache, bn_cache, relu_cache = self.cache
        da = relu_backward(dprevious, relu_cache)
        dbn, dgamma, dbeta = batchnorm_backward_alt(da, bn_cache)
        dx, dw, db = affine_backward(dbn, fc_cache)
        return dx, dw, db, dgamma,dbeta
    
    def affine_layernorm_relu_forward(self, layer_input):
        a, fc_cache = affine_forward(layer_input, self.W, self.b)
        ln, ln_cache = layernorm_forward(a, self.gamma, self.beta, self.ln_param)
        out, relu_cache = relu_forward(ln)
        self.cache = (fc_cache, ln_cache, relu_cache)
        return out

    def affine_layernorm_relu_backward(self, dprevious):
        if not self.cache:
            raise "The forward method was not called beforehand!"
        fc_cache, ln_cache, relu_cache = self.cache
        da = relu_backward(dprevious, relu_cache)
        dln, dgamma, dbeta = layernorm_backward(da, ln_cache)
        dx, dw, db = affine_backward(dln, fc_cache)
        return dx, dw, db, dgamma,dbeta
      
    
class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """
    
    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        self.layers = []
        
        forward_func = affine_relu_forward
        backward_func = affine_relu_backward
        
        layer_input_dim = input_dim
        for layer_index in range(self.num_layers):
            if layer_index < len(hidden_dims):
                layer_output_dim = hidden_dims[layer_index]
            else:
                layer_output_dim = num_classes
                forward_func = affine_forward
                backward_func = affine_backward
            
            layer = Layer(layer_index,
                          self.params,
                          layer_input_dim,
                          layer_output_dim,
                          weight_scale,
                          forward_func,
                          backward_func)
            self.layers.append(layer)
            layer_input_dim = layer_output_dim
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed
                
        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
            for (i, layer) in enumerate(self.layers[:-1]):
                layer.bn_param = self.bn_params[i]
                self.params[layer.gamma_key] = np.ones(layer.size[1])
                self.params[layer.beta_key] = np.zeros(layer.size[1])
                layer.forward = layer.affine_batchnorm_relu_forward
                layer.backward = layer.affine_batchnorm_relu_backward
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]
            for (i, layer) in enumerate(self.layers[:-1]):
                layer.ln_param = {}
                self.params[layer.gamma_key] = np.ones(layer.size[1])
                self.params[layer.beta_key] = np.zeros(layer.size[1])
                layer.forward = layer.affine_layernorm_relu_forward
                layer.backward = layer.affine_layernorm_relu_backward

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
    
        
    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        layer_input = X
        for (layer_index, layer) in enumerate(self.layers):
            layer_output = layer.forward(layer_input)
            if layer_index < len(self.layers) - 1:
                if self.use_dropout:
                    layer_output, layer.dropout_cache = dropout_forward(layer_output, self.dropout_param)
                layer_input = layer_output
            else:
                scores = layer_output

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        loss, dscores = softmax_loss(scores, y)
        
        w_sum = sum([np.sum(layer.W**2) for layer in self.layers])
        loss += 0.5 * self.reg * w_sum
        
        dprevious = dscores
        last_layer = True
        for layer in reversed(self.layers):
            if self.use_dropout and not last_layer:
                dprevious = dropout_backward(dprevious, layer.dropout_cache)
            else:
                if self.normalization and not last_layer:
                    dprevious, grads[layer.w_key], grads[layer.b_key], grads[layer.gamma_key], grads[layer.beta_key] = \
                        layer.backward(dprevious)
                else:
                    dprevious, grads[layer.w_key], grads[layer.b_key] = layer.backward(dprevious)
            grads[layer.w_key] += self.reg * layer.W
            last_layer = False
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

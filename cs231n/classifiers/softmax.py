import numpy as np
from random import shuffle

def softmax(z,axis=None):
  z     -=  np.max (z,axis=axis)
  return np.exp(z)/np.sum( np.exp(z), axis=axis)


def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train   = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    f        = W.dot(X[:, i])
    scores   = softmax(f)
    loss    -= np.log(scores[y[i]])

    for j in xrange(num_classes):
      dW[j,:]    += X[:,i]*scores[j]
      if j ==y[i]:
        dW[j,:] -= X[:,i]  
        

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW   /= num_train

  # Add regularization to the loss.
  loss +=  0.5 * reg * np.sum(W * W)
  dW   +=  reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  num_classes = W.shape[0]
  num_train   = X.shape[1]
  dW   = np.zeros_like(W)

  f        = W.dot(X)
  scores   = softmax(f,axis=0)
  s        = [np.log(scores[y[i],i]) for i in xrange(num_train)]
  loss    -= np.sum( s ) 

  dW = scores.dot(X.T)
  
  for i in xrange(num_train):
    dW[y[i],:] -= X[:,i] 
    
  ''' AN equivalet loop to the above but takes more computation time
  for j in xrange(num_classes):
    dW[j,:] -= np.sum(X[:,y==j],axis=1)
  '''
 
  
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW   /= num_train

  # Add regularization to the loss.
  loss +=  0.5 * reg * np.sum(W * W)
  dW   +=  reg * W

  return loss, dW

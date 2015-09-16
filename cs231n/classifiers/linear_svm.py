import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train   = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    scores              = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    count               = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss       += margin
      	dW[j,:]    += X[:, i]
      	count += 1
    dW[y[i],:] -= count*X[:, i]
      

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW   /= num_train

  # Add regularization to the loss.
  loss +=  0.5 * reg * np.sum(W * W)
  dW   +=  reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  delta = 1.0
  loss  = 0.0
  dW    = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train   = X.shape[1]


  loss        = 0.0
  wx          = W.dot(X)
  corr_scores = np.zeros((1,num_train))
  for i in xrange(num_train):
  	corr_scores[0,i] = wx[y[i],i]
  margins = np.maximum(0, wx - corr_scores + delta)
  loss    = np.sum(margins) - delta*num_train # n_train times we returned delta instead of zero

  pos_margin = wx - corr_scores + delta > 0
  for i in xrange(num_train):
    pos_margin[y[i],i]    = 0

  dW = pos_margin.dot(X.T)
  for i in xrange(num_train):
  	dW[y[i],:] -= np.sum(pos_margin[:,i])*X[:, i]
 

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW   /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW   +=  reg * W


  return loss, dW

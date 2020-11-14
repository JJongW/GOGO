# -*- coding: utf-8 -*- 
import numpy as np
from random import shuffle

#dot 연산을 통해 score을 얻고
#score점수를 확률로 변환해서 이를 out에 저장
#class에 대한 확률값 중에 정답인 class의 확률 값만을 -log취해
#loss함수에 더한 후 모든 사진에 대해 반복함
#1/N을 구현하고 overfitting방지를 위해 regularization을 추가

def softmax_loss_naive(W, X, y, reg):
  Num,D = X.shape[0]
  Class=W.shape[1]
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  out=np.zeros((Num,Class))
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(Num):
    for j in range(Class):
      for k in range(D):
        out[i, j] += X[i, k] * W[k, j]
    out[i, :] = np.exp(out[i, :])
    out[i, :] /= np.sum(out[i, :])

  loss -= np.sum(np.log(out[np.arange(Num), y])) 
  loss /= Num
  loss += reg * np.sum(W**2)
  
  out[np.arange(N), y] -= 1
 
  for i in range(Num):
    for j in range(D):
      for k in range(Class):
        dW[j, k] += X[i, j] * out[i, k]  

  dW /= Num
  dW += 2*reg * W 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  Num = X.shape[0]
  loss = 0.0
  Class=W.shape[1]
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  score = np.dot(X, W)   
  out = np.exp(score)
  out /= np.sum(out, axis=1, keepdims=True)  
  loss -= np.sum(np.log(out[np.arange(N), y]))
  loss /= Num
  loss += reg * np.sum(W**2)
    
  dout = np.copy(out)   
  dout[np.arange(Num), y] -= 1
  dW = np.dot(X.T, dout)  
  dW /= Num
  dW +=2*reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


import numpy as np
import tensorflow as tf

def test():
 a = tf.placeholder("float")
 b = tf.placeholder("float")
   
 y = tf.mul(a, b)
   
 sess = tf.Session()
   
 print sess.run(y, feed_dict={a: 3, b: 3})


def linear_regression_1d():
 num_points = 1000
 vectors_set = []
 for i in xrange(num_points):
          x1= np.random.normal(0.0, 0.55)
          y1= x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
          vectors_set.append([x1, y1])
  
 x_data = [v[0] for v in vectors_set]
 y_data = [v[1] for v in vectors_set]

 W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
 b = tf.Variable(tf.zeros([1]))
 y = W * x_data + b
 print 'Type of y is        ',type(y)
 print 'Type of y-y_data is ',type(y-y_data)
 
 loss      = tf.reduce_mean(tf.square(y - y_data))
 optimizer = tf.train.GradientDescentOptimizer(0.5)
 train     = optimizer.minimize(loss)
 
 init = tf.initialize_all_variables()
 
 sess = tf.Session()
 sess.run(init)
 
 for step in xrange(20):
      sess.run(train)
      print(step, sess.run(W), sess.run(b))
      print(step, sess.run(loss))

test()
linear_regression_1d()
# tf_a_plus_b()
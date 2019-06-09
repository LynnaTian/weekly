# coding=utf-8
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/', one_hot=True)
x = tf.placeholder("float",[None,784])
y = tf.placeholder("float",[None,10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#softmax 多分类的问题
actv = tf.nn.softmax(tf.matmul(x,W)+ b)

#loss function:交叉熵
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(actv),reduction_indices=1))

learning_rate = 0.01
#梯度下降优化器求解，训练的过程就是最小化损失函数cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

pred = tf.equal(tf.argmax(actv,1),tf.argmax(y,1))
#cast将true和false转换为float类型，true为1和False为0，累加衡量准确率
accr = tf.reduce_mean(tf.cast(pred, "float"))

init = tf.global_variables_initializer()
epochs = 50
batch_size = 100
display_step = 5

sess = tf.Session()
sess.run(init)

for epoch in range(epochs):
    avg_cost = 0
    num_batch = int(mnist.train.num_examples/batch_size)
    for i in range(num_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x:batch_xs, y: batch_ys})
        feeds = {x:batch_xs, y: batch_ys}
        avg_cost += sess.run(cost, feed_dict=feeds)/num_batch
    if epoch % display_step == 0:
        feeds_train = {x: batch_xs, y:batch_ys}
        feeds_test = {x: mnist.test.images, y: mnist.test.labels}
        train_acc = sess.run(accr, feed_dict=feeds_train)
        test_acc = sess.run(accr, feed_dict=feeds_test)
        print("Epoch: %03d/%03d cost: %.9f train_acc: %.3f test_acc: %.3f" % (epoch, epochs, avg_cost,  train_acc, test_acc))
print 'done'
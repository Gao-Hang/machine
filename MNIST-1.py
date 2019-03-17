import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
import tensorflow as tf
import input_data


def machine():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # 交叉熵
    y_ = tf.placeholder("float", [None, 10])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

    # 梯度下降
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        result = sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        # if not result:
        #     print('预测的值是：',
        #           sess.run(y, feed_dict={x: np.array([mnist.test.images[i]]), y_: np.array([mnist.test.labels[i]])}))
        #     print('实际的值是：',
        #           sess.run(y_, feed_dict={x: np.array([mnist.test.images[i]]), y_: np.array([mnist.test.labels[i]])}))
        #     one_pic_arr = np.reshape(mnist.test.images[i], (28, 28))
        #     pic_matrix = np.matrix(one_pic_arr, dtype="float")
        #     plt.imshow(pic_matrix)
        #     pylab.show()
        #     break

    # 评估模型
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    # 把布尔值转换成浮点数，然后取平均值 [True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75.
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # 最后，我们计算所学习到的模型在测试数据集上面的正确率。

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
    machine()

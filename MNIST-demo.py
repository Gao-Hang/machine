import input_data


def machine():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    import matplotlib.pyplot as plt

    type(mnist)

    mnist.train.images

    # 查看mnist中训练集中的图像数量
    mnist.train.num_examples

    # 查看mnist中训练集中的图像数量
    mnist.train.num_examples

    # 查看mnist中第401张图片的矩阵
    mnist.train.images[400]

    # 同理，将train改为test便可以查看测试集中的各属性
    mnist.test.images[120]

    # 先将该处矩阵变成[28,28]形状，在通过matplotlib中内置函数imshow()显示图像
    plt.imshow(mnist.train.images[400].reshape(-1, 28))

    # 显示灰度的图像
    plt.imshow(mnist.train.images[400].reshape(-1, 28), cmap='gist_gray')

    # 显示转换成线性的图像
    plt.imshow(mnist.train.images[400].reshape(784, 1))


if __name__ == '__main__':
    machine()

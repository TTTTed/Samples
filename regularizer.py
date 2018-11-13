import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
BATCH_SIZE = 30
SEED = 2
# 基于seed产生随机数
rdm = np.random.RandomState(SEED)
# 返回300 * 2 的矩阵，表示300组坐标点（x0,x1）作为数据输入集
X = rdm.randn(300, 2)
#  两数平方和小于2，则为1，其余为0
Y_ = [int(x0 * x0 + x1 * x1 < 2) for x0, x1 in X]
# 1 赋值为red，其余为‘blue
Y_c = [['red' if y else 'blue'] for y in Y_]
# 对数据集X和标签Y进行shape整理，第一个元素为-1表示，第二个参数计算得到，
# 第二个参数表示多少列，把X整理为n行两列，把Y整理为n行1列
X = np.vstack(X).reshape(-1, 2)
Y_ = np.vstack(Y_).reshape(-1, 1)

print(X)
print(Y_)
print(Y_c)
# 用plt.scatter画出数据集X和行中第0列元素，和第1列元素（x0,x1）用Y_c表示
# np.squeeze()删除指定维度
plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
plt.show()


def get_weight(shape, regularizer):
    w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    # get_variable 共享相同的W
    with tf.variable_scope("",reuse=tf.AUTO_REUSE) :
        w2 = tf.get_variable(name="",shape=(),initializer=())

    # 加入正则化参数              
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.constant(0.01, shape=shape))
    return b


x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = get_weight([2, 11], 0.01)
b1 = get_bias([11])
y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = get_weight([11, 1], 0.01)
b2 = get_bias([1])
y = tf.matmul(y1, w2) + b2  # 输出层不过激活函数

# 定义损失函数
loss_mse = tf.reduce_mean(tf.square(y - y_))
loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

# 定义反向传播方法：不含正则化
LEARNING_RATE_BASE = 0.1  # 最初学习率
    LEARNING_RATE_DECAY = 0.99  # 学习率衰减率
    LEARNING_RATE_STEP = 1  # 喂入多少轮后 ，更新一次学习率，设为总样本数/batch_size

    # 设置一个计数器，跑了几轮，初值为0，设为不被训练
    global_step = tf.Variable(0, trainable=False)

    # 定义指数下降学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               LEARNING_RATE_STEP,
                                               LEARNING_RATE_DECAY,
                                               staircase=True)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)                                            
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    for i in range(40000):
        start = (i * BATCH_SIZE) % 300
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})

        if i % 2000 == 0:
            loss_mse_v = sess.run(loss_mse, feed_dict={x: X, y_: Y_})
            print('loss is ', i, loss_mse_v)

    # xx在-3到3之间，步长为0.01，yy在-3到3之间以步长0.01，生成2维网格坐标点
    xx, yy = np.mgrid[-3:3:.01, -3:3:.01]

    grid = np.c_[xx.ravel(), yy.ravel()]

    probs = sess.run(y, feed_dict={x: grid})

    probs = probs.reshape(xx.shape)
    print('w1', sess.run(w1))
    print('b1', sess.run(b1))
    print('w2', sess.run(w2))
    print('b2', sess.run(b2))

plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
plt.contour(xx, yy, probs, levels=[.5])
plt.show()

# 定义反向传播方法：包含正则化
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_total)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    for i in range(40000):
        start = (i * BATCH_SIZE) % 300
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})

        if i % 2000 == 0:
            loss_v = sess.run(loss_total, feed_dict={x: X, y_: Y_})
            print('loss is ', i, loss_v)

    xx, yy = np.mgrid[-3:3:.01, -3:3:.01]

    grid = np.c_[xx.ravel(), yy.ravel()]

    probs = sess.run(y, feed_dict={x: grid})

    probs = probs.reshape(xx.shape)

plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))

plt.contour(xx, yy, probs, levels=[.5])
plt.show()
def shallow():
    # 定义变量及滑动平均类
    # 定义一个32位浮点变量，初始值为0.0，
    w1 = tf.Variable(0, dtype=tf.float32)
    # 定义一个计数器，记录轮数，初始值为0，不可以被优化
    global_step = tf.Variable(0, trainable=False)
    # 实例化滑动平均类，衰减率为0.99，当前轮数为global_step
    Moving_Average_Decay = 0.99
    ema = tf.train.ExponentialMovingAverage(Moving_Average_Decay, global_step)
    # ema.apply()后的括号里是更新列表，每次运行sess.run(ema_op)时，就会对更新列表中的元素邱华栋平均值
    # 在实际的应用是会使用tf.trainable_variables()自动将带训练参数汇总为列表
    # ema_op = ema.apply([w1])
    ema_op = ema.apply(tf.trainable_variables())

    # 2.查看不同迭代器中变量取值的变化
    with tf.Session() as sess:
        # 初始化
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # 用ema.average(w1)获取w1的滑动平均值 （要运行多个节点，作为列表的元素列出，卸载sess.run()中）
        # 打印出当前参数w1 和 w1 的滑动平均值
        print(sess.run([w1, ema.average(w1)]))

        # 参数w1的值赋为1
        sess.run([tf.assign(w1, 1)])
        sess.run(ema_op)
        print(sess.run([w1, ema.average(w1)]))

        # 更新step 和 w1 的值，模拟100论迭代后参数变为10
        sess.run(tf.assign(global_step, 100))
        sess.run(tf.assign(w1, 10))
        sess.run(ema_op)
        print(sess.run([w1, ema.average(w1)]))

        # 每次sess.run 更新一次w1的滑动平均值
        sess.run(ema_op)
        print(sess.run([w1, ema.average(w1)]))

        sess.run(ema_op)
        print(sess.run([w1, ema.average(w1)]))

        sess.run(ema_op)
        print(sess.run([w1, ema.average(w1)]))

        sess.run(ema_op)
        print(sess.run([w1, ema.average(w1)]))

        sess.run(ema_op)
        print(sess.run([w1, ema.average(w1)]))

        sess.run(ema_op)
        print(sess.run([w1, ema.average(w1)]))
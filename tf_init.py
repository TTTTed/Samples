import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 为x,y占位
def first_step():
    x_ = tf.placeholder(dtype=tf.float32,shape=[],name="")
    y_ = tf.placeholder(dtype=tf.float32,shape=[],name="")

# 初始化权重
def second_step():
    # 常规定义
    w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))

    # get_variable 共享相同的W
    with tf.variable_scope("",reuse=tf.AUTO_REUSE) :
        w2 = tf.get_variable(name="",shape=(),initializer=())

    # 加入正则化参数
    def get_weight(shape, regularizer):
        w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
        return w

# 搭建计算图
def compute():
    pass
    return "y"

# 定义损失函数
def loss_func():
    #loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * COST, (y_ - y) * PROFIT))
    loss_mse = tf.reduce_mean(tf.square(y - y_))
    # 初始化权重时，如果加入了正则化
    loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

# 定义优化函数
def better():
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
    train_step_2 = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

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
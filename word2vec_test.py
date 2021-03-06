from conf import *

data_index = 0


def maybe_download(file_name, expected_bytes):
    if not os.path.exists(file_name):
        file_name, _ = request.urlretrieve(url + file_name, file_name)
    stat_info = os.stat(file_name)
    if stat_info.st_size == expected_bytes:
        print("Found and verified", file_name)
    else:
        print(stat_info.st_size)
        raise Exception(
            "Failed to verify" + file_name + ".Can you get to it with a browser"
        )
    return file_name


def read_data(file_name):
    with zipfile.ZipFile(file_name) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def build_dataset(words):
    count = [["UNK", -1]]
    # 取出现频次最高的前5000个词[[word，频次]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    # 用组成索引和词的键值对[word:index]
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    # 生成一个列表，将最初文件中的词全部转化为5000个词的索引，未登录此为0
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    # 计算未登录词出现的次数
    count[0][1] = unk_count

    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    # 利用词频字典统计后的data，每个单词的频数统计count，词与之对应的序号dictionary，和反转字典
    return data, count, dictionary, reverse_dictionary


# batch_size 训练轮数，num_skips，生成样本数量，skip_window窗口大小（最大关联距离）
# batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    # batch_size 中要包含每个单词的所有样本
    assert batch_size % num_skips == 0
    # 样本数量不能超过窗口大小的两倍。
    assert num_skips <= 2 * skip_window
    # 初始化变量
    batch = np.ndarray(shape=batch_size, dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    # 生成一个最大容量为span的双向队列
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        # 对队列对象使用append放法时，只会保留最后的SPAN个对象
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


def main_func():
    graph = tf.Graph()
    with graph.as_default():
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        with tf.device("/cpu:0"):
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
            )
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size))
            )
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                             biases=nce_biases,
                                             labels=train_labels,
                                             inputs=embed,
                                             num_sampled=num_sampled,
                                             num_classes=vocabulary_size))
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset
        )
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True
        )
        init_op = tf.global_variables_initializer()
    with tf.Session(graph=graph) as sess:
        init_op.run()
        print("initialized!!!")
        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(
                batch_size, num_skips, skip_window)
            feed_dict = {train_inputs: batch_inputs,
                         train_labels: batch_labels}
            _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0 and step:
                average_loss /= 2000
                print("Average loss at step", step, ":", average_loss)
                average_loss = 0

            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = "Nearest to %s :" % (valid_word)
                    for k in range(top_k):
                        closed_word = reverse_dictionary[nearest[k]]
                        log_str = "%s,%s" % (log_str, closed_word)
                    print(log_str)
        final_embeddings = normalized_embeddings.eval()
        return final_embeddings


def plot_with_labels(low_dim_embs, labels, filename="tsne.png"):
    assert low_dim_embs.shape[0] >= len(labels), "more labels than embeddings"
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords="offset points",
            ha="right",
            va="bottom"
        )
    plt.savefig(filename)


if __name__ == '__main__':
    # filename = maybe_download("text8.zip", 31344016)
    words = read_data("text8.zip")
    # print("data size is ", len(words))
    data, count, dictionary, reverse_dictionary = build_dataset(words)
    del words
    # print("Most common words (+UNK)", count[:5])
    # print("Sample data", data[:10], [reverse_dictionary[i] for i in data[:10]])
    batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
    print("batch:", batch)
    print("labels:", labels)
    # for i in range(8):
    #     print(batch[i], reverse_dictionary[batch[i]], "->", labels[i, 0], reverse_dictionary[labels[i, 0]])
    # final_embeddings = main_func()
    # tsne = TSNE(perplexity=30, n_components=2, init="pca", n_iter=5000)
    # plot_only = 100
    # low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    # labels = [reverse_dictionary[i] for i in range(plot_only)]
    # plot_with_labels(low_dim_embs, labels)



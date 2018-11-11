import jieba
from gensim.models import word2vec


def initialize():
    with open(r'docs\4422774.txt', 'rb') as f:
        document_cut = jieba.cut(f.read(), cut_all=False)
        result = ' '.join(document_cut)
        result = result.encode('utf-8')
        with open('4422774.txt', 'wb+') as f1:
            f1.write(result)  # 读取的方式和写入的方式要一致


def create_model():
    # 加载分此后的文本，使用的是Ttext2Corpus类
    sentences = word2vec.Text8Corpus(r'4422774.txt')
    # 训练模型，部分参数如下
    model = word2vec.Word2Vec(sentences, size=100, hs=1, min_count=1, window=3)
    model.save(u'4422774.model')


def use_model():
    model = word2vec.Word2Vec.load('4422774.model')
    # 计算两个词向量的相似度
    return model


def similarity(model, a, b):
    try:
        sim1 = model.similarity(a, b)
    except KeyError:
        sim1 = 0
        sim2 = 0
    print(u'%s 和 %s 的相似度为%s ' % (a, b, sim1))
    print('----------------分割线---------------------------')


def likely(model, keyword):
    # 与某个词（李达康）最相近的3个字的词
    print(u'与李达康最相近的3个字的词')
    req_count = 5
    for key in model.similar_by_word(u'李达康', topn=100):
        if len(key[0]) == 3:
            req_count -= 1
            print(key[0], key[1])
            if req_count == 0:
                break
    print('-----------------分割线---------------------------')


def related(model, keyword):
    try:
        sim3 = model.most_similar(keyword, topn=20)
        print(u'和 %s 与相关的词有：\n' % (keyword))
        for key in sim3:
            print(key[0], key[1])
    except:
        print(' error')
    print('-----------------分割线---------------------------')


def differ(model, *tuple):
    # 找出不同类的词
    sim4 = model.doesnt_match(u'泡沫 触屏 开关'.split())
    print(u'这类人物中不同类的人名', sim4)
    print("-" * 30 + '分割线' + "-" * 30)


if __name__ == '__main__':
    initialize()
    create_model()
    model = use_model()
    differ(model)

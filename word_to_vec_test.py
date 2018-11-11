# -*-coding:utf-8 -*-
import jieba
from gensim.models import word2vec


def initialize():
    name_list = ['沙瑞金', '田国富', '高育良', '侯亮平', '钟小艾', '陈岩石', '欧阳菁', '易学习', '王大路', '蔡成功', '孙连城',
                 '季昌明', '丁义珍', '郑西坡', '赵东来', '高小琴', '赵瑞龙', '林华华', '陆亦可', '刘新建', '刘庆祝', '京州市',
                 '副市长']
    for i in name_list:
        jieba.suggest_freq(i, True)
    with open('in_the_name_of_people.txt', 'rb') as f:
        document_cut = jieba.cut(f.read(), cut_all=False)
        result = ' '.join(document_cut)
        result = result.encode('utf-8')
        with open('in_the_name_of_people_segment.txt', 'wb+') as f1:
            f1.write(result)  # 读取的方式和写入的方式要一致

    # 加载分此后的文本，使用的是Ttext2Corpus类
    sentences = word2vec.Text8Corpus(r'in_the_name_of_people_segment.txt')
    print(sentences)

    # 训练模型，部分参数如下
    model = word2vec.Word2Vec("", size=100, hs=1, min_count=1, window=3)
    model.save(u'人民的名义.model')


def use_model():
    model = word2vec.Word2Vec.load('人民的名义.model')
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
    # 计算某个词(侯亮平)的相关列表
    try:
        sim3 = model.most_similar(u'侯亮平', topn=20)
        print(u'和 侯亮平 与相关的词有：\n')
        for key in sim3:
            print(key[0], key[1])
    except:
        print(' error')
    print('-----------------分割线---------------------------')


def differ(model, *tuple):
    # 找出不同类的词
    sim4 = model.doesnt_match(u'沙瑞金 高育良 李达康 刘庆祝'.split())
    print(u'这类人物中不同类的人名', sim4)
    print('-----------------分割线---------------------------')


# 以一种c语言可以解析的形式存储词向量
# model.save_word2vec_format(u"书评.model.bin", binary=True)
# 对应的加载方式
# model_3 =word2vec.Word2Vec.load_word2vec_format("text8.model.bin",binary=True)


if __name__ == '__main__':
    initialize()

    model = use_model()
    related()

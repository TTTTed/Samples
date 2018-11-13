import jieba
from gensim.models import word2vec
import jieba.posseg as pseg
import os
import sys
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

class tre_word2_vec():
	# 语料生成
	def initialize():
	    with open(r'docs\\4422774.txt', 'rb') as f:
	        document_cut = jieba.cut(f.read(), cut_all=False)

	        result = ' '.join(document_cut)

	        result = result.encode('utf-8')

	        with open('4422774.txt', 'wb+') as f1:

	            f1.write(result)  # 读取的方式和写入的方式要一致

	# 训练模型
	def create_model():
	    # 加载分此后的文本，使用的是Ttext2Corpus类
	    sentences = word2vec.Text8Corpus(r'4422774.txt')
	    # 训练模型，部分参数如下
	    model = word2vec.Word2Vec(sentences, size=100, hs=1, min_count=1, window=3)
	    model.save(u'4422774.model')

	# 加载模型
	def use_model():
	    model = word2vec.Word2Vec.load('4422774.model')
	    # 计算两个词向量的相似度
	    return model



	def use_model():
		# 计算模型中，a,b两个词的相似性
		sim1 = model.similarity(a, b)
	    
	    # 与某个词最相近的3个字的词
	    sim2 = model.similar_by_word(u'李达康', topn=100):
		
		# 相关的词
		sim3 = model.most_similar(keyword, topn=20)

		# 找出不同类的词
		sim4 = model.doesnt_match(u'泡沫 触屏 开关'.split())


class tf_idf():

    corpus = ["我 来到 北京 清华大学",  # 第一类文本切词后的结果，词之间以空格隔开
              "他 来到 了 网易 杭研 大厦",  # 第二类文本的切词结果
              "小明 硕士 毕业 与 中国 科学院",  # 第三类文本的切词结果
              "我 爱 北京 天安门"]  # 第四类文本的切词结果
    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(
    vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    

    for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        print(u"-------这里输出第", i+1, u"类文本的词语tf-idf权重------")
        for j in range(len(word)):
            print(word[j], weight[i][j])


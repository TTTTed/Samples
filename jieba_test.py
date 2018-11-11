import jieba

# 全模式
seg_list = jieba.cut("我来到北京清华大学", cut_all=True, HMM=False)
print("Full Mode: " + "/ ".join(seg_list))

# 默认模式
seg_list = jieba.cut("我来到北京清华大学", cut_all=False, HMM=True)
print("Default Mode: " + "/ ".join(seg_list))

seg_list = jieba.cut("他来到了网易杭研大厦", HMM=False)
print(", ".join(seg_list))

# 搜索引擎模式
seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造", HMM=False)
print(", ".join(seg_list))

# jieba.cut的默认参数只有三个,jieba源码如下
# cut(self, sentence, cut_all=False, HMM=True)
# 分别为:输入文本 是否为全模式分词 与是否开启HMM进行中文分词
TestStr = "2010年底部队友谊篮球赛结束"
# 因为在汉语中没有空格进行词语的分隔，所以经常会出现中文歧义，比如年底-底部-部队-队友
# jieba 默认启用了HMM（隐马尔科夫模型）进行中文分词，实际效果不错

# 全模式
seg_list = jieba.cut(TestStr, cut_all=True)
print("Full Mode:", "/ ".join(seg_list))

# 默认模式# 在默认模式下有对中文歧义有较好的分类方式
seg_list = jieba.cut(TestStr, cut_all=False)
print("Default Mode:", "/ ".join(seg_list))

# 搜索引擎模式
seg_list = jieba.cut_for_search(TestStr)
print("cut for Search", "/".join(seg_list))

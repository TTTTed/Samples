import pymysql
from jieba import cut
import logging
import csv

# 打开数据库连接
db = pymysql.connect("localhost",
                     "root",
                     "ted2333",
                     "avc-qa",
                     charset='utf8')

# 使用cursor()方法获取操作游标
cursor = db.cursor()

csv_file = csv.reader(open(r"C:\Users\nan_h\Desktop\traindata.csv", "r+", errors="ignore"))

for each_line in csv_file:
    if len(each_line) == 4:
        try:
            sent_1 = " ".join(cut(each_line[1].replace("\"\'", ""), cut_all=False))

            sent_2 = " ".join(cut(each_line[2].replace("\'\"", ""), cut_all=False))
            similarity = float(each_line[3])

            sql = """INSERT INTO traindata(sentence1,
                                                        sentence2, 
                                                        similarity) 
                                 VALUES ('%s', '%s',%f)""" % (sent_1, sent_2, similarity)

            # 执行sql语句
            cursor.execute(sql)
            # 提交到数据库
            db.commit()


        except BaseException as e:
            print("%s", e)
            db.rollback()

# 关闭数据库连接
db.close()

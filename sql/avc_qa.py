import pymysql
import codecs
from jieba import cut

# 打开数据库连接
db = pymysql.connect("localhost",
                     "root",
                     "ted2333",
                     "avc-qa",
                     charset='utf8')

# 使用cursor()方法获取操作游标
cursor = db.cursor()


def first():
    # 使用execute方法执行SQL语句
    cursor.execute("SELECT VERSION()")

    # 使用 fetchone() 方法获取一条数据
    data = cursor.fetchone()

    print("Database version : %s " % data)

    cursor.execute("DROP TABLE IF EXISTS traindata")

    # 创建数据表SQL语句
    sql = """CREATE TABLE traindata (
             id  int NOT NULL,
             sentence  CHAR(20),
             AGE INT,  
             SEX CHAR(1),
             INCOME FLOAT )"""
    cursor.execute(sql)


with open(r"C:\Users\nan_h\Desktop\traindata.csv", "r+") as f:
    for i in f:
        i = i.split(",")
        sent_1 = " ".join(cut(i[1], cut_all=False))
        print(sent_1)
        sent_2 = " ".join(cut(i[2], cut_all=False))
        print(sent_2)
        sim = float(i[3])

        print("sent1:  %s ;sent2:  %s;similarity:  %s" % (sent_1, sent_2, sim))
        sql = """INSERT INTO traindata(sentence1,
                                        sentence2, 
                                        similarity) 
                 VALUES ({}, {},{})""".format(sent_1, sent_2, float(i[3]))

        # 执行sql语句
        cursor.execute(sql)
        # 提交到数据库执行
        db.commit()
        break

# 关闭数据库连接
db.close()

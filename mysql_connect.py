import pymysql
import codecs

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


def second():
    cursor.execute("DROP TABLE IF EXISTS traindata")

    # 创建数据表SQL语句
    sql = """CREATE TABLE traindata (
             id  int NOT NULL,
             sentence  CHAR(20),
             AGE INT,  
             SEX CHAR(1),
             INCOME FLOAT )"""
    cursor.execute(sql)


with codecs.open("data.csv", "rb+") as f:
    for i in f:
        print(i[0])
        if i[0] =="id":
            continue

        i = eval(i.decode("utf-8"))
        id = int(i[0])
        sent_1 = i[1]
        sent_2 = i[2]
        sim = float(i[3])
        sql = """INSERT INTO traindata(id,
                                        sentence_1, 
                                        sentence_2,
                                        similarity)
                 VALUES (id, sent_1, sent_2,sim)"""
        try:
            # 执行sql语句
            cursor.execute(sql)
            # 提交到数据库执行
            db.commit()
        except:
            # Rollback in case there is any error
            db.rollback()

# 关闭数据库连接
db.close()

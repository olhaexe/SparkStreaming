#export SPARK_KAFKA_VERSION=0.10
#/spark2.4/bin/pyspark --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.5,com.datastax.spark:spark-cassandra-connector_2.11:2.4.2 --driver-memory 512m --driver-cores 1 --master local[1]

from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StringType, IntegerType, FloatType, TimestampType
from pyspark.sql import functions as F
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import OneHotEncoderEstimator, VectorAssembler, CountVectorizer, StringIndexer, IndexToString

spark = SparkSession.builder.appName("gogin_spark").getOrCreate()
kafka_brokers = "bigdataanalytics-worker-1.novalocal:6667"

def console_output(df, freq):
    return df.writeStream \
        .format("console") \
        .trigger(processingTime='%s seconds' % freq ) \
        .options(truncate=True) \
        .start()

schema = StructType() \
    .add("Id", IntegerType()) \
    .add("age", FloatType()) \
    .add("years_of_experience", FloatType()) \
    .add("lesson_price", FloatType())

# csv - чтение из файла, sample
teachers_info = spark \
    .readStream\
    .format("csv")\
    .schema(schema)\
    .options(path="input_teachers_csv",
             sep=",",
             header=True,
             maxFilesPerTrigger=1)\
    .load()

out = console_output(teachers_info, 5)
out.stop()

# запись в Кафку - sink
def kafka_sink(df, freq):
    return df.selectExpr("CAST(null AS STRING) as key", "CAST(struct(*) AS STRING) as value") \
        .writeStream \
        .format("kafka") \
        .trigger(processingTime='%s seconds' % freq ) \
        .option("topic", "lesson8_ob") \
        .option("kafka.bootstrap.servers", kafka_brokers) \
        .option("checkpointLocation", "checkpoints/kafka_checkpoint") \
        .start()

stream = kafka_sink(teachers_info, 5)
stream.stop()

#подготавливаем DataFrame для запросов к касандре с историческими данными
#нужны заранее keyspace и таблица того же размера, что и в кафке или файле-источнике
"""
/cassandra/bin/cqlsh 10.0.0.18 — запуск

создать схему
CREATE  KEYSPACE  lesson8
   WITH REPLICATION = {
      'class' : 'SimpleStrategy', 'replication_factor' : 1 } ;

use lesson8;

DROP TABLE test_teacher_points;
DROP TABLE test_teacher_points_predicted;

CREATE TABLE test_teacher_points
(Id int, 
age float,
years_of_experience float,
lesson_price float,
qualification float,
physics float,
chemistry float,
biology float,
english float,
geography float,
history float,
primary key (Id));

# большие буквы переводит в маленькие, тот же формат, что и фичи для записи
CREATE TABLE test_teacher_points_predicted
(id int primary key, 
age float,
years_of_experience float,
lesson_price float,
qualification float,
physics float,
chemistry float,
biology float,
english float,
geography float,
history float,
mean_exam_points float);

insert into test_teacher_points (Id, qualification, physics, chemistry, biology, english, geography, history)
values (10000, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
insert into test_teacher_points (Id, qualification, physics, chemistry, biology, english, geography, history)
values (10001, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
insert into test_teacher_points (Id, qualification, physics, chemistry, biology, english, geography, history)
values (10002, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
insert into test_teacher_points (Id, qualification, physics, chemistry, biology, english, geography, history)
values (10003, 3.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0);
insert into test_teacher_points (Id, qualification, physics, chemistry, biology, english, geography, history)
values (10004, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
insert into test_teacher_points (Id, qualification, physics, chemistry, biology, english, geography, history)
values (10005, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
insert into test_teacher_points (Id, qualification, physics, chemistry, biology, english, geography, history)
values (10006, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
insert into test_teacher_points (Id, qualification, physics, chemistry, biology, english, geography, history)
values (10007, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
insert into test_teacher_points (Id, qualification, physics, chemistry, biology, english, geography, history)
values (10008, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
insert into test_teacher_points (Id, qualification, physics, chemistry, biology, english, geography, history)
values (10009, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
"""

# положить "исторические" данные в кассандру


cassandra_features_raw = spark.read \
    .format("org.apache.spark.sql.cassandra") \
    .options(table="test_teacher_points", keyspace="lesson8" ) \
    .load()

cassandra_features_raw.show()

cassandra_features_selected = cassandra_features_raw.selectExpr("Id", "age", "years_of_experience", "lesson_price",
                                                                "qualification", "physics", "chemistry",
                                                                "biology", "english", "geography", "history")
cassandra_features_selected.show()

#подгружаем ML из HDFS
pipeline_model = PipelineModel.load("my_LR_model8_ob")

##########
#вся логика в этом foreachBatch
def writer_logic(df, epoch_id):
    df.persist()
    print("---------I've got new batch--------")
    print("This is what I've got from file source:")
    df.show()
    features_from_file_source = df.groupBy("Id") \
        .agg(F.max("age").alias("age"), F.max("years_of_experience").alias("years_of_experience"), \
             F.max("lesson_price").alias("lesson_price"), \
             F.lit(0.).alias("qualification"), \
             F.lit(0.).alias("physics"),
             F.lit(0.).alias("chemistry"), \
             F.lit(0.).alias("biology"), \
             F.lit(0.).alias("english"), \
             F.lit(0.).alias("geography"), \
             F.lit(0.).alias("history"))
    print("Here is the sums from file_source:")
    features_from_file_source.show()
    teachsers_list_df = features_from_file_source.select("Id").distinct()
    #превращаем DataFrame(Row) в Array(Row)
    teachers_list_rows = teachsers_list_df.collect()
    #превращаем Array(Row) в Array(String)
    teachers_list = map( lambda x: str(x.__getattr__("Id")) , teachers_list_rows )
    where_string = " Id = " + " or Id = ".join(teachers_list)
    print("I'm gonna select this from Cassandra:")
    print(where_string)
    print("Here is what I've got from Cassandra:")
    cassandra_features_selected.where(where_string).show()
    features_from_cassandra = cassandra_features_selected.where(where_string).na.fill(0)
    features_from_cassandra.persist()
    print("I've replaced nulls with 0 from Cassandra:")
    features_from_cassandra.show()
    #объединяем микробатч из кафки и микробатч касандры
    cassandra_file_union = features_from_file_source.union(features_from_cassandra)
    cassandra_file_aggregation = cassandra_file_union.groupBy("Id") \
        .agg(F.max("age").alias("age"), F.max("years_of_experience").alias("years_of_experience"), \
             F.max("lesson_price").alias("lesson_price"), \
             F.max("qualification").alias("qualification"), \
             F.max("physics").alias("physics"),
             F.max("chemistry").alias("chemistry"), \
             F.max("biology").alias("biology"), \
             F.max("english").alias("english"), \
             F.max("geography").alias("geography"), \
             F.max("history").alias("history"))
    print("Here is how I aggregated Cassandra and file:")
    cassandra_file_aggregation.show()
    predict = pipeline_model.transform(cassandra_file_aggregation)
    print("I've got the prediction:")
    predict.show()
    predict_short = predict.select(F.col('Id').alias('id'), 'age', 'years_of_experience', 'lesson_price',
                                   'qualification', 'physics', 'chemistry', 'biology',
                                   'english', 'geography', 'history',
                                   F.col('predicted_points').cast(FloatType()).alias('mean_exam_points'))
    print("Here is what I've got after model transformation:")
    predict_short.show()
    #обновляем исторический агрегат в касандре
    predict_short.write \
        .format("org.apache.spark.sql.cassandra") \
        .options(table="test_teacher_points_predicted", keyspace="lesson8") \
        .mode("append") \
        .save()
    features_from_cassandra.unpersist()
    print("I saved the prediction and aggregation in Cassandra. Continue...")
    df.unpersist()

#связываем источник и foreachBatch функцию, не забываем удалять чекпоинт
stream = teachers_info \
    .writeStream \
    .trigger(processingTime='30 seconds') \
    .foreachBatch(writer_logic) \
    .option("checkpointLocation", "checkpoints/test_teacher_checkpoint")

#поехали
s = stream.start()

# ждет нового от первого источника (файла или Кафки) - по обновлениям в Кассандре не пересчитывaет
s.stop()
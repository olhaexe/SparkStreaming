#export SPARK_KAFKA_VERSION=0.10
#/spark2.4/bin/pyspark --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.5,com.datastax.spark:spark-cassandra-connector_2.11:2.4.2 --driver-memory 512m --driver-cores 1 --master local[1]
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StringType, IntegerType, FloatType, TimestampType
from pyspark.sql import functions as F
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from pyspark.ml.feature import OneHotEncoderEstimator, VectorAssembler, CountVectorizer, StringIndexer, IndexToString

spark = SparkSession.builder.appName("gogin_spark").getOrCreate()
kafka_brokers = "bigdataanalytics-worker-1.novalocal:6667"

def console_output(df, freq):
    return df.writeStream \
        .format("console") \
        .trigger(processingTime='%s seconds' % freq ) \
        .options(truncate=True, numRows=100) \
        .start()

schema = StructType() \
    .add("Id", StringType()) \
    .add("MSSubClass", IntegerType()) \
    .add("MSZoning", StringType()) \
    .add("LotFrontage", StringType()) \
    .add("LotArea", IntegerType()) \
    .add("Street", StringType()) \
    .add("Alley", StringType()) \
    .add("LotShape", StringType()) \
    .add("LandContour", StringType()) \
    .add("Utilities", StringType()) \
    .add("LotConfig", StringType()) \
    .add("LandSlope", StringType()) \
    .add("Neighborhood", StringType()) \
    .add("Condition1", StringType()) \
    .add("Condition2", StringType()) \
    .add("BldgType", StringType()) \
    .add("HouseStyle", StringType()) \
    .add("OverallQual", IntegerType()) \
    .add("OverallCond", IntegerType()) \
    .add("YearBuilt", IntegerType()) \
    .add("YearRemodAdd", IntegerType()) \
    .add("RoofStyle", StringType()) \
    .add("RoofMatl", StringType()) \
    .add("Exterior1st", StringType()) \
    .add("Exterior2nd", StringType()) \
    .add("MasVnrType", StringType()) \
    .add("MasVnrArea", FloatType()) \
    .add("ExterQual", StringType()) \
    .add("ExterCond", StringType()) \
    .add("Foundation", StringType()) \
    .add("BsmtQual", StringType()) \
    .add("BsmtCond", StringType()) \
    .add("BsmtExposure", StringType()) \
    .add("BsmtFinType1", StringType()) \
    .add("BsmtFinSF1", IntegerType()) \
    .add("BsmtFinType2", StringType()) \
    .add("BsmtFinSF2", IntegerType()) \
    .add("BsmtUnfSF", IntegerType())\
    .add("TotalBsmtSF", IntegerType()) \
    .add("Heating", StringType()) \
    .add("HeatingQC", StringType()) \
    .add("CentralAir", StringType()) \
    .add("Electrical", StringType()) \
    .add("1stFlrSF", IntegerType()) \
    .add("2ndFlrSF", IntegerType()) \
    .add("LowQualFinSF", IntegerType()) \
    .add("GrLivArea", IntegerType()) \
    .add("BsmtFullBath", IntegerType()) \
    .add("BsmtHalfBath", IntegerType()) \
    .add("FullBath", IntegerType()) \
    .add("HalfBath", IntegerType()) \
    .add("BedroomAbvGr", IntegerType()) \
    .add("KitchenAbvGr", IntegerType()) \
    .add("KitchenQual", StringType()) \
    .add("TotRmsAbvGrd", IntegerType()) \
    .add("Functional", StringType()) \
    .add("Fireplaces", IntegerType()) \
    .add("FireplaceQu", StringType()) \
    .add("GarageType", StringType()) \
    .add("GarageYrBlt", FloatType()) \
    .add("GarageFinish", StringType()) \
    .add("GarageCars", IntegerType()) \
    .add("GarageArea", IntegerType()) \
    .add("GarageQual", StringType()) \
    .add("GarageCond", StringType()) \
    .add("PavedDrive", StringType()) \
    .add("WoodDeckSF", IntegerType()) \
    .add("OpenPorchSF", IntegerType()) \
    .add("EnclosedPorch", IntegerType()) \
    .add("3SsnPorch", IntegerType()) \
    .add("ScreenPorch", IntegerType()) \
    .add("PoolArea", IntegerType()) \
    .add("PoolQC", StringType()) \
    .add("Fence", StringType()) \
    .add("MiscFeature", StringType()) \
    .add("MiscVal", IntegerType()) \
    .add("MoSold", IntegerType()) \
    .add("YrSold", IntegerType()) \
    .add("SaleType", StringType()) \
    .add("SaleCondition", StringType())

# csv - стрим из файла
test_start = spark \
    .readStream\
    .format("csv")\
    .schema(schema)\
    .options(path="input_houses_csv",
             sep=",",
             header=True,
             maxFilesPerTrigger=1)\
    .load()

# если в числовом поле LotFrontage есть null, всю строку записывает нулевой
# поэтому задаем в схеме строчный тип, потом меняем на Float и заменяем null-значения
# визуально нет Id c null-значением, откуда потом вылетают - не понимаю
test = test_start\
    .withColumn("LotFrontage", F.expr("CAST(LotFrontage as FLOAT)"))\
    .na.fill({"LotFrontage": 60.0, "Id": 0})

out = console_output(test.select("Id", "LotFrontage", "LotArea"), 100)
out.stop

# довольно долго подгружается
pipeline_model = PipelineModel.load("my_GB_model8_ob")

"""
/cassandra/bin/cqlsh 10.0.0.18 — запуск

#создать схему
#CREATE  KEYSPACE  lesson8
#   WITH REPLICATION = {
#      'class' : 'SimpleStrategy', 'replication_factor' : 1 } ;

use lesson8;

DROP TABLE houses_price_prediction;
CREATE TABLE IF NOT EXISTS houses_price_prediction
(Id int primary key, 
SalePrice int);
"""

#вся логика в этом foreachBatch
def writer_logic(df, epoch_id):
    df.persist()
    print("---------I've got new batch--------")
    print("This is what I've got from streaming file source:")
    df.show()
    predict = pipeline_model.transform(df)
    print("I've got the prediction:")
    predict.show()
    # в названиях столбцов только маленькие буквы, большие Кассандра не принимает
    predict_short = predict.select(F.col('Id').alias('id'),
                                   F.col('SalePricePred').cast(IntegerType()).alias('saleprice'))
    print("Here is what I've got after model transformation:")
    predict_short.show()
    #запись данных в Кассандру
    predict_short.write \
        .format("org.apache.spark.sql.cassandra") \
        .options(table="houses_price_prediction", keyspace="lesson8") \
        .mode("append") \
        .save()
    print("I saved the prediction in Cassandra. Continue...")
    df.unpersist()

#связываем источник и foreachBatch функцию, не забываем удалять чекпоинт
stream = test \
    .writeStream \
    .trigger(processingTime='5 seconds') \
    .foreachBatch(writer_logic) \
    .option("checkpointLocation", "checkpoints/test_houses_checkpoint")

#поехали
s = stream.start()

# ждет нового из источника файлов, при появлении - отрабатывает
s.stop()

def killAll():
    for active_stream in spark.streams.active:
        print("Stopping %s by killAll" % active_stream)
        active_stream.stop()

killAll()
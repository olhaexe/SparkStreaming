#export SPARK_KAFKA_VERSION=0.10
#/spark2.4/bin/pyspark --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.5,com.datastax.spark:spark-cassandra-connector_2.11:2.4.2 --driver-memory 512m --driver-cores 1 --master local[1]
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StringType, IntegerType, TimestampType
from pyspark.sql import functions as F
from pyspark.ml.regression import LinearRegression
#from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import OneHotEncoderEstimator, VectorAssembler, CountVectorizer, StringIndexer, IndexToString

spark = SparkSession.builder.appName("gogin_spark").getOrCreate()

train = spark.read\
    .option("header", True)\
    .csv("teachers/train.csv", sep=',')\
    .withColumn("Id", F.expr("CAST(Id as INTEGER)"))\
    .withColumn("age", F.expr("CAST(age as FLOAT)"))\
    .withColumn("years_of_experience", F.expr("CAST(years_of_experience as FLOAT)"))\
    .withColumn("lesson_price", F.expr("CAST(lesson_price as FLOAT)"))\
    .withColumn("qualification", F.expr("CAST(qualification as FLOAT)"))\
    .withColumn("physics", F.expr("CAST(physics as FLOAT)"))\
    .withColumn("chemistry", F.expr("CAST(chemistry as FLOAT)"))\
    .withColumn("biology", F.expr("CAST(biology as FLOAT)"))\
    .withColumn("english", F.expr("CAST(english as FLOAT)"))\
    .withColumn("geography", F.expr("CAST(geography as FLOAT)"))\
    .withColumn("history", F.expr("CAST(history as FLOAT)"))\
    .cache()

train.show(5, False)
train.printSchema()

stages = []
label_stringIdx = StringIndexer(inputCol = 'mean_exam_points', outputCol = 'label').setHandleInvalid("keep")
stages += [label_stringIdx]

assemblerInputs = ['age', 'years_of_experience', 'lesson_price', 'qualification', 'physics', 'chemistry', 'biology', 'english', 'geography', 'history']
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features").setHandleInvalid("keep")
stages += [assembler]

lr = LinearRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)
stages += [lr]

label_stringIdx_fit = label_stringIdx.fit(train)
indexToStringEstimator = IndexToString().setInputCol("prediction").setOutputCol("predicted_points").setLabels(label_stringIdx_fit.labels)

stages +=[indexToStringEstimator]

pipeline = Pipeline().setStages(stages)
pipelineModel = pipeline.fit(train)

#сохраняем модель на HDFS
pipelineModel.write().overwrite().save("my_LR_model8_ob")

###для наглядности
pipelineModel.transform(train).select("mean_exam_points", "predicted_points").show(100)  #можно посчитать процент полной сходимости

# rmse-метрика
from pyspark.ml.evaluation import RegressionEvaluator
regressionEvaluator = RegressionEvaluator(
    predictionCol="predicted_points",
    labelCol="mean_exam_points",
    metricName="rmse")

prediction = pipelineModel.transform(train).select(F.col("mean_exam_points").cast("Float"),
                                                   F.col("predicted_points").cast("Float"))
rmse = regressionEvaluator.evaluate(prediction)
print("RMSE is " + str(rmse))

pipelineModel.transform(train).show()

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit

paramGrid = ParamGridBuilder()  \
    .addGrid(lr.regParam, [0.01, 0.1])\
    .build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(),
                          numFolds=3)  # use 3+ folds in practice

# Run cross-validation, and choose the best set of parameters.
cvModel = pipeline.fit(train)

prediction = cvModel.transform(train).select(F.col("mean_exam_points").cast("Float"),
                                                   F.col("predicted_points").cast("Float"))

rmse = regressionEvaluator.evaluate(prediction)
print("RMSE is " + str(rmse))
# не улучшается - RMSE is 14.0434860345


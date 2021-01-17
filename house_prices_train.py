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

# сразу задаем числовые колонки
train = spark.read\
    .option("header", True)\
    .csv("houses/train.csv", sep=',')\
    .withColumn("MSSubClass", F.expr("CAST(MSSubClass as INTEGER)")) \
    .withColumn("LotFrontage", F.expr("CAST (LotFrontage as INTEGER)"))\
    .withColumn("LotArea", F.expr("CAST(LotArea as INTEGER)")) \
    .withColumn("OverallQual", F.expr("CAST(OverallQual as INTEGER)")) \
    .withColumn("OverallCond", F.expr("CAST(OverallCond as INTEGER)")) \
    .withColumn("YearBuilt", F.expr("CAST(YearBuilt as INTEGER)")) \
    .withColumn("YearRemodAdd", F.expr("CAST(YearRemodAdd as INTEGER)")) \
    .withColumn("MasVnrArea", F.expr("CAST(MasVnrArea as FLOAT)"))\
    .withColumn("BsmtFinSF1", F.expr("CAST(BsmtFinSF1 as INTEGER)"))\
    .withColumn("BsmtFinSF2", F.expr("CAST(BsmtFinSF2 as INTEGER)")) \
    .withColumn("BsmtUnfSF", F.expr("CAST(BsmtUnfSF as INTEGER)")) \
    .withColumn("TotalBsmtSF", F.expr("CAST(TotalBsmtSF as INTEGER)")) \
    .withColumn("1stFlrSF", F.expr("CAST(1stFlrSF as INTEGER)"))\
    .withColumn("2ndFlrSF", F.expr("CAST(2ndFlrSF as INTEGER)")) \
    .withColumn("LowQualFinSF", F.expr("CAST(LowQualFinSF as INTEGER)")) \
    .withColumn("GrLivArea", F.expr("CAST(GrLivArea as INTEGER)")) \
    .withColumn("BsmtFullBath", F.expr("CAST(BsmtFullBath as INTEGER)"))\
    .withColumn("BsmtHalfBath", F.expr("CAST(BsmtHalfBath as INTEGER)"))\
    .withColumn("FullBath", F.expr("CAST(FullBath as INTEGER)"))\
    .withColumn("HalfBath", F.expr("CAST(HalfBath as INTEGER)"))\
    .withColumn("BedroomAbvGr", F.expr("CAST(BedroomAbvGr as INTEGER)")) \
    .withColumn("KitchenAbvGr", F.expr("CAST(KitchenAbvGr as INTEGER)"))\
    .withColumn("TotRmsAbvGrd", F.expr("CAST(TotRmsAbvGrd as INTEGER)"))\
    .withColumn("Fireplaces", F.expr("CAST(Fireplaces as INTEGER)"))\
    .withColumn("GarageYrBlt", F.expr("CAST(GarageYrBlt as FLOAT)"))\
    .withColumn("GarageCars", F.expr("CAST(GarageCars as INTEGER)"))\
    .withColumn("GarageArea", F.expr("CAST(GarageArea as INTEGER)")) \
    .withColumn("WoodDeckSF", F.expr("CAST(WoodDeckSF as INTEGER)")) \
    .withColumn("OpenPorchSF", F.expr("CAST(OpenPorchSF as INTEGER)")) \
    .withColumn("EnclosedPorch", F.expr("CAST(EnclosedPorch as INTEGER)")) \
    .withColumn("3SsnPorch", F.expr("CAST(3SsnPorch as INTEGER)")) \
    .withColumn("ScreenPorch", F.expr("CAST(ScreenPorch as INTEGER)")) \
    .withColumn("PoolArea", F.expr("CAST(PoolArea as INTEGER)")) \
    .withColumn("MiscVal", F.expr("CAST(MiscVal as INTEGER)")) \
    .withColumn("MoSold", F.expr("CAST(MoSold as INTEGER)")) \
    .withColumn("YrSold", F.expr("CAST(YrSold as INTEGER)")) \
    .withColumn("SalePrice", F.expr("CAST(SalePrice as INTEGER)")) \
    .cache()

train.show(5)
train.printSchema()

#в общем - все анализируемые колонки заносим в колонку-вектор features
# пробуем скормить неполные столбцы - получается, модель обучается и без обработки данных
categoricalColumns = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
                      'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
                      'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                      'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
                      'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                      'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
                      'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
                      'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence',
                      'MiscFeature', 'SaleType', 'SaleCondition']

stages = []
for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index').setHandleInvalid("keep")
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"]).setHandleInvalid("keep")
    stages += [stringIndexer, encoder]

label_stringIdx = StringIndexer(inputCol = 'SalePrice', outputCol = 'label').setHandleInvalid("keep")
stages += [label_stringIdx]

numericCols = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
               'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
               'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
               'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
               'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',
               'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
               '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features").setHandleInvalid("keep")
stages += [assembler]

gbtr = GBTRegressor(featuresCol = 'features', labelCol = 'label', maxIter=10)
stages += [gbtr]

label_stringIdx_fit = label_stringIdx.fit(train)
indexToStringEstimator = IndexToString().setInputCol("prediction").setOutputCol("SalePricePred").setLabels(label_stringIdx_fit.labels)

stages +=[indexToStringEstimator]

pipeline = Pipeline().setStages(stages)
pipelineModel = pipeline.fit(train)

#сохраняем модель на HDFS
pipelineModel.write().overwrite().save("my_GB_model8_ob")

###для наглядности
pipelineModel.transform(train).select("Id", "SalePrice", "SalePricePred").show(100)

regressionEvaluator = RegressionEvaluator(
    predictionCol="SalePricePred",
    labelCol="SalePrice",
    metricName="rmse")

# кушает только Float или Double
prediction = pipelineModel.transform(train).select(F.col("SalePrice").cast("Float"),
                                                   F.col("SalePricePred").cast("Float"))
rmse = regressionEvaluator.evaluate(prediction)
print("RMSE is " + str(rmse))

crossval = CrossValidator(estimator=pipeline,
                          evaluator=RegressionEvaluator(),
                          numFolds=3)  # use 3+ folds in practice

# Run cross-validation, and choose the best set of parameters.
cvModel = pipeline.fit(train)

prediction = cvModel.transform(train).select(F.col("SalePrice").cast("Float"),
                                                   F.col("SalePricePred").cast("Float"))

rmse = regressionEvaluator.evaluate(prediction)
print("RMSE is " + str(rmse))
# значение не поменялось

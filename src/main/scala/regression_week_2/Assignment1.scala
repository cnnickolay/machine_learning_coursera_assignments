package regression_week_2

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionModel, LinearRegressionWithSGD}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.{SparkConf, SparkContext}


object Assignment1 {
  def main(args: Array[String]) {
    val trainDataFile = "/Users/niko/projects/machine-learning/course_2/kc_house/kc_house_train_data.csv"
    val testDataFile = "/Users/niko/projects/machine-learning/course_2/kc_house/kc_house_test_data.csv"
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[6]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val customSchema = StructType(Array(
      StructField("id", StringType, nullable = false),
      StructField("date", StringType, nullable = false),
      StructField("price", FloatType, nullable = false),
      StructField("bedrooms", FloatType, nullable = false),
      StructField("bathrooms", FloatType, nullable = false),
      StructField("sqft_living", FloatType, nullable = false),
      StructField("sqft_lot", IntegerType, nullable = false),
      StructField("floors", StringType, nullable = false),
      StructField("waterfront", IntegerType, nullable = false),
      StructField("view", IntegerType, nullable = false),
      StructField("condition", IntegerType, nullable = false),
      StructField("grade", IntegerType, nullable = false),
      StructField("sqft_above", IntegerType, nullable = false),
      StructField("sqft_basement", IntegerType, nullable = false),
      StructField("yr_built", IntegerType, nullable = false),
      StructField("yr_renovated", IntegerType, nullable = false),
      StructField("zipcode", StringType, nullable = false),
      StructField("lat", FloatType, nullable = false),
      StructField("long", FloatType, nullable = false),
      StructField("sqft_living15", FloatType, nullable = false),
      StructField("sqft_lot15", FloatType, nullable = false)
    ))

    val df = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true") // Use first line of all files as header
      .schema(customSchema)
      .load(trainDataFile)
    val testDf = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true") // Use first line of all files as header
      .schema(customSchema)
      .load(testDataFile)

    val toIntSingleArgument = udf[Float, Float](a => a * a)
    val log = udf[Float, Float](a => Math.log(a.toDouble).toFloat)
    val toIntDoubleArgument = udf[Float, Float, Float]((a, b) => a * b)
    val toIntDoubleArgumentSumm = udf[Float, Float, Float]((a, b) => a + b)

    val testDataAugmented = testDf
      .withColumn("bedrooms_squared", toIntSingleArgument(testDf("bedrooms")))
      .withColumn("bed_bath_rooms", toIntDoubleArgument(testDf("bedrooms"), testDf("bathrooms")))
      .withColumn("log_sqft_living", log(testDf("sqft_living")))
      .withColumn("lat_plus_long", toIntDoubleArgumentSumm(testDf("lat"), testDf("long")))

    val df_augmented = df
      .withColumn("bedrooms_squared", toIntSingleArgument(df("bedrooms")))
      .withColumn("bed_bath_rooms", toIntDoubleArgument(df("bedrooms"), df("bathrooms")))
      .withColumn("log_sqft_living", log(df("sqft_living")))
      .withColumn("lat_plus_long", toIntDoubleArgumentSumm(df("lat"), df("long")))

    val dataTrain1 = df_augmented.rdd.zipWithIndex().collect { case (row, idx) =>
      LabeledPoint(row.getAs[Float]("price"),
        Vectors.dense(
          row.getAs[Float]("sqft_living"),
          row.getAs[Float]("bedrooms"),
          row.getAs[Float]("bathrooms"),
          row.getAs[Float]("lat"),
          row.getAs[Float]("long")))
    }.cache()
    val dataTrain2 = df_augmented.rdd.zipWithIndex().collect { case (row, idx) =>
      LabeledPoint(row.getAs[Float]("price"),
        Vectors.dense(
          row.getAs[Float]("sqft_living"),
          row.getAs[Float]("bedrooms"),
          row.getAs[Float]("bathrooms"),
          row.getAs[Float]("lat"),
          row.getAs[Float]("long"),
          row.getAs[Float]("bed_bath_rooms")
        ))
    }.cache()
    val dataTrain3 = df_augmented.rdd.zipWithIndex().collect { case (row, idx) =>
      LabeledPoint(row.getAs[Float]("price"),
        Vectors.dense(
          row.getAs[Float]("sqft_living"),
          row.getAs[Float]("bedrooms"),
          row.getAs[Float]("bathrooms"),
          row.getAs[Float]("lat"),
          row.getAs[Float]("long"),
          row.getAs[Float]("bed_bath_rooms"),
          row.getAs[Float]("bedrooms_squared"),
          row.getAs[Float]("log_sqft_living"),
          row.getAs[Float]("lat_plus_long")
        ))
    }.cache()
    val dataTest3 = testDataAugmented.rdd.zipWithIndex().collect { case (row, idx) =>
      LabeledPoint(row.getAs[Float]("price"),
        Vectors.dense(
          row.getAs[Float]("sqft_living"),
          row.getAs[Float]("bedrooms"),
          row.getAs[Float]("bathrooms"),
          row.getAs[Float]("lat"),
          row.getAs[Float]("long"),
          row.getAs[Float]("bed_bath_rooms"),
          row.getAs[Float]("bedrooms_squared"),
          row.getAs[Float]("log_sqft_living"),
          row.getAs[Float]("lat_plus_long")
        ))
    }.cache()
    val dataTest2 = testDataAugmented.rdd.zipWithIndex().collect { case (row, idx) =>
      LabeledPoint(row.getAs[Float]("price"),
        Vectors.dense(
          row.getAs[Float]("sqft_living"),
          row.getAs[Float]("bedrooms"),
          row.getAs[Float]("bathrooms"),
          row.getAs[Float]("lat"),
          row.getAs[Float]("long"),
          row.getAs[Float]("bed_bath_rooms")
        ))
    }.cache()
    val dataTest1 = testDataAugmented.rdd.zipWithIndex().collect { case (row, idx) =>
      LabeledPoint(row.getAs[Float]("price"),
        Vectors.dense(
          row.getAs[Float]("sqft_living"),
          row.getAs[Float]("bedrooms"),
          row.getAs[Float]("bathrooms"),
          row.getAs[Float]("lat"),
          row.getAs[Float]("long")))
    }.cache()


    println("==== bedrooms_squared")
    testDataAugmented.describe("bedrooms_squared").show()
    println("==== bed_bath_rooms")
    testDataAugmented.describe("bed_bath_rooms").show()
    println("==== log_sqft_living")
    testDataAugmented.describe("log_sqft_living").show()
    println("==== lat_plus_long")
    testDataAugmented.describe("lat_plus_long").show()

    def createModel(trainingSet: RDD[LabeledPoint], numIterations: Int, step: Double): LinearRegressionModel = {
      val modelPreparation: LinearRegressionWithSGD = new LinearRegressionWithSGD().setIntercept(true)
      modelPreparation.optimizer.setStepSize(step)
      modelPreparation.optimizer.setNumIterations(numIterations)
      modelPreparation.run(trainingSet)
    }

//    val step: Double = 0.0000001
//    val numIterations: Int = 5000
//    val step: Double = 0.001
//    val numIterations: Int = 100

    def calculateMSE(trainingSet: RDD[LabeledPoint], model: LinearRegressionModel): Double = {
      val valuesAndPreds = trainingSet.map { point =>
        val prediction = model.predict(point.features)
        (point.label, prediction)
      }
      valuesAndPreds.map { case (v, p) => math.pow(v - p, 2)}.mean()
    }

    for (
      step <- List(0.00000001);
      iterations <- 1000 to 5000 by 1000
    ) {
      val model1 = createModel(dataTrain1, iterations, step)
      val model2 = createModel(dataTrain2, iterations, step)
      val model3 = createModel(dataTrain3, iterations, step)

      println(s"========= step $step iterations $iterations")
      println(s"predicted = ${Math.round(model1.predict(Vectors.dense(1180.0, 3.0, 1.0, 47.5112, -122.257)))}")
      println(s"predicted = ${Math.round(model2.predict(Vectors.dense(1180.0, 3.0, 1.0, 47.5112, -122.257, 3)))}")
      println(s"predicted = ${Math.round(model3.predict(Vectors.dense(1180.0, 3.0, 1.0, 47.5112, -122.257, 3, 9, 7.07326971745971,-5808.576778400001)))}")
      println(s"model1 weights ${model1.weights}, intercept ${model1.intercept}")
      println(s"model2 weights ${model2.weights}, intercept ${model2.intercept}")
      println(s"model3 weights ${model3.weights}, intercept ${model3.intercept}")
      println(s"Error for model 1 on train data is ${calculateMSE(dataTrain1, model1)}")
      println(s"Error for model 2 on train data is ${calculateMSE(dataTrain2, model2)}")
      println(s"Error for model 3 on train data is ${calculateMSE(dataTrain3, model3)}")
      println(s"Error for model 1 on test data is ${calculateMSE(dataTest1, model1)}")
      println(s"Error for model 2 on test data is ${calculateMSE(dataTest2, model2)}")
      println(s"Error for model 3 on test data is ${calculateMSE(dataTest3, model3)}")
    }

  }
}
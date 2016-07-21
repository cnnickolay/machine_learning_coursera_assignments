package regression_week_2

import org.apache.spark.sql.functions.{lit, _}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Column, DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}




object Assignment2 {
  def main(args: Array[String]) {
    val trainDataFile = "/Users/niko/projects/sandbox/spark1/src/main/scala/regression_week_2/kc_house_train_data.csv"
    val testDataFile = "/Users/niko/projects/sandbox/spark1/src/main/scala/regression_week_2/kc_house_test_data.csv"
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

    val trainData = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true") // Use first line of all files as header
      .schema(customSchema)
      .load(trainDataFile)
    val testData = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true") // Use first line of all files as header
      .schema(customSchema)
      .load(testDataFile)

    def getNumpyData(df: DataFrame, features: List[String], output: String): (DataFrame, List[Float]) = {
      val _df = df.withColumn("constant", lit(1f)).select((new Column("constant") :: features.map(feature => new Column(feature))).toArray: _*)
      val observations = df.select(output).rdd.map(row => row(0).asInstanceOf[Float]).collect().toList
      (_df, observations)
    }

    def predictOutcome(df: DataFrame, weights: List[Float]): List[Float] = {
      df.rdd.map {
        row => (weights, row.toSeq).zipped.map { case (weight, feature: Float) => feature * weight }.sum
      }.collect().toList
    }

    def featureDerivative(errors: List[Float], feature: List[Float]): Float = {
      2 * (errors, feature).zipped.map(_ * _).sum
    }

    def extractFeatureValues(column: String, df: DataFrame): List[Float] = {
      df.select(column).rdd.map(row => row.getFloat(0)).collect().toList
    }

    def regressionGradientDescent(df: DataFrame, outputs: List[Float], initialWeights: List[Float], stepSize: Float, tolerance: Float, maxIterations: Integer, currentIteration: Integer): (Boolean, List[Float]) = {
      val predictions: List[Float] = predictOutcome(df, initialWeights)
      val errors: List[Float] = (predictions, outputs).zipped.map { case (prediction, output) => output - prediction }

      df.rdd.map { row => (row.toSeq, errors).zipped.map{ case (observation: Float, error) => observation * error} }
      val partials = df.columns.map { column =>
        val featureValues = extractFeatureValues(column, df)
        -featureDerivative(errors, featureValues)
      }.toList
      val updatedWeights: List[Float] = (partials, initialWeights).zipped.map {
        case (partial, initialWeight) => initialWeight - stepSize * partial
      }
      val derivative = partials.map(Math.pow(_, 2)).sum
      val gradientMagnitude = Math.sqrt(derivative)
//      println(s"Calculating iteration $currentIteration with weights $initialWeights")
//      println(s"Gradient magnitude $gradientMagnitude")

      if (gradientMagnitude < tolerance) (true, initialWeights) else {
        if (currentIteration > maxIterations) (false, initialWeights)
        else regressionGradientDescent(df, outputs, updatedWeights, stepSize, tolerance, maxIterations, currentIteration + 1)
      }
    }

    val (df1, output1) = getNumpyData(trainData, List("sqft_living"), "price")
    val (converged1, weights1) = regressionGradientDescent(df1, output1, List(-47000f, 1f), 7e-12f, 2.5e7f, 50, 0)

    println(s"$converged1 with weights $weights1")

    val applyModel1 = udf[Float, Float](sqftLiving => weights1.head + sqftLiving * weights1(1))
    val augmentedTestData = testData.withColumn("model1", applyModel1(testData("sqft_living")))
    augmentedTestData.head(1).toSeq.foreach(print)
    val rssModel1 = augmentedTestData.rdd.map { row =>
      Math.pow(row.getFloat(2) - row.getFloat(21), 2)
    }.sum
    println(s"\nRSS for model 1 is $rssModel1")


    ///////////////
    val (df2, output2) = getNumpyData(trainData, List("sqft_living", "sqft_living15"), "price")
    println(df2.columns.toList)
    val (converged2, weights2) = regressionGradientDescent(df2, output2, List(-100000f, 1f, 1f), 4e-12f, 1e9f, 500, 0)

    println(s"$converged2 with weights $weights2")

    val applyModel2 = udf[Float, Float, Float]((sqftLiving, sqftLiving15) => weights2.head + sqftLiving * weights2(1) + sqftLiving15 * weights2(2))
    val augmentedTestData2 = testData.withColumn("model2", applyModel2(testData("sqft_living"), testData("sqft_living15")))
    augmentedTestData2.head(1).toSeq.foreach(print)
    val rssModel2 = augmentedTestData2.rdd.map { row =>
      Math.pow(row.getFloat(2) - row.getFloat(21), 2)
    }.sum
    println(s"\nRSS for model 2 is $rssModel2")

  }
}
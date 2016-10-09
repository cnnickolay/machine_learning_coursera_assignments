package regression_week_4

import org.apache.spark.sql.functions.{lit, _}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Column, DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import scala.math._

object Week4Assignment2 {
  def main(args: Array[String]) {
    val salesDataFile = "/Users/niko/projects/sandbox/spark1/src/main/scala/regression_week_2/kc_house_data.csv"
    val trainDataFile = "/Users/niko/projects/sandbox/spark1/src/main/scala/regression_week_2/kc_house_train_data.csv"
    val testDataFile = "/Users/niko/projects/sandbox/spark1/src/main/scala/regression_week_2/kc_house_test_data.csv"
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[6]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val decimalType = DataTypes.createDecimalType(38, -10)
    val customSchema = StructType(Array(
      StructField("id", StringType, nullable = false),
      StructField("date", StringType, nullable = false),
      StructField("price", decimalType, nullable = false),
      StructField("bedrooms", decimalType, nullable = false),
      StructField("bathrooms", decimalType, nullable = false),
      StructField("sqft_living", decimalType, nullable = false),
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
      StructField("lat", decimalType, nullable = false),
      StructField("long", decimalType, nullable = false),
      StructField("sqft_living15", decimalType, nullable = false),
      StructField("sqft_lot15", decimalType, nullable = false)
    ))

    val salesData = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true") // Use first line of all files as header
      .schema(customSchema)
      .load(salesDataFile)
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

    def getNumpyData(df: DataFrame, features: List[String], output: String): (DataFrame, List[BigDecimal]) = {
      val _df = df.withColumn("constant", lit(BigDecimal("1.0"))).select((new Column("constant") :: features.map(feature => new Column(feature))).toArray: _*)
      val observations = df.select(output).rdd.map(row => toScalaBigDecimal(row.getDecimal(0))).collect().toList
      (_df, observations)
    }

    def predictOutcome(df: DataFrame, weights: List[BigDecimal]): List[BigDecimal] = {
      df.rdd.map {
        row => (weights, row.toSeq).zipped.map { case (weight, feature: BigDecimal) => feature * weight }.sum
      }.collect().toList
    }

    def featureDerivative(errors: List[BigDecimal], feature: List[BigDecimal], weight: BigDecimal, l2Penalty: BigDecimal, featureIsConstant: Boolean): BigDecimal = {
//      2 * (errors, feature).zipped.map(_ * _).sum + { if (!featureIsConstant) 2 * l2Penalty * weight else 0 }
      if (featureIsConstant) {
        2 * errors.sum
      } else {
        2 * (errors, feature).zipped.map(_ * _).sum + 2 * l2Penalty * weight
      }
    }

    def toScalaBigDecimal(decimal: java.math.BigDecimal): scala.math.BigDecimal = scala.math.BigDecimal(decimal.toString)

    def extractFeatureValues(column: String, df: DataFrame): List[BigDecimal] = {
      df.select(column).rdd.map(row => row.getDecimal(0)).collect().toList.map(toScalaBigDecimal)
    }

    //////////// TEST
    val (exampleFeatures, exampleOutput) = getNumpyData(salesData, List("sqft_living"), "price")
    val weights = List(BigDecimal("1"), BigDecimal("10"))
    val testPredictions = predictOutcome(exampleFeatures, weights)
    val errors = (testPredictions, exampleOutput).zipped.map(_ - _)

    val constantsFeature = extractFeatureValues("constant", exampleFeatures)
    val firstAnswer1 = featureDerivative(errors, constantsFeature, weights.head, 1, featureIsConstant = false)
    val firstAnswer2 = (errors, constantsFeature).zipped.map(_*_).sum * 2 + 20
    println(s"First answer is $firstAnswer1 and $firstAnswer2")

    val sqftFeature = extractFeatureValues("sqft_living", exampleFeatures)
    val secondAnswer1 = featureDerivative(errors, sqftFeature, weights.reverse.head, 1, featureIsConstant = true)
    val secondAnswer2 = errors.sum * 2
    println(s"Second answer is $secondAnswer1 and $secondAnswer2")
    /////////////////
/*

    def regressionGradientDescent(df: DataFrame, outputs: List[BigDecimal], initialWeights: List[BigDecimal], stepSize: BigDecimal,
                                  l2Penalty: BigDecimal, maxIterations: Integer = 100, currentIteration: Integer = 0): List[BigDecimal] = {
      val predictions: List[BigDecimal] = predictOutcome(df, initialWeights)
      val errors: List[BigDecimal] = (predictions, outputs).zipped.map { case (prediction, output) => output - prediction }

      val partials = (df.columns, initialWeights, (0 to df.columns.length).map(_ == 0)).zipped.map { case (column, weight, intercept) =>
        val featureValues = extractFeatureValues(column, df)
        -featureDerivative(errors, featureValues, weight, l2Penalty, intercept)
      }.toList
      val updatedWeights: List[BigDecimal] = (partials, initialWeights).zipped.map {
        case (partial, initialWeight) => initialWeight - stepSize * partial
      }
//      val derivative = partials.map(Math.pow(_, 2)).sum
//      val gradientMagnitude = Math.sqrt(derivative)
//      println(s"Calculating iteration $currentIteration with weights $initialWeights")
//      println(s"Gradient magnitude $gradientMagnitude")

      if (currentIteration >= (maxIterations - 1)) {
        updatedWeights
      } else {
        regressionGradientDescent(df, outputs, updatedWeights, stepSize, l2Penalty, maxIterations, currentIteration + 1)
      }
    }

    val (testNumpyData, testObservations) = getNumpyData(testData, List("sqft_living"), "price")

    List(BigDecimal(0), BigDecimal(1e11)).foreach { l2 =>
      println(s"========= L2 Penalty $l2")

      val (df1, output1) = getNumpyData(trainData, List("sqft_living"), "price")
      val weights1 = regressionGradientDescent(df1, output1, List(0, 0), 1e-12, l2)

      println(s"Weights are $weights1")

      val applyModel1 = udf[BigDecimal, BigDecimal](sqftLiving => weights1.head + sqftLiving * weights1(1))
      val augmentedTestData = testData.withColumn("model1", applyModel1(testData("sqft_living")))
      augmentedTestData.head(1).toSeq.foreach(print)
      val rssModel1 = augmentedTestData.rdd.map { row =>
        Math.pow(row.getDecimal(2) - row.getDecimal(21), 2)
      }.sum
      println(s"\nRSS for model 1 is $rssModel1")
    }
*/

  }
}
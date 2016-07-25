package regression_week_3

import com.quantifind.charts.Highcharts
import com.quantifind.charts.Highcharts._
import com.quantifind.charts.highcharts.{Exporting, Zoom, _}
import com.quantifind.charts.repl.IterablePair
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD, RidgeRegressionWithSGD}
import org.apache.spark.sql.{Column, DataFrame, Row, SQLContext}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.{SparkConf, SparkContext}
import com.quantifind.charts.Highcharts
import com.quantifind.charts.Highcharts._
import com.quantifind.charts.highcharts._
import com.quantifind.charts.repl.IterablePair
import org.apache.spark.rdd.RDD

import scala.collection.immutable.NumericRange
import scala.collection.immutable.Range._




object Assignment {
  def main(args: Array[String]) {
    val kcHouseDataFile = "src/main/scala/regression_week_3/kc_house_data.csv"
    val set1DataFile = "src/main/scala/regression_week_3/wk3_kc_house_set_1_data.csv"
    val set2DataFile = "src/main/scala/regression_week_3/wk3_kc_house_set_2_data.csv"
    val set3DataFile = "src/main/scala/regression_week_3/wk3_kc_house_set_3_data.csv"
    val set4DataFile = "src/main/scala/regression_week_3/wk3_kc_house_set_4_data.csv"
    val testDataFile = "src/main/scala/regression_week_3/wk3_kc_house_test_data.csv"
    val trainDataFile = "src/main/scala/regression_week_3/wk3_kc_house_train_data.csv"
    val validDataFile = "src/main/scala/regression_week_3/wk3_kc_house_valid_data.csv"
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[6]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val customSchema = StructType(Array(
      StructField("id", StringType, nullable = false),
      StructField("date", StringType, nullable = false),
      StructField("price", DoubleType, nullable = false),
      StructField("bedrooms", DoubleType, nullable = false),
      StructField("bathrooms", DoubleType, nullable = false),
      StructField("sqft_living", DoubleType, nullable = false),
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
      StructField("lat", DoubleType, nullable = false),
      StructField("long", DoubleType, nullable = false),
      StructField("sqft_living15", DoubleType, nullable = false),
      StructField("sqft_lot15", DoubleType, nullable = false)
    ))

    val kcHouseData = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").schema(customSchema).load(kcHouseDataFile)
    val set1Data = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").schema(customSchema).load(set1DataFile)
    val set2Data = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").schema(customSchema).load(set2DataFile)
    val set3Data = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").schema(customSchema).load(set3DataFile)
    val set4Data = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").schema(customSchema).load(set4DataFile)
    val testData = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").schema(customSchema).load(testDataFile)
    val trainData = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").schema(customSchema).load(trainDataFile)
    val validData = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").schema(customSchema).load(validDataFile)

    val toIntDoubleArgument = udf[Double, Int, Double]{(power, value) => Math.pow(power, value)}

    def polynomialDataFrame(originalDf: DataFrame, column: String, degree: Integer): DataFrame = {
      var df = originalDf.withColumn("power_1", originalDf(column))
      for (power <- 1 to degree) {
        df = df.withColumn(s"power_$power", toIntDoubleArgument(originalDf(column), lit(power)))
      }
      df
    }

    def extractDoubleList(df: DataFrame, column: String): List[Double] = df.rdd.map(_.getAs[Double]("column")).collect().toList

    def buildPlot(initialDataFrame: DataFrame, labeledPoint: (DataFrame => RDD[LabeledPoint]),
                  degree: Int, numIterations: Int, stepSize: Double, transformationColumn: String,
                  feature: String, targetColumn: String, graphicXColumn: String) = {
      val poly1Data = polynomialDataFrame(initialDataFrame, transformationColumn, degree)
      val poly1DataLabeledPointsRDD = labeledPoint(poly1Data)
      val poly1DataModelConfig = new LinearRegressionWithSGD().setIntercept(true)
      poly1DataModelConfig.optimizer.setNumIterations(numIterations).setStepSize(stepSize)
      val poly1DataModel = poly1DataModelConfig.run(poly1DataLabeledPointsRDD)
      println(s"${poly1DataModel.intercept} ${poly1DataModel.weights}")

      val calculatePrediction = udf[Double, Double](sqftLiving => poly1DataModel.intercept + poly1DataModel.weights(0) * sqftLiving)
      val poly1DataAugmented = poly1Data.withColumn(s"${targetColumn}_predicted", calculatePrediction(poly1Data(feature)))

      poly1DataAugmented.show()

      val series1: Series = Series(poly1DataAugmented.select(graphicXColumn, targetColumn).collect().zipWithIndex.collect { case (row, idx) if idx % 100 == 0 => Data(row.getAs[Double](0), row.getAs[Double](1)) }, chart = Some(SeriesType.scatter))
      val series2: Series = Series(poly1DataAugmented.select(graphicXColumn, s"${targetColumn}_predicted").collect().zipWithIndex.collect { case (row, idx) if idx % 100 == 0 => Data(row.getAs[Double](0), row.getAs[Double](1)) }, chart = Some(SeriesType.scatter))

      val chart = Highchart(Seq(series1, series2), chart = Some(Chart(zoomType = Some(Zoom.xy))), yAxis = None)
      plot(chart)
    }

    buildPlot(
      kcHouseData.select("sqft_living", "price"),
      df => df.select("price", "sqft_living").rdd.map(row => LabeledPoint(row.getAs[Double](0), Vectors.dense(row.getAs[Double](1)))),
      1, 50, 0.0000001, "sqft_living", "sqft_living", "price", "sqft_living")
    buildPlot(
      kcHouseData.select("sqft_living", "price"),
      df => df.select("price", "power_2").rdd.map(row => LabeledPoint(row.getAs[Double](0), Vectors.dense(row.getAs[Double](1)))),
      2, 1000, 0.00000000000001, "sqft_living", "power_2", "price", "sqft_living")
    buildPlot(
      kcHouseData.select("sqft_living", "price"),
      df => df.select("price", "power_3").rdd.map(row => LabeledPoint(row.getAs[Double](0), Vectors.dense(row.getAs[Double](1)))),
      3, 10000, 0.000000000000000000001, "sqft_living", "power_3", "price", "sqft_living")
  }
}
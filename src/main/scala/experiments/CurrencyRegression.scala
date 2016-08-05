package experiments


import java.lang.Math
import java.text.SimpleDateFormat

import com.quantifind.charts.Highcharts
import com.quantifind.charts.Highcharts._
import com.quantifind.charts.highcharts.{Exporting, Zoom, _}
import com.quantifind.charts.repl.IterablePair
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionModel, LinearRegressionWithSGD, RidgeRegressionWithSGD}
import org.apache.spark.sql.{Column, DataFrame, Row, SQLContext}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.{SparkConf, SparkContext}
import com.quantifind.charts.Highcharts
import com.quantifind.charts.Highcharts._
import com.quantifind.charts.highcharts._
import com.quantifind.charts.repl.IterablePair
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.param.ParamPair
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.rdd.RDD
import org.joda.time.DateTime
import org.joda.time.format.DateTimeFormat
import regression_week_3.Assignment
import util.Regression

import scala.collection.immutable.NumericRange
import scala.collection.immutable.Range._


object CurrencyRegression {
  def main(args: Array[String]) {
    val csv = "/Users/niko/Downloads/HISTDATA_COM_ASCII_EURUSD_T201512/DAT_ASCII_EURUSD_T_201512_10000.csv"

    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[6]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val customSchema = StructType(Array(
      StructField("date", StringType, nullable = false),
      StructField("rate", DoubleType, nullable = false)
    ))

    val originalDf = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "false")
      .schema(customSchema)
      .load(csv)

    val dateFormatter = new SimpleDateFormat("yyyyMMdd HHmmssSSS")

    val timeOffset = dateFormatter.parse(originalDf.first().getAs[String]("date")).getTime
    val normalizeDate = udf[Double, String](rawDate => dateFormatter.parse(rawDate).getTime - timeOffset)
    val powerValue = udf[Double, Double, Double]((power, rawDate) => Math.pow(rawDate, power))
    val dfProcessed = originalDf
      .withColumn("date_norm", normalizeDate(originalDf("date")))

    val dfAugmented = dfProcessed
      .withColumn("power_1", dfProcessed("date_norm"))
      .withColumn("power_2", powerValue(lit(2), dfProcessed("date_norm")))
//      .withColumn("power_3", powerValue(lit(3), dfProcessed("date_norm")))
//      .withColumn("power_4", powerValue(lit(4), dfProcessed("date_norm")))
//      .withColumn("power_5", powerValue(lit(5), dfProcessed("date_norm")))
//      .withColumn("power_6", powerValue(lit(6), dfProcessed("date_norm")))

    val trainData = dfAugmented.rdd
//      .zipWithIndex()
//      .filter{ case (row, idx) => idx < 100 }
//      .map{ case (row, idx) => row }
      .map{ row => LabeledPoint(row.getAs[Double]("rate"), Vectors.dense(row.getAs[Double]("power_1"), row.getAs[Double]("power_2"))) }
      .cache()


//    val model = RidgeRegressionWithSGD.train(trainData, 100, stepSize = 1E-50, regParam = 0.01)
//    val predictionUdf = udf { (x: Double, y: Double) => model.predict(Vectors.dense(x, y)) }
//    val predictedDf = dfAugmented
//      .withColumn("prediction", predictionUdf(dfAugmented("power_1"), dfAugmented("power_2")))
//
    def renderPlot(title: String, xAxis: String, yAxis: String, predictionYAxis: String*)(df: DataFrame): Unit = {
      val series1: Series = Series(df.select(xAxis, yAxis).rdd
        .zipWithIndex
        .filter{ case (row, idx) => idx % 10000 == 0 }
        .map(_._1)
        .map { row => Data(row.getAs[Double](0), row.getAs[Double](1)) }.collect(), chart = Some(SeriesType.scatter))

      val series2: List[Series] = predictionYAxis.map { prediction => Series(df.select(xAxis, prediction).rdd
        .zipWithIndex
        .filter { case (row, idx) => idx % 10000 == 0 }
        .map(_._1)
        .map { row => Data(row.getAs[Double](0), row.getAs[Double](1)) }.collect(), chart = Some(SeriesType.scatter))
      }.toList

      val chart = Highchart(series1 :: series2 ::: Nil, chart = Some(Chart(zoomType = Some(Zoom.xy))), yAxis = None, title = Some(Title(title)))
      plot(chart)
    }

//    renderPlot("test", "date_norm", "rate", "prediction", "prediction2", "prediction3")(predictedDf)

    Assignment.buildPlot(dfProcessed, row => LabeledPoint(row.getAs[Double]("rate"), Vectors.dense(row.getAs[Double]("power_1"), row.getAs[Double]("power_2"), row.getAs[Double]("power_3"), row.getAs[Double]("power_4"), row.getAs[Double]("power_5"))), 5, 1E-7, "date_norm", Assignment.renderPlot("", "date_norm", "rate"))
  }
}
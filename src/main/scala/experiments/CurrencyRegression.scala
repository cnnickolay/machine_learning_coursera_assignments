package experiments


import java.text.SimpleDateFormat

import com.quantifind.charts.Highcharts._
import com.quantifind.charts.highcharts._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.{SparkConf, SparkContext}

import scala.annotation.tailrec


object CurrencyRegression {
  def renderPlot(title: String, xAxis: String, yAxis: String, predictedYAxis: String, filterRatio: Int)(df: DataFrame): Unit = {
    val series1: Series = Series(name = Some("observations"), data = df.select(xAxis, yAxis).collect().zipWithIndex.collect { case (row, idx) if idx % filterRatio == 0 => Data(row.getAs[Double](0), row.getAs[Double](1)) }, chart = Some(SeriesType.scatter))
    val series2: Series = Series(name = Some("prediction"), data = df.select(xAxis, predictedYAxis).collect().zipWithIndex.collect { case (row, idx) if idx % filterRatio == 0 => Data(row.getAs[Double](0), row.getAs[Double](1)) }, chart = Some(SeriesType.scatter))

    val chart = Highchart(Seq(series1, series2), chart = Some(Chart(zoomType = Some(Zoom.xy))), yAxis = None, title = Some(Title(title)))
    plot(chart)
  }


  def main(args: Array[String]) {
    val csv = "/Users/niko/Downloads/HISTDATA_COM_ASCII_EURUSD_T201512/DAT_ASCII_EURUSD_T_201512_10000.csv"

    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[6]")
    val sc = new SparkContext(conf)
    val sqlContext = SparkSession.builder().getOrCreate()

    val customSchema = StructType(Array(
      StructField("date", StringType, nullable = false),
      StructField("label", DoubleType, nullable = false)
    ))

    val rawDf = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "false")
      .schema(customSchema)
      .load(csv)

    val dateFormatter = new SimpleDateFormat("yyyyMMdd HHmmssSSS")

    val timeOffset = dateFormatter.parse(rawDf.first().getAs[String]("date")).getTime
    val normalizeDate = udf[Double, String](rawDate => dateFormatter.parse(rawDate).getTime - timeOffset)
    val powerValue = udf[Double, Double, Int]((rawDate, power) => Math.pow(rawDate, power))
    val dfProcessed = rawDf
      .withColumn("date_norm", normalizeDate(rawDf("date")))

    @tailrec def generatePolynomial(df: DataFrame, currentDegree: Int, maxDegree: Int): DataFrame = {
      if (currentDegree <= maxDegree) generatePolynomial(df.withColumn(s"power_$currentDegree", powerValue(dfProcessed("date_norm"), lit(currentDegree))), currentDegree + 1, maxDegree)
      else df
    }

    val df = generatePolynomial(dfProcessed, 1, 10)
    df.show()

    val assembler = new VectorAssembler()
      .setInputCols(Array("power_1", "power_2", "power_3", "power_4", "power_5", "power_6", "power_7", "power_8", "power_9", "power_10"))
      .setOutputCol("features")

    val dfPrepared = assembler.transform(df)

    val lr = new LinearRegression()
      .setMaxIter(100)
      .setRegParam(0)
      .setElasticNetParam(1)
      .setFitIntercept(true)

    val model = lr.fit(dfPrepared)
    val dfAugmented = model.transform(dfPrepared)

    dfAugmented.show()

    renderPlot("test", "date_norm", "label", "prediction", 20)(dfAugmented)

  }
}
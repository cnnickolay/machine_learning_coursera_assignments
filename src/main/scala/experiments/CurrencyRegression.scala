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
  def renderPlot(filterRatio: Int, title: String, xAxis: String, yAxis: String, predictedYAxis: String*)(df: DataFrame): Unit = {
    val observationSeries: Series = Series(name = Some("observations"), data = df.select(xAxis, yAxis).collect().zipWithIndex.collect { case (row, idx) if idx % filterRatio == 0 => Data(row.getAs[Double](0), row.getAs[Double](1)) }, chart = Some(SeriesType.scatter))
    val predictedSeries: List[Series] = predictedYAxis.map{ predicted => Series(name = Some(predicted), data = df.select(xAxis, predicted).collect().zipWithIndex.collect { case (row, idx) if idx % filterRatio == 0 => Data(row.getAs[Double](0), row.getAs[Double](1)) }, chart = Some(SeriesType.scatter)) }.toList

    val chart = Highchart(predictedSeries ::: observationSeries :: Nil, chart = Some(Chart(zoomType = Some(Zoom.xy))), yAxis = None, title = Some(Title(title)))
    plot(chart)
  }


  def main(args: Array[String]) {
    val csv = "DAT_ASCII_EURUSD_T_201512.csv"

    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[6]")
    val sc = new SparkContext(conf)
    val sqlContext = SparkSession.builder().getOrCreate()

    val customSchema = StructType(Array(
      StructField("date", StringType, nullable = false),
      StructField("label", DoubleType, nullable = false)
    ))

    val _rawDf = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "false")
      .schema(customSchema)
      .load(csv)

    List("01", "02", "03", "04", "07", "08", "09", "10", "11",  "14", "15", "16", "17", "18", "21", "22", "23", "24", "25", "28", "29", "30", "31")
      .foreach { value =>
        val rawDf = _rawDf.filter(row => row.getAs[String]("date").startsWith(s"201512$value"))

        val dateFormatter = new SimpleDateFormat("yyyyMMdd HHmmssSSS")

        val timeOffset = dateFormatter.parse(rawDf.first().getAs[String]("date")).getTime
        val normalizeDate = udf[Double, String](rawDate => Math.round((dateFormatter.parse(rawDate).getTime - timeOffset)/(1000*60)))
        val powerValue = udf[Double, Double, Int]((rawDate, power) => Math.pow(rawDate, power))
        val dfProcessed = rawDf.withColumn("date_norm", normalizeDate(rawDf("date")))

        @tailrec def generatePolynomial(df: DataFrame, currentDegree: Int, maxDegree: Int): DataFrame = {
          if (currentDegree <= maxDegree) generatePolynomial(df.withColumn(s"power_$currentDegree", powerValue(dfProcessed("date_norm"), lit(currentDegree))), currentDegree + 1, maxDegree)
          else df
        }

        val df = generatePolynomial(dfProcessed, 1, 20)
        df.show()

        val assembler = new VectorAssembler()

        val dfPrepared1 = assembler.setInputCols((1 to 10).map(i => s"power_$i").toArray).setOutputCol("features1").transform(df)
        val dfPrepared2 = assembler.setInputCols((1 to 5).map(i => s"power_$i").toArray).setOutputCol("features2").transform(dfPrepared1)

        val lr = new LinearRegression()
          .setMaxIter(100)
          .setRegParam(0)
          .setElasticNetParam(0)
          .setFitIntercept(true)

        val model1 = lr.setFeaturesCol("features1").setPredictionCol("prediction1").fit(dfPrepared2)
        val model2 = lr.setFeaturesCol("features2").setPredictionCol("prediction2").fit(dfPrepared2)
        val dfAugmented1 = model1.transform(dfPrepared2)
        val dfAugmented2 = model2.transform(dfAugmented1)

        renderPlot(100, value, "date_norm", "label", "prediction1", "prediction2")(dfAugmented2)
      }
  }
}
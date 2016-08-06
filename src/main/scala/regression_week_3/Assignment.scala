package regression_week_3

import java.lang.Math

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
import org.apache.spark.rdd.RDD
import util.Regression

import scala.collection.immutable.NumericRange
import scala.collection.immutable.Range._




object Assignment {

  def renderPlot(title: String, xAxis: String, yAxis: String)(df: DataFrame): Unit = {
    val series1: Series = Series(df.select(xAxis, yAxis).collect().zipWithIndex.collect { case (row, idx) if idx % 100 == 0 => Data(row.getAs[Double](0), row.getAs[Double](1)) }, chart = Some(SeriesType.scatter))
    val series2: Series = Series(df.select(xAxis, "predicted").collect().zipWithIndex.collect { case (row, idx) if idx % 100 == 0 => Data(row.getAs[Double](0), row.getAs[Double](1)) }, chart = Some(SeriesType.scatter))

    val chart = Highchart(Seq(series1, series2), chart = Some(Chart(zoomType = Some(Zoom.xy))), yAxis = None, title = Some(Title(title)))
    plot(chart)
  }

  val toIntDoubleArgument = udf[Double, Double, Int]{(value, power) => Math.pow(value, power)}

  def polynomialDataFrame(originalDf: DataFrame, column: String, degree: Integer): DataFrame = {
    var df = originalDf.withColumn("power_1", originalDf(column))
    for (power <- 0 to degree) {
      df = df.withColumn(s"power_$power", toIntDoubleArgument(originalDf(column), lit(power)))
    }
    df
  }

  def buildPlot(initialDataFrame: DataFrame, labeledPoint: (Row => LabeledPoint),
                degree: Int, initialStep: Double, transformationColumn: String,
                renderGraph: (DataFrame => Unit)) = {
    val poly1Data = polynomialDataFrame(initialDataFrame, transformationColumn, degree)
    poly1Data.show()
    val poly1DataLabeledPointsRDD = poly1Data.rdd.map(row => labeledPoint(row))
    val poly1DataModelConfig = new LinearRegressionWithSGD().setIntercept(true)
    poly1DataModelConfig.optimizer.setNumIterations(100)

    def findApproximation(prevModel: Option[LinearRegressionModel], step: Double): LinearRegressionModel = {
      poly1DataModelConfig.optimizer.setStepSize(step)
      val model = poly1DataModelConfig.run(poly1DataLabeledPointsRDD)
      prevModel match {
        case Some(_prevModel) => println(s"step $step prevmodel weight ${_prevModel.weights(0)}, nextmodel weight ${model.weights(0)}")
        case None => println(s"step $step nextmodel weight ${model.weights(0)}")
      }

      prevModel match {
//        case Some(_prevModel) if step < 1E-100 => _prevModel
//        case Some(_prevModel) if _prevModel.weights.toArray.product.isNaN && model.weights.toArray.product.isNaN => findApproximation(Some(model), step / 10)
//        case Some(_prevModel) if !_prevModel.weights.toArray.product.isNaN && model.weights.toArray.product.isNaN => _prevModel
//        case Some(_prevModel) if _prevModel.weights.toArray.product.isNaN && !model.weights.toArray.product.isNaN => findApproximation(Some(model), step / 10)
        case Some(_prevModel) =>
          val firstRow = poly1DataLabeledPointsRDD.first()
          val calculateRMSE: ((LinearRegressionModel, RDD[LabeledPoint]) => Double) = {
            (model, points) =>
              val predicted = model.predict(points.map(_.features)).collect()
              val averageError = (predicted, points.map(_.label).collect()).zipped.map {
                case (predict, observation) => Math.pow(predict - observation, 2)
              }.sum / points.count()
              Math.sqrt(averageError)
          }
          val prevModelError = calculateRMSE(_prevModel, poly1DataLabeledPointsRDD)
          val modelError = calculateRMSE(model, poly1DataLabeledPointsRDD)
          println(s"label ${firstRow.label}, prev predict error $prevModelError, next predict error $modelError")
          println(s"Model weights ${model.intercept} ${model.weights}")
          if (prevModelError.isNaN && modelError.isNaN) findApproximation(Some(model), step / 10)
          else if (prevModelError.isNaN && !modelError.isNaN) findApproximation(Some(model), step / 10)
          else if (!prevModelError.isNaN && modelError.isNaN) _prevModel
          else if (prevModelError.isInfinity || modelError.isInfinity) findApproximation(Some(model), step / 10)
          else if (prevModelError > modelError) {
//            println(s"label ${firstRow.label}, prev predict error $prevModelError, next predict error $modelError")
            findApproximation(Some(model), step / 10)
          } else _prevModel
        case None => findApproximation(Some(model), step / 10)
      }
    }
    val model = findApproximation(None, initialStep)

    val augmentedRows = poly1Data.rdd.map { row =>
      val prediction = model.predict(labeledPoint(row).features)
      val values = row.toSeq.toList ::: prediction :: Nil
      Row(values: _*)
    }
    val poly1DataAugmented = poly1Data.sqlContext.createDataFrame(augmentedRows, poly1Data.schema.add("predicted", DoubleType))

    poly1DataAugmented.show()

    renderGraph(poly1DataAugmented)
    poly1DataAugmented
  }


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

    def extractDoubleList(df: DataFrame, column: String): List[Double] = df.rdd.map(_.getAs[Double](column)).collect().toList

//    buildPlot(
//      kcHouseData.select("sqft_living", "price"),
//      df => df.select("price", "sqft_living").rdd.map(row => LabeledPoint(row.getAs[Double](0), Vectors.dense(row.getAs[Double](1)))),
//      1, 1E-5, "sqft_living", renderPlot("1st order", "sqft_living"))
//    buildPlot(
//      kcHouseData.select("sqft_living", "price"),
//      df => df.select("price", "power_1", "power_2").rdd.map(row => LabeledPoint(row.getAs[Double](0), Vectors.dense(row.getAs[Double](1), row.getAs[Double](2)))),
//      2, 1E-10, "sqft_living", renderPlot("2nd order", "sqft_living"))
//    buildPlot(
//      kcHouseData.select("sqft_living", "price"),
//      df => df.select("price", "power_1", "power_2", "power_3").rdd.map(row => LabeledPoint(row.getAs[Double](0), Vectors.dense(row.getAs[Double](1), row.getAs[Double](2), row.getAs[Double](3)))),
//      3, 1E-10, "sqft_living", renderPlot("3rd order", "sqft_living"))
//    buildPlot(
//      kcHouseData.select("sqft_living", "price"),
//      df => df.select("price", "power_1", "power_2", "power_3", "power_4").rdd.map(row => LabeledPoint(row.getAs[Double](0), Vectors.dense(row.getAs[Double](1), row.getAs[Double](2), row.getAs[Double](3), row.getAs[Double](4)))),
//      4, 1E-25, "sqft_living", renderPlot("4rd order", "sqft_living"))
//    buildPlot(
//      kcHouseData.select("sqft_living", "price"),
//      df => df.select("price", "power_1", "power_2", "power_3", "power_4", "power_5").rdd.map(row => LabeledPoint(row.getAs[Double](0), Vectors.dense(row.getAs[Double](1), row.getAs[Double](2), row.getAs[Double](3), row.getAs[Double](4), row.getAs[Double](5)))),
//      5, 1E-30, "sqft_living", renderPlot("5rd order", "sqft_living"))

//    val weights_1 = Regression.regressionGradientDescent(kcHouseData.select("sqft_living"), extractDoubleList(kcHouseData, "price"), List(263.0850134528269), 0.0000000000001, 0.0001)
//    val kcHouseDataAugmented_1 = Regression.predict(kcHouseData, List("sqft_living"), weights_1)
//    kcHouseDataAugmented_1.select("sqft_living", "price", "predicted").show()

    def calculateRSS(df: DataFrame): Double = {
      df.rdd.map(row => Math.pow(row.getAs[Double]("price") - row.getAs[Double]("predicted"), 2)).sum()
    }

    val kcHouseDataPol = polynomialDataFrame(kcHouseData, "sqft_living", 15)
    val trainDataPol = polynomialDataFrame(trainData, "sqft_living", 15)
    val validationDataPol = polynomialDataFrame(validData, "sqft_living", 15)
    val testDataPol = polynomialDataFrame(testData, "sqft_living", 15)
    validationDataPol.show()

    (1 to 16 by 1).foreach { idx =>
      val features = (0 to idx).map("power_"+_).toList
      val weights = Regression.regressionGradientDescent(trainDataPol.select(features.head, features.tail: _*), extractDoubleList(trainDataPol, "price"), List.fill(idx-1)(0d) ::: 100d :: Nil, Nil, 0.00001, 0.00000001, maxSteps = Some(100))
      val trainDataAugmented = Regression.predict(trainDataPol, features, weights)
      val testDataAugmented = Regression.predict(testDataPol, features, weights)
      val validationDataAugmented = Regression.predict(validationDataPol, features, weights)
      println(s"----------- $idx")
      renderPlot(idx.toString, "sqft_living", "price")(validationDataAugmented)
      println(s"weights $weights")
      println(s"RSS train ${calculateRSS(trainDataAugmented)}")
      println(s"RSS validation ${calculateRSS(validationDataAugmented)}")
      println(s"RSS test ${calculateRSS(testDataAugmented)}")
    }

  }

}
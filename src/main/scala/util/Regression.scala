package util

import com.quantifind.charts.Highcharts._
import com.quantifind.charts.highcharts._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, Row}

import scala.annotation.tailrec

object Regression {

  private def predictOutcome(df: DataFrame, weights: List[Double]): List[Double] = {
    df.rdd.map {
      row => (weights, row.toSeq).zipped.map { case (weight, feature: Double) => feature * weight }.sum
    }.collect().toList
  }

  private def extractFeatureValues(column: String, df: DataFrame): List[Double] = {
    df.select(column).rdd.map(row => row.getDouble(0)).collect().toList
  }

  private def featureDerivative(errors: List[Double], feature: List[Double]): Double = {
    2 * (errors, feature).zipped.map(_ * _).sum
  }

  private def RMSE(df: DataFrame, observations: List[Double], weights: List[Double]): Double = {
    val predictions = df.rdd.map(row => row.toSeq.map(value => value.asInstanceOf[Double]).toList)
      .map(values => (values, weights).zipped.map(_*_).sum).collect()

    Math.sqrt((observations, predictions).zipped.map(_-_).map(Math.pow(_, 2)).sum / observations.size)
  }

  val correction = 1000000

  @tailrec def regressionGradientDescent(df: DataFrame, observations: List[Double], initialWeights: List[Double], historicalWeights: List[List[Double]], stepSize: Double, errorDiffThreshold: Double, currentStep: Option[Int] = None, maxSteps: Option[Int] = None): List[Double] = {
    val _initialWeights = if (initialWeights.size == df.columns.length) initialWeights else List.fill(df.columns.length)(0d)
    val predictions: List[Double] = predictOutcome(df, _initialWeights)
    val errors: List[Double] = (predictions, observations).zipped.map { case (prediction, output) => output - prediction }

    df.rdd.map { row => (row.toSeq, errors).zipped.map{ case (observation: Double, error) => observation * error} }
    val partials = df.columns.map { column =>
      val featureValues = extractFeatureValues(column, df)
      -featureDerivative(errors, featureValues)
    }.toList
    val updatedWeights: List[Double] = (partials, _initialWeights).zipped.map { case (partial, initialWeight) =>
        initialWeight - stepSize * partial
    }

    val weightsDiff = _initialWeights.zip(updatedWeights).map { case (initial, updated) => (initial - updated) * correction }
    //    val _updatedWeights = if (currentStep.getOrElse(0) > 10 && currentStep.getOrElse(0) % 20 == 0) {
    //      println("---------")
    val _updatedWeights = if (historicalWeights.size >= 3) {
      val beforeLast = historicalWeights.reverse.tail.head
      val current = (_initialWeights, updatedWeights).zipped.map((x, y) => Math.abs(x - y))
      val prev = (beforeLast, _initialWeights).zipped.map((x, y) => Math.abs(x - y))
      val percentChange = (current, prev).zipped.map((y, x) => x / y * 100)
      val diff = percentChange.map(Math.round).map(_ - 100).sum
      if (diff == 0) {
        println("---------")
        initialWeights.zip(weightsDiff).map { case (x, y) => x - y }
      } else updatedWeights
    } else updatedWeights

    val rmse = RMSE(df, observations, _initialWeights)
    val rmseWithUpdatedWeights = RMSE(df, observations, updatedWeights)

    val errorDiff = rmse - rmseWithUpdatedWeights
    val gradientMagnitude = Math.sqrt(partials.map(Math.pow(_, 2)).sum)
    println(s"magnitude $gradientMagnitude, error diff $errorDiff weights ${_initialWeights}")
//    println(s"gradient magnitude $gradientMagnitude, weights ${_initialWeights}")

    if (errorDiff > 0 && errorDiff > errorDiffThreshold) {
      if (currentStep.getOrElse(0) >= maxSteps.getOrElse(Int.MaxValue)) _initialWeights else regressionGradientDescent(df, observations, _updatedWeights, historicalWeights ::: _updatedWeights :: List[List[Double]](), stepSize, errorDiffThreshold, Some(currentStep.getOrElse(0) + 1), maxSteps)
    }
    else if (errorDiff < 0) { println(s"adjusting step $stepSize"); regressionGradientDescent(df, observations, _initialWeights, historicalWeights, stepSize / 10, errorDiffThreshold, currentStep, maxSteps) }
    else { println(s"weights found ${_initialWeights} with error diff $errorDiff. Steps required ${currentStep.getOrElse(0)}"); _initialWeights }
  }

  def predict(df: DataFrame, columns: List[String], weights: List[Double]): DataFrame = {
    val rddAugmented = df.rdd.map(row => {
      val rowDouble = columns.map(row.getAs[Double])
      val prediction = (rowDouble, weights).zipped.map(_*_).sum
      val values = row.toSeq.toList ::: prediction :: Nil
      Row(values: _*)
    })
    df.sqlContext.createDataFrame(rddAugmented, df.schema.add("predicted", DoubleType))
  }

  def renderPlot(title: String, xAxis: String, yAxis: String, predictedYAxis: String, filterRatio: Int)(df: DataFrame): Unit = {
    val series1: Series = Series(name = Some("observations"), data = df.select(xAxis, yAxis).collect().zipWithIndex.collect { case (row, idx) if idx % filterRatio == 0 => Data(row.getAs[Double](0), row.getAs[Double](1)) }, chart = Some(SeriesType.scatter))
    val series2: Series = Series(name = Some("prediction"), data = df.select(xAxis, predictedYAxis).collect().zipWithIndex.collect { case (row, idx) if idx % filterRatio == 0 => Data(row.getAs[Double](0), row.getAs[Double](1)) }, chart = Some(SeriesType.scatter))

    val chart = Highchart(Seq(series1, series2), chart = Some(Chart(zoomType = Some(Zoom.xy))), yAxis = None, title = Some(Title(title)))
    plot(chart)
  }

}


package util

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

  @tailrec def regressionGradientDescent(df: DataFrame, observations: List[Double], initialWeights: List[Double], stepSize: Double, errorDiffThreshold: Double, currentStep: Option[Int] = None, maxSteps: Option[Int] = None): List[Double] = {
    val predictions: List[Double] = predictOutcome(df, initialWeights)
    val errors: List[Double] = (predictions, observations).zipped.map { case (prediction, output) => output - prediction }

    df.rdd.map { row => (row.toSeq, errors).zipped.map{ case (observation: Double, error) => observation * error} }
    val partials = df.columns.map { column =>
      val featureValues = extractFeatureValues(column, df)
      -featureDerivative(errors, featureValues)
    }.toList
    val updatedWeights: List[Double] = (partials, initialWeights).zipped.map {
      case (partial, initialWeight) => initialWeight - stepSize * partial
    }

    val rmse = RMSE(df, observations, initialWeights)
    val rmseWithUpdatedWeights = RMSE(df, observations, updatedWeights)

    val errorDiff = rmse - rmseWithUpdatedWeights
    val gradientMagnitude = Math.sqrt(partials.map(Math.pow(_, 2)).sum)
//    println(s"error diff $errorDiff")
//    println(s"gradient magnitude $gradientMagnitude, weights $initialWeights")

    if (errorDiff > 0 && errorDiff > errorDiffThreshold) {
      if (currentStep.getOrElse(0) >= maxSteps.getOrElse(Int.MaxValue)) initialWeights else regressionGradientDescent(df, observations, updatedWeights, stepSize, errorDiffThreshold, Some(currentStep.getOrElse(0) + 1), maxSteps)
    }
    else if (errorDiff < 0) regressionGradientDescent(df, observations, initialWeights, stepSize / 10, errorDiffThreshold, currentStep, maxSteps)
    else { println(s"weights found $initialWeights with error diff $errorDiff"); initialWeights }
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

}


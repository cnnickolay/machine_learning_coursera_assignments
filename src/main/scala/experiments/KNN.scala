package experiments

import java.lang.Math.exp

import breeze.numerics.{abs, pow}

import scala.annotation.tailrec

object KNN {

  def weightedRegression(from: Int, to: Int): Double = 1d/(abs(from - to) + 1)

  def gaussianKernel(lambda: Double, from: Int, to: Int): Double = exp(-pow(from - to, 2.0)/lambda)

  @tailrec
  def knn(values: List[Double], radius: Int, power: Int = 1, kernel: ((Int, Int) => Double) = weightedRegression): List[Double] = {
    if (power == 0)
      values
    else {
      val means = values.indices.par.map { currentElementIdx =>
        findMeansForIndex(values, radius, currentElementIdx, kernel)
      }.toList

      knn(means, radius, power - 1)
    }
  }

  def findMeansForIndex(values: List[Double], radius: Int, currentElementIdx: Int, kernel: ((Int, Int) => Double)): Double = {
    val start = if (currentElementIdx - radius < 0) 0 else currentElementIdx - radius
    val end = if (currentElementIdx + radius >= values.size) values.size - 1 else currentElementIdx + radius
    val res = (start to end).par.map { idx =>
      val value = values(idx)
      val weight = kernel(currentElementIdx, idx)
      (value * weight, weight)
    }.toList

    val numerator = res.map(_._1).sum
    val denominator = res.map(_._2).sum
    numerator / denominator
  }

  @tailrec
  def backKnn(values: List[Double], radius: Int, power: Int = 1, kernel: ((Int, Int) => Double) = weightedRegression): List[Double] = power match {
    case 0 => values
    case _ =>
      val means = values.indices.par.map { currentElementIdx =>
        val sliced = values.slice(0, currentElementIdx + 1)
        findMeansForIndex(sliced, radius, currentElementIdx, kernel)
      }.toList
      backKnn(means, radius, power - 1)
  }

  def derivative(means: List[Double]): List[Double] = {
    val derivative = means.zipWithIndex.par.map { case (value, idx) =>
      if (idx <= 1) 0 else (means(idx) - means(idx - 2))/2
//      if (idx == 0 || idx == means.size - 1) 0 else (means(idx + 1) - means(idx - 1))/2
    }.toList
    backKnn(derivative, 5, 2)
    derivative
  }

  def findTurnOverPoints(values: List[Double], derivatives: List[Double], minFluctuation: Double) = {
    val turnOverPoints = (derivatives.reverse.tail.reverse zip derivatives.tail).zipWithIndex.collect {
      case ((before, after), idx) if (before < 0 && after > 0) || (before > 0 && after < 0) => idx
    }

    val fluctuations = (turnOverPoints.reverse.tail.reverse zip turnOverPoints.tail).map { case (firstIdx, secondIdx) => (values(secondIdx) - values(firstIdx), values(firstIdx), values(secondIdx), firstIdx, secondIdx, secondIdx - firstIdx) }
    val filteredFluctuations = fluctuations.collect{case tuple @ (price, _, _, _, _, _) if abs(price) >= minFluctuation => tuple}
    filteredFluctuations
  }

  def geometricMean(values: List[Double], range: Int): List[Double] = {
    values sliding range map { slice =>
      pow(slice.product, 1d/range)
    } toList
  }

}

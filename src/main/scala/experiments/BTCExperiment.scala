package experiments

import experiments.KNN._

import scala.collection.mutable.ListBuffer
import scala.io.Source

object BTCExperiment extends App {

  val data = Source.fromFile("/Users/niko/projects/zeppelin/data/fx/coinbaseEUR.csv").getLines().slice(0, 30000).toList
  val ticks = data.par.map{ line => val values = line.split(","); values(0).toLong }.map(_*1000).toList
  val fluctuations = data.par.map{ line => val values = line.split(","); values(1).toDouble }.toList

  ///////////////
  val ticksToMinutes = ticks.par.map{ tick => tick / 1000 / 60 }.toList
  val minutes = (ticksToMinutes zip fluctuations).par.groupBy{case (minute, fluctuation) => minute}.mapValues( list => list.map(_._2) ).toList.sortBy(_._1)
  println(s"Days ${minutes.size / 60 / 24}")
  val fluctuationsMinutes = minutes.par.map{ case (minute, values) => values.sum / values.size }.toList

  val input = fluctuationsMinutes

  val operationFee = 0.002
  case class OperationHistory(minute: Integer, eurAmount: Double, firstDerivative: Double, secondDerivative: Double)

  var counter = 0
  def incrementCounter = synchronized {
    counter = counter + 1
  }

  def evaluate(knnRange: Integer, power: Integer, firstDerivativeThreshold: Double, secondDerivativeThreshold: Double) = {
    val mean = knn(input, knnRange, power)
    val derivatives = derivative(mean)
    val second_derivatives = derivative(derivatives)

    val inputDerivatives = derivatives
    val inputSecondDerivatives = second_derivatives

    var eurAmount = 1000d
    var bitcoinAmount = 0d

    val zippedDerivatives = inputDerivatives zip inputSecondDerivatives
    val zippedDerivativesShifted = zippedDerivatives.tail
    val zippedDerivativesNotShifted = zippedDerivatives.reverse.tail.reverse

    def buyBitcoins(idx: Integer) = { if (eurAmount > 0) { bitcoinAmount = eurAmount * (1 - operationFee) / input(idx); eurAmount = 0d; } }
    def buyEuro(idx: Integer) = { if (bitcoinAmount > 0) { eurAmount = bitcoinAmount * (1 - operationFee) * input(idx); bitcoinAmount = 0d; println(s"Euros $eurAmount") } }

    (zippedDerivativesNotShifted zip zippedDerivativesShifted).zipWithIndex.foreach {
      case (((derivative, _), (pastDerivative, _)), idx) if pastDerivative > firstDerivativeThreshold && derivative < firstDerivativeThreshold => print("1");buyEuro(idx)
      case (((derivative, _), (pastDerivative, _)), idx) if pastDerivative < firstDerivativeThreshold && derivative > firstDerivativeThreshold => print("2");buyBitcoins(idx)
      case (((_, secondDerivative), (_, pastSecondDerivative)), idx) if pastSecondDerivative < secondDerivativeThreshold && secondDerivative > secondDerivativeThreshold => print("3");buyEuro(idx)
      case (((_, secondDerivative), (_, pastSecondDerivative)), idx) if pastSecondDerivative > secondDerivativeThreshold && secondDerivative < secondDerivativeThreshold => print("4");buyBitcoins(idx)
      case _ =>
    }
    buyEuro(input.size - 1)
    incrementCounter
    eurAmount
  }

/*
  val parameters = for {
    knnRange <- 10 to 200 by 10
    power <- 3 to 10
    firstDerivativeThreshold <- 0d to -0.5 by -0.01
    secondDerivativeThreshold <- 0d to 0.5 by 0.01
  } yield(knnRange, power, firstDerivativeThreshold, secondDerivativeThreshold)

  val result = parameters.par.map { case (knn, power, firstDerivativeThreshold, secondDerivativeThreshold) => evaluate(knn, power, firstDerivativeThreshold, secondDerivativeThreshold) }

  result.toList.sortBy(_._1).reverse
  result.take(1000).foreach(println)
*/

/*
  val output = evaluate(120, 8, 0, 0.0)
  println
  println(output)
*/

  val means1 = backKnn(input, 5, 4)
  val means2 = backKnn(input, 15, 4)
  val means3 = backKnn(input, 25, 4)
  val means4 = backKnn(input, 50, 4)
  val means5 = backKnn(input, 100, 4)

  val bears = (means1 zip means2 zip means3 zip means4 zip means5).zipWithIndex.collect {
    case (((((mean1, mean2), mean3), mean4), mean5), idx) if mean3 < mean4 && mean4 < mean5 => (idx, input(idx))
  }
  val bulls = (means1 zip means2 zip means3 zip means4 zip means5).zipWithIndex.collect {
    case (((((mean1, mean2), mean3), mean4), mean5), idx) if mean3 > mean4 && mean4 > mean5 => (idx, input(idx))
  }

  (bears zip bears.tail).foreach { case ((idx1, price1), (idx2, price2)) =>
    if (idx2 != idx1 + 1) {
      println(s"$idx1 - $price1")
      println(s"$idx2 - $price2")
    }
  }

}

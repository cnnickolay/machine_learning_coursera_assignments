package util
/*

import com.holdenkarau.spark.testing.{RDDGenerator, SharedSparkContext}
import com.quantifind.charts.Highcharts
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{Row, SQLContext}
import org.scalatest.{FlatSpec, FunSpec, FunSuite, Matchers}

import scala.util.Random

/**
  * Created by niko on 26/07/2016.
  */
class Regression$Test extends FunSuite with Matchers {

  test("test") {
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[6]")
    val sc = new SparkContext(conf)
    val sql = new SQLContext(sc)
    val rows = (10d to 1000 by 1).map(value => Row(1d, value))
    val rdd = sc.parallelize(rows)
    val df = sql.createDataFrame(rdd, StructType(Seq(StructField("const", DoubleType), StructField("x", DoubleType))))
    val observations = (10d to 1000 by 1).map(x => Math.sin(x) + x*x/100 + (Random.nextInt(100)-50)).toList

    val weights = Regression.regressionGradientDescent(df, observations, List(0d, 0d), stepSize = 0.000000001, 0.0000001)

    val dfAugmented = Regression.predict(df, df.columns.toList, weights)
    println(s"weights $weights")
    println(s"observations $observations")
    dfAugmented.show()
  }

}
*/

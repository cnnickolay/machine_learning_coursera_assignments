import java.lang.Math.pow
import org.apache.spark.sql.functions.{lit, _}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Column, DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

import org.apache.spark.ml.feature.{PCA, VectorAssembler}
import org.apache.spark.ml.param.{Param, ParamMap, ParamPair}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{Row, SQLContext, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import util.Regression

import scala.util.Random

/**
  * Created by niko on 04/08/2016.
  */
object SimpleTest {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[6]")
    val sc = new SparkContext(conf)
    val sqlContext = SparkSession.builder().getOrCreate()



    val rnd = new Random()
    val xValues = (1 to 200).map(_.toDouble).toList
    val yValues = xValues.map { x =>
//      val result = x*x*x-280*x*x+21100*x-408000
      val result = -((114943399d*pow(x, 5))/256466972160000d)+((175076509d*pow(x, 4))/929228160000d)-((1655323100983d*pow(x, 3))/64116743040000d)+((4799080868d*pow(x, 2))/4174267125d)-((4524702943d*x)/22898836800d) + 1
      val range = result * 0.2
      val fluctuation = rnd.nextDouble()*range
      result + fluctuation - fluctuation/2 + 1200
    }
    val rdd = sc.parallelize((xValues, yValues).zipped.map{(x, y) => Row(1d, x, pow(x, 2), pow(x, 3), pow(x, 4), pow(x, 5), y)})
    val structure = StructType(Seq(
      StructField("const", DoubleType),
      StructField("x", DoubleType),
      StructField("x2", DoubleType),
      StructField("x3", DoubleType),
      StructField("x4", DoubleType),
      StructField("x5", DoubleType),
      StructField("y", DoubleType)
    ))
    val df = sqlContext.createDataFrame(rdd, structure)

    val sdf = new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
    val dateToMsec = udf[Double, String](date => sdf.parse(date).getTime())
    df.withColumn("test", dateToMsec(df("bucketDate"))).show()

    val predictionColumns = List("const", "x", "x2", "x3", "x4", "x5")
    val assembler = new VectorAssembler()
      .setInputCols(predictionColumns.toArray)
      .setOutputCol("features")

    val transformedDf = assembler.transform(df).withColumn("label", df("y"))
    transformedDf.show()

    val lr = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0)
      .setElasticNetParam(0.8)
      .setFitIntercept(true)

    val model = lr.fit(transformedDf)
    val augmentedDf = model.transform(transformedDf)

    augmentedDf.show()
    val table = augmentedDf.select("x", "y").rdd.collect().map(row => row.mkString("\t")).reduce((x, y) => s"$x\n$y")

    Regression.renderPlot("test", "x", "y", "prediction", 1)(augmentedDf)

    println(model)
  }

}

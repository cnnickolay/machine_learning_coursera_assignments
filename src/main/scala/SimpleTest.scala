import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{Row, SQLContext}
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
    val sqlContext = new SQLContext(sc)

    val rnd = new Random()
    val xValues = (1 to 200).map(_.toDouble).toList
    val yValues = xValues.map { x =>
      val square = x*x
      val range = square * 0.2
      val fluctuation = rnd.nextDouble()*range
      square + fluctuation - fluctuation/2 + 1200
    }
    val rdd = sc.parallelize((xValues, yValues).zipped.map{(x, y) => Row(1d, x, x*x, y)})
    val structure = StructType(Seq(
      StructField("const", DoubleType),
      StructField("x", DoubleType),
      StructField("x2", DoubleType),
      StructField("y", DoubleType)
    ))
    val df = sqlContext.createDataFrame(rdd, structure)

    df.show()

    val predictionColumns = List("x", "x2")
    val model = Regression.regressionGradientDescent(df.select(predictionColumns.head, predictionColumns.tail: _*), yValues, List(0), Nil, 1E-6, 1E-15, None, Some(1000))
    val augmentedRows = df.rdd.map { row =>
      val prediction = model.indices.zip(predictionColumns).map { case(i, column) =>
        row.getAs[Double](column) * model(i)
      }.sum
      val values = row.toSeq.toList ::: prediction :: Nil
      Row(values: _*)
    }
    val augmentedDf = sqlContext.createDataFrame(augmentedRows, structure.add(StructField("prediction", DoubleType)))
    augmentedDf.show()

    Regression.renderPlot("test", "x", "y", "prediction", 1)(augmentedDf)

    println(model)
  }

}

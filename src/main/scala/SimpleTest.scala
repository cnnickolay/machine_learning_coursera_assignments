import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.{LeastSquaresGradient, LogisticGradient, SquaredL2Updater}
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import util.Regression

/**
  * Created by niko on 04/08/2016.
  */
object SimpleTest {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[6]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val xValues = (1 to 200).map(_.toDouble).toList
    val yValues = xValues.map(Math.sin(_)*10).reverse
    val rdd = sc.parallelize((xValues, yValues).zipped.map{(x, y) => Row(1d, x, Math.pow(x, 2), Math.pow(x, 3), Math.pow(x, 4), y)})
    val structure = StructType(Seq(StructField("const", DoubleType), StructField("x", DoubleType), StructField("x2", DoubleType), StructField("x3", DoubleType), StructField("x4", DoubleType), StructField("y", DoubleType)))
    val df = sqlContext.createDataFrame(rdd, structure)

    df.show()

    val model = Regression.regressionGradientDescent(df.select("const", "x", "x2", "x3", "x4"), yValues, List(-1, 5, -10, 10, 1), 0.00001, 0.00000000000001, None, Some(1))
    val augmentedRows = df.rdd.map { row =>
      val prediction = model.indices.map { i =>
        row.getAs[Double](i) * model(i)
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

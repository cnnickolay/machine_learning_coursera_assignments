package regression_week_3

import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.{SparkConf, SparkContext}



object Assignment {
  def main(args: Array[String]) {
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
      StructField("price", FloatType, nullable = false),
      StructField("bedrooms", FloatType, nullable = false),
      StructField("bathrooms", FloatType, nullable = false),
      StructField("sqft_living", FloatType, nullable = false),
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
      StructField("lat", FloatType, nullable = false),
      StructField("long", FloatType, nullable = false),
      StructField("sqft_living15", FloatType, nullable = false),
      StructField("sqft_lot15", FloatType, nullable = false)
    ))

    val set1Data = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").schema(customSchema).load(set1DataFile)
    val set2Data = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").schema(customSchema).load(set2DataFile)
    val set3Data = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").schema(customSchema).load(set3DataFile)
    val set4Data = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").schema(customSchema).load(set4DataFile)
    val testData = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").schema(customSchema).load(testDataFile)
    val trainData = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").schema(customSchema).load(trainDataFile)
    val validData = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").schema(customSchema).load(validDataFile)

    val toIntDoubleArgument = udf[Float, Int, Float]((power, value) => List.fill(power)(value).product)

    def polynomialDataFrame(feature: List[Any], degree: Integer) = {
      val parallelize = sc.parallelize(feature)
      val rdd = parallelize.map(v => Row(v))
      val df = sqlContext.createDataFrame(rdd, StructType(StructField("power_1", FloatType, nullable = false) :: Nil))
      def addColumn(power: Int, to: Int, df: DataFrame): DataFrame = {
        if (power <= to) {
          val _df = df.withColumn(s"power_$power", toIntDoubleArgument(df("power_1"), lit(power)))
          addColumn(power + 1, to, _df)
        } else df
      }
      addColumn(2, degree, df)
    }

    val frame = polynomialDataFrame(List(1f, 2f, 3f, 4f, 5f, 6f), 4)
    frame.show()
  }
}
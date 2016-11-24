import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

object DecisionTrees extends App {

  val conf = new SparkConf().setAppName("Simple Application").setMaster("local[6]")
  val sc = new SparkContext(conf)
  val session = SparkSession.builder().appName("Spark SQL basic example").getOrCreate()

  val data = Seq(
    Seq(30000, 35, 120, 1),
    Seq(20000, 30, 100, 0),
    Seq(40000, 42, 80, 1),
    Seq(32000, 37, 60, 0),
    Seq(42000, 37, 99, 1),
    Seq(22000, 47, 70, 1),
    Seq(32000, 57, 60, 1),
    Seq(52000, 67, 30, 0),
    Seq(82000, 37, 40, 0),
    Seq(72000, 47, 80, 1),
    Seq(62000, 57, 90, 0),
    Seq(32000, 17, 70, 0)
  )

  val structType = StructType(List(StructField("income", IntegerType, nullable = false), StructField("age", IntegerType, nullable = false), StructField("weight", IntegerType, nullable = false), StructField("label", IntegerType, nullable = false)))
  val rows = data.map { list => Row(list: _*) }

  val df = session.createDataFrame(sc.parallelize(rows), structType)

  val vectorAssembler = new VectorAssembler()
  val preparedDf = vectorAssembler.setInputCols(Array("income", "age", "weight")).setOutputCol("features").transform(df)
  val categoricalFeaturesInfo = Map[Int, Int]()

  val labelIndexer = new StringIndexer()
    .setInputCol("label")
    .setOutputCol("indexedLabel")
    .fit(preparedDf)
  // Automatically identify categorical features, and index them.
  val featureIndexer = new VectorIndexer()
    .setInputCol("features")
    .setOutputCol("indexedFeatures")
    .setMaxCategories(4) // features with > 4 distinct values are treated as continuous.
    .fit(preparedDf)

  // Train a DecisionTree model.
  val dt = new DecisionTreeClassifier()
    .setLabelCol("indexedLabel")
    .setFeaturesCol("indexedFeatures")

  // Convert indexed labels back to original labels.
  val labelConverter = new IndexToString()
    .setInputCol("prediction")
    .setOutputCol("predictedLabel")
    .setLabels(labelIndexer.labels)

  val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

  // Train model. This also runs the indexers.
  val model = pipeline.fit(preparedDf)

  // Make predictions.
  val predictions = model.transform(preparedDf)

  predictions.show()

}

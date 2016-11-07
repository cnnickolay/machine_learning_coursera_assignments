import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object PerceptronTest extends App
{
  val conf = new SparkConf().setAppName("Simple Application").setMaster("local[6]")
  val sc = new SparkContext(conf)
  val sqlContext = SparkSession.builder().getOrCreate()

  // Load the data stored in LIBSVM format as a DataFrame.
  val data: DataFrame = sqlContext.read.format("libsvm")
    .load("src/main/resources/sample_multiclass_classification_data.txt")
  // Split the data into train and test
  data.show(20)
  val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
  val train: Dataset[Row] = splits(0)
  val test = splits(1)
  // specify layers for the neural network:
  // input layer of size 4 (features), two intermediate of size 5 and 4
  // and output of size 3 (classes)
  val layers = Array[Int](4, 5, 4, 5, 6, 3)
  // create the trainer and set its parameters
  val trainer = new MultilayerPerceptronClassifier()
    .setLayers(layers)
    .setBlockSize(128)
    .setSeed(1234L)
    .setMaxIter(100)
  // train the model
  val model = trainer.fit(train)
  // compute accuracy on the test set
  val result = model.transform(test)
  val predictionAndLabels = result.select("prediction", "label")
  val evaluator = new MulticlassClassificationEvaluator()
    .setMetricName("accuracy")
  println("Accuracy: " + evaluator.evaluate(predictionAndLabels))
}

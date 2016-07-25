name := "spark1"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.6.2"
libraryDependencies += "org.apache.spark" % "spark-sql_2.11" % "1.6.2"
libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "1.6.2"
libraryDependencies += "com.databricks" % "spark-csv_2.11" % "1.4.0"
libraryDependencies += "log4j" % "log4j" % "1.2.17"
libraryDependencies += "com.quantifind" %% "wisp" % "0.0.4"

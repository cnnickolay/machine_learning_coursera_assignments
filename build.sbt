name := "spark1"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.6.2"
libraryDependencies += "org.apache.spark" % "spark-sql_2.11" % "1.6.2"
libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "1.6.2"
libraryDependencies += "com.databricks" % "spark-csv_2.11" % "1.4.0"
libraryDependencies += "log4j" % "log4j" % "1.2.17"
libraryDependencies += "com.quantifind" %% "wisp" % "0.0.4"
//libraryDependencies += "org.scalatest" %% "scalatest" % "2.2.6" % "test"
//libraryDependencies += "com.holdenkarau" % "spark-testing-base_2.11" % "1.6.1_0.3.3"
//libraryDependencies += "org.eclipse.jetty" % "jetty-util" % "9.3.2.v20150730"

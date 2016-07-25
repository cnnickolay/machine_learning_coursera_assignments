import com.quantifind.charts.Highcharts
import com.quantifind.charts.Highcharts._
import com.quantifind.charts.highcharts._
import com.quantifind.charts.repl.IterablePair

import scala.collection.immutable.NumericRange
import scala.collection.immutable.Range._

/**
  * Created by niko on 23/07/2016.
  */
object PlotTest extends App {

  val topWords = Array(("alpha", 14), ("beta", 23), ("omega", 18))
//  val numberedColumns = column(topWords.map(_._2).toList)
  val range = 0.1d to 10 by 0.1
//  val x = scatter(IterablePair.mkIterableIterable(range, range.map(1/_)))

//  val axisType: com.quantifind.charts.highcharts.AxisType.Type = "category"
//  val namedColumns = numberedColumns.copy(xAxis = numberedColumns.xAxis.map {
//    axisArray => axisArray.map { _.copy(axisType = Option(axisType),
//      categories = Option(topWords.map(_._1))) }
//  })
//  Highcharts.plot(numberedColumns)


  val series1 = Series((0d to 10 by 0.5).map(i => Data(i, i * i)), chart = Some(SeriesType.scatter))
  val series2 = Series((0.01d to 10 by 0.1).map(i => Data(i, 1 / i)), chart = Some(SeriesType.areaspline))
  val chart = Highchart(Seq(series1, series2), chart = Some(Chart(zoomType = Some(Zoom.xy))), yAxis = None,
    exporting = Some(Exporting(filename = "target/output.jpg", _type = Some("jpeg"))))

  plot(chart)

}

package experiments

import experiments.KNN.{backKnn, knn}
import org.scalatest.FunSuite

class KNN$Test extends FunSuite {

  test("knn for same elements") {
    val result = knn(List(1, 1, 1), 1)
    assert(result === List(1, 1, 1))
  }

  test("knn") {
    val result = knn(List(1, 5, 9), 1)
    assert(result === List(2.3333333333333335, 5.0, 7.666666666666667))
  }

  test("knn powered") {
    val input: List[Double] = List(1, 5, 9)
    val result1 = knn(input, 1, 1)
    val result2 = knn(result1, 1, 1)
    val result3 = knn(result2, 1, 1)
    val powered = knn(input, 1, 3)
    assert(powered === result3)
  }

  test("knn for a single element") {
    val result = knn(List(1, 5, 9), 0)
    assert(result === List(1, 5, 9))
  }

  test("backKnn for a single element") {
    val result = backKnn(List(1), 1)
    assert(result === List(1))
  }

  test("backKnn for several elements") {
    val result = backKnn(List(1, 2, 3), 1)
    assert(result === List(1.0, 1.6666666666666667, 2.6666666666666665))
  }

  test("backKnn with gaussian kernel") {
    val result = backKnn(List(10, 20, 60, 80, 100, 80, 60, 40), 3, 1, KNN.gaussianKernel(3, _, _))
    assert(result === List(1.0, 1.6666666666666667, 2.6666666666666665))
  }

  test("f") {
    println(KNN.gaussianKernel(10, 10, 4))
    println(KNN.gaussianKernel(100, 10, 4))
    println(KNN.gaussianKernel(1000, 10, 4))
  }

}

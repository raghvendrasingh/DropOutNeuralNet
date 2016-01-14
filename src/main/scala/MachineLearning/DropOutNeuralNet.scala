package MachineLearning

/**
  * Created by raghvendra.singh on 1/9/16.
  */

import breeze.linalg._
import breeze.stats.distributions._
import breeze.stats._


class DropOutNeuralNet(numNodesinLayers: Vector[Int], dropoutProbability: Double) {

  private var layerWeights = List[DenseMatrix[Double]]()
  private var layerBiases = List[DenseVector[Double]]()
  private var delta = List[DenseMatrix[Double]]()
  private var layerOutputs = List[DenseMatrix[Double]]()

  def initializeNet(): Unit = {
    if (numNodesinLayers.size <= 2) {
      throw new Exception("Not enough layers in neural network. Please provide atleast three layers including input and output layer.")
    }
    for (i <- 0 to numNodesinLayers.size - 2) {
      val b = math.sqrt(6.toDouble / (numNodesinLayers(i) + numNodesinLayers(i + 1)))
      val vec = DenseVector.zeros[Double](numNodesinLayers(i) * numNodesinLayers(i + 1)) map (x => math.random * (b + b) - b)
      val weight = new DenseMatrix[Double](numNodesinLayers(i), numNodesinLayers(i + 1), vec.toArray)
      layerWeights = layerWeights :+ weight
      val bias: DenseVector[Double] = DenseVector.rand(numNodesinLayers(i+1))
      layerBiases = layerBiases :+ bias
    }
    assert(layerWeights.size == numNodesinLayers.size-1)
    assert(layerBiases.size == numNodesinLayers.size-1)
  }

  def printLayerWeightsAndLayerBiases(): Unit = {
    for (i <- layerWeights.indices) {
      println(s"Layer ${i} - ${i+1} weights=")
      println(layerWeights(i))
      println(s"Layer ${i} - ${i+1} biases=")
      println(layerBiases(i))
    }
  }

  def forward(input: DenseMatrix[Double], dropNodeList: List[DenseMatrix[Double]]): Unit = {
    layerOutputs = layerOutputs.drop(layerOutputs.size)
    layerOutputs = layerOutputs :+ input
    for (i <- 0 to numNodesinLayers.size-3) {
      assert(layerWeights(i).t.cols == layerOutputs(i).rows)
      var tempOut: DenseMatrix[Double] = layerWeights(i).t * layerOutputs(i)
      tempOut = tempOut(::,*) + layerBiases(i)
      layerOutputs = layerOutputs :+  (NNUtils.sigmoidMatrix(tempOut) :* dropNodeList(i))
    }
    /** Calculate output for output layer using softmax function*/
    var softmaxInput: DenseMatrix[Double] = layerWeights.last.t * layerOutputs.last
    softmaxInput = softmaxInput(::,*) + layerBiases.last
    val output = NNUtils.softmaxMatrix(softmaxInput)
    layerOutputs = layerOutputs :+ output
  }

  def backward(targetLabels: DenseMatrix[Double], dropNodeList: List[DenseMatrix[Double]]): Unit = {
    val outputDelta = -(targetLabels - layerOutputs.last) :* (layerOutputs.last :* (DenseMatrix.ones[Double](layerOutputs.last.rows,layerOutputs.last.cols)- layerOutputs.last) )
    delta = outputDelta +: delta
    for (i <- numNodesinLayers.size-2 to 1 by -1) {
      val temp1: DenseMatrix[Double] = layerWeights(i) * delta(0)
      val hiddenDelta =  temp1 :* (layerOutputs(i) :* (DenseMatrix.ones[Double](layerOutputs(i).rows,layerOutputs(i).cols) - layerOutputs(i)))
      delta = hiddenDelta +: delta
    }

  }

  def printLayerOutputs(): Unit = {
    for (i <- layerOutputs.indices) {
      println(layerOutputs(i))
      println()
    }
  }

  def testLibrary(): Unit = {
    val x = DenseVector.zeros[Double](5)
    println(x)
    val poi = Poisson(3.0)
    println(poi.draw())
    val poiSample = poi.sample(10)
    println(poiSample)
    println(poi.probabilityOf(2))
    println(poi.probabilityOf(3))
    val gau = Gaussian(2, 1.0)
    val gaussianSamples = gau.sample(10).toArray
    println("gaussianSamples=")
    gaussianSamples foreach (x => print(x.toString + " "))
    val meanGau = mean(gaussianSamples)
    val varGau = variance(gaussianSamples)
    println(meanGau)
    println(varGau)
    val tup = meanAndVariance(gaussianSamples)
    println(tup.mean + " " + tup.variance + " " + tup.count)

    val m = new DenseMatrix(4, 4, linspace(1, 16, 16).toArray)
    println(m)
    val n = m.t
    //println(n)
    val mulMN: DenseMatrix[Double] = m * n
    println(mulMN.rows + " " + mulMN.cols)
    val row0 = mulMN(0, ::)
    //println(m(0,::) * m(0,::).t)
    m(0 to 1, 0 to 1) := DenseMatrix((12.0, 13.0), (14.0, 15.0))
    println(m)
    val shuffledGaussianSamples = shuffle(gaussianSamples)
    println("shuffledGaussianSamples=")
    shuffledGaussianSamples foreach (x => print(x.toString + " "))
    println()
    val shuffledDenseVector = shuffle(DenseVector[Int](1, 2, 3, 4, 5, 6))
    shuffledDenseVector foreach (x => print(x.toString + " "))
    val alp = shuffledDenseVector :*= 2
    println()
    alp foreach (x => print(x.toString + " "))
    println()
    println(sum(DenseMatrix((1.0, 2.0), (3.0, 4.0))))
    val alp1 = DenseVector[Int](1, 2, 3, 4) - DenseVector[Int](2, 3, 4, 5)
    alp1 foreach (x => print(x.toString + " "))
    println()
    val alp2 = alp1 map (x => x * x)
    alp2 foreach (x => print(x.toString + " "))
    println()
    val b = math.sqrt(6.toDouble / (5 + 5))
    val alp3 = DenseVector.zeros[Double](10) map (x => math.random * (b + b) - b)
    alp3 foreach (x => print(x.toString + " "))
    println()
    val alp4 = DenseVector.rand(10)
    alp4 foreach (x => print(x.toString + " "))
    println()
    val alp5 = DenseMatrix((1,2),(3,4)) * DenseVector(1,2)
    println(alp5)
    val alp6 = DenseMatrix((1.0,2.0),(3.0,4.0),(5.0,6.0))
    println(NNUtils.sigmoidMatrix(alp6))

    println("++++++++++++++++++++++++++++++++++++")
    val alp7 = NNUtils.makeDropNodes(10,Vector(1,5,3,2),0.5)
    println(alp7(0))
    println()
    println()
    println(alp7(1))

    println("------------------------------------------")
    val alp8 = DenseMatrix((0,0,1,0,0,1),(1,0,1,0,1,0))
    val s = sum(alp8(::,0))
    println(s)

    val alp9 = DenseMatrix((1,1),(2,2)) :* DenseMatrix((3,3),(4,4))
    println(alp9)

    val alp10 = softmax(DenseVector(1.0,2.0,3.0))
    println(alp10)
    println(math.log(math.exp(1)+math.exp(2)+math.exp(3)))
    println(math.log(math.exp(1)+math.exp(4)+math.exp(0)))

    val alp11 = DenseMatrix((1.0,2.0,3.0),(4.0,5.0,6.0),(0.0,1.0,0.0))
    println(alp11)
    val vec: DenseVector[Double] = DenseVector.zeros(alp11.cols)
    val numer = alp11 map (x => math.exp(x))
    for (i <- 0 to alp11.cols-1) {
      vec(i) = softmax(alp11(::,i))
      numer(::,i) :*= 1.toDouble/vec(i)
    }
    println(numer)
    val alp12 = DenseMatrix((1.0,2.0),(3.0,4.0))
    val alp13 = DenseMatrix((5.0,6.0),(7.0,8.0))
    val alp14 = alp12-alp13
    println(DenseMatrix.ones[Double](alp14.rows,alp14.cols)-alp14)

  }
}


object ExecuteDropOutNeuralNet extends App {
  val obj = new DropOutNeuralNet(Vector(3,2,2,2),0.5)
  obj.testLibrary()
  /*obj.initializeNet()
  println("Layer weights and baises=")
  obj.printLayerWeightsAndLayerBiases()
  println("drop node list is=")
  val dropNodeList = NNUtils.makeDropNodes(2,Vector(3,2,2,2),0.5)
  NNUtils.printList(dropNodeList)
  println("forward phase=")
  obj.forward(DenseMatrix((1.0,2.0,3.0),(4.0,5.0,6.0)).t, dropNodeList)
  println("Layer outputs=")
  obj.printLayerOutputs()*/
  //obj.backward()
}

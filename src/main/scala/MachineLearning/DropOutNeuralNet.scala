package MachineLearning

/**
  * Created by raghvendra.singh on 1/9/16.
  */

import breeze.linalg._



class DropOutNeuralNet(numNodesInLayers: Vector[Int], dropoutProbability: Double) {
  private var layerWeights = List[DenseMatrix[Double]]()
  private var layerBiases = List[DenseVector[Double]]()
  private var layerOutputs = List[DenseMatrix[Double]]()
  private var deltaLayerWeights = List[DenseMatrix[Double]]()
  private var deltaLayerBiases = List[DenseVector[Double]]()

  def initializeNet(): Unit = {
    if (numNodesInLayers.size <= 2) {
      throw new Exception("Not enough layers in neural network. Please provide atleast three layers including input and output layer.")
    }
    for (i <- 0 to numNodesInLayers.size - 2) {
      val b = math.sqrt(6.toDouble / (numNodesInLayers(i) + numNodesInLayers(i + 1)))
      val vec = DenseVector.zeros[Double](numNodesInLayers(i) * numNodesInLayers(i + 1)) map (x => math.random * (b + b) - b)
      val weight = new DenseMatrix[Double](numNodesInLayers(i), numNodesInLayers(i + 1), vec.toArray)
      layerWeights = layerWeights :+ weight
      val bias: DenseVector[Double] = DenseVector.rand(numNodesInLayers(i + 1))
      layerBiases = layerBiases :+ bias
    }
    assert(layerWeights.size == numNodesInLayers.size - 1)
    assert(layerBiases.size == numNodesInLayers.size - 1)
  }

  def printLayerWeightsAndLayerBiases(): Unit = {
    for (i <- layerWeights.indices) {
      println(s"Layer ${i} - ${i + 1} weights=")
      println(layerWeights(i))
      println(s"Layer ${i} - ${i + 1} biases=")
      println(layerBiases(i))
    }
  }

  def printDeltaLayerWeightsAndDeltaLayerBiases(): Unit = {
    for (i <- deltaLayerWeights.indices) {
      println(s"delta layer ${i} - ${i + 1} weights=")
      println(deltaLayerWeights(i))
      println(s"delta layer ${i} - ${i + 1} biases=")
      println(deltaLayerBiases(i))
    }
  }

  def forward(input: DenseMatrix[Double], dropNodeList: List[DenseMatrix[Double]]): Unit = {
    layerOutputs = layerOutputs.drop(layerOutputs.size)
    layerOutputs = layerOutputs :+ input
    for (i <- 0 to numNodesInLayers.size - 3) {
      assert(layerWeights(i).t.cols == layerOutputs(i).rows)
      var tempOut: DenseMatrix[Double] = layerWeights(i).t * layerOutputs(i)
      tempOut = tempOut(::, *) + layerBiases(i)
      layerOutputs = layerOutputs :+ (NNUtils.sigmoidMatrix(tempOut) :* dropNodeList(i))
    }
    /** Calculate output for output layer using softmax function */
    var softmaxInput: DenseMatrix[Double] = layerWeights.last.t * layerOutputs.last
    softmaxInput = softmaxInput(::, *) + layerBiases.last
    val output = NNUtils.softmaxMatrix(softmaxInput)
    layerOutputs = layerOutputs :+ output
  }

  def backward(targetLabels: DenseMatrix[Double], dropNodeList: List[DenseMatrix[Double]]): Unit = {
    val outputDelta = -(targetLabels - layerOutputs.last) :* (layerOutputs.last :* (DenseMatrix.ones[Double](layerOutputs.last.rows, layerOutputs.last.cols) - layerOutputs.last))
    var delta = outputDelta
    val size = numNodesInLayers.size - 2
    val temp: DenseMatrix[Double] = layerOutputs(size) * delta.t
    deltaLayerWeights = temp +: deltaLayerWeights
    deltaLayerBiases = sum(delta(*, ::)) +: deltaLayerBiases
    for (i <- numNodesInLayers.size - 2 to 1 by -1) {
      val temp1: DenseMatrix[Double] = layerWeights(i) * delta
      val hiddenDelta = temp1 :* (layerOutputs(i) :* (DenseMatrix.ones[Double](layerOutputs(i).rows, layerOutputs(i).cols) - layerOutputs(i)))
      delta = hiddenDelta
      val temp2: DenseMatrix[Double] = layerOutputs(i - 1) * delta.t
      deltaLayerWeights = temp2 +: deltaLayerWeights
      val temp3: DenseMatrix[Double] = delta :* dropNodeList(i - 1)
      deltaLayerBiases = sum(temp3(*, ::)) +: deltaLayerBiases
    }
  }

  def updateWeights(numSamples: Int, learningRate: Double, weightDecay: Double): Unit = {
    var newLayerWeights = List[DenseMatrix[Double]]()
    var newLayerBiases = List[DenseVector[Double]]()
    try {
      for (i <- layerWeights.indices) {
        deltaLayerWeights(i) :*= (1.toDouble / numSamples)
        val temp3: DenseMatrix[Double] = deltaLayerWeights(i) + (layerWeights(i) * weightDecay)
        val temp4 = layerWeights(i) - (temp3 * learningRate)
        newLayerWeights = newLayerWeights :+ temp4
        newLayerBiases = newLayerBiases :+ (layerBiases(i) - (deltaLayerBiases(i) :*= (learningRate / numSamples)))
      }
      layerWeights = newLayerWeights
      layerBiases = newLayerBiases
    } catch {
      case ex: Exception => {
        println("exception is:", ex)
        throw new Exception("Unexpected execution error while executing method updateWeights()", ex)
      }
    }
  }

  def printLayerOutputs(): Unit = {
    println("Layer outputs:")
    println()
    for (i <- layerOutputs.indices) {
      println(layerOutputs(i))
      println()
    }
  }

  def train(trainingSamples: DenseMatrix[Double], trainingLabels: DenseMatrix[Double], numNodesInLayers: Vector[Int], maxEpochs: Int, learningRate: Double, weightDecay: Double, isDropOut: Boolean): Unit = {
    var epoch = 0
    val numSamples = trainingSamples.cols
    var k = 1
    while (epoch < maxEpochs) {
      if (epoch == k * 100) {
        println(s"${k * 100} epochs completed.")
        k = k + 1
      }
      deltaLayerWeights = List[DenseMatrix[Double]]()
      deltaLayerBiases = List[DenseVector[Double]]()
      val dropNodeList = NNUtils.makeDropNodes(numSamples, numNodesInLayers, dropoutProbability, isDropOut)
      val shuffledTrainingSamples = NNUtils.shuffleMatrix(trainingSamples)
      forward(shuffledTrainingSamples, dropNodeList)
      backward(trainingLabels, dropNodeList)
      updateWeights(numSamples, learningRate, weightDecay)
      epoch = epoch + 1
    }

  }
}


object ExecuteDropOutNeuralNet extends App {
  val obj = new DropOutNeuralNet(Vector(4,3,3,2),0.5)
  obj.initializeNet()
  obj.train(DenseMatrix((1.0,1.0),(2.0,2.0),(-1.0,-1.0),(-2.0,-2.0)), DenseMatrix((0.0,1.0),(1.0,0.0)), Vector(4,3,3,2), 5, 0.1, 0, true)
}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * Created by raghvendra.singh on 12/25/15.
  */




class DropOutNN(unitsPerLayer: List[Int], dropOutProb: Double = -1) {
  /** It is a list of number of units per layer. The first element gives the number of units
    * in input layer and the last element of this list gives the number of units in output layer.
    * The size of this list is = (input + numHiddenLayers + output) = (1 + numHiddenLayers + 1).
    */
  private val numUnitsPerLayer = unitsPerLayer

  /** It is a probability with which we drop a unit in any hidden layer */
  private val dropOutProbability = if (dropOutProb != -1) dropOutProb else 0.5

  private val inputs = ArrayBuffer[Double]()
  /** weightList is a list of all the weight
    * matrices between all the layers. The first element of this list is the weight between input layer and the first
    * hidden layer. The last element of this list is the weight between last hidden layer and the output layer.
    */
  private var weightList = List[Array[Array[Double]]]()
  /** biasList is a list of biases between all the layers. The first element of this list provides the biases
    * for input layer and the last element of this list provides the biases for last hidden layer.
    */
  private var biasList = List[Array[Double]]()

  /** A list of mutable HashSet containing the indices of drop nodes per hidden layer. */
  //private var dropNodeIndicesPerHiddenLayer = List[mutable.HashSet[Int]]()

  private var outputPerLayer = List[Array[Double]]()

  private var deltaWeightList = List[Array[Array[Double]]]()
  private var deltaBiasList =  List[Array[Double]]()

  /** Initializes neural net weights and biases. */
  def initializeNeuralNet(): Unit = {
    if (numUnitsPerLayer.size <= 2) {
      throw new Exception("Not enough layers in neural network. Please provide atleast three layers including input and output layer.")
    }
    for (i <- 0 to numUnitsPerLayer.size - 2) {
      val weight = Array.ofDim[Double](numUnitsPerLayer(i), numUnitsPerLayer(i+1))
      val bias = Array.fill[Double](numUnitsPerLayer(i + 1))(0)
      val b = math.sqrt(6.toDouble / (weight.length + weight(0).length))
      initializeWeight(weight, b)
      initializeBias(bias, 1)
      weightList = weightList :+ weight
      biasList = biasList :+ bias
    }
  }

  private def initializeDeltaWeightListAndDeltaBiasList(): Unit = {
    if (numUnitsPerLayer.size <= 2) throw new Exception("Not enough layers in neural network. Please provide atleast three layers including input and output layer.")

    for (i <- 0 to numUnitsPerLayer.size - 2) {
      val weight = Array.ofDim[Double] (numUnitsPerLayer (i), numUnitsPerLayer (i + 1) )
      val bias = Array.fill[Double] (numUnitsPerLayer (i + 1) ) (0)
      initializeWeight (weight, 0)
      initializeBias (bias, 0)
      deltaWeightList = deltaWeightList :+ weight
      deltaBiasList = deltaBiasList :+ bias
    }
  }

  /** Print the size and weight for all the layers. Also print the size and bias for all the layers. */

  def printAllWeightsAndBiases(): Unit = {
    for (i <- weightList.indices) {
      showSizeMatrix(weightList(i))
      printMatrix(weightList(i))
    }
    biasList foreach (x => {
      showSizeVector(x)
      printVector(x)
    })
  }

  /** Initializes the weight matrix between layer L and (L+1) in a neural network
    *
    * @param weight - A weight matrix with dimension |L+1| * |L| is provided where |L+1| is the number of units in layer
    *               (L+1) and |L| is the number of units in layer L. The weight matrix is between layer L and (L+1).
    *
    */

  private def initializeWeight(weight: Array[Array[Double]], b: Double): Unit = {
    assert(b >= 0)
    for (i <- weight.indices) {
      for (j <- weight(0).indices) {
        weight(i)(j) = math.random * (b + b) - b
      }
    }
  }

  /** Initializes the bias vector for the layer L
    *
    * @param bias - A bias vector with dimension 1*|L+1| is provided where |L+1| is the number of units in layer
    *             L+1. This bias is for layer L.
    */
  private def initializeBias(bias: Array[Double], b: Double): Unit = {
    for (i <- bias.indices)
      bias(i) = math.random * (2*b) - b
  }

  private def showSizeMatrix(matrix: Array[Array[Double]]): Unit = {
    println("Size of matrix is =" + matrix.length + "x" + matrix(0).length)
  }

  private def showSizeVector(vector: Array[Double]): Unit = {
    println("Size of vector is =" + vector.length)
  }

  private def printMatrix(matrix: Array[Array[Double]]): Unit = {
    for (i <- matrix.indices) {
      for (j <- matrix(0).indices) {
        print(matrix(i)(j) + " ")
      }
      println()
    }
    println()
  }

  private def printVector(vector: Array[Double]): Unit = {
    for (i <- vector.indices)
      print(vector(i) + " ")
    println()
    println()
  }

  /** This method return the a list of all the weight matrices of this neural network. */
  def getWeights: List[Array[Array[Double]]] = weightList

  /** This method fills a list of per hidden layer HashSet called dropNodeIndicesPerHiddenLayer. This HashSet
    * contains the indices of the drop nodes of a particular hidden layer.
    */
  def makeDropNodes(): List[mutable.HashSet[Int]] = {
    var dropNodeIndicesPerHiddenLayer = List[mutable.HashSet[Int]]()
    for (i <- 1 to numUnitsPerLayer.size - 2) {
      var dropNodeIndices = mutable.HashSet[Int]()
      for (j <- 0 to numUnitsPerLayer(i) - 1) {
        val rnd = Random.nextDouble()
        if (rnd < dropOutProbability) dropNodeIndices += j
      }
      if (dropNodeIndices.isEmpty) dropNodeIndices += Random.nextInt(numUnitsPerLayer(i))
      else if (dropNodeIndices.size == numUnitsPerLayer(i)) dropNodeIndices -= Random.nextInt(numUnitsPerLayer(i))
      dropNodeIndicesPerHiddenLayer = dropNodeIndicesPerHiddenLayer :+ dropNodeIndices
    }
    dropNodeIndicesPerHiddenLayer
  }

  /** This function checks whether the index passed is present in passed hidden layer index.
    *
    * @param nodeIndex - It is the index of the node about which we determine whether it is a drop node.
    * @param dropNodes - It is a HashSet containing indices of drop nodes for a particular hidden layer.
    * @return - if nodeIndex is found in dropNodes then return true else false.
    */
  private def isDropNode(nodeIndex: Int, dropNodes: mutable.HashSet[Int]): Boolean = {
    if (dropNodes.contains(nodeIndex)) true else false
  }

  private def printOutputsPerLayer(outputPerLayer: List[Array[Double]]): Unit = {
    for (i <- outputPerLayer.indices) {
      for (j <- outputPerLayer(i).indices) print(outputPerLayer(i)(j) + " ")
      println()
    }
  }

  def forward(inpValues: Array[Double], dropNodeList: List[mutable.HashSet[Int]] ): Unit = {
    if (inpValues.length != numUnitsPerLayer.head) throw new Exception("Bad Input length.")
    var hOut = inpValues
    outputPerLayer = outputPerLayer :+ hOut
    for (i <- 0 to numUnitsPerLayer.size - 2) {
      if (i != numUnitsPerLayer.size-2 ) {
        hOut = multiply(hOut, weightList(i), biasList(i), dropNodeList(i))
        sigmoid(hOut, dropNodeList(i))
      } else {
        hOut = multiply(hOut, weightList(i), biasList(i))
        softMax(hOut)
      }
      outputPerLayer = outputPerLayer :+ hOut
    }
    printOutputsPerLayer(outputPerLayer)
    printAllWeightsAndBiases()
  }

  private def softMax(inp: Array[Double]): Unit = {
    val denom = inp map (x => math.exp(x)) sum
    val ind = inp.indices
    for(i <- ind)
      inp(i) = math.exp(inp(i))/denom
  }

  private def sigmoid(inp: Array[Double], dropNodes: mutable.HashSet[Int]): Unit = {
    for (i <- inp.indices) {
      if (!isDropNode(i, dropNodes))
         inp(i) = 1.0 / (1 + math.exp(-1 * inp(i)))
    }
  }

  private def multiply(a: Array[Double], b: Array[Array[Double]], bias: Array[Double] = Array[Double](),
                       dropNodes: mutable.HashSet[Int] = mutable.HashSet[Int]()): Array[Double] = {
    val hSum = Array.fill(b(0).length)(0.0)
    for (j <- b(0).indices) {
      if (!isDropNode(j, dropNodes)) {
        for (i <- a.indices) hSum(j) = hSum(j) + a(i) * b(i)(j)
        if(bias.length > 0)
          hSum(j) = hSum(j) + bias(j)
      }
    }
    hSum
  }

  private def toArrayBuffer(inp: Array[Double]): ArrayBuffer[Double] = {
    var out = ArrayBuffer[Double]()
    inp foreach (x => out += x)
    out
  }

  private def hadamardProduct(a: Array[Double], b: Array[Double]): Array[Double] = {
    assert(a.length == b.length)
    val result = Array.fill(a.length)(0.0)
    for (i <- a.indices) result(i) = a(i) * b(i)
    result
  }

  private def subtractVectors(a: Array[Double], b: Array[Double]): Array[Double] = {
    assert(a.length == b.length)
    val result = Array.fill(a.length)(0.0)
    for(i <- a.indices) result(i) = a(i) - b(i)
    result
  }

  private def makeIdentity(size: Int): Array[Double] = {
    val result = Array.fill(size)(1.0)
    result
  }

  private def negate(a: Array[Double]): Array[Double] = {
    for (i <- a.indices) a(i) = -1 * a(i)
    a
  }

  def backward(targetVals: Array[Double], dropNodeList: List[mutable.HashSet[Int]]): Unit = {
    var newWeightList = List[Array[Array[Double]]]()
    var newBiasList = List[Array[Double]]()
    if (targetVals.length != numUnitsPerLayer.last) throw new Exception("Bad Output length.")
    var del = targetVals
    for (i <- numUnitsPerLayer.size-1 to 1 by -1) {
      if (i == numUnitsPerLayer.size-1) {
        val identity = makeIdentity(outputPerLayer(i).size)
        del = hadamardProduct(negate(subtractVectors(del, outputPerLayer(i))), hadamardProduct(outputPerLayer(i), subtractVectors(identity, outputPerLayer(i))))
      } else {
        val identity = makeIdentity(outputPerLayer(i).size)
        val res = multiply(del, weightList(i).transpose, Array[Double](), dropNodeList(i-1))
        del = hadamardProduct(res, hadamardProduct(outputPerLayer(i), subtractVectors(identity, outputPerLayer(i))))
      }
      println(s"del ${i}" + printVector(del))
      newWeightList = addMatrices(deltaWeightList(i-1), multiplyAndReturnMatrix(del, outputPerLayer(i-1))) +: newWeightList
      newBiasList = addVectors(del, deltaBiasList(i-1)) +: newBiasList
    }
    deltaWeightList = newWeightList
    deltaBiasList = newBiasList
   // println("dbl is=" + printVector(deltaBiasList(0)) + ",,," + println(deltaBiasList(1)))
  }

  private def addVectors(a: Array[Double], b: Array[Double]): Array[Double] = {
    val result = Array.fill(a.length)(0.0)
    for (i <- a.indices) result(i) = a(i) + b(i)
    result
  }


  private def multiplyAndReturnMatrix(a: Array[Double], b: Array[Double]): Array[Array[Double]] = {
    val result = Array.ofDim[Double](b.length, a.length)
    for (i <- a.indices) {
      for (j <- b.indices)
        result(j)(i) = b(j) * a(i)
    }
    result
  }

  private def subtractMatrices(a: Array[Array[Double]], b: Array[Array[Double]]): Array[Array[Double]] = {
    assert( (a.length == b.length) && (a(0).length == b(0).length) )
    val result = Array.ofDim[Double](a.length, a(0).length)
    for (i <- a.indices) {
      for (j <- a(0).indices)
        result(i)(j) = a(i)(j) - b(i)(j)
    }
    result
  }

  private def multiplyScalarWithMatrix(a: Double, b: Array[Array[Double]]): Array[Array[Double]] = {
    for (i <- b.indices) {
      for (j <- b(0).indices)
        b(i)(j) = a * b(i)(j)
    }
    b
  }

  private def addMatrices(a: Array[Array[Double]], b: Array[Array[Double]]): Array[Array[Double]] = {
    assert( (a.length == b.length) && (a(0).length == b(0).length) )
    val result = Array.ofDim[Double](a.length, a(0).length)
    for (i <- a.indices) {
      for (j <- a(0).indices)
        result(i)(j) = a(i)(j) + b(i)(j)
    }
    result
  }

  private def multiplyScalarWithVector(a: Double, b: Array[Double]): Array[Double] = {
    for(i <- b.indices) {
      b(i) = a * b(i)
    }
    b
  }

  private def updateParameters(alpha: Double, lambda: Double, n: Int): Unit = {
    var newWeightList = List[Array[Array[Double]]]()
    var newBiasList = List[Array[Double]]()

    for (i <- weightList.indices) {
      newWeightList = newWeightList :+ subtractMatrices(weightList(i), multiplyScalarWithMatrix(alpha, addMatrices(multiplyScalarWithMatrix(1.toDouble/n, deltaWeightList(i)), multiplyScalarWithMatrix(lambda, weightList(i)))))
      newBiasList = newBiasList :+ subtractVectors(biasList(i), multiplyScalarWithVector(alpha.toDouble/n, deltaBiasList(i)))
    }
    weightList = newWeightList
    biasList = newBiasList
  }

  private def printDropNodes(list: List[mutable.HashSet[Int]]): Unit = {
    println("size="+list.size)
    println("dropnodes are==")
    for (i <- list.indices) {
      for (j <- list(i)) {
        print(j + " ")
      }
      println()
    }
    println("dropnodes identified.")
  }

  def train(trainData: Array[Array[Double]], trainOutputs: Array[Array[Double]], maxEpochs: Int, learningRate: Double, weightDecay: Double): Unit = {
    var epoch = 0
    val numSamples = trainData.length
    var sequence = List[Int]()
    for (i <- trainData.indices)
      sequence = sequence :+ i
    var k = 1
    while (epoch < maxEpochs) {
      if(epoch == k*100) {
        println(s"${k*100} epochs completed.")
        k = k + 1
      }
      initializeDeltaWeightListAndDeltaBiasList()
      sequence = Random.shuffle(sequence)
      for (i <- trainData.indices) {
        val idx = sequence(i)
        val inpData = trainData(idx).clone
        val targetData = trainOutputs(idx).clone
        val dropNodeList = makeDropNodes()
        //printDropNodes(dropNodeList)
        forward(inpData, dropNodeList)
        println("forward done")
        backward(targetData, dropNodeList)
        println("backward done")
      }
      checkNumericalGradient(trainData(0), trainOutputs(0), learningRate, weightDecay)
      /** update the weights and biases */
      updateParameters(learningRate, weightDecay, numSamples)
      epoch = epoch + 1
    }
  }

  private def checkNumericalGradient(data: Array[Double], outputs: Array[Double], alpha: Double, lambda: Double): Unit = {
     val serializedParameters = serializeParameters(weightList, biasList)
     println("done1")
     val paramGrads = computeNumericalGradient(serializedParameters, outputs, lambda)
     println("done2")
     var gradWeightList = List[Array[Array[Double]]]()
     for (i <- deltaWeightList.indices) {
       gradWeightList = gradWeightList :+ addMatrices(deltaWeightList(i), multiplyScalarWithMatrix(lambda, weightList(i)))
     }
     println("deltaBiasList is =" + printVector(deltaBiasList(0)) + ",,,,"+ printVector(deltaBiasList(1)))
     val backPropGrads = serializeParameters(gradWeightList, deltaBiasList)
     assert (paramGrads.length == backPropGrads.length)
     for (i <- paramGrads.indices) println(paramGrads(i) + "  " + backPropGrads(i))
  }

  private def computeCost(params: Array[Double], output: Array[Double], lambda: Double): Double = {
    val observed = outputPerLayer(numUnitsPerLayer.size -1)
    val expected = output
    val cost = 0.5 * (subtractVectors(observed, expected) map (x => x*x) sum) + (lambda/2) * (params map (x => x*x) sum)
    cost
  }
  private def computeNumericalGradient(serializedParameters: Array[Double], output: Array[Double], lambda: Double): Array[Double] = {
    val paramGrads = Array.fill(serializedParameters.length)(0.0)
    val eps = 0.0001
    for (i <- serializedParameters.indices) {
      val newParam1 = serializedParameters.clone
      newParam1(i) = newParam1(i) + eps
      val newParam2 = serializedParameters.clone
      newParam2(i) = newParam2(i) - eps
      paramGrads(i) = (computeCost(newParam1, output, lambda) - computeCost(newParam2, output, lambda))/ (2*eps)
    }
    paramGrads
  }

  private def serializeParameters(wList: List[Array[Array[Double]]], bList: List[Array[Double]]): Array[Double] = {
    var result = Array[Double]()
    for (i <- wList.indices) {
      for (j <- wList(i).indices) {
        for (k <- wList(i)(j).indices) result = result :+ wList(i)(j)(k)
      }
    }

    for (i <- bList.indices) {
      for (j <- bList(i).indices) result = result :+ bList(i)(j)
    }

    result
  }
}


object ExecuteDropOutNN extends App {
  val nn = new DropOutNN(List(2,2,2))
  nn.initializeNeuralNet()
  //nn.forward(Array(0.2,0.5,0.7), dropNodeList)
  //println("forward done")
  //nn.backward(Array(1.2,2.2,3.2,4.1), dropNodeList)
  nn.train(Array(Array(0.1,0.5)), Array(Array(0.1,0.9)), 1, 0.1, 0.1)
  //nn.printAllWeightsAndBiases()
}

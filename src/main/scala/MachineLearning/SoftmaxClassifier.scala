package MachineLearning

/**
  * Created by raghvendra.singh on 1/28/16.
  */
import breeze.linalg._

object ExecuteSoftmaxClassifier extends App {
  NNUtils.initializeNet(Vector(5,4,3))
  /** Training data with one sample per column. No. of rows = No. of features per sample. */
  val trainingSamples = DenseMatrix((1.0,1.5),(2.0,2.2),(-1.0,-1.6),(-2.0,-2.1),(-3.0,-3.4))
  /** Training output with one output sample per column. No. of rows = No. of output units in neural net.
    * For Softmax output make sure that training labels are in one hot vector format.
    */
  val trainingLabels = DenseMatrix((0.0,1.0),(1.0,0.0),(0.0,0.0))
  /** Maximum number of iterations while training the model */
  val maxEpochs = 1
  /** Learning rate used in the optimization algorithm. */
  val learningRate = 0.1
  /** It is a regularization parameter to avoid over fitting in neural net. */
  val weightDecay = 0.3
  /** This a boolean flag to indicate numerical gradient check. */
  val checkNumericalGrad = true
  /** This is a boolean flag to indicate use of dropout neural net. */
  val isDropOut = false
  /** This is a dropout probability. Each node in hidden layer can be chucked out with this probability. */
  val dropOutProbability = 0.0

  train()

  /** This method trains the neural network using all the parameters defined above */
  def train(): Unit = {
    var epoch = 0
    val numSamples = trainingSamples.cols
    var k = 1
    while (epoch < maxEpochs) {
      if (epoch == k * 100) {
        println(s"${k * 100} epochs completed.")
        k = k + 1
      }
      val dropNodeList = NNUtils.makeDropNodes(numSamples, dropOutProbability, isDropOut)
      val shuffledTuple = NNUtils.shuffleMatrix(trainingSamples, trainingLabels)
      NNUtils.forward(shuffledTuple._1, dropNodeList)
      /** Output layer uses softmax function.*/
      NNUtils.softmaxLossForward()
      /** We are trying to minimize softmax cost function */
      val outputDelta = NNUtils.softmaxLossBackward(shuffledTuple._2)
      NNUtils.backward(numSamples, outputDelta, dropNodeList, weightDecay)
      if (checkNumericalGrad) {
        var numGrad: DenseVector[Double] = null
        numGrad = NNUtils.computeNumericalGradientSE(numSamples, shuffledTuple._1, shuffledTuple._2, weightDecay)
        NNUtils.checkNumericalGradient(numGrad)
        epoch = 1
      }
      NNUtils.updateWeightsGradientDescent(numSamples, learningRate, weightDecay)
      epoch = epoch + 1
    }
  }
}
package MachineLearning

/**
  * Created by raghvendra.singh on 1/9/16.
  */

import breeze.linalg._

object ExecuteDropOutNeuralNet extends App {
  NNUtils.initializeNet(Vector(2,3,3))
  /** Training data with one sample per column. No. of rows = No. of features per sample. */
  val trainingSamples = DenseMatrix((1.0,1.5,2.0,2.2,-1.0),(-1.6,-2.0,-2.1,-3.0,-3.4))
  /** Training output with one output sample per column. No. of rows = No. of output units in neural net. */
  val trainingLabels = DenseMatrix((0.2,1.0,1.1,1.5,2.1),(1.0,2.1,1.4,0.4,0.6),(0.7,0.2,1.1,1.5,1.9))
  /** Maximum number of iterations while training the model */
  val maxEpochs = 1
  /** Learning rate used in the optimization algorithm. */
  val learningRate = 0.1
  /** It is a regularization parameter to avoid over fitting in neural net. */
  val weightDecay = 0.2
  /** This a boolean flag to indicate numerical gradient check. If it is true then the algorithm executes for only
    * one iteration just to make sure that gradient through back propagation is calculated correctly
    */
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
      /** Output layer uses non linear sigmoid function.*/
      NNUtils.meanSquaredErrorLossForward()
      /** We are trying to minimize mean squared error cost function */
      val outputDelta: DenseMatrix[Double] = NNUtils.meanSquaredErrorLossBackward(shuffledTuple._2)
      NNUtils.backward(numSamples, outputDelta, dropNodeList, weightDecay)
      if (checkNumericalGrad) {
        val numGrad: DenseVector[Double] = NNUtils.computeNumericalGradientMSE(numSamples, shuffledTuple._1, shuffledTuple._2, weightDecay)
        NNUtils.checkNumericalGradient(numGrad)
        epoch = 1
      }
      NNUtils.updateWeightsGradientDescent(numSamples, learningRate, weightDecay)
      epoch = epoch + 1
    }
  }
}

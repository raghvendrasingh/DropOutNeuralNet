package MachineLearning

/**
  * Created by raghvendra.singh on 1/9/16.
  */

import breeze.linalg._

object ExecuteDropOutNeuralNet extends App {
  NNUtils.initializeNet(Vector(5,4,4,3))
  /** For Softmax output make sure that training labels are in one hot vector format. */
  NNUtils.train(DenseMatrix((1.0,1.5),(2.0,2.2),(-1.0,-1.6),(-2.0,-2.1),(-3.0,-3.4)), DenseMatrix((0.0,1.0),(1.0,0.0),(0.0,0.0)), "SE", 1, 0.1, 0.3, true)
}

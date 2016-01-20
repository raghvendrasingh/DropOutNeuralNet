package MachineLearning

/**
  * Created by raghvendra.singh on 1/9/16.
  */

import breeze.linalg._

object ExecuteDropOutNeuralNet extends App {
  NNUtils.initializeNet(Vector(4,3,2))
  NNUtils.train(DenseMatrix((1.0,1.0),(2.0,2.0),(-1.0,-1.0),(-2.0,-2.0)), DenseMatrix((0.0,1.0),(1.0,0.0)), 1, 0.1, 0.3, true, false, 0.5)
}

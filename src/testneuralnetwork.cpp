#include "testneuralnetwork.h"

namespace cave
{
    TestNeuralNetwork::TestNeuralNetwork()
    {
        const int inputSize = 4;
        const int outputSize = 5;

        _neuralNetwork.setScaleWeights(0.1);

        _neuralNetwork.add(NeuralNetwork::DENSE, 5, inputSize);
        _neuralNetwork.add(NeuralNetwork::RELU);
        _neuralNetwork.add(NeuralNetwork::DENSE, 4);
        _neuralNetwork.add(NeuralNetwork::RELU);
        _neuralNetwork.add(NeuralNetwork::DENSE, 3);
        _neuralNetwork.add(NeuralNetwork::SOFTMAX);
        _neuralNetwork.add(NeuralNetwork::CROSS_ENTROPY_LOSS);
    }
}
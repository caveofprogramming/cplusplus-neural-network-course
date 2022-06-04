#include <iostream>
#include "testneuralnetwork.h"
#include "matrixfunctions.h"

namespace cave
{
    TestNeuralNetwork::TestNeuralNetwork()
    {
        _neuralNetwork.setScaleWeights(0.1);

        _neuralNetwork.add(NeuralNetwork::DENSE, 5, _inputSize);
        _neuralNetwork.add(NeuralNetwork::RELU);
        _neuralNetwork.add(NeuralNetwork::DENSE, 4);
        _neuralNetwork.add(NeuralNetwork::RELU);
        _neuralNetwork.add(NeuralNetwork::DENSE, 3);
        _neuralNetwork.add(NeuralNetwork::SOFTMAX);
        _neuralNetwork.add(NeuralNetwork::CROSS_ENTROPY_LOSS);
    }

    bool TestNeuralNetwork::testRunForwards()
    {
        auto testData = MatrixFunctions::generateTestData(_numberItems, _inputSize, _outputSize);

        Matrix &input = testData.input;

        BatchResult batchResult;

        _neuralNetwork.runForwards(batchResult, input);

        return batchResult.io.size() == 7;
    }
}
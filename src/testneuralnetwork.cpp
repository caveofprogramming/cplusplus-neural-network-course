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
        _neuralNetwork.add(NeuralNetwork::DENSE, _outputSize);
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

    bool TestNeuralNetwork::testRunBackwards()
    {
        auto testData = MatrixFunctions::generateTestData(_numberItems, _inputSize, _outputSize);

        Matrix &input = testData.input;
        Matrix &expected = testData.output;

        BatchResult batchResult;

        _neuralNetwork.runForwards(batchResult, input);
        _neuralNetwork.runBackwards(batchResult, expected);

        Matrix &calculatedError = batchResult.error.front();

        std::cout << calculatedError << std::endl;

        auto approximatedError = MatrixFunctions::gradient(input, [&]{
            BatchResult result;
            _neuralNetwork.runForwards(result, input);

            return MatrixFunctions::crossEntropyLoss(result.io.back(), expected);
        });

        std::cout << approximatedError << std::endl;

        return calculatedError == approximatedError;
    }
}
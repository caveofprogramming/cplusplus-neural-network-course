#include <iostream>
#include <map>
#include <string>
#include <functional>

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

    bool TestNeuralNetwork::all()
    {
        std::map<std::string, std::function<bool()>> tests{
            {"RunForwards", std::bind(&TestNeuralNetwork::testRunForwards, this)},
            {"RunBackwards", std::bind(&TestNeuralNetwork::testRunBackwards, this)},
            {"Adjust", std::bind(&TestNeuralNetwork::testAdjust, this)},
        };

        bool result = true;

        for (const auto &[name, function] : tests)
        {
            std::cout << "Running " << name << ": ";
            bool r = function();
            result &= r;
            std::cout << (r ? "passed": "failed") << std::endl;
        }

        std::cout << "\n" << (result ? "All tests passed": "Some tests failed!") << std::endl;

        return result;
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
        _neuralNetwork.runBackwards(batchResult, expected, true);

        Matrix &calculatedError = batchResult.error.front();

        auto approximatedError = MatrixFunctions::gradient(input, [&]
                                                           {
            BatchResult result;
            _neuralNetwork.runForwards(result, input);

            return MatrixFunctions::crossEntropyLoss(result.io.back(), expected); });

        return calculatedError == approximatedError;
    }

    bool TestNeuralNetwork::testAdjust()
    {
        auto testData = MatrixFunctions::generateTestData(_numberItems, _inputSize, _outputSize);

        Matrix &input = testData.input;
        Matrix &expected = testData.output;

        Matrix predictions1 = _neuralNetwork.predict(input);
        double totalLoss1 = MatrixFunctions::crossEntropyLoss(predictions1, expected).rowSums().get(0);

        _neuralNetwork.runBatch(input, expected);

        Matrix predictions2 = _neuralNetwork.predict(input);
        double totalLoss2 = MatrixFunctions::crossEntropyLoss(predictions2, expected).rowSums().get(0);

        return totalLoss1 > totalLoss2;
    }
}
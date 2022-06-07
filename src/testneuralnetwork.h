#pragma once

#include "neuralnetwork.h"

namespace cave
{
    class TestNeuralNetwork
    {
    private:
        int _inputSize{6};
        int _outputSize{6};
        int _numberItems{4};
        cave::NeuralNetwork _neuralNetwork;

    public:
        TestNeuralNetwork();
        bool testRunForwards();
        bool testRunBackwards();
        bool testAdjust();
    };
}
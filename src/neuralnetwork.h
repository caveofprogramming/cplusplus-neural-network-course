#pragma once

#include <vector>

#include "matrix.h"
#include "matrixfunctions.h"

namespace cave
{
    class NeuralNetwork
    {
    private:
        std::vector<Matrix> _weights;
        std::vector<Matrix> _bias;
        std::vector<int> _weightIndices;
    };
}
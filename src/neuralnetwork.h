#pragma once

#include <vector>
#include <iostream>
#include <string>

#include "matrix.h"
#include "matrixfunctions.h"

namespace cave
{
    class NeuralNetwork
    {
    public:
        const std::vector<std::string> _transformNames{"DENSE", "RELU", "SOFTMAX", "MEAN_SQUARES_LOSS", "CROSS_ENTROPY_LOSS"};

        enum Transform
        {
            DENSE,
            RELU,
            SOFTMAX,
            MEAN_SQUARE_LOSS,
            CROSS_ENTROPY_LOSS,
        };

    private:
        std::vector<Matrix> _weights;
        std::vector<Matrix> _biases;
        std::vector<int> _weightIndices;
        std::vector<Transform> _transforms;
        double _scaleWeights{0.2};

    public:
        void setScaleWeights(double scale) { _scaleWeights = scale; }
        void add(Transform transform, int rows = 0, int cols = 0);
        friend std::ostream &operator<<(std::ostream &out, NeuralNetwork &nn);
    };

}
#pragma once

#include <vector>
#include <iostream>
#include <string>
#include <deque>


#include "matrix.h"
#include "vector.h"
#include "matrixfunctions.h"

namespace cave
{
    struct BatchResult
    {
        std::vector<Matrix> io;
        std::deque<Matrix> error;
        int numberItems{0};
        int numberCorrect{0};
        double totalLoss{0};
    };

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
        std::vector<Vector> _biases;
        std::vector<int> _weightIndices;
        std::vector<Transform> _transforms;
        double _scaleWeights{0.2};

    public:
        Matrix predict(Matrix &input);
        BatchResult runBatch(Matrix &input, Matrix &expected);
        void setScaleWeights(double scale) { _scaleWeights = scale; }
        void runForwards(BatchResult &result, Matrix &input);
        void runBackwards(BatchResult &result, Matrix &expected, bool bGenerateInputError = false);
        void adjust(BatchResult &result, double learningRate);
        void add(Transform transform, int rows = 0, int cols = 0);
        friend std::ostream &operator<<(std::ostream &out, NeuralNetwork &nn);
    };

}
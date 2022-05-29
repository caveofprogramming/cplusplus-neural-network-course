#include "matrixfunctions.h"

#include <random>

namespace cave
{
    void MatrixFunctions::modify(Matrix &m, std::function<double(double)> mod)
    {
        int index = 0;

        for(int row = 0; row < m._rows; ++row)
        {
            for(int col = 0; col < m._cols; ++col)
            {
                m._v[index] = mod(m._v[index]);
                ++index;
            }
        }
    }

    Matrix MatrixFunctions::meanSquareLoss(const Matrix &actual, const Matrix &expected)
    {
        Matrix difference = actual - expected;

        modify(difference, [&](double value){ return value * value/actual._rows; });

        return difference.colSums();
    }

    IO MatrixFunctions::generateTestData(int numberItems, int inputSize, int outputSize)
    {
        std::default_random_engine generator;
        std::random_device rd;
        generator.seed(rd());

        std::uniform_int_distribution<int> uniform(1, outputSize);
        std::normal_distribution<double> normal(0, 1);
    }
}
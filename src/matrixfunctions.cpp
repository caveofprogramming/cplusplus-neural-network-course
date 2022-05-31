#include "matrixfunctions.h"

#include <random>
#include <cmath>

namespace cave
{
    void MatrixFunctions::modify(Matrix &m, std::function<double(double)> mod)
    {
        int index = 0;

        for (int row = 0; row < m._rows; ++row)
        {
            for (int col = 0; col < m._cols; ++col)
            {
                m._v[index] = mod(m._v[index]);
                ++index;
            }
        }
    }

    Matrix MatrixFunctions::meanSquareLoss(const Matrix &actual, const Matrix &expected)
    {
        Matrix difference = actual - expected;

        modify(difference, [&](double value)
               { return value * value / actual._rows; });

        return difference.colSums();
    }

    IO MatrixFunctions::generateTestData(int numberItems, int inputSize, int outputSize)
    {
        std::default_random_engine generator;
        std::random_device rd;
        generator.seed(rd());

        std::uniform_int_distribution<int> uniform(1, outputSize);
        std::normal_distribution<double> normal(0, 1);

        Matrix input(inputSize, numberItems);
        Matrix output(outputSize, numberItems);

        for (int item = 0; item < numberItems; ++item)
        {
            int radius = uniform(generator);

            output.set(radius - 1, item, 1);

            double sumsquares = 0.0;

            for (int row = 0; row < inputSize; ++row)
            {
                double value = normal(generator);

                sumsquares += value * value;

                input.set(row, item, value);
            }

            double distance = std::sqrt(sumsquares);

            for (int row = 0; row < inputSize; ++row)
            {
                double value = input.get(row, item);

                input.set(row, item, radius * value / distance);
            }
        }

        return IO(input, output);
    }

    Matrix MatrixFunctions::gradient(Matrix &input, std::function<Matrix()> f)
    {
        Matrix result(input.rows(), input.cols());

        Matrix losses1 = f();

        const double inc = 0.001;

        for(int row = 0; row < input.rows(); ++row)
        {
            for(int col = 0; col < input.cols(); ++ col)
            {
                const double value = input.get(row, col);

                input.set(row, col, value + inc);

                Matrix losses2 = f();

                double rate = (losses2.get(0, col) - losses1.get(0, col))/inc;

                result.set(row, col, rate);

                input.set(row, col, value);
            }
        }

        return result;
    }
}
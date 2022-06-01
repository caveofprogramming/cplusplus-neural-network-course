#include "matrixfunctions.h"

#include <random>
#include <cmath>

namespace cave
{

    Matrix MatrixFunctions::greatestRowIndex(const Matrix &input)
    {
        Matrix result(1, input.cols());

        for(int col = 0; col < input.cols(); ++col)
        {
            double greatest = 0.0;

            for(int row = 0; row < input.rows(); ++row)
            {
                double value = input.get(row, col);

                if(value > greatest)
                {
                    result._v[col] = row;
                    greatest = value;
                }
            }
        }

        return result;
    }

    Matrix MatrixFunctions::softmax(const Matrix &input)
    {
        Matrix result(input.rows(), input.cols());

        for (int i = 0; i < input._v.size(); ++i)
        {
            result._v[i] = std::exp(input._v[i]);
        }

        Matrix colSums = result.colSums();

        int index = 0;
        for (int row = 0; row < result.rows(); ++row)
        {
            for (int col = 0; col < result.cols(); ++col)
            {
                result._v[index] = result._v[index] / colSums._v[col];

                ++index;
            }
        }

        return result;
    }

    Matrix MatrixFunctions::relu(const Matrix &input)
    {
        Matrix result(input.rows(), input.cols());

        for (int i = 0; i < input._v.size(); ++i)
        {
            double value = input._v[i];

            if (value > 0)
            {
                result._v[i] = value;
            }
        }

        return result;
    }

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

    Matrix MatrixFunctions::gradient(Matrix &input, std::function<Matrix()> f, int useLossColumn)
    {
        Matrix result(input.rows(), input.cols());

        Matrix losses1 = f();

        const double inc = 0.001;

        for (int row = 0; row < input.rows(); ++row)
        {
            for (int col = 0; col < input.cols(); ++col)
            {
                const double value = input.get(row, col);

                input.set(row, col, value + inc);

                Matrix losses2 = f();

                int lossColumn = col;

                if (useLossColumn >= 0)
                {
                    lossColumn = useLossColumn;
                }

                double rate = (losses2.get(0, lossColumn) - losses1.get(0, lossColumn)) / inc;

                result.set(row, col, rate);

                input.set(row, col, value);
            }
        }

        return result;
    }
}
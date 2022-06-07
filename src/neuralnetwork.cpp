#include "neuralnetwork.h"
#include <random>
#include <stdexcept>
#include <assert.h>

namespace cave
{
    BatchResult NeuralNetwork::runBatch(Matrix &input, Matrix &expected)
    {
        BatchResult result;

        runForwards(result, input);
        runBackwards(result, expected);
        adjust(result, 0.01);

        Matrix losses = MatrixFunctions::crossEntropyLoss(result.io.back(), expected);
      
        return result;
    }

    void NeuralNetwork::runForwards(BatchResult &result, Matrix &input)
    {
        result.io.push_back(input);

        int weightIndex = 0;

        for (auto transform : _transforms)
        {
            Matrix &output = result.io.back();

            switch (transform)
            {
            case DENSE:
            {
                Matrix &weight = _weights[weightIndex];
                Vector &bias = _biases[weightIndex];

                Matrix product = weight * output;
                product += bias;
                result.io.push_back(product);

                ++weightIndex;
            }
            break;
            case SOFTMAX:
                result.io.push_back(MatrixFunctions::softmax(output));
                break;
            case RELU:
                result.io.push_back(MatrixFunctions::relu(output));
                break;
            case MEAN_SQUARE_LOSS:
                break;
            case CROSS_ENTROPY_LOSS:
                break;
            }
        }
    }

    void NeuralNetwork::runBackwards(BatchResult &result, Matrix &expected, bool bGenerateInputError)
    {
        auto transformsIt = _transforms.rbegin();

        if (*transformsIt != CROSS_ENTROPY_LOSS || *(++transformsIt) != SOFTMAX)
        {
            throw std::logic_error("Last activation must be softmax and loss must be cross entropy.");
        }

        Matrix error;

        bool usedSoftmax = false;

        auto weightIt = _weights.rbegin();
        int weightIndex = _weights.size() - 1;

        for (int i = _transforms.size() - 2; i >= 0; --i)
        {
            Transform transform = _transforms[i];
            Matrix &input = result.io[i];
            Matrix &output = result.io[i + 1];

            switch (transform)
            {
            case DENSE:

                if (bGenerateInputError || weightIndex != 0)
                {
                    Matrix &weight = *weightIt;

                    error = weight.transpose() * error;

                    weightIt++;
                    --weightIndex;
                }
                else
                {
                    error = Matrix();
                }
                break;

            case RELU:

                error = MatrixFunctions::transform(error, [&](int index, double value)
                                                   {
                    if(input.get(index) < 0)
                    {
                        return 0.0;
                    }

                    return value; });
                break;

            case SOFTMAX:
                if (usedSoftmax)
                {
                    throw std::logic_error("Softmax can only be used for output layer.");
                }

                error = output - expected;
                usedSoftmax = true;
                break;

            case CROSS_ENTROPY_LOSS:
                continue;
                break;

            case MEAN_SQUARE_LOSS:
                continue;
                break;
            }

            result.error.push_front(error);
        }
    }

    void NeuralNetwork::adjust(BatchResult &result, double learningRate)
    {
        for (int i = 0; i < _weights.size(); ++i)
        {
            int weightIndex = _weightIndices[i];

            Matrix &weight = _weights[i];
            Matrix &bias = _biases[i];
            Matrix &input = result.io[weightIndex];
            Matrix &error = result.error[weightIndex + 1];

            assert(weight.rows() == error.rows());
            assert(weight.cols() == input.rows());

            bias -= learningRate * error.rowMeans();
            weight -= learningRate / input.cols() * (error * input.transpose());
        }
    }

    std::ostream &operator<<(std::ostream &out, NeuralNetwork &nn)
    {
        out << "Scale weights: " << nn._scaleWeights << std::endl;
        out << std::endl;

        int weightIndex = 0;

        for (auto &t : nn._transforms)
        {
            out << nn._transformNames[t];

            if (t == nn.DENSE)
            {
                Matrix &weight = nn._weights[weightIndex];
                Matrix &bias = nn._biases[weightIndex];

                out << " " << weight.str() << " " << bias.str();

                weightIndex++;
            }

            out << std::endl;
        }

        return out;
    }

    void NeuralNetwork::add(Transform transform, int rows, int cols)
    {
        std::default_random_engine generator;
        std::random_device rd;
        generator.seed(rd());

        std::normal_distribution<double> normal(0, _scaleWeights);

        if (transform == DENSE)
        {
            _weightIndices.push_back(_transforms.size());

            if (cols == 0)
            {
                if (_weights.size() == 0)
                {
                    throw std::invalid_argument("Must specify number of columns for initial dense layer.");
                }

                cols = _weights.back().rows();
            }

            Matrix weight(rows, cols, [&](int)
                          { return normal(generator); });
            Vector bias(rows);

            _weights.push_back(weight);
            _biases.push_back(bias);
        }

        _transforms.push_back(transform);
    }
}
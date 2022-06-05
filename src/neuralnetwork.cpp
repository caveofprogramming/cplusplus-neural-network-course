#include "neuralnetwork.h"
#include <random>
#include <stdexcept>

namespace cave
{
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

    void NeuralNetwork::runBackwards(BatchResult &result, Matrix &expected)
    {
        Matrix error = result.io.back() - expected;

        result.error.push_front(error);


        for(int i = _transforms.size() - 2; i >= 0; --i)
        {
            Transform transform = _transforms[i];
            Matrix &input = result.io[i];
            Matrix &output = result.io[i + 1];

            //std::cout << _transformNames[transform] << std::endl;
            //std::cout << input << std::endl;
            //std::cout << output << std::endl;
            //std::cout << "\n" << std::endl;
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
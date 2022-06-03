#include "neuralnetwork.h"
#include <random>
#include <stdexcept>

namespace cave
{
    std::ostream& operator<<(std::ostream &out, NeuralNetwork &nn)
    {
        out << "Scale weights: " << nn._scaleWeights << std::endl;
        out << std::endl;

        int weightIndex = 0;

        for(auto &t: nn._transforms)
        {
            out << nn._transformNames[t];

            if(t == nn.DENSE)
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

            if(cols == 0)
            {
                if(_weights.size() == 0)
                {
                    throw std::invalid_argument("Must specify number of columns for initial dense layer.");
                }

                cols = _weights.back().rows();
            }

            Matrix weight(rows, cols, [&](int){ return normal(generator); });
            Matrix bias(rows, cols);

            _weights.push_back(weight);
            _biases.push_back(bias);
        }

        _transforms.push_back(transform);
    }
}
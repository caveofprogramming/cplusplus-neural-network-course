#include <functional>
#include <iostream>
#include <vector>
#include <cmath>
#include "matrixfunctions.h"
#include "matrix.h"
#include "neuralnetwork.h"
#include "vector.h"

#include <cmath>

using namespace cave;

int main()
{
    Vector v(3, {2, 4, 5});
    Matrix m(3, 2, {1, 2, 3, 4, 5, 6});

    std::cout << v << std::endl;
    std::cout << m << std::endl;
    m += v;
    std::cout << m << std::endl;

    return 0;
    NeuralNetwork nn;

    const int inputSize = 4;
    const int outputSize = 5;

    nn.setScaleWeights(0.1);

    nn.add(NeuralNetwork::DENSE, 5, inputSize);
    nn.add(NeuralNetwork::RELU);
    nn.add(NeuralNetwork::DENSE, 4);
    nn.add(NeuralNetwork::RELU);
    nn.add(NeuralNetwork::DENSE, 3);
    nn.add(NeuralNetwork::SOFTMAX);
    nn.add(NeuralNetwork::CROSS_ENTROPY_LOSS);

    std::cout << nn << std::endl;

    return 0;
}

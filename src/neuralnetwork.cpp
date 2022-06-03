#include "neuralnetwork.h"

namespace cave
{
    void NeuralNetwork::add(Transform transform, int rows, int cols)
    {
        _transforms.push_back(transform);
    }
}
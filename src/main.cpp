#include <functional>
#include <iostream>
#include <vector>
#include <cmath>
#include "matrixfunctions.h"
#include "matrix.h"
#include "neuralnetwork.h"
#include "vector.h"
#include "testneuralnetwork.h"

#include <cmath>

using namespace cave;

int main()
{
    cave::TestNeuralNetwork tests;

    auto result = tests.testRunBackwards();
    std::cout << (result ? "test passed": "test failed") << std::endl;

    return 0;
}

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

    auto result = tests.testRunForwards();
    std::cout << result << std::endl;

    return 0;
}

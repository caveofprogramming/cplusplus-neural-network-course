#include <functional>
#include <iostream>
#include <vector>
#include <cmath>
#include "matrixfunctions.h"
#include "matrix.h"

#include <cmath>

int main()
{
    const int inputSize = 2;
    const int outputSize = 3;

    auto testData = cave::MatrixFunctions::generateTestData(5, inputSize, outputSize);

    std::cout << testData.input << std::endl;
    std::cout << testData.output << std::endl;

    cave::Matrix &input = testData.input;
    cave::Matrix &expected = testData.output;


    cave::Matrix weight(outputSize, inputSize, [](int i){ return 0.01 * i * i + i; });

    auto network = [&]{
        cave::Matrix result = weight * input;

        return cave::MatrixFunctions::meanSquareLoss(result, expected);
    };

    cave::Matrix grad = cave::MatrixFunctions::gradient(input, network);

    std::cout << network() << std::endl;
    std::cout << grad << std::endl;
    
    return 0;
}


#include <functional>
#include <iostream>
#include <vector>
#include <cmath>
#include "matrixfunctions.h"
#include "matrix.h"

#include <cmath>

int main()
{
    const int inputSize = 4;
    const int outputSize = 3;
    const int numberItems = 2;

    auto testData = cave::MatrixFunctions::generateTestData(numberItems, inputSize, outputSize);

    std::cout << testData.input << std::endl;
    std::cout << testData.output << std::endl;

    cave::Matrix &input = testData.input;
    cave::Matrix &expected = testData.output;

    cave::Matrix weight(outputSize, inputSize, [](int i)
                        { return 0.01 * i * i + i; });

    auto network = [&]
    {
        cave::Matrix result = weight * input;

        return cave::MatrixFunctions::meanSquareLoss(result, expected);
    };

    cave::Matrix grad1 = cave::MatrixFunctions::gradient(weight, network, 0);
    cave::Matrix grad2 = cave::MatrixFunctions::gradient(weight, network, 1);

    std::cout << weight << std::endl;
    std::cout << grad1 + grad2 << std::endl;
    std::cout << (2.0 / outputSize * (weight * input - expected)) * input.transpose() << std::endl;

    return 0;
}

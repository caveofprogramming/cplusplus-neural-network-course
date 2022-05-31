#include <functional>
#include <iostream>
#include <vector>
#include <cmath>
#include "matrixfunctions.h"
#include "matrix.h"

#include <cmath>

int main()
{
    auto testData = cave::MatrixFunctions::generateTestData(5, 2, 3);

    std::cout << testData.input << std::endl;
    std::cout << testData.output << std::endl;

    cave::Matrix &input = testData.input;
    cave::Matrix &expected = testData.output;


    cave::Matrix weight(0, 0);

    auto network = [&]{
        cave::Matrix result = weight * input;

        return cave::MatrixFunctions::meanSquareLoss(result, expected);
    };

    cave::Matrix grad = cave::MatrixFunctions::gradient(input, network);
    
    return 0;
}


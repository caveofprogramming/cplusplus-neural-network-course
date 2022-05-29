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

    double radius = std::sqrt(std::pow(-1.616188, 2) + std::pow(1.178106, 2));

    std::cout << radius << std::endl;
    
    return 0;
}


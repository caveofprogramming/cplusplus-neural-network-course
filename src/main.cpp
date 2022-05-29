#include <functional>
#include <iostream>
#include <vector>
#include <cmath>
#include "matrixfunctions.h"
#include "matrix.h"

int main()
{
    cave::Matrix actual(2, 3, [](int i){ return i; });
    cave::Matrix expected(2, 3, [](int i){ return i*i-2; });

    std::cout << actual << std::endl;
    std::cout << expected << std::endl;
    std::cout << actual - expected << std::endl;

    Matrix losses = cave::MatrixFunctions::meanSquareLoss(actual, expected);

    std::cout << losses << std::endl;

    
    
    return 0;
}


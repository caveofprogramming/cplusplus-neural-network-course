#include <functional>
#include <iostream>
#include <vector>
#include <cmath>
#include "matrixfunctions.h"
#include "matrix.h"

int main()
{
    cave::Matrix m1(2, 3, [](int i){ return 2 * i; });

    std::cout << m1 << std::endl;

    cave::MatrixFunctions::modify(m1, [](double value){
        return 2*value;
    });

    std::cout << m1 << std::endl;
    
    return 0;
}


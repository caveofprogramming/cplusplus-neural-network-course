#pragma once

#include <vector>
#include "matrix.h"

namespace cave
{
    class Vector: public Matrix
    {
    public:
        Vector(int rows): Matrix(rows, 1){};
        Vector(int rows, std::vector<double> values): Matrix(rows, 1, values){};
    };
}
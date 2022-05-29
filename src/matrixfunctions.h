#pragma once

#include <functional>

#include "matrix.h"

using namespace cave;

namespace cave
{
    class MatrixFunctions
    {
    public:
        static void modify(Matrix &m, std::function<double(double)> mod);
    };
}
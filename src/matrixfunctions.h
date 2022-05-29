#pragma once

#include <functional>

#include "matrix.h"

using namespace cave;

namespace cave
{
    struct IO
    {
        cave::Matrix input;
        cave::Matrix output;
    };

    class MatrixFunctions
    {
    public:
        static void modify(Matrix &m, std::function<double(double)> mod);
        static Matrix meanSquareLoss(const Matrix &actual, const Matrix &expected);
        static IO generateTestData(int numberItems, int inputSize, int outputSize);
    };
}
#include "matrixfunctions.h"

namespace cave
{
    void MatrixFunctions::modify(Matrix &m, std::function<double(double)> mod)
    {
        int index = 0;

        for(int row = 0; row < m._rows; ++row)
        {
            for(int col = 0; col < m._cols; ++col)
            {
                m._v[index] = mod(m._v[index]);
                ++index;
            }
        }
    }

    Matrix MatrixFunctions::meanSquareLoss(const Matrix &actual, const Matrix &expected)
    {
        Matrix difference = actual - expected;

        modify(difference, [&](double value){ return value * value/actual._rows; });

        return difference.colSums();
    }
}
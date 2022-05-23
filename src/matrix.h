#pragma once

#include <vector>

namespace cave
{
    class Matrix
    {
    private:
        int _rows{0};
        int _cols{0};
        std::vector<double> _v;

    public:
        Matrix(int rows, int cols): _rows(rows), _cols{cols}{}
    };
}
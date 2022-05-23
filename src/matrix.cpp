#include "matrix.h"
#include <iomanip>

namespace cave
{
    Matrix::Matrix(int rows, int cols, std::function<double(int)> init) : _rows(rows), _cols{cols}
    {
        _v.resize(rows * cols);

        for(int i = 0; i < _v.size(); ++i)
        {
            _v[i] = init(i);
        }
    }

    std::ostream &operator<<(std::ostream &out, cave::Matrix m)
    {
        const int maxRows = 8;
        const int maxCols = 8;

        out << std::fixed;
        out << std::showpos;

        for (int row = 0; row < m._rows; ++row)
        {
            if (row == maxRows)
            {
                out << "...\n";
            }

            if (row >= maxRows)
            {
                continue;
            }

            for (int col = 0; col < m._cols; ++col)
            {
                if (col == maxCols)
                {
                    out << " ...";
                }

                if (col >= maxCols)
                {
                    continue;
                }

                out << std::setprecision(6);
                out << std::setw(12);
                out << m._v[row * m._cols + col];
            }

            out << "\n";
        }
        return out;
    }
}
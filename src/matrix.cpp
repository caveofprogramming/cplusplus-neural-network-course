#include "matrix.h"
#include <iomanip>
#include <sstream>
#include <stdexcept>

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

    /*
     * Arithmetical operators
     */

    Matrix operator+(const Matrix &m1, const Matrix &m2)
    {
        if(m1._rows != m2._rows || m1._cols != m2._cols)
        {
            std::stringstream ss;
            ss << "Cannot add matrices of different size.\n";
            ss << "Matrix 1: " << m1.str() << "\n";
            ss << "Matrix 2: " << m2.str() << "\n";
            throw std::logic_error(ss.str());
        }

        Matrix result(m1._rows, m2._cols);

        for(int i = 0; i < m1._v.size(); ++i)
        {
            result._v[i] = m1._v[i] + m2._v[i];
        }

        return result;
    }

    /*********************************/

    std::string Matrix::str() const
    {
        std::stringstream ss;

        ss << _rows << "x" << _cols;

        return ss.str();
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
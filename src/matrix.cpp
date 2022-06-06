#include "matrix.h"
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace cave
{
    Matrix::Matrix(int rows, int cols, std::function<double(int)> init) : _rows(rows), _cols(cols)
    {
        _v.resize(rows * cols);

        for (int i = 0; i < _v.size(); ++i)
        {
            _v[i] = init(i);
        }
    }

    Matrix::Matrix(int rows, int cols, std::vector<double> values, bool byRowOrder)
        : _rows(rows), _cols(cols)
    {
        if (byRowOrder)
        {
            _v = values;
        }
        else
        {
            Matrix tmp(_cols, _rows);
            tmp._v = values;
            Matrix transposed = tmp.transpose();
            _v = std::move(transposed._v);
        }
    }

    double Matrix::get(int index) const
    {
        return _v[index];
    }

    double Matrix::get(int row, int col) const
    {
        return _v[row * _cols + col];
    }

    void Matrix::set(int index, double value)
    {
        _v[index] = value;
    }

    void Matrix::set(int row, int col, double value)
    {
        _v[row * _cols + col] = value;
    }

    Matrix Matrix::transpose()
    {
        Matrix result(_cols, _rows);

        int index = 0;

        for (int row = 0; row < _rows; ++row)
        {
            for (int col = 0; col < _cols; ++col)
            {
                result._v[col * _rows + row] = _v[index++];
            }
        }

        return result;
    }

    Matrix Matrix::rowSums()
    {
        Matrix result(_rows, 1);

        int index = 0;

        for (int row = 0; row < _rows; ++row)
        {
            for (int col = 0; col < _cols; ++col)
            {
                result._v[row] += _v[index++];
            }
        }

        return result;
    }

    Matrix Matrix::colSums()
    {
        Matrix result(1, _cols);

        int index = 0;

        for (int row = 0; row < _rows; ++row)
        {
            for (int col = 0; col < _cols; ++col)
            {
                result._v[col] += _v[index++];
            }
        }

        return result;
    }

    Matrix Matrix::rowMeans()
    {
        Matrix result(_rows, 1);

        int index = 0;

        for (int row = 0; row < _rows; ++row)
        {
            for (int col = 0; col < _cols; ++col)
            {
                result._v[row] += _v[index++] / _cols;
            }
        }

        return result;
    }

    /*
     * Arithmetical operators
     */

    Matrix operator*(const Matrix &m1, const Matrix &m2)
    {
        if (m1._cols != m2._rows)
        {
            std::stringstream ss;
            ss << "Cannot multiply these matrices: incompatible sizes.\n";
            ss << "Matrix 1: " << m1.str() << "\n";
            ss << "Matrix 2: " << m2.str() << "\n";
            throw std::logic_error(ss.str());
        }

        Matrix result(m1._rows, m2._cols);

        for (int row = 0; row < result._rows; ++row)
        {
            for (int col = 0; col < result._cols; ++col)
            {
                for (int n = 0; n < m1._cols; ++n)
                {
                    result._v[row * result._cols + col] += m2._v[col + n * m2._cols] * m1._v[row * m1._cols + n];
                }
            }
        }

        return result;
    }

    Matrix operator*(double multiplier, const Matrix &m)
    {
        Matrix result(m._rows, m._cols);

        for (int i = 0; i < m._v.size(); ++i)
        {
            result._v[i] = m._v[i] * multiplier;
        }

        return result;
    }

    Matrix &operator-=(Matrix &m1, const Matrix &m2)
    {
        if (m1._rows != m2._rows || m1._cols != m2._cols)
        {
            std::stringstream ss;
            ss << "Cannot subtract matrices of different sizes.\n";
            ss << "Matrix 1: " << m1.str() << "\n";
            ss << "Matrix 2: " << m2.str() << "\n";
            throw std::logic_error(ss.str());
        }

        for (int i = 0; i < m1._v.size(); ++i)
        {
            m1._v[i] -= m2._v[i];
        }

        return m1;
    }

    Matrix operator-(const Matrix &m1, const Matrix &m2)
    {
        if (m1._rows != m2._rows || m1._cols != m2._cols)
        {
            std::stringstream ss;
            ss << "Cannot subtract matrices of different sizes.\n";
            ss << "Matrix 1: " << m1.str() << "\n";
            ss << "Matrix 2: " << m2.str() << "\n";
            throw std::logic_error(ss.str());
        }

        Matrix result(m1._rows, m2._cols);

        for (int i = 0; i < m1._v.size(); ++i)
        {
            result._v[i] = m1._v[i] - m2._v[i];
        }

        return result;
    }

    Matrix operator+(const Matrix &m1, const Matrix &m2)
    {
        if (m1._rows != m2._rows || m1._cols != m2._cols)
        {
            std::stringstream ss;
            ss << "Cannot add matrices of different sizes.\n";
            ss << "Matrix 1: " << m1.str() << "\n";
            ss << "Matrix 2: " << m2.str() << "\n";
            throw std::logic_error(ss.str());
        }

        Matrix result(m1._rows, m2._cols);

        for (int i = 0; i < m1._v.size(); ++i)
        {
            result._v[i] = m1._v[i] + m2._v[i];
        }

        return result;
    }

    /*********************************/

    bool operator==(const Matrix &m1, const Matrix &m2)
    {
        if(m1.rows() != m2.rows() || m1.cols() != m2.cols())
        {
            return false;
        }

        const double tolerance = 0.01;

        for(int i = 0; i < m1._v.size(); ++i)
        {
            if(std::abs(m1._v[i] - m2._v[i]) > tolerance)
            {
                return false;
            }
        }

        return true;
    }

    std::string Matrix::str() const
    {
        std::stringstream ss;

        ss << _rows << "x" << _cols;

        return ss.str();
    }

    std::ostream &operator<<(std::ostream &out, const cave::Matrix &m)
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
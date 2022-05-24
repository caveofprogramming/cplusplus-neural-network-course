#pragma once

#include <vector>
#include <iostream>
#include <functional>
#include <string>

namespace cave
{
    class Matrix
    {
    private:
        int _rows{0};
        int _cols{0};
        std::vector<double> _v;

    public:
        Matrix(int rows, int cols) : _rows(rows), _cols{cols}
        {
            _v.resize(rows * cols);
        };

        Matrix(int rows, int cols, std::vector<double> values) : _rows(rows), _cols{cols}, _v(values){

                                                                                           };

        std::string str() const;

        Matrix(int rows, int cols, std::function<double(int)> init);

        friend std::ostream &operator<<(std::ostream &out, cave::Matrix m);

        /*
         * Arithmetical operators
         */

        friend Matrix operator*(const Matrix &m1, const Matrix &m2);
        friend Matrix operator+(const Matrix &m1, const Matrix &m2);
        friend Matrix operator-(const Matrix &m1, const Matrix &m2);
        friend Matrix &operator-=(Matrix &m1, const Matrix &m2);
        friend Matrix operator*(double multiplier, const Matrix &m);
    };
}
#pragma once

#include <vector>
#include <iostream>
#include <functional>
#include <string>

class MatrixFunctions;

namespace cave
{
    class Matrix
    {
    private:
        int _rows{0};
        int _cols{0};
        std::vector<double> _v;

    public:
        Matrix(){}
        
        Matrix(int rows, int cols) : _rows(rows), _cols(cols)
        {
            _v.resize(rows * cols);
        };

        Matrix(int rows, int cols, std::vector<double> values, bool byRowOrder = true); 

        double rows() const { return _rows; }
        double cols() const { return _cols; }
        std::string str() const;

        Matrix(int rows, int cols, std::function<double(int)> init);

        friend std::ostream &operator<<(std::ostream &out, const cave::Matrix &m);

        Matrix transpose();

        Matrix rowSums();
        Matrix colSums();
        Matrix rowMeans();

        double get(int index) const;
        double get(int row, int col) const;
        void set(int index, double value);
        void set(int row, int col, double value);


        /*
         * Arithmetical operators
         */

        friend Matrix operator*(const Matrix &m1, const Matrix &m2);
        friend Matrix operator+(const Matrix &m1, const Matrix &m2);
        friend Matrix operator-(const Matrix &m1, const Matrix &m2);
        friend Matrix &operator-=(Matrix &m1, const Matrix &m2);
        friend Matrix operator*(double multiplier, const Matrix &m);
        friend bool operator==(const Matrix &m1, const Matrix &m2);

        friend class MatrixFunctions;
    };
}
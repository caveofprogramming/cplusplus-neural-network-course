#pragma once

#include <functional>
#include <vector>

#include "matrix.h"

using namespace cave;

namespace cave
{
    struct IO
    {
        IO(Matrix input, Matrix output)
        {
            this->input = input;
            this->output = output;
        }

        cave::Matrix input;
        cave::Matrix output;
    };

    class MatrixFunctions
    {
    public:
        static void modify(Matrix &m, std::function<double(double)> mod);
        static void modify(Matrix &m, std::function<double(int, double)> mod);
        static Matrix transform(Matrix &m, std::function<double(int, double)> mod);
        static Matrix meanSquareLoss(const Matrix &actual, const Matrix &expected);
        static Matrix crossEntropyLoss(const Matrix &actual, const Matrix &expected);
        static std::vector<bool> itemsCorrect(const Matrix &actual, const Matrix &expected);
        static int numberCorrect(const Matrix &actual, const Matrix &expected);
        static IO generateTestData(int numberItems, int inputSize, int outputSize);
        static Matrix gradient(Matrix &input, std::function<Matrix()> f, int useLossColumn = -1);
        static Matrix relu(const Matrix &input);
        static Matrix softmax(const Matrix &input);
        static Matrix greatestRowIndex(const Matrix &input);
    };
}
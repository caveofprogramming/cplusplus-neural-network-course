#include <functional>
#include <iostream>
#include <vector>
#include <cmath>
#include "matrixfunctions.h"
#include "matrix.h"
#include "neuralnetwork.h"

#include <cmath>

int main()
{
    const int inputSize = 4;
    const int outputSize = 4;
    const int numberItems = 8;

    auto testData = cave::MatrixFunctions::generateTestData(numberItems, inputSize, outputSize);

    cave::Matrix &input = testData.input;
    cave::Matrix &expected = testData.output;

    std::cout << input << std::endl;
    std::cout << expected << std::endl;

    auto correct = cave::MatrixFunctions::itemsCorrect(input, expected);

    for(auto b: correct)
    {
        std::cout << b;
    }

    std::cout << std::endl;

    std::cout << "Number correct: " << cave::MatrixFunctions::numberCorrect(input, expected) << std::endl;

    Matrix softmaxOutput = cave::MatrixFunctions::softmax(input);
    std::cout << softmaxOutput << std::endl;
    std::cout << cave::MatrixFunctions::crossEntropyLoss(softmaxOutput, expected) << std::endl;

    auto network = [&]
    {
        Matrix softmaxOutput = cave::MatrixFunctions::softmax(input);
        return cave::MatrixFunctions::crossEntropyLoss(softmaxOutput, expected);
    };

    cave::Matrix grad = cave::MatrixFunctions::gradient(input, network);

    std::cout << grad << std::endl;

    std::cout << softmaxOutput - expected << std::endl;

    return 0;
}

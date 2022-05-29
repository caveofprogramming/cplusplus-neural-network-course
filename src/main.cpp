#include <functional>
#include <iostream>
#include <vector>
#include <cmath>
#include "matrix.h"

void neuronDemo();
void gradientDemo();
void backpropDemo();

int main()
{
    cave::Matrix m1(2, 2, {1, 2, 3, 4}, false);
    cave::Matrix m2(2, 2, {2, 3, 5, 6}, false);
    
    std::cout << m1 * m2 << std::endl;
    std::cout << m2 * m1 << std::endl;

    return 0;
}

double loss(double actual, double expected)
{
    return std::pow(actual - expected, 2);
}

double gradient(double &x, std::function<double()> func)
{
    const double inc = 0.0001;
    const double originalValue = x;

    double y1 = func();

    x += inc;

    double y2 = func();

    x = originalValue;

    double rate = (y2 - y1) / inc;

    return rate;
}

void gradientDemo()
{
    double x = -6;

    auto func = [&x]
    {
        double y = 3.7 * x + 4;

        return y;
    };

    std::cout << "x: " << x << std::endl;
    std::cout << "func: " << func() << std::endl;
    std::cout << "gradient: " << gradient(x, func) << std::endl;
}

double singleNeuron(std::vector<double> inputs, std::vector<double> weights,
                    double bias)
{
    double weightedSum = 0.0;

    for (int i = 0; i < inputs.size(); ++i)
    {
        weightedSum += inputs[i] * weights[i];
    }

    weightedSum += bias;

    return weightedSum;
}

void neuronDemo()
{
    std::vector<double> inputs{0.2, -0.1, 0.4};
    std::vector<double> weights{0.3, 0.2, -0.3};
    double bias = 0;

    std::cout << singleNeuron(inputs, weights, bias) << std::endl;

    const double expected = 0.9;
    const double learningRate = 0.01;

    for (int epoch = 0; epoch < 1000; epoch++)
    {
        for (int i = 0; i < inputs.size(); ++i)
        {
            auto func = [&]
            { return loss(singleNeuron(inputs, weights, bias), expected); };

            double error = gradient(weights[i], func);

            weights[i] -= learningRate * error;
        }

        auto func = [&]
        { return loss(singleNeuron(inputs, weights, bias), expected); };

        double error = gradient(bias, func);

        bias -= learningRate * error;

        double output = singleNeuron(inputs, weights, bias);
        double lossValue = loss(output, expected);

        std::cout << epoch << ": " << epoch << "; output: " << output << ": loss: " << lossValue << std::endl;
    }
}

void backpropDemo()
{
    const double expected = 3.54;

    double input1 = 2.34;
    double weight1 = 0.87;
    double weight2 = 0.23;
    double bias1 = 0.1;
    double bias2 = 0.2;

    auto network = [&]
    {
        double output1 = singleNeuron({input1}, {weight1}, bias1);
        double output2 = singleNeuron({output1}, {weight2}, bias2);
        return loss(output2, expected);
    };

    double error = gradient(weight1, [&]
                            { return network(); });

    double output1 = singleNeuron({input1}, {weight1}, bias1);
    double output2 = singleNeuron({output1}, {weight2}, bias2);

    std::cout << "Error: " << error << std::endl;
    std::cout << "Calculated Error: " << input1 * weight2 * (2 * (output2 - expected)) << std::endl;
}

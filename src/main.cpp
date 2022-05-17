#include <iostream>
#include <vector>
#include <functional>

void neuronDemo();
void gradientDemo();

int main() {
  gradientDemo();
  return 0;
}

double gradient(double &x, std::function<double()> func) {
  return 0.0;
}

void gradientDemo() {
  double x = 2.0;

  auto func = [x]{
    double y = 0.5 * x;
    y *= y;
    y -= 4;

    return y;
  };

  std::cout << "x: " << x << std::endl;
  std::cout << "func: " << func() << std::endl;
  std::cout << "gradient: " << gradient(x, func) << std::endl;
}

double singleNeuron(std::vector<double> inputs, std::vector<double> weights,
                    double bias) {
  double weightedSum = 0.0;

  for (int i = 0; i < inputs.size(); ++i) {
    weightedSum += inputs[i] * weights[i];
  }

  weightedSum += bias;

  return weightedSum;
}

void neuronDemo() {

  std::vector<double> inputs{0.2, -0.1, 0.4};
  std::vector<double> weights{0.3, 0.2, -0.3};
  double bias = 0;

  std::cout << singleNeuron(inputs, weights, bias) << std::endl;
}

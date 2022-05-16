#include <iostream>
#include <vector>

void neuronDemo();

int main() {
  neuronDemo();
  return 0;
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

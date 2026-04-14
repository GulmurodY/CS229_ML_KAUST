#include "neural_network.h"

NeuralNetwork::NeuralNetwork(int inputDim,
                             int hiddenDim,
                             int outputDim,
                             Activation hiddenActivation,
                             TaskType task)
    : inputDim_(inputDim),
      hiddenDim_(hiddenDim),
      outputDim_(outputDim),
      hiddenActivation_(hiddenActivation),
      task_(task) {}

std::vector<double> NeuralNetwork::forward(const std::vector<double>& x)
{
    (void)x;
    return {};
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& X,
                          const std::vector<std::vector<double>>& Y,
                          int epochs,
                          double learningRate,
                          int batchSize)
{
    (void)X; (void)Y; (void)epochs; (void)learningRate; (void)batchSize;
}

double NeuralNetwork::loss(const std::vector<std::vector<double>>& X,
                           const std::vector<std::vector<double>>& Y)
{
    (void)X; (void)Y;
    return 0.0;
}

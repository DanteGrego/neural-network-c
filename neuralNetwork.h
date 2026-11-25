#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#define MAX_LAYER_SIZE 100
#define MAX_LAYER_CANT 20

#define LEAKY_RELU_CONST 0.1

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct{
    char* name;
    double (*function)(double);
    double (*derivative)(double);
}function;

typedef struct{
    double weights[MAX_LAYER_CANT][MAX_LAYER_SIZE][MAX_LAYER_SIZE];
    double bias[MAX_LAYER_CANT][MAX_LAYER_SIZE];
    double activatedValues[MAX_LAYER_CANT][MAX_LAYER_SIZE];
    double deactivatedValues[MAX_LAYER_CANT][MAX_LAYER_SIZE];
    double weightChanges[MAX_LAYER_CANT][MAX_LAYER_SIZE][MAX_LAYER_SIZE];
    double biasChanges[MAX_LAYER_CANT][MAX_LAYER_SIZE];
    int layerSizes[MAX_LAYER_CANT];
    int cantLayers;
    int activationFunctions[MAX_LAYER_CANT];
}neuralNetwork;

typedef neuralNetwork* NeuralNetwork;

NeuralNetwork createNeuralNetwork(int cantLayers, int* layerSizes, int* activationFunctions, double minWeight, double maxWeight, double minBias, double maxBias);
void deleteNeuralNetwork(NeuralNetwork* net);

void forwardPropagate(NeuralNetwork net, double* input);
void backPropagate(NeuralNetwork net, double* desiredOutput);
void applyChanges(NeuralNetwork net, double learningRate);
void resetChanges(NeuralNetwork net);
double getCost(NeuralNetwork net, double* input, double* desiredOutput);
double getCostBatch(NeuralNetwork net, double* inputs, double* desiredOutputs, int batchSize);
void learn(NeuralNetwork net, double* input, double* desiredOutput, double learningRate);
void learnBatch(NeuralNetwork net, double* inputs, double* desiredOutputs, double learningRate, int batchSize);

void saveNetwork(NeuralNetwork net, char* filePath);
void loadNetwork(NeuralNetwork net, char* filePath);

void showStructure(NeuralNetwork net);
void showWeightsBias(NeuralNetwork net);
void showDeactivatedValues(NeuralNetwork net);
void showActivatedValues(NeuralNetwork net);

double ReLU(double x);
double ReLUDerivative(double x);
double leakyReLU(double x);
double leakyReLUDerivative(double x);
double sigmoid(double x);
double sigmoidDerivative(double x);
double errorFunction(double desiredValue, double actualValue);
double errorFunctionDerivative(double desiredValue, double actualValue);

static const function reluFunc = {"ReLU", &ReLU, &ReLUDerivative};
static const function sigmoidFunc = {"sigmoid", &sigmoid, &sigmoidDerivative};
static const function leakyReLUFunc = {"leaky ReLU", &leakyReLU, &leakyReLUDerivative};

static const function activationFunctions[] = {reluFunc, leakyReLUFunc, sigmoidFunc};

#endif

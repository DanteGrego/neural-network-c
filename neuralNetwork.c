#include "neuralNetwork.h"

double randomValue(double min, double max){
    return ((double)rand() / RAND_MAX) * (max - min) + min;
}

NeuralNetwork createNeuralNetwork(int cantLayers, int* layerSizes, int* activationFunctions, double minWeight, double maxWeight, double minBias, double maxBias){
    NeuralNetwork net = (NeuralNetwork)malloc(sizeof(neuralNetwork));

    net->cantLayers = cantLayers;
    for(int l = 0; l < cantLayers; l++){
        net->layerSizes[l] = layerSizes[l];
        if(l > 0)
            net->activationFunctions[l] = activationFunctions[l];
        for(int n = 0; n < layerSizes[l]; n++){
            net->activatedValues[l][n] = net->deactivatedValues[l][n] = 0; 

            if(l > 0){
                net->bias[l][n] = randomValue(minBias, maxBias);
                net->biasChanges[l][n] = 0;
                for(int w = 0; w < layerSizes[l-1]; w++){
                    net->weights[l][n][w] = randomValue(minWeight, maxWeight);
                    net->weightChanges[l][n][w] = 0;
                }
            }
        }
    }
    
    return net;
}

void deleteNeuralNetwork(NeuralNetwork* net){
    free(*net);
    *net = NULL;
}


void forwardPropagate(NeuralNetwork net, double* input){
    for(int n = 0; n < net->layerSizes[0]; n++){
        net->deactivatedValues[0][n] = net->activatedValues[0][n] = input[n];
    }

    for(int l = 1; l < net->cantLayers; l++)
        for(int n = 0; n < net->layerSizes[l]; n++){
            net->deactivatedValues[l][n] = net->bias[l][n];
            for(int w = 0; w < net->layerSizes[l-1]; w++)
                net->deactivatedValues[l][n] += net->activatedValues[l-1][w] * net->weights[l][n][w];
            net->activatedValues[l][n] = activationFunctions[net->activationFunctions[l]].function(net->deactivatedValues[l][n]);
        }
}

void backPropagate(NeuralNetwork net, double* desiredOutput){
    double backPropValues[MAX_LAYER_CANT][MAX_LAYER_SIZE];

    int l = net->cantLayers - 1;

    for(int n = 0; n < net->layerSizes[l]; n++){
        backPropValues[l][n] = errorFunctionDerivative(desiredOutput[n], net->activatedValues[l][n]);
        backPropValues[l][n] *= activationFunctions[net->activationFunctions[l]].derivative(net->deactivatedValues[l][n]);
        
        net->biasChanges[l][n] = backPropValues[l][n];
        for(int w = 0; w < net->layerSizes[l-1]; w++){
            net->weightChanges[l][n][w] += net->activatedValues[l-1][w] * backPropValues[l][n];
        }
    }

    for(l = net->cantLayers-2; l > 0; l--){
        for(int n = 0; n < net->layerSizes[l]; n++){
            backPropValues[l][n] = 0;
            for(int nn = 0; nn < net->layerSizes[l+1]; nn++){
                backPropValues[l][n] += net->weights[l+1][nn][n] * backPropValues[l+1][nn];
            }
            backPropValues[l][n] *= activationFunctions[net->activationFunctions[l]].derivative(net->deactivatedValues[l][n]);
            
            net->biasChanges[l][n] = backPropValues[l][n];
            for(int w = 0; w < net->layerSizes[l-1]; w++){
                net->weightChanges[l][n][w] += net->activatedValues[l-1][w] * backPropValues[l][n];
            }
        }
    }
}

void applyChanges(NeuralNetwork net, double learningRate){
    for(int l = 1; l < net->cantLayers; l++)
        for(int n = 0; n < net->layerSizes[l]; n++){
            for(int w = 0; w < net->layerSizes[l-1]; w++)
                net->weights[l][n][w] -= net->weightChanges[l][n][w] * learningRate;
            net->bias[l][n] -= net->biasChanges[l][n] * learningRate;
        }
}

void resetChanges(NeuralNetwork net){
    for(int l = 1; l < net->cantLayers; l++)
        for(int n = 0; n < net->layerSizes[l]; n++){
            for(int w = 0; w < net->layerSizes[l-1]; w++)
                net->weightChanges[l][n][w] = 0;
            net->biasChanges[l][n] = 0;
        }
}

double getCost(NeuralNetwork net, double* input, double* desiredOutput){
    double cost = 0;

    forwardPropagate(net, input);

    for(int n = 0; n < net->layerSizes[net->cantLayers-1]; n++)
        cost += errorFunction(desiredOutput[n], net->activatedValues[net->cantLayers-1][n]);

    return cost;
}

double getCostBatch(NeuralNetwork net, double* inputs, double* desiredOutputs, int batchSize){
    double cost = 0;
    int inputSize = net->layerSizes[0], outputSize = net->layerSizes[net->cantLayers-1];
    for(int i = 0; i < batchSize; i++)
        cost += getCost(net, inputs + i * inputSize, desiredOutputs + i * outputSize);

    return cost / batchSize;
}

void learn(NeuralNetwork net, double* input, double* desiredOutput, double learningRate){
    resetChanges(net);
    forwardPropagate(net, input);
    backPropagate(net, desiredOutput);
    applyChanges(net, learningRate);
}

void learnBatch(NeuralNetwork net, double* inputs, double* desiredOutputs, double learningRate, int batchSize){
    int inputSize = net->layerSizes[0], outputSize = net->layerSizes[net->cantLayers-1];
    resetChanges(net);
    for(int i = 0; i < batchSize; i++){
        forwardPropagate(net, inputs + i * inputSize);
        backPropagate(net, desiredOutputs + i * outputSize);
    }
    applyChanges(net, learningRate);
}


void saveNetwork(NeuralNetwork net, char* filePath){
    FILE* file = fopen(filePath, "wb");

    if(file != NULL){
        fwrite(net, sizeof(neuralNetwork), 1, file);
        fclose(file);
        printf("Neural network saved\n");
    }else
        printf("Error: Cannot open file\n");
}

void loadNetwork(NeuralNetwork net, char* filePath){
    FILE* file = fopen(filePath, "rb");

    if(file != NULL){
        fread(net, sizeof(neuralNetwork), 1, file);
        fclose(file);
        printf("Neural network loaded\n");
    }else
        printf("Error: Cannot open file\n");
}


void showStructure(NeuralNetwork net){
    printf("Cant layers: %d\n", net->cantLayers);
    for(int l = 0; l < net->cantLayers; l++)
        printf("Layer %d size: %d activation function: %s\n", l, net->layerSizes[l], activationFunctions[net->activationFunctions[l]].name);
}

void showWeightsBias(NeuralNetwork net){
    for(int l = 1; l < net->cantLayers; l++){
        printf("Layer %d:\n", l);
        for(int n = 0; n < net->layerSizes[l]; n++){
            printf("  Node %d:\n", n);
            printf("  Bias: %f\n", net->bias[l][n]);
            for(int w = 0; w < net->layerSizes[l-1]; w++){
                printf("Weight %d: %f\n", w, net->weights[l][n][w]);
            }
        }
    }
}

void showDeactivatedValues(NeuralNetwork net){
    for(int l = 0; l < net->cantLayers; l++){
        for(int n = 0; n < net->layerSizes[l]; n++)
            printf("%f ", net->deactivatedValues[l][n]);
        printf("\n");
    }
}

void showActivatedValues(NeuralNetwork net){
    for(int l = 0; l < net->cantLayers; l++){
        for(int n = 0; n < net->layerSizes[l]; n++)
            printf("%f ", net->activatedValues[l][n]);
        printf("\n");
    }
}


double ReLU(double x){
    if(x < 0) return 0;
    else return x;
}

double ReLUDerivative(double x){
    if(x < 0) return 0;
    else return 1;
}

double leakyReLU(double x){
    if(x < 0) return x * LEAKY_RELU_CONST;
    else return x;
}

double leakyReLUDerivative(double x){
    if(x < 0) return LEAKY_RELU_CONST;
    else return 1;
}

double sigmoid(double x){
    return 1 / (1 + exp(-x));
}   

double sigmoidDerivative(double x){
    double val = sigmoid(x);
    return val * (1 - val);
}

double errorFunction(double desiredValue, double actualValue){
    double error = actualValue - desiredValue;
    return 0.5 * error * error;
}

double errorFunctionDerivative(double desiredValue, double actualValue){
    double error = actualValue - desiredValue;
    return error;
}
#include "neuralNetwork.h"
#include "raylib.h"

#define WINDOW_W 400
#define WINDOW_H 400

double targetFunction(double x){
    return sin(x / 50) * 100 + WINDOW_H / 2;
}

double targetOutput(double x, double y){
    if(y < targetFunction(x)) return 1;
    else return 0;
    /*
    x -= WINDOW_W / 2;
    y -= WINDOW_H / 2;
    if(x*x + y*y < WINDOW_W * WINDOW_W / (3 * 3)) return 1;
    else return 0;
    */
    /*
    x -= WINDOW_W / 2;
    y -= WINDOW_H / 2;
    return __max(1 - (x*x + y*y) / (WINDOW_W * WINDOW_W / 8 + WINDOW_H * WINDOW_H / 8), 0);
    */
}


int main(){
    int dx = WINDOW_W / 75;
    int dy = WINDOW_H / 75;
    const int cantPoints = WINDOW_W * WINDOW_H / (dx * dy);

    const int cantLayers = 7;

    int netSizes[] = {2, 20, 20, 20, 10, 10, 1};
    
    int activationFunctions[cantLayers];

    for(int i = 0; i < cantLayers-1; i++)
        activationFunctions[i] = 1;

    activationFunctions[cantLayers-1] = 2;

    NeuralNetwork net = createNeuralNetwork(cantLayers, netSizes, activationFunctions, -0.1, 0.1, -0.1, 0.1);

    loadNetwork(net, "datas/networkdata1.bin");

    showStructure(net);

    double *desiredOutputs, *inputs;

    desiredOutputs = (double*)malloc(sizeof(double) * cantPoints);
    inputs = (double*)malloc(sizeof(double) * cantPoints * 2);

    int i = 0;
    for(int x = 0; x < WINDOW_W; x+=dx)
        for(int y = 0; y < WINDOW_H; y+=dy){
            inputs[i*2] = (double)x / WINDOW_W;
            inputs[i*2+1] = (double)y / WINDOW_H;
            desiredOutputs[i] = targetOutput(x, y);
            i++;
        }

    const int batchSize = 200;
    double desiredOutputsBatch[batchSize], inputsBatch[batchSize*2];

    InitWindow(WINDOW_W, WINDOW_H, "Monke 2");

    RenderTexture2D tex = LoadRenderTexture(WINDOW_W, WINDOW_H);

    double initialLearningRate = 0.01, learningRateDecay = 0.005;

    int t = 0;
    int r = 0;
    while(!WindowShouldClose()){
        for(int it = 0; it < 40; it++){
            for(int i = 0; i < batchSize; i++){
                r = ((double)rand() / RAND_MAX) * cantPoints;
                inputsBatch[i*2] = inputs[r*2];
                inputsBatch[i*2+1] = inputs[r*2+1];
                desiredOutputsBatch[i] = desiredOutputs[r];
            }
            
            //double prevCost = getCostBatch(net, inputs, desiredOutputs, WINDOW_H * WINDOW_W / (dx * dy));

            double learningRate = initialLearningRate / (exp(learningRateDecay * t));
            learnBatch(net, inputsBatch, desiredOutputsBatch, learningRate, batchSize);

            //double cost = getCostBatch(net, inputs, desiredOutputs, WINDOW_H * WINDOW_W / (dx * dy));

            //printf("%f %f %d\n", cost, learningRate, t);
        }
        BeginTextureMode(tex);
        for(int x = 0; x < WINDOW_W; x += dx)
            for(int y = 0; y < WINDOW_H; y += dy){
                forwardPropagate(net, (double[]){(double)x / WINDOW_W, (double)y / WINDOW_H});
                double output = net->activatedValues[net->cantLayers-1][0];
                int c = output * 255;
                DrawRectangle(x, y, dx, dy, (Color){c, c, c, 255});
            }
         
        /*    
        for(int x = 0; x < WINDOW_W; x++){
            DrawRectangle(x, targetFunction(x), 1, 1, GREEN);
        }
        */
            
        for(int i = 0; i < batchSize; i++){
            DrawRectangle(inputsBatch[i*2] * WINDOW_W - 2.5, inputsBatch[i*2+1] * WINDOW_H - 2.5, 5, 5, (Color){255 * desiredOutputsBatch[i], 0, 255 * (1 - desiredOutputsBatch[i]), 255});
        }

        EndTextureMode();

        BeginDrawing();
        DrawTextureRec(tex.texture, (Rectangle){0, 0, tex.texture.width, -tex.texture.height}, (Vector2){0, 0}, WHITE);
        DrawFPS(10, 10);
        EndDrawing();
        t++;
    }

    UnloadRenderTexture(tex);
    CloseWindow();

    saveNetwork(net, "datas/networkdata1.bin");
    
    free(desiredOutputs);
    free(inputs);
    deleteNeuralNetwork(&net);

    return 0;
}
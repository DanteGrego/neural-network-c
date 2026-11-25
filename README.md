# Red neuronal en C

Red neuronal implementada en C sin librerías de aprendizaje automático, con entrenamiento por retropropagación. raylib se usa solo para la visualización.

## Características

- Cantidad de capas y de neuronas por capa configurable (hasta 20 capas de 100 neuronas).
- Función de activación por capa: ReLU, leaky ReLU o sigmoide.
- Descenso por gradiente por lotes, con error cuadrático.
- Guardado y carga de la red en un archivo binario.

La red está en `neuralNetwork.h` y `neuralNetwork.c`, y no depende del ejemplo.

## Ejemplo

`main.c` entrena una red que clasifica si un punto del plano está por encima o por debajo de una curva senoidal. La ventana muestra en tiempo real la salida de la red en todo el plano y los puntos del lote actual. El objetivo se cambia en `targetOutput`.

Al iniciar carga los pesos de `datas/networkdata1.bin` y al cerrar la ventana los vuelve a guardar. Para entrenar desde cero, borrar ese archivo.

## Compilación (Windows, GCC)

Requiere [raylib](https://www.raylib.com/). Reemplazar `RAYLIB` por su carpeta `src`:

```
gcc main.c neuralNetwork.c -o main.exe -I RAYLIB -L RAYLIB -lraylib -lopengl32 -lgdi32 -lwinmm
```

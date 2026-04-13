#include <cstddef>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <iomanip>

#define ll long long
#define CSC(call)                                                   \
do {                                                                \
    cudaError_t res = call;                                         \
    if (res != cudaSuccess) {                                       \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));       \
        exit(0);                                                    \
    }                                                               \
} while(0)

__global__ void kernel(double* arr, double* new_arr, ll n) {
    ll idx = blockIdx.x * blockDim.x + threadIdx.x;
    ll stride = blockDim.x * gridDim.x; // Общий размер сетки
    
    // Grid-Stride Loop: позволяет маленькой сетке обработать большой массив
    for (ll i = idx; i < n; i += stride) {
        new_arr[i] = fabs(arr[i]);
    }
}

int main(int argc, char* argv[]) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    // Значения по умолчанию
    int gridDimX = 256;  // Количество блоков
    int blockDimX = 256; // Количество нитей в блоке

    // Считываем конфигурацию ядра из аргументов (./a.out <blocks> <threads>)
    if (argc >= 3) {
        gridDimX = std::atoi(argv[1]);
        blockDimX = std::atoi(argv[2]);
    }

    ll n = 0;
    if (!(std::cin >> n)) return 0;

    double* arr = (double*)malloc(sizeof(double) * n);
    for (ll i = 0; i < n; ++i) {
        std::cin >> arr[i];
    }

    double* dev_arr;
    double* new_arr;
    double* ans = (double*)malloc(sizeof(double) * n);

    CSC(cudaMalloc(&dev_arr, sizeof(double) * n));
    CSC(cudaMalloc(&new_arr, sizeof(double) * n));
    CSC(cudaMemcpy(dev_arr, arr, sizeof(double) * n, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CSC(cudaEventCreate(&start));
    CSC(cudaEventCreate(&stop));

    CSC(cudaEventRecord(start));
    kernel<<<gridDimX, blockDimX>>>(dev_arr, new_arr, n);
    CSC(cudaEventRecord(stop));
    CSC(cudaEventSynchronize(stop));
    CSC(cudaGetLastError());

    float t;
    CSC(cudaEventElapsedTime(&t, start, stop));
    CSC(cudaEventDestroy(start));
    CSC(cudaEventDestroy(stop));

    fprintf(stdout, "Size: %llu GPU time (<<<%d, %d>>>) = %f ms\n", n, gridDimX, blockDimX, t);

    CSC(cudaMemcpy(ans, new_arr, sizeof(double) * n, cudaMemcpyDeviceToHost));
    
    if (n <= 100) {
        for (ll i = 0; i < n; ++i) {
            std::cout << std::setprecision(10) << ans[i] << ' ';
        }
        std::cout << '\n';
    }

    CSC(cudaFree(dev_arr));
    CSC(cudaFree(new_arr));
    free(arr);
    free(ans);

    return 0;
}
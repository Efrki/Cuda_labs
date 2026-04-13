#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <chrono>

#define ll long long

void kernel(double* arr, double* new_arr, ll n){
    for (ll i = 0; i < n; ++i) {
        new_arr[i] = std::fabs(arr[i]);
    }
} 

int main(int argc, char* argv[]) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    
    ll n = 0;
    if (!(std::cin >> n)) return 0;

    // Выделяем память в куче (heap), чтобы избежать SegFault на больших файлах
    double* arr = (double*)malloc(sizeof(double) * n);
    double* new_arr = (double*)malloc(sizeof(double) * n);
    
    for (ll i = 0; i < n; ++i) {
        std::cin >> arr[i];
    }

    auto start = std::chrono::high_resolution_clock::now();
    kernel(arr, new_arr, n);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> duration = end - start;
    
    // Выводим время в поток ошибок stderr (всегда будет видно в терминале)
    fprintf(stdout, "Size: %llu CPU time = %f ms\n", n,  duration.count());

    // Печатаем массив, только если он маленький (для проверки)
    if (n <= 100 || (argc > 1 && strcmp(argv[1], "--print") == 0)) {
        for (ll i = 0; i < n; ++i) {
            std::cout << std::setprecision(10) << new_arr[i] << ' ';
        }
        std::cout << '\n';
    }

    free(arr);
    free(new_arr);
    return 0;
}
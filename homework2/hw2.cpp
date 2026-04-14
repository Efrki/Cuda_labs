#include <iomanip>
#include <ios>
#include <iostream>
#include <vector>
#include <algorithm> 

void bubble_sort(std::vector<float>& array){
    int n = array.size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n - i - 1; ++j) {
            if (array[j] > array[j+1]) {
                std::swap(array[j], array[j+1]);
            }
        }
    }
}

int main(){
    int n = 0;
    std::vector<float> vec;

    std::ios_base::sync_with_stdio(false);
    std::cin.tie(0);
    std::cout << std::scientific << std::setprecision(6);

    std::cin >> n;
    vec.resize(n);

    for (int i = 0; i < n; ++i) {
        std::cin >> vec[i];
    }

    bubble_sort(vec);
    for (float x : vec) {
        std::cout << x << ' ';
    }

    return 0;
}
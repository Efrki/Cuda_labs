#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cstdint>

#define MAX_CLASSES 32
#define BLOCK_SIZE 256

struct ClassAverage {
    float r, g, b;
};

__constant__ ClassAverage d_class_avg[MAX_CLASSES];
__constant__ int d_num_classes;

__global__ void classifyMinDistance(const uint8_t* __restrict__ input,
                                    uint8_t* __restrict__ output,
                                    int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    if (idx >= total_pixels) return;

    const uint8_t* in_pixel = input + idx * 4;
    uint8_t* out_pixel = output + idx * 4;

    float r = in_pixel[0];
    float g = in_pixel[1];
    float b = in_pixel[2];

    int best_class = 0;
    float min_dist = 1e30f;

    for (int c = 0; c < d_num_classes; ++c) {
        float dr = r - d_class_avg[c].r;
        float dg = g - d_class_avg[c].g;
        float db = b - d_class_avg[c].b;
        float dist = dr*dr + dg*dg + db*db;
        if (dist < min_dist) {
            min_dist = dist;
            best_class = c;
        }
    }

    out_pixel[0] = in_pixel[0];
    out_pixel[1] = in_pixel[1];
    out_pixel[2] = in_pixel[2];
    out_pixel[3] = static_cast<uint8_t>(best_class);
}

inline void checkCuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " - "
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
#define CHECK_CUDA(err) checkCuda(err, __FILE__, __LINE__)

std::vector<uint8_t> readImage(const std::string& path, int& width, int& height) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open image file: " << path << std::endl;
        exit(EXIT_FAILURE);
    }
    file.read(reinterpret_cast<char*>(&width), sizeof(int));
    file.read(reinterpret_cast<char*>(&height), sizeof(int));
    size_t data_size = static_cast<size_t>(width) * height * 4;
    std::vector<uint8_t> data(data_size);
    file.read(reinterpret_cast<char*>(data.data()), data_size);
    file.close();
    return data;
}

void writeImage(const std::string& path, int width, int height, const std::vector<uint8_t>& data) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot create output file: " << path << std::endl;
        exit(EXIT_FAILURE);
    }
    file.write(reinterpret_cast<const char*>(&width), sizeof(int));
    file.write(reinterpret_cast<const char*>(&height), sizeof(int));
    file.write(reinterpret_cast<const char*>(data.data()), data.size());
    file.close();
}

int main() {
    std::string input_image_path, output_image_path;

    if (!std::getline(std::cin, input_image_path) || input_image_path.empty()) {
        std::cerr << "Error reading input image path" << std::endl;
        return EXIT_FAILURE;
    }
    if (!std::getline(std::cin, output_image_path) || output_image_path.empty()) {
        std::cerr << "Error reading output image path" << std::endl;
        return EXIT_FAILURE;
    }

    int num_classes;
    if (!(std::cin >> num_classes)) {
        std::cerr << "Error reading number of classes" << std::endl;
        return EXIT_FAILURE;
    }
    if (num_classes > MAX_CLASSES) {
        std::cerr << "Number of classes exceeds MAX_CLASSES (" << MAX_CLASSES << ")" << std::endl;
        return EXIT_FAILURE;
    }
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    int width, height;
    std::vector<uint8_t> image_data = readImage(input_image_path, width, height);

    std::vector<ClassAverage> host_avg(num_classes, {0.0f, 0.0f, 0.0f});

    for (int c = 0; c < num_classes; ++c) {
        int np;
        if (!(std::cin >> np)) {
            std::cerr << "Error reading number of samples for class " << c << std::endl;
            return EXIT_FAILURE;
        }
        if (np <= 0) {
            std::cerr << "Invalid sample count for class " << c << std::endl;
            return EXIT_FAILURE;
        }

        double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
        for (int i = 0; i < np; ++i) {
            int x, y;
            if (!(std::cin >> x >> y)) {
                std::cerr << "Error reading coordinates for sample " << i << " class " << c << std::endl;
                return EXIT_FAILURE;
            }
            if (x < 0 || x >= width || y < 0 || y >= height) {
                std::cerr << "Coordinates (" << x << "," << y << ") out of bounds" << std::endl;
                return EXIT_FAILURE;
            }
            size_t idx = (static_cast<size_t>(y) * width + x) * 4;
            sum_r += image_data[idx];
            sum_g += image_data[idx + 1];
            sum_b += image_data[idx + 2];
        }
        host_avg[c].r = static_cast<float>(sum_r / np);
        host_avg[c].g = static_cast<float>(sum_g / np);
        host_avg[c].b = static_cast<float>(sum_b / np);
    }

    size_t image_bytes = static_cast<size_t>(width) * height * 4;
    uint8_t *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, image_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, image_bytes));

    CHECK_CUDA(cudaMemcpy(d_input, image_data.data(), image_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(d_class_avg, host_avg.data(), num_classes * sizeof(ClassAverage)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_num_classes, &num_classes, sizeof(int)));

    int total_pixels = width * height;
    int grid_size = (total_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    classifyMinDistance<<<grid_size, BLOCK_SIZE>>>(d_input, d_output, width, height);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<uint8_t> output_data(image_bytes);
    CHECK_CUDA(cudaMemcpy(output_data.data(), d_output, image_bytes, cudaMemcpyDeviceToHost));

    writeImage(output_image_path, width, height, output_data);

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return EXIT_SUCCESS;
}
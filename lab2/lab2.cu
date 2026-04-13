#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CSC(call)                                                   \
do {                                                                \
    cudaError_t res = call;                                         \
    if (res != cudaSuccess) {                                       \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));       \
        exit(0);                                                    \
    }                                                               \
} while(0)

__global__ void kernel(cudaTextureObject_t tex, uchar4 *out, int w, int h) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= w || y >= h) return;

    int Gx[3][3] = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}};

    float gradX = 0.0f;
    float gradY = 0.0f;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            uchar4 p = tex2D<uchar4>(tex, x + j, y + i);
            float gray = 0.299f * p.x + 0.587f * p.y + 0.114f * p.z;
            
            gradX += gray * Gx[i + 1][j + 1];
            gradY += gray * Gy[i + 1][j + 1];
        }
    }

    float res = sqrtf(gradX * gradX + gradY * gradY);
    unsigned char final_val = (unsigned char)fminf(255.0f, res);

    uchar4 p_center = tex2D<uchar4>(tex, x, y);
    out[y * w + x] = make_uchar4(final_val, final_val, final_val, p_center.w);
}

int main() {
    char path_in[1024], path_out[1024];
    if (scanf("%1023s %1023s", path_in, path_out) != 2) return 0;

    int w, h;
    FILE *fp = fopen(path_in, "rb");
    if (!fp) return 0;
    
    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);
    
    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), (size_t)w * h, fp);
    fclose(fp);

    cudaArray *arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));
    CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    cudaTextureObject_t tex = 0;
    CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    uchar4 *dev_out;
    CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    kernel<<<grid, block>>>(tex, dev_out, w, h);
    CSC(cudaGetLastError());
    CSC(cudaDeviceSynchronize());

    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    fp = fopen(path_out, "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), (size_t)w * h, fp);
    fclose(fp);

    CSC(cudaDestroyTextureObject(tex));
    CSC(cudaFreeArray(arr));
    CSC(cudaFree(dev_out));
    free(data);

    return 0;
}
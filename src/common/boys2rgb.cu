#include "common/boys2rgb.h"
#include <thrust/device_vector.h>
#include "common/cuda_check.h"

#define T_PER_BLOCK 16
#define PI 3.14159265f

// Code ported from https://github.com/fury-gl/fury/blob/2a00ef29b33a76ef7b804ac6360d72701beb87d2/fury/colormap.py#L95

__device__ float ss(float na, float nd) {
    return na * sinf(nd * PI / 180.0f);
}

__device__ float cc(float na, float nd) {
    return na * cosf(nd * PI / 180.0f);
}

__global__ void Boys2RGBKernel(unsigned char* out_rgb, const float* in_dir_xyz, std::size_t num_points) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_points)
        return;

    float x = in_dir_xyz[3 * i];
    float y = in_dir_xyz[3 * i + 1];
    float z = in_dir_xyz[3 * i + 2];

    const float length = norm3df(x, y, z);
    if (length > 0.0f) {
        x /= length;
        y /= length;
        z /= length;
    }

    float x2 = x * x;
    float y2 = y * y;
    float z2 = z * z;

    float x3 = x * x2;
    float y3 = y * y2;
    float z3 = z * z2;

    float z4 = z2 * z2;

    float xy = x * y;
    float xz = x * z;
    float yz = y * z;

    float hh1 = 0.5 * (3 * z2 - 1) / 1.58;
    float hh2 = 3 * xz / 2.745;
    float hh3 = 3 * yz / 2.745;
    float hh4 = 1.5 * (x2 - y2) / 2.745;
    float hh5 = 6 * xy / 5.5;
    float hh6 = (1 / 1.176) * 0.125 * (35 * z4 - 30 * z2 + 3);
    float hh7 = 2.5 * x * (7 * z3 - 3 * z) / 3.737;
    float hh8 = 2.5 * y * (7 * z3 - 3 * z) / 3.737;
    float hh9 = ((x2 - y2) * 7.5 * (7 * z2 - 1)) / 15.85;
    float hh10 = ((2 * xy) * (7.5 * (7 * z2 - 1))) / 15.85;
    float hh11 = 105 * (4 * x3 * z - 3 * xz * (1 - z2)) / 59.32;
    float hh12 = 105 * (-4 * y3 * z + 3 * yz * (1 - z2)) / 59.32;

    float s0 = -23.0;
    float s1 = 227.9;
    float s2 = 251.0;
    float s3 = 125.0;

    float ss23 = ss(2.71, s0);
    float cc23 = cc(2.71, s0);
    float ss45 = ss(2.12, s1);
    float cc45 = cc(2.12, s1);
    float ss67 = ss(0.972, s2);
    float cc67 = cc(0.972, s2);
    float ss89 = ss(0.868, s3);
    float cc89 = cc(0.868, s3);

    float X = 0.0;

    X = X + hh2 * cc23;
    X = X + hh3 * ss23;

    X = X + hh5 * cc45;
    X = X + hh4 * ss45;

    X = X + hh7 * cc67;
    X = X + hh8 * ss67;

    X = X + hh10 * cc89;
    X = X + hh9 * ss89;

    float Y = 0.0;

    Y = Y + hh2 * -ss23;
    Y = Y + hh3 * cc23;

    Y = Y + hh5 * -ss45;
    Y = Y + hh4 * cc45;

    Y = Y + hh7 * -ss67;
    Y = Y + hh8 * cc67;

    Y = Y + hh10 * -ss89;
    Y = Y + hh9 * cc89;

    float Z = 0.0;

    Z = Z + hh1 * -2.8;
    Z = Z + hh6 * -0.5;
    Z = Z + hh11 * 0.3;
    Z = Z + hh12 * -2.5;

    // scale and normalize to fit
    // in the rgb space

    float w_x = 4.1925;
    float trl_x = -2.0425;
    float w_y = 4.0217;
    float trl_y = -1.8541;
    float w_z = 4.0694;
    float trl_z = -2.1899;

    float fr = 0.9 * fabs((X - trl_x) / w_x) + 0.05;
    float fg = 0.9 * fabs((Y - trl_y) / w_y) + 0.05;
    float fb = 0.9 * fabs((Z - trl_z) / w_z) + 0.05;

    unsigned char r = static_cast<unsigned char>(fmax(0.0, fmin(fr, 1.0)) * 255);
    unsigned char g = static_cast<unsigned char>(fmax(0.0, fmin(fg, 1.0)) * 255);
    unsigned char b = static_cast<unsigned char>(fmax(0.0, fmin(fb, 1.0)) * 255);

    out_rgb[3*i] = r;
    out_rgb[3*i+1] = g;
    out_rgb[3*i+2] = b;
}

void Boys2RGB(unsigned char* out_rgb, const float* in_dir_xyz, std::size_t num_points) {
    if (num_points == 0)
        return;

    const std::size_t count = num_points * 3;
    thrust::device_vector<float> d_dir(in_dir_xyz, in_dir_xyz + count);
    thrust::device_vector<unsigned char> d_rgb(count);
    float* d_dir_ptr = thrust::raw_pointer_cast(d_dir.data());
    unsigned char* d_rgb_ptr = thrust::raw_pointer_cast(d_rgb.data());

    dim3 block(T_PER_BLOCK);
    dim3 grid((num_points + block.x - 1) / block.x);
    Boys2RGBKernel<<<grid, block>>>(d_rgb_ptr, d_dir_ptr, num_points);

    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(out_rgb, d_rgb_ptr, count, cudaMemcpyDeviceToHost));
}

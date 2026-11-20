struct Layer {
    double offset_x;
    double offset_y;
    double offset_z;
    double xz_scale;
    double y_scale;
    float amplitude;
    unsigned char permutations[256];
    unsigned char padding[4];
};

static_assert(sizeof(Layer) == 304, "Layer layout must match Rust");

#include "perlin_gradients.cuh"

__device__ double wrap(double x) {
    constexpr double round_off = 33554432.0;
    constexpr double half_round_off = 0x1.fffffffffffffp+23;
    return x >= -half_round_off && x < half_round_off
        ? x
        : x - floor(x / round_off + 0.5) * round_off;
}

__device__ int permute(const Layer& layer, int index) {
    return layer.permutations[index & 255];
}

__device__ float dot(const Layer& layer, int hash, float x, float y, float z) {
    const float* gradient = gradients[permute(layer, hash) & 15];
    // Match the volume sampler's XZ dot, then add the Y term.
    return (gradient[0] * x + gradient[2] * z) + gradient[1] * y;
}

__device__ float noise_lerp(float t, float start, float end) {
    return start + t * (end - start);
}

__device__ float smoothstep(float x) {
    return x * x * x * (x * (x * 6.0f - 15.0f) + 10.0f);
}

// A column shares X/Z coordinates and the first two permutation lookups across Y.
struct ColumnState {
    int x0, x1, floor_z;
    float rx, rz, tx, tz;
};

__device__ ColumnState prepare_column(const Layer& layer, int block_x, int block_z) {
    const double x = wrap(block_x * layer.xz_scale) + layer.offset_x;
    const double z = wrap(block_z * layer.xz_scale) + layer.offset_z;
    const int floor_x = static_cast<int>(floor(x));
    const int floor_z = static_cast<int>(floor(z));
    const float rx = static_cast<float>(x - floor_x);
    const float rz = static_cast<float>(z - floor_z);
    return {permute(layer, floor_x), permute(layer, floor_x + 1), floor_z,
            rx, rz, smoothstep(rx), smoothstep(rz)};
}

__device__ float sample_y_relative(
    const Layer& layer, const ColumnState& column, int floor_y, float ry, float ty
) {
    const int x0 = column.x0, x1 = column.x1, floor_z = column.floor_z;
    const float rx = column.rx, rz = column.rz, tx = column.tx, tz = column.tz;
    const int xy00 = permute(layer, x0 + floor_y);
    const int xy10 = permute(layer, x1 + floor_y);
    const int xy01 = permute(layer, x0 + floor_y + 1);
    const int xy11 = permute(layer, x1 + floor_y + 1);

    const float z0y0 = noise_lerp(tx, dot(layer, xy00 + floor_z, rx, ry, rz),
                               dot(layer, xy10 + floor_z, rx - 1.0f, ry, rz));
    const float z0y1 = noise_lerp(tx, dot(layer, xy01 + floor_z, rx, ry - 1.0f, rz),
                               dot(layer, xy11 + floor_z, rx - 1.0f, ry - 1.0f, rz));
    const float z1y0 = noise_lerp(tx, dot(layer, xy00 + floor_z + 1, rx, ry, rz - 1.0f),
                               dot(layer, xy10 + floor_z + 1, rx - 1.0f, ry, rz - 1.0f));
    const float z1y1 = noise_lerp(tx, dot(layer, xy01 + floor_z + 1, rx, ry - 1.0f, rz - 1.0f),
                               dot(layer, xy11 + floor_z + 1, rx - 1.0f, ry - 1.0f, rz - 1.0f));
    return noise_lerp(tz, noise_lerp(ty, z0y0, z0y1), noise_lerp(ty, z1y0, z1y1));
}

__device__ float sample_y(const Layer& layer, const ColumnState& column, int block_y) {
    const double y = wrap(block_y * layer.y_scale) + layer.offset_y;
    const int floor_y = static_cast<int>(floor(y));
    const float ry = static_cast<float>(y - floor_y);
    return sample_y_relative(layer, column, floor_y, ry, smoothstep(ry));
}

extern "C" __global__ void fill_noise(
    const Layer* layers,
    unsigned int layer_count,
    const int* origins,
    int size_x,
    int size_y,
    int size_z,
    int step_xz,
    int step_y,
    unsigned int count,
    float* output
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }

    const int samples_per_volume = size_x * size_y * size_z;
    const unsigned int volume = index / samples_per_volume;
    const int local = index % samples_per_volume;
    const int x = origins[3 * volume] + (local / size_y % size_x) * step_xz;
    const int y = origins[3 * volume + 1] + (local % size_y) * step_y;
    const int z = origins[3 * volume + 2] + (local / (size_x * size_y)) * step_xz;
    float value = 0.0f;
    for (unsigned int layer = 0; layer < layer_count; ++layer) {
        value += layers[layer].amplitude * sample_y(layers[layer], prepare_column(layers[layer], x, z), y);
    }
    output[index] = value;
}

__device__ float accumulate_column(
    const Layer* layers, unsigned int layer_count, const ColumnState* columns, int y
) {
    float value = 0.0f;
    for (unsigned int layer = 0; layer < layer_count; ++layer) {
        value += layers[layer].amplitude * sample_y(layers[layer], columns[layer], y);
    }
    return value;
}

template <class Finish>
__device__ void sample_column(
    const Layer* layers, unsigned int layer_count, int x, int base_y, int z,
    int size_y, int step_y, float* column_output, ColumnState* columns,
    const Finish& finish
) {
    for (unsigned int layer = threadIdx.x; layer < layer_count; layer += blockDim.x) {
        columns[layer] = prepare_column(layers[layer], x, z);
    }
    __syncthreads();

    for (int iy = threadIdx.x; iy < size_y; iy += blockDim.x) {
        const int y = base_y + iy * step_y;
        column_output[iy] = finish(accumulate_column(layers, layer_count, columns, y));
    }
}

// One block per (volume, X, Z).
template <class Finish>
__device__ void evaluate_column(
    const Layer* layers, unsigned int layer_count, const int* origins,
    int size_x, int size_y, int size_z, int step_x, int step_y, int step_z,
    float* output, ColumnState* columns, const Finish& finish
) {
    const unsigned int volume = blockIdx.z;
    const int x = origins[3 * volume] + static_cast<int>(blockIdx.x) * step_x;
    const int z = origins[3 * volume + 2] + static_cast<int>(blockIdx.y) * step_z;
    const unsigned int base = ((volume * size_z + blockIdx.y) * size_x + blockIdx.x) * size_y;
    sample_column(layers, layer_count, x, origins[3 * volume + 1], z, size_y, step_y,
                  output + base, columns, finish);
}

struct Identity {
    __device__ float operator()(float value) const { return value; }
};

extern "C" __global__ void fill_columns(
    const Layer* layers, unsigned int layer_count, const int* origins,
    int size_x, int size_y, int size_z, int step_xz, int step_y, float* output
) {
    extern __shared__ ColumnState columns[];
    evaluate_column(layers, layer_count, origins, size_x, size_y, size_z,
                    step_xz, step_y, step_xz, output, columns, Identity{});
}
static_assert(sizeof(ColumnState) == 28, "shared memory size must match Rust");

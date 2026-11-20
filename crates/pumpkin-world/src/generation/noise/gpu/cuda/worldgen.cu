#include <math.h>
#include "density.cuh"

struct Operation {
    int kind;
    float a, b;
};

struct Tail {
    Operation ops[8];
    int count;
    __device__ float operator()(float value) const;
};

static_assert(sizeof(Tail) == 100, "Rust/CUDA tail layout mismatch");

__device__ float Tail::operator()(float value) const {
    // The host walks the router from parent to leaf. Evaluate in reverse order.
    for (int i = count - 1; i >= 0; --i) {
        const Operation op = ops[i];
        switch (op.kind) {
            case 1: value += op.a; break;
            case 2: value *= op.a; break;
            case 3: value = fabsf(value); break;
            case 4: value *= value; break;
            case 5: value = value * value * value; break;
            case 6: value = value > 0 ? value : value * op.a; break;
            case 7: value = value < op.a ? op.a : (value > op.b ? op.b : value); break;
            case 8: value = -value; break;
            case 9: {
                const float clamped = value < -1.0f ? -1.0f : (value > 1.0f ? 1.0f : value);
                value = clamped / 2.0f - clamped * clamped * clamped / 24.0f;
                break;
            }
            case 10: value = value == 0.0f ? INFINITY : 1.0f / value; break;
            case 11: value = sqrtf(value); break;
            case 12: value = value > 0.0f ? 1.0f : (value < 0.0f ? -1.0f : 0.0f); break;
        }
    }
    return value;
}

struct Origins { int xyz[64 * 3]; };

extern "C" __global__ void fill_worldgen(
    const Layer* layers, unsigned int layer_count, Origins origins,
    int size_x, int size_y, int size_z, int step_x, int step_y, int step_z,
    Tail tail, float* output
) {
    extern __shared__ ColumnState columns[];
    evaluate_column(layers, layer_count, origins.xyz, size_x, size_y, size_z,
                    step_x, step_y, step_z, output, columns, tail);
}

struct PlaneDescriptor {
    const Layer* layers;
    unsigned int layer_count;
    Tail tail;
};

constexpr unsigned int MAX_PLANES = 25;
struct GridDescriptor { PlaneDescriptor planes[MAX_PLANES]; };
static_assert(sizeof(PlaneDescriptor) == 112, "Rust/CUDA plane layout mismatch");
static_assert(sizeof(GridDescriptor) == 2800, "Rust/CUDA grid layout mismatch");

extern "C" __global__ void fill_grid(
    const __grid_constant__ GridDescriptor grid, int origin_x, int origin_y, int origin_z,
    int size_x, int size_y, int size_z, int step_x, int step_y, int step_z,
    float* output
) {
    extern __shared__ ColumnState columns[];
    const PlaneDescriptor& plane = grid.planes[blockIdx.z];
    const int x = origin_x + static_cast<int>(blockIdx.x) * step_x;
    const int z = origin_z + static_cast<int>(blockIdx.y) * step_z;
    const unsigned int base =
        ((blockIdx.z * size_z + blockIdx.y) * size_x + blockIdx.x) * size_y;
    sample_column(plane.layers, plane.layer_count, x, origin_y, z, size_y,
                  step_y, output + base, columns, plane.tail);
}

struct GridOrigins { int xyz[8 * 3]; };

extern "C" __global__ void fill_grid_pipeline(
    const __grid_constant__ GridDescriptor grid, const __grid_constant__ GridOrigins origins,
    unsigned int plane_count, int size_x, int size_y, int size_z, int step_x, int step_y,
    int step_z, float* output
) {
    extern __shared__ ColumnState columns[];
    const unsigned int chunk = blockIdx.z / plane_count;
    const PlaneDescriptor& plane = grid.planes[blockIdx.z % plane_count];
    const int x = origins.xyz[3 * chunk] + static_cast<int>(blockIdx.x) * step_x;
    const int z = origins.xyz[3 * chunk + 2] + static_cast<int>(blockIdx.y) * step_z;
    const unsigned int base =
        ((blockIdx.z * size_z + blockIdx.y) * size_x + blockIdx.x) * size_y;
    sample_column(plane.layers, plane.layer_count, x, origins.xyz[3 * chunk + 1], z,
                  size_y, step_y, output + base, columns, plane.tail);
}

struct BlendedDescriptor {
    const Layer* layers;
    const double* smear;
    Tail tail;
    unsigned int main, min, total;
};

static_assert(sizeof(BlendedDescriptor) == 128, "Rust/CUDA blended layout mismatch");

struct SmearY {
    int floor_y;
    float fudged_y, alpha;
};

__device__ SmearY prepare_smear_y(const Layer& layer, double scale, int block_y) {
    const double original_y = block_y * layer.y_scale;
    const double y = wrap(original_y) + layer.offset_y;
    const int floor_y = static_cast<int>(floor(y));
    const double relative_y = y - floor_y;
    const double limit = original_y >= 0.0 && original_y < relative_y
        ? original_y : relative_y;
    const double fudge = floor(limit / scale + static_cast<double>(1.0e-7f)) * scale;
    return {floor_y, static_cast<float>(relative_y - fudge),
            smoothstep(static_cast<float>(relative_y))};
}

static_assert(sizeof(SmearY) == 12, "Rust/CUDA smear layout mismatch");

struct BaseYs { int base[8]; };
struct TableRows { unsigned int row[8]; };

extern "C" __global__ void prepare_terrain_y(
    const __grid_constant__ BlendedDescriptor blend, BaseYs bases,
    int size_y, int step_y, unsigned int row_entries, SmearY* table
) {
    const unsigned int entry = blockIdx.x * blockDim.x + threadIdx.x;
    if (entry >= row_entries) {
        return;
    }
    const unsigned int octave = entry / size_y;
    const int y = bases.base[blockIdx.y] + static_cast<int>(entry % size_y) * step_y;
    table[blockIdx.y * row_entries + entry] =
        prepare_smear_y(blend.layers[octave], blend.smear[octave], y);
}

template<bool CachedY>
__device__ float sample_smear_stack(
    const BlendedDescriptor& blend, const ColumnState* columns,
    unsigned int begin, unsigned int end, int block_y,
    const SmearY* row, int size_y, int iy
) {
    float value = 0.0f;
    for (unsigned int i = begin; i < end; ++i) {
        const Layer& layer = blend.layers[i];
        const SmearY y = CachedY ? row[i * size_y + iy]
            : prepare_smear_y(layer, blend.smear[i], block_y);
        value += layer.amplitude *
            sample_y_relative(layer, columns[i], y.floor_y, y.fudged_y, y.alpha);
    }
    return value;
}

template<bool CachedY = false>
__device__ void sample_blended_column(
    const BlendedDescriptor& blend, int x, int base_y, int z, int size_y, int step_y,
    float* output, ColumnState* columns, const SmearY* row = nullptr
) {
    for (unsigned int i = threadIdx.x; i < blend.total; i += blockDim.x) {
        columns[i] = prepare_column(blend.layers[i], x, z);
    }
    __syncthreads();

    const unsigned int mid = blend.main + blend.min;
    for (int iy = threadIdx.x; iy < size_y; iy += blockDim.x) {
        const int y = base_y + iy * step_y;
        const float alpha = sample_smear_stack<CachedY>(blend, columns, 0, blend.main, y, row, size_y, iy) + 0.5f;
        float value;
        if (alpha <= 0.0f) {
            value = sample_smear_stack<CachedY>(blend, columns, blend.main, mid, y, row, size_y, iy);
        } else if (alpha >= 1.0f) {
            value = sample_smear_stack<CachedY>(blend, columns, mid, blend.total, y, row, size_y, iy);
        } else {
            const float low = sample_smear_stack<CachedY>(blend, columns, blend.main, mid, y, row, size_y, iy);
            const float high = sample_smear_stack<CachedY>(blend, columns, mid, blend.total, y, row, size_y, iy);
            value = noise_lerp(alpha, low, high);
        }
        output[iy] = blend.tail(value);
    }
}

struct Pair {
    float first, second;
};

__device__ Pair sample_smear_pair(
    const BlendedDescriptor& blend, const ColumnState* columns,
    unsigned int begin, unsigned int end, int block_y, bool first, bool second
) {
    Pair values{0.0f, 0.0f};
    if (!first && !second) {
        return values;
    }
    for (unsigned int i = begin; i < end; ++i) {
        const Layer& layer = blend.layers[i];
        const SmearY y = prepare_smear_y(layer, blend.smear[i], block_y);
        // Each column keeps the same octave accumulation order as the single-column path.
        if (first) {
            values.first += layer.amplitude *
                sample_y_relative(layer, columns[i], y.floor_y, y.fudged_y, y.alpha);
        }
        if (second) {
            values.second += layer.amplitude *
                sample_y_relative(layer, columns[blend.total + i], y.floor_y, y.fudged_y, y.alpha);
        }
    }
    return values;
}

__device__ float finish_blend(float alpha, float low, float high) {
    if (alpha <= 0.0f) return low;
    if (alpha >= 1.0f) return high;
    return noise_lerp(alpha, low, high);
}

__device__ void sample_blended_pair(
    const BlendedDescriptor& blend, int x, int next_x, int base_y, int z, int size_y,
    int step_y, float* output, ColumnState* columns
) {
    for (unsigned int i = threadIdx.x; i < blend.total; i += blockDim.x) {
        columns[i] = prepare_column(blend.layers[i], x, z);
        columns[blend.total + i] = prepare_column(blend.layers[i], next_x, z);
    }
    __syncthreads();

    const unsigned int mid = blend.main + blend.min;
    for (int iy = threadIdx.x; iy < size_y; iy += blockDim.x) {
        const int y = base_y + iy * step_y;
        const Pair main = sample_smear_pair(blend, columns, 0, blend.main, y, true, true);
        const float alpha_first = main.first + 0.5f;
        const float alpha_second = main.second + 0.5f;
        const Pair low = sample_smear_pair(
            blend, columns, blend.main, mid, y, !(alpha_first >= 1.0f), !(alpha_second >= 1.0f));
        const Pair high = sample_smear_pair(
            blend, columns, mid, blend.total, y, !(alpha_first <= 0.0f), !(alpha_second <= 0.0f));
        output[iy] = blend.tail(finish_blend(alpha_first, low.first, high.first));
        output[size_y + iy] = blend.tail(finish_blend(alpha_second, low.second, high.second));
    }
}

struct BinaryOperands {
    PlaneDescriptor companions[4];
    unsigned int operations[4];
    unsigned int slots[MAX_PLANES];
};

static_assert(sizeof(BinaryOperands) == 568, "Rust/CUDA binary operand layout mismatch");

__device__ void sample_binary_column(
    const PlaneDescriptor& left, const PlaneDescriptor& right, unsigned int operation,
    int x, int base_y, int z, int size_y, int step_y, float* output, ColumnState* columns
) {
    ColumnState* right_columns = columns + left.layer_count;
    for (unsigned int i = threadIdx.x; i < left.layer_count; i += blockDim.x) {
        columns[i] = prepare_column(left.layers[i], x, z);
    }
    for (unsigned int i = threadIdx.x; i < right.layer_count; i += blockDim.x) {
        right_columns[i] = prepare_column(right.layers[i], x, z);
    }
    __syncthreads();
    for (int iy = threadIdx.x; iy < size_y; iy += blockDim.x) {
        const int y = base_y + iy * step_y;
        const float lhs = left.tail(accumulate_column(left.layers, left.layer_count, columns, y));
        const float rhs = right.tail(
            accumulate_column(right.layers, right.layer_count, right_columns, y));
        switch (operation) {
            case 1: output[iy] = lhs + rhs; break;
            case 2: output[iy] = lhs - rhs; break;
            case 3: output[iy] = lhs * rhs; break;
        }
    }
}

struct Selector {
    unsigned int threshold_count;
    float thresholds[4];
    unsigned int alternatives[5];
};

struct Selectors {
    Selector entries[3];
    unsigned int slots[MAX_PLANES];
};

static_assert(sizeof(Selectors) == 220, "Rust/CUDA selector layout mismatch");

__device__ void sample_selected_column(
    const PlaneDescriptor& input, const Selector& selector, const GridDescriptor& grid,
    int x, int base_y, int z, int size_y, int step_y, float* output, ColumnState* columns
) {
    for (unsigned int i = threadIdx.x; i < input.layer_count; i += blockDim.x) {
        columns[i] = prepare_column(input.layers[i], x, z);
    }
    unsigned int offset = input.layer_count;
    for (unsigned int candidate = 0; candidate <= selector.threshold_count; ++candidate) {
        const PlaneDescriptor& alternative = grid.planes[selector.alternatives[candidate]];
        for (unsigned int i = threadIdx.x; i < alternative.layer_count; i += blockDim.x) {
            columns[offset + i] = prepare_column(alternative.layers[i], x, z);
        }
        offset += alternative.layer_count;
    }
    __syncthreads();

    for (int iy = threadIdx.x; iy < size_y; iy += blockDim.x) {
        const int y = base_y + iy * step_y;
        const float value = input.tail(accumulate_column(input.layers, input.layer_count, columns, y));
        unsigned int choice = 0;
        unsigned int offset = input.layer_count;
        while (choice < selector.threshold_count && !(value < selector.thresholds[choice])) {
            offset += grid.planes[selector.alternatives[choice]].layer_count;
            ++choice;
        }
        const PlaneDescriptor& alternative = grid.planes[selector.alternatives[choice]];
        output[iy] = alternative.tail(
            accumulate_column(alternative.layers, alternative.layer_count, columns + offset, y));
    }
}

template<bool CachedY = false, bool Fuse = false, bool Select = false>
__device__ void terrain_block(
    const GridDescriptor& grid, const BlendedDescriptor& blend, const GridOrigins& origins,
    unsigned int plane_count, int size_x, int size_y, int size_z, int step_x, int step_y,
    int step_z, bool pair_columns, float* output, ColumnState* columns,
    const SmearY* table = nullptr, const TableRows* table_rows = nullptr,
    const BinaryOperands* binary = nullptr, const Selectors* selectors = nullptr
) {
    const unsigned int chunk = blockIdx.z / plane_count;
    const unsigned int index = blockIdx.z % plane_count;
    const int column_x = static_cast<int>(blockIdx.x);
    const int x = origins.xyz[3 * chunk] + column_x * step_x;
    const int y = origins.xyz[3 * chunk + 1];
    const int z = origins.xyz[3 * chunk + 2] + static_cast<int>(blockIdx.y) * step_z;
    const unsigned int base =
        ((blockIdx.z * size_z + blockIdx.y) * size_x + blockIdx.x) * size_y;
    if (blend.total != 0 && index == plane_count - 1) {
        if constexpr (CachedY) {
            if (table != nullptr) {
                const SmearY* row = table + table_rows->row[chunk] * blend.total * size_y;
                sample_blended_column<true>(blend, x, y, z, size_y, step_y, output + base, columns, row);
                return;
            }
        }
        if (pair_columns && column_x % 2 != 0) {
            return;
        }
        if (pair_columns && column_x + 1 < size_x) {
            sample_blended_pair(blend, x, x + step_x, y, z, size_y, step_y, output + base, columns);
        } else {
            sample_blended_column(blend, x, y, z, size_y, step_y, output + base, columns);
        }
    } else {
        const PlaneDescriptor& plane = grid.planes[index];
        if constexpr (Select) {
            const unsigned int slot = selectors->slots[index];
            if (slot != 0) {
                sample_selected_column(plane, selectors->entries[slot - 1], grid, x, y, z,
                                       size_y, step_y, output + base, columns);
                return;
            }
        }
        if constexpr (Fuse) {
            const unsigned int slot = binary->slots[index];
            if (slot != 0) {
                sample_binary_column(plane, binary->companions[slot - 1],
                                     binary->operations[slot - 1], x, y, z, size_y,
                                     step_y, output + base, columns);
                return;
            }
        }
        sample_column(plane.layers, plane.layer_count, x, y, z, size_y, step_y,
                      output + base, columns, plane.tail);
    }
}

extern "C" __global__ void fill_terrain(
    const __grid_constant__ GridDescriptor grid, const __grid_constant__ BlendedDescriptor blend,
    const __grid_constant__ GridOrigins origins, unsigned int plane_count, int size_x,
    int size_y, int size_z, int step_x, int step_y, int step_z, float* output
) {
    extern __shared__ ColumnState columns[];
    terrain_block(grid, blend, origins, plane_count, size_x, size_y, size_z, step_x, step_y,
                  step_z, false, output, columns);
}

extern "C" __global__ void fill_terrain_pair(
    const __grid_constant__ GridDescriptor grid, const __grid_constant__ BlendedDescriptor blend,
    const __grid_constant__ GridOrigins origins, unsigned int plane_count, int size_x,
    int size_y, int size_z, int step_x, int step_y, int step_z, unsigned int pair_columns,
    float* output
) {
    extern __shared__ ColumnState columns[];
    terrain_block(grid, blend, origins, plane_count, size_x, size_y, size_z, step_x, step_y,
                  step_z, pair_columns != 0, output, columns);
}


extern "C" __global__ void fill_terrain_table(
    const __grid_constant__ GridDescriptor grid, const __grid_constant__ BlendedDescriptor blend,
    const __grid_constant__ GridOrigins origins, unsigned int plane_count, int size_x,
    int size_y, int size_z, int step_x, int step_y, int step_z, TableRows rows, const SmearY* table,
    float* output
) {
    extern __shared__ ColumnState columns[];
    terrain_block<true>(grid, blend, origins, plane_count, size_x, size_y, size_z, step_x, step_y,
                        step_z, false, output, columns, table, &rows);
}


extern "C" __global__ void fill_terrain_fused(
    const __grid_constant__ GridDescriptor grid, const __grid_constant__ BlendedDescriptor blend,
    const __grid_constant__ GridOrigins origins, unsigned int plane_count, int size_x,
    int size_y, int size_z, int step_x, int step_y, int step_z, TableRows rows, const SmearY* table,
    const __grid_constant__ BinaryOperands binary, float* output
) {
    extern __shared__ ColumnState columns[];
    terrain_block<true, true>(grid, blend, origins, plane_count, size_x, size_y, size_z,
                              step_x, step_y, step_z, false, output, columns, table, &rows, &binary);
}

extern "C" __global__ void fill_terrain_select(
    const __grid_constant__ GridDescriptor grid, const __grid_constant__ BlendedDescriptor blend,
    const __grid_constant__ GridOrigins origins, unsigned int plane_count, int size_x,
    int size_y, int size_z, int step_x, int step_y, int step_z, TableRows rows, const SmearY* table,
    const __grid_constant__ BinaryOperands binary, const __grid_constant__ Selectors selectors,
    float* output
) {
    extern __shared__ ColumnState columns[];
    terrain_block<true, true, true>(grid, blend, origins, plane_count, size_x, size_y, size_z,
                                   step_x, step_y, step_z, false, output, columns, table, &rows,
                                   &binary, &selectors);
}

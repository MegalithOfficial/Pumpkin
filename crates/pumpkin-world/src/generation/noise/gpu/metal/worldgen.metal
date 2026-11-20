#include <metal_stdlib>
using namespace metal;

constant float3 GRADIENTS[16] = {
    float3(1.0f, 1.0f, 0.0f),
    float3(-1.0f, 1.0f, 0.0f),
    float3(1.0f, -1.0f, 0.0f),
    float3(-1.0f, -1.0f, 0.0f),
    float3(1.0f, 0.0f, 1.0f),
    float3(-1.0f, 0.0f, 1.0f),
    float3(1.0f, 0.0f, -1.0f),
    float3(-1.0f, 0.0f, -1.0f),
    float3(0.0f, 1.0f, 1.0f),
    float3(0.0f, -1.0f, 1.0f),
    float3(0.0f, 1.0f, -1.0f),
    float3(0.0f, -1.0f, -1.0f),
    float3(1.0f, 1.0f, 0.0f),
    float3(0.0f, -1.0f, 1.0f),
    float3(-1.0f, 1.0f, 0.0f),
    float3(0.0f, -1.0f, -1.0f),
};

struct MetalLayer {
    float amplitude;
    uchar permutations[256];
};

struct CoordX {
    int x0;
    int x1;
    float rx;
    float tx;
};

struct CoordZ {
    int floor_z;
    float rz;
    float tz;
};

struct CoordY {
    int floor_y;
    float ry;
    float ty;
};

struct Operation {
    int kind;
    float a;
    float b;
};

struct Tail {
    Operation ops[8];
    int count;
};

struct PlaneDesc {
    uint layer_offset;
    uint layer_count;
    Tail tail;
};

struct BlendedDesc {
    uint layer_offset;
    uint main_count;
    uint min_count;
    uint total_count;
    Tail tail;
};

struct SelectorDesc {
    uint threshold_count;
    float thresholds[4];
    uint alternatives[5];
};

struct BatchParams {
    uint plane_count;
    uint chunk_count;
    uint y_row_count;
    uint size_x;
    uint size_y;
    uint size_z;
    uint y_rows[8];
    PlaneDesc planes[25];
    BlendedDesc blended;
    PlaneDesc companions[4];
    uint binary_ops[4];
    uint binary_slots[25];
    SelectorDesc selectors[3];
    uint selector_slots[25];
};

inline int permute(const device MetalLayer& layer, int index) {
    return int(layer.permutations[index & 255]);
}

inline float grad_dot(const device MetalLayer& layer, int hash, float x, float y, float z) {
    float3 g = GRADIENTS[permute(layer, hash) & 15];
    return (g.x * x + g.z * z) + g.y * y;
}

inline float noise_lerp(float t, float start_v, float end_v) {
    return start_v + t * (end_v - start_v);
}

inline float sample_layer(
    const device MetalLayer& layer,
    CoordX cx,
    CoordZ cz,
    CoordY cy
) {
    int xy00 = permute(layer, cx.x0 + cy.floor_y);
    int xy10 = permute(layer, cx.x1 + cy.floor_y);
    int xy01 = permute(layer, cx.x0 + cy.floor_y + 1);
    int xy11 = permute(layer, cx.x1 + cy.floor_y + 1);

    float z0y0 = noise_lerp(
        cx.tx,
        grad_dot(layer, xy00 + cz.floor_z, cx.rx, cy.ry, cz.rz),
        grad_dot(layer, xy10 + cz.floor_z, cx.rx - 1.0f, cy.ry, cz.rz)
    );
    float z0y1 = noise_lerp(
        cx.tx,
        grad_dot(layer, xy01 + cz.floor_z, cx.rx, cy.ry - 1.0f, cz.rz),
        grad_dot(layer, xy11 + cz.floor_z, cx.rx - 1.0f, cy.ry - 1.0f, cz.rz)
    );
    float z1y0 = noise_lerp(
        cx.tx,
        grad_dot(layer, xy00 + cz.floor_z + 1, cx.rx, cy.ry, cz.rz - 1.0f),
        grad_dot(layer, xy10 + cz.floor_z + 1, cx.rx - 1.0f, cy.ry, cz.rz - 1.0f)
    );
    float z1y1 = noise_lerp(
        cx.tx,
        grad_dot(layer, xy01 + cz.floor_z + 1, cx.rx, cy.ry - 1.0f, cz.rz - 1.0f),
        grad_dot(layer, xy11 + cz.floor_z + 1, cx.rx - 1.0f, cy.ry - 1.0f, cz.rz - 1.0f)
    );
    return noise_lerp(cz.tz, noise_lerp(cy.ty, z0y0, z0y1), noise_lerp(cy.ty, z1y0, z1y1));
}

inline float apply_tail(Tail tail, float v) {
    for (int i = tail.count - 1; i >= 0; --i) {
        Operation op = tail.ops[i];
        switch (op.kind) {
            case 1: v = v + op.a; break;
            case 2: v = v * op.a; break;
            case 3: v = fabs(v); break;
            case 4: v = v * v; break;
            case 5: v = (v * v) * v; break;
            case 6: v = v > 0.0f ? v : v * op.a; break;
            case 7: v = v < op.a ? op.a : (v > op.b ? op.b : v); break;
            case 8: v = -v; break;
            case 9: {
                float c = v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
                v = c / 2.0f - ((c * c) * c) / 24.0f;
                break;
            }
            case 10: v = v == 0.0f ? INFINITY : 1.0f / v; break;
            case 11: v = sqrt(v); break;
            case 12: v = v > 0.0f ? 1.0f : (v < 0.0f ? -1.0f : 0.0f); break;
        }
    }
    return v;
}

inline float accumulate_layers(
    const device MetalLayer* layers,
    const device CoordX* coords_x,
    const device CoordZ* coords_z,
    const device CoordY* coords_y,
    constant BatchParams& batch,
    uint layer_offset,
    uint layer_count,
    uint chunk,
    uint y_row,
    uint col_x,
    uint col_z,
    uint iy
) {
    float value = 0.0f;
    uint x_stride = batch.chunk_count * batch.size_x;
    uint z_stride = batch.chunk_count * batch.size_z;
    uint y_stride = batch.y_row_count * batch.size_y;
    uint x_local = chunk * batch.size_x + col_x;
    uint z_local = chunk * batch.size_z + col_z;
    uint y_local = y_row * batch.size_y + iy;

    for (uint i = 0u; i < layer_count; ++i) {
        uint idx = layer_offset + i;
        CoordX cx = coords_x[idx * x_stride + x_local];
        CoordZ cz = coords_z[idx * z_stride + z_local];
        CoordY cy = coords_y[idx * y_stride + y_local];
        value += layers[idx].amplitude * sample_layer(layers[idx], cx, cz, cy);
    }
    return value;
}

kernel void fill_worldgen_metal(
    const device MetalLayer* layers [[buffer(0)]],
    const device CoordX* coords_x [[buffer(1)]],
    const device CoordZ* coords_z [[buffer(2)]],
    const device CoordY* coords_y [[buffer(3)]],
    constant BatchParams& batch [[buffer(4)]],
    device float* out_values [[buffer(5)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    uint col_x = group_id.x;
    uint col_z = group_id.y;
    uint chunk = group_id.z / batch.plane_count;
    uint plane_idx = group_id.z % batch.plane_count;
    uint y_row = batch.y_rows[chunk];
    uint base_out = ((group_id.z * batch.size_z + col_z) * batch.size_x + col_x) * batch.size_y;

    if (batch.blended.total_count != 0u && plane_idx == batch.plane_count - 1u) {
        uint base_layer = batch.blended.layer_offset;
        uint main_cnt = batch.blended.main_count;
        uint min_cnt = batch.blended.min_count;
        uint max_cnt = batch.blended.total_count - main_cnt - min_cnt;
        for (uint iy = tid; iy < batch.size_y; iy += 64u) {
            float alpha = accumulate_layers(
                layers, coords_x, coords_z, coords_y, batch,
                base_layer, main_cnt, chunk, y_row, col_x, col_z, iy
            ) + 0.5f;
            float value;
            if (alpha <= 0.0f) {
                value = accumulate_layers(
                    layers, coords_x, coords_z, coords_y, batch,
                    base_layer + main_cnt, min_cnt, chunk, y_row, col_x, col_z, iy
                );
            } else if (alpha >= 1.0f) {
                value = accumulate_layers(
                    layers, coords_x, coords_z, coords_y, batch,
                    base_layer + main_cnt + min_cnt, max_cnt, chunk, y_row, col_x, col_z, iy
                );
            } else {
                float low = accumulate_layers(
                    layers, coords_x, coords_z, coords_y, batch,
                    base_layer + main_cnt, min_cnt, chunk, y_row, col_x, col_z, iy
                );
                float high = accumulate_layers(
                    layers, coords_x, coords_z, coords_y, batch,
                    base_layer + main_cnt + min_cnt, max_cnt, chunk, y_row, col_x, col_z, iy
                );
                value = noise_lerp(alpha, low, high);
            }
            out_values[base_out + iy] = apply_tail(batch.blended.tail, value);
        }
        return;
    }

    PlaneDesc plane = batch.planes[plane_idx];
    uint sel_slot = batch.selector_slots[plane_idx];
    if (sel_slot != 0u) {
        SelectorDesc sel = batch.selectors[sel_slot - 1u];
        for (uint iy = tid; iy < batch.size_y; iy += 64u) {
            float selector_val = apply_tail(
                plane.tail,
                accumulate_layers(
                    layers, coords_x, coords_z, coords_y, batch,
                    plane.layer_offset, plane.layer_count, chunk, y_row, col_x, col_z, iy
                )
            );
            uint choice = 0u;
            while (choice < sel.threshold_count && !(selector_val < sel.thresholds[choice])) {
                ++choice;
            }
            PlaneDesc alt = batch.planes[sel.alternatives[choice]];
            out_values[base_out + iy] = apply_tail(
                alt.tail,
                accumulate_layers(
                    layers, coords_x, coords_z, coords_y, batch,
                    alt.layer_offset, alt.layer_count, chunk, y_row, col_x, col_z, iy
                )
            );
        }
        return;
    }

    uint bin_slot = batch.binary_slots[plane_idx];
    if (bin_slot != 0u) {
        PlaneDesc right = batch.companions[bin_slot - 1u];
        uint op = batch.binary_ops[bin_slot - 1u];
        for (uint iy = tid; iy < batch.size_y; iy += 64u) {
            float lhs = apply_tail(
                plane.tail,
                accumulate_layers(
                    layers, coords_x, coords_z, coords_y, batch,
                    plane.layer_offset, plane.layer_count, chunk, y_row, col_x, col_z, iy
                )
            );
            float rhs = apply_tail(
                right.tail,
                accumulate_layers(
                    layers, coords_x, coords_z, coords_y, batch,
                    right.layer_offset, right.layer_count, chunk, y_row, col_x, col_z, iy
                )
            );
            float res = 0.0f;
            if (op == 1u) {
                res = lhs + rhs;
            } else if (op == 2u) {
                res = lhs - rhs;
            } else if (op == 3u) {
                res = lhs * rhs;
            }
            out_values[base_out + iy] = res;
        }
        return;
    }

    for (uint iy = tid; iy < batch.size_y; iy += 64u) {
        out_values[base_out + iy] = apply_tail(
            plane.tail,
            accumulate_layers(
                layers, coords_x, coords_z, coords_y, batch,
                plane.layer_offset, plane.layer_count, chunk, y_row, col_x, col_z, iy
            )
        );
    }
}

#include "perlin_gradients.glsl"

struct Layer {
    double offset_x;
    double offset_y;
    double offset_z;
    double xz_scale;
    double y_scale;
    float amplitude;
    uint _pad0;
    uint permutations[64];
};

struct SmearY {
    int floor_y;
    float fudged_y;
    float alpha;
};

struct ColumnState {
    int x0;
    int x1;
    int floor_z;
    float rx;
    float rz;
    float tx;
    float tz;
};

double wrap_coord(double x) {
    const double round_off = 33554432.0LF;
    const double half_round_off = 16777215.999999998LF;
    if (x >= -half_round_off && x < half_round_off) {
        return x;
    }
    precise double q = floor(x / round_off + 0.5LF);
    precise double wrapped = x - q * round_off;
    return wrapped;
}

float smoothstep_noise(float x) {
    precise float a = x * 6.0;
    precise float b = a - 15.0;
    precise float c = x * b;
    precise float d = c + 10.0;
    precise float x2 = x * x;
    precise float x3 = x2 * x;
    precise float res = x3 * d;
    return res;
}

SmearY compute_smear_y(double offset_y, double y_scale, double scale, int block_y) {
    precise double original_y = double(block_y) * y_scale;
    precise double y = wrap_coord(original_y) + offset_y;
    double fy = floor(y);
    int floor_y = int(fy);
    precise double relative_y = y - fy;
    double limit = (original_y >= 0.0LF && original_y < relative_y) ? original_y : relative_y;
    precise double fudge = floor(limit / scale + double(1.0e-7)) * scale;
    precise double fudged_d = relative_y - fudge;
    SmearY out_y;
    out_y.floor_y = floor_y;
    out_y.fudged_y = float(fudged_d);
    out_y.alpha = smoothstep_noise(float(relative_y));
    return out_y;
}

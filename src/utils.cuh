#pragma once

#include <cstddef>
#include <cuda/std/array>

namespace Params {
    constexpr float AVOID_FACTOR = 2.f;
    constexpr float MATCHING_FACTOR = .6f;
    constexpr float CENTERING_FACTOR = .15f;

    constexpr float  VISION_DISTANCE = 2.f;
    constexpr float  AVOID_DISTANCE = .5f;
    constexpr float  MAX_SPEED = .7f;
    constexpr float  MIN_SPEED = .3f;

    constexpr float  TOP_BOUND = 150.f;
    constexpr float  RIGHT_BOUND = 150.f;
    constexpr float  FAR_BOUND = 150.f;
    constexpr float  BOTTOM_BOUND = -150.f;
    constexpr float  LEFT_BOUND = -150.f;
    constexpr float  NEAR_BOUND = -150.f;


    constexpr int FLOCK_SIZE = 5000000;
    constexpr int BLOCK_SIZE = 256;
    
    constexpr unsigned int uint_ceil(float f)
    {
        const unsigned int i = static_cast<int>(f);
        return f > i ? i + 1 : i;
    }


    constexpr unsigned int X_GRIDS = uint_ceil((Params::RIGHT_BOUND - Params::LEFT_BOUND) / Params::VISION_DISTANCE);
    constexpr unsigned int Y_GRIDS = uint_ceil((Params::TOP_BOUND - Params::BOTTOM_BOUND) / Params::VISION_DISTANCE);
    constexpr unsigned int Z_GRIDS = uint_ceil((Params::FAR_BOUND - Params::NEAR_BOUND) / Params::VISION_DISTANCE);
    constexpr unsigned int AREA_GRIDS = X_GRIDS * Y_GRIDS * Z_GRIDS;

    constexpr float WORLD_WIDTH = Params::RIGHT_BOUND - Params::LEFT_BOUND;
    constexpr float WORLD_HEIGHT = Params::TOP_BOUND - Params::BOTTOM_BOUND;
    constexpr float WORLD_DEPTH = Params::FAR_BOUND - Params::NEAR_BOUND;
};

namespace DeviceHelpers {
    //helpers
    __device__ inline float4 add(float4 a, float4 b) {
        return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }

    __device__ inline float4 sub(float4 a, float4 b) {
        return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
    }

    __device__ inline float4 scale(float4 a, float s) {
        return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
    }
};

struct Accumulator {
    float4 pos_avg{0.0f, 0.0f, 0.0f};
    float4 vel_avg{0.0f, 0.0f, 0.0f};
    unsigned int neighboring_boids = 0.0;
    float4 close{0.0f, 0.0f, 0.0f};
};

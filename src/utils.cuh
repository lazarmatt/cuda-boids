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

    constexpr float  TOP_BOUND = 130.f;
    constexpr float  RIGHT_BOUND = 130.f;
    constexpr float  FAR_BOUND = 130.f;
    constexpr float  BOTTOM_BOUND = -130.f;
    constexpr float  LEFT_BOUND = -130.f;
    constexpr float  NEAR_BOUND = -130.f;


    constexpr int FLOCK_SIZE = 4000000;
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
    __device__ inline float3 add(float3 a, float3 b) {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    __device__ inline float3 sub(float3 a, float3 b) {
        return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    __device__ inline float3 scale(float3 a, float s) {
        return make_float3(a.x * s, a.y * s, a.z * s);
    }

    __device__ inline float dot(float3 a, float3 b) {
        return a.x*b.x + a.y*b.y + a.z*b.z;
    }

    __device__ inline float3 cross(float3 a, float3 b) {
        return make_float3(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        );
    }

    __device__ inline float3 normalize(float3 a) {
        float invLen = rnorm3df(a.x,a.y,a.z);
        return scale(a, invLen);
    }

    __device__ inline cuda::std::array<float4,4> transform(float3 translate, float3 rotate) {
        float3 normRotate = normalize(rotate);
        cuda::std::array<float4,4> ret;
        //column major
        
        float denom = fmaxf(1.0f + normRotate.y, 1e-6f);
        float invnp = __fdividef(1.0f, denom);
        ret[0] = make_float4(1-(normRotate.x*normRotate.x) * invnp, -1*normRotate.x, -1*(normRotate.x*normRotate.z) * invnp, 0.0f);
        ret[1] = make_float4(normRotate.x, normRotate.y, normRotate.z, 0.0f);
        ret[2] = make_float4(-1*(normRotate.x*normRotate.z) * invnp, -1*normRotate.z, 1-(normRotate.z*normRotate.z) * invnp, 0.0f);
        ret[3] = make_float4(translate.x,translate.y,translate.z,1.0f);
   
        return ret;
    };
};

struct Accumulator {
    float3 pos_avg{0.0f, 0.0f, 0.0f};
    float3 vel_avg{0.0f, 0.0f, 0.0f};
    unsigned int neighboring_boids = 0.0;
    float3 close{0.0f, 0.0f, 0.0f};
};

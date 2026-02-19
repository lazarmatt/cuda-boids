#pragma once

#include <cstddef>
#include <cuda/std/array>

namespace Hyperparams {
    constexpr float AVOID_FACTOR = 2.f;
    constexpr float MATCHING_FACTOR = .5f;
    constexpr float CENTERING_FACTOR = .1f;

    constexpr float  VISION_DISTANCE = 2.f;
    constexpr float  AVOID_DISTANCE = .5f;
    constexpr float  MAX_SPEED = .7f;
    constexpr float  MIN_SPEED = .3f;

    constexpr float  TOP_BOUND = 75.f;
    constexpr float  RIGHT_BOUND = 75.f;
    constexpr float  FAR_BOUND = 75.f;
    constexpr float  BOTTOM_BOUND = -75.f;
    constexpr float  LEFT_BOUND = -75.f;
    constexpr float  NEAR_BOUND = -75.f;


    constexpr size_t FLOCK_SIZE = 500000;

    
    constexpr unsigned int uint_ceil(float f)
    {
        const unsigned int i = static_cast<int>(f);
        return f > i ? i + 1 : i;
    }


    constexpr unsigned int X_GRIDS = uint_ceil((Hyperparams::RIGHT_BOUND - Hyperparams::LEFT_BOUND) / Hyperparams::VISION_DISTANCE);
    constexpr unsigned int Y_GRIDS = uint_ceil((Hyperparams::TOP_BOUND - Hyperparams::BOTTOM_BOUND) / Hyperparams::VISION_DISTANCE);
    constexpr unsigned int Z_GRIDS = uint_ceil((Hyperparams::FAR_BOUND - Hyperparams::NEAR_BOUND) / Hyperparams::VISION_DISTANCE);

    
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
        if (normRotate.y < -0.99f) {
            //edge case
            ret[0] = make_float4(1.0f,0.0f,0.0f,0.0f);
            ret[1] = make_float4(0.0f,-1.0f,0.0f,0.0f);
            ret[2] = make_float4(0.0f,0.0f,-1.0f,0.0f);
            ret[3] = make_float4(translate.x,translate.y,translate.z,1.0f);
        } else {
            //ret[0] = {1-(normRotate.x*normRotate.x)/(1+normRotate.y), normRotate.x, -1*(normRotate.x*normRotate.z)/(1+normRotate.y), translate.x};
            //ret[1] = {-1*normRotate.x, normRotate.y, -1*normRotate.z, translate.y};
            //ret[2] = {-1*(normRotate.x*normRotate.z)/(1+normRotate.y), normRotate.z, 1-(normRotate.z*normRotate.z)/(1+normRotate.y), translate.z};
            ret[0] = make_float4(1-(normRotate.x*normRotate.x)/(1+normRotate.y), -1*normRotate.x, -1*(normRotate.x*normRotate.z)/(1+normRotate.y), 0.0f);
            ret[1] = make_float4(normRotate.x, normRotate.y, normRotate.z, 0.0f);
            ret[2] = make_float4(-1*(normRotate.x*normRotate.z)/(1+normRotate.y), -1*normRotate.z, 1-(normRotate.z*normRotate.z)/(1+normRotate.y), 0.0f);
            ret[3] = {translate.x,translate.y,translate.z,1.0f};
        }
        return ret;
    };
};

struct Accumulator {
    float3 pos_avg{0.0f, 0.0f, 0.0f};
    float3 vel_avg{0.0f, 0.0f, 0.0f};
    unsigned int neighboring_boids = 0.0;
    float3 close{0.0f, 0.0f, 0.0f};
};

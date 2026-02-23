#pragma once

#include "params.cuh"
#include <array>
#include <random>

// a flock of boids
// set # of boids in Hyperparams.FLOCK_SIZE
class Flock {
    private:
        int* mpd_gridIndices;
        int* mpd_gridStarts;
        int* mpd_gridEnds;
        int* mpd_boidIndices;
        __half2* mpd_xBuffer;
        __half2* mpd_yBuffer;
        __half2* mpd_zBuffer;
        std::array<float,3> randomPos(std::mt19937& rng);
        std::array<float,3> randomVel(std::mt19937& rng);

    
    public:
        //will copy new boid positions to "draw" params
        void step(__half2* cudaXvx, __half2* cudaYvy, __half2* cudaZvz);
        void genRand(__half2* cudaXvx, __half2* cudaYvy, __half2* cudaZvz);
        Flock();
        ~Flock();

};

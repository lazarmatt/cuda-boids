#pragma once

#include "utils.cuh"
#include <random>

// a flock of boids
// set # of boids in Hyperparams.FLOCK_SIZE
class Flock {
    private:
        int* mpd_gridIndices;
        int* mpd_gridStarts;
        int* mpd_gridEnds;
        int* mpd_boidIndices;
        float4* mpd_newVels;
        float4* mpd_velBuffer;
        float4* mpd_posBuffer;
        float4 randomPos(std::mt19937& rng);
        float4 randomVel(std::mt19937& rng);

        size_t m_xGrids;
        size_t m_yGrids;
        size_t m_zGrids;
    
    public:
        void step(float4* cudaPos, float4* cudaVel);
        void genRand(float4* cudaPos, float4* cudaVel);
        Flock();
        ~Flock();

};

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
        float3* mpd_newVels;
        float3* mpd_velBuffer;
        float3* mpd_posBuffer;
        float3 randomPos(std::mt19937& rng);
        float3 randomVel(std::mt19937& rng);

        size_t m_xGrids;
        size_t m_yGrids;
        size_t m_zGrids;
    
    public:
        void step(float3* cudaPos, float3* cudaVel);
        void genRand(float3* cudaPos, float3* cudaVel);
        Flock();
        ~Flock();

};

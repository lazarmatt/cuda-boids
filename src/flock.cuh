#pragma once

#include "boid.cuh"
#include "utils.cuh"

// a flock of boids
// set # of boids in Hyperparams.FLOCK_SIZE
class Flock {
    private:
        Boid* mpd_boids;
        int* mpd_gridIndices;
        int* mpd_gridStarts;
        int* mpd_boidIndices;
        Boid randomBoid();

        size_t m_xGrids;
        size_t m_yGrids;
        size_t m_zGrids;
    
    public:
        void step(float4* transforms);
        Flock();
        ~Flock();

};

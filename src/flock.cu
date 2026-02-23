#include "flock.cuh"
#include <cmath>
#include <cuda/std/cmath>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/binary_search.h>

float4 Flock::randomVel(std::mt19937& rng) {
    
    std::normal_distribution<float> dist(0.0f, 1.0f);

    float vxUnscaled = dist(rng);
    float vyUnscaled = dist(rng);
    float vzUnscaled = dist(rng);

    float len = std::sqrt(vxUnscaled*vxUnscaled + vyUnscaled*vyUnscaled + vzUnscaled*vzUnscaled);
    float magnitude = (Params::MIN_SPEED + Params::MAX_SPEED) / 2;
    float scale = magnitude / len;

    float4 veloc(vxUnscaled*scale,vyUnscaled*scale,vzUnscaled*scale);

    return veloc;
};

float4 Flock::randomPos(std::mt19937& rng) {

    std::uniform_real_distribution<float> distx(Params::LEFT_BOUND,Params::RIGHT_BOUND);
    std::uniform_real_distribution<float> disty(Params::BOTTOM_BOUND,Params::TOP_BOUND);
    std::uniform_real_distribution<float> distz(Params::NEAR_BOUND,Params::FAR_BOUND);

    float4 posit(distx(rng),disty(rng),distz(rng));

    return posit;
}

__global__ void calcNewVeloc(const float4* __restrict__ pos, const float4* __restrict__ vel, float4* newVel, 
    const int* __restrict__ grids, const int* __restrict__ gridStarts, int* __restrict__ gridEnds) {
        
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < Params::FLOCK_SIZE) {
        Accumulator accum;

        //get data for later
        float4 boidPos = __ldg(&pos[idx]);
        int gridIdx = __ldg(&grids[idx]);
        
        // Get 3D grid space
        int gx = gridIdx % Params::X_GRIDS;
        int gy = (gridIdx / Params::X_GRIDS) % Params::Y_GRIDS;
        int gz = gridIdx / (Params::X_GRIDS * Params::Y_GRIDS);
        
        //surrounding 9 grids
        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
                for (int k = -1; k <= 1; k++)
                {   
                    int nx = (gx + i + Params::X_GRIDS) % Params::X_GRIDS;
                    int ny = (gy + j + Params::Y_GRIDS) % Params::Y_GRIDS;
                    int nz = (gz + k + Params::Z_GRIDS) % Params::Z_GRIDS;
                    int neighborGridIdx = nx + ny * Params::X_GRIDS + nz * Params::X_GRIDS * Params::Y_GRIDS;

                    int neighborStart = __ldg(&gridStarts[neighborGridIdx]);
                    int neighborEnd = __ldg(&gridEnds[neighborGridIdx]);
                    //empty cell
                    if(neighborStart == neighborEnd)
                        continue;
                    
                    
                    
                    for(int neighborIdx = neighborStart;neighborIdx < neighborEnd;neighborIdx++)
                    {
                        float4 neighborPos = __ldg(&pos[neighborIdx]);

                        float4 d = DeviceHelpers::sub(boidPos, neighborPos);

                        // for each axis: if the gap is more than half the world, the short path is through the wrap
                        if (d.x >  Params::WORLD_WIDTH * 0.5f) d.x -= Params::WORLD_WIDTH;
                        if (d.x < -Params::WORLD_WIDTH * 0.5f) d.x += Params::WORLD_WIDTH;
                        if (d.y >  Params::WORLD_HEIGHT * 0.5f) d.y -= Params::WORLD_HEIGHT;
                        if (d.y < -Params::WORLD_HEIGHT * 0.5f) d.y += Params::WORLD_HEIGHT;
                        if (d.z >  Params::WORLD_DEPTH * 0.5f) d.z -= Params::WORLD_DEPTH;
                        if (d.z < -Params::WORLD_DEPTH * 0.5f) d.z += Params::WORLD_DEPTH;

                        float sqDist = d.x*d.x + d.y*d.y + d.z*d.z;

                        if (sqDist < Params::AVOID_DISTANCE*Params::AVOID_DISTANCE) {
                            //Avoiding
                            accum.close = DeviceHelpers::add(accum.close, d);
                        } else if (sqDist < Params::VISION_DISTANCE*Params::VISION_DISTANCE) {
                            // Centering/Matching
                            float4 wrappedNeighborPos = { boidPos.x - d.x, boidPos.y - d.y, boidPos.z - d.z };
                            accum.pos_avg = DeviceHelpers::add(accum.pos_avg, wrappedNeighborPos);
                            accum.vel_avg = DeviceHelpers::add(accum.vel_avg, __ldg(&vel[neighborIdx]));
                            accum.neighboring_boids += 1;
                        }
                    }
                }
        
        float4 boidVel = __ldg(&vel[idx]);

        if (accum.neighboring_boids > 0) {
            //add centering/matching
            accum.pos_avg = DeviceHelpers::scale(accum.pos_avg,1.0f/((float) accum.neighboring_boids));
            accum.vel_avg = DeviceHelpers::scale(accum.vel_avg,1.0f/((float) accum.neighboring_boids));

            boidVel.x = boidVel.x + 
                (accum.pos_avg.x - boidPos.x) * Params::CENTERING_FACTOR +
                (accum.vel_avg.x - boidVel.x) * Params::MATCHING_FACTOR;
            boidVel.y = boidVel.y + 
                (accum.pos_avg.y - boidPos.y) * Params::CENTERING_FACTOR +
                (accum.vel_avg.y - boidVel.y) * Params::MATCHING_FACTOR;
            boidVel.z = boidVel.z + 
                (accum.pos_avg.z - boidPos.z) * Params::CENTERING_FACTOR +
                (accum.vel_avg.z - boidVel.z) * Params::MATCHING_FACTOR;
        }

        // add avoiding
        boidVel.x = boidVel.x + accum.close.x*Params::AVOID_FACTOR;
        boidVel.y = boidVel.y + accum.close.y*Params::AVOID_FACTOR;
        boidVel.z = boidVel.z + accum.close.z*Params::AVOID_FACTOR;

        float invSpeed = rnorm3df(boidVel.x, boidVel.y, boidVel.z);
        float speed = 1.0f / invSpeed;
        if (speed < Params::MIN_SPEED) {
            boidVel.x = boidVel.x * invSpeed * Params::MIN_SPEED;
            boidVel.y = boidVel.y * invSpeed * Params::MIN_SPEED;
            boidVel.z = boidVel.z * invSpeed * Params::MIN_SPEED;
        }
        else if (speed > Params::MAX_SPEED) {
            boidVel.x = boidVel.x * invSpeed * Params::MAX_SPEED;
            boidVel.y = boidVel.y * invSpeed * Params::MAX_SPEED;
            boidVel.z = boidVel.z * invSpeed * Params::MAX_SPEED;
        }

        // move boid
        newVel[idx] = boidVel;
    }
};

__global__ void assignGrid(float4* pos, int* gridIndices) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < Params::FLOCK_SIZE) {
        float4 p = pos[i];
        unsigned int x = cuda::std::floor((p.x - Params::LEFT_BOUND) / Params::VISION_DISTANCE);
        unsigned int y = cuda::std::floor((p.y - Params::BOTTOM_BOUND) / Params::VISION_DISTANCE);
        unsigned int z = cuda::std::floor((p.z - Params::NEAR_BOUND) / Params::VISION_DISTANCE);

        gridIndices[i] = x + y * Params::X_GRIDS + z * Params::X_GRIDS * Params::Y_GRIDS;
    }
};

__global__ void updateBoid(float4* pos, float4* vel, float4* newVel) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < Params::FLOCK_SIZE) {
        float4 d = newVel[i];
        vel[i] = d;
        float4 p = pos[i];
        p.x += d.x;
        p.y += d.y;
        p.z += d.z;

        //keep bounded, just teleport to other side
        p.x = p.x > Params::RIGHT_BOUND ? p.x - (Params::RIGHT_BOUND - Params::LEFT_BOUND) : p.x;
        p.x = p.x < Params::LEFT_BOUND ? p.x + (Params::RIGHT_BOUND - Params::LEFT_BOUND) : p.x;
        p.y = p.y > Params::TOP_BOUND ? p.y - (Params::TOP_BOUND - Params::BOTTOM_BOUND) : p.y;
        p.y = p.y < Params::BOTTOM_BOUND ? p.y + (Params::TOP_BOUND - Params::BOTTOM_BOUND) : p.y;
        p.z = p.z > Params::FAR_BOUND ? p.z - (Params::FAR_BOUND - Params::NEAR_BOUND) : p.z;
        p.z = p.z < Params::NEAR_BOUND ? p.z + (Params::FAR_BOUND - Params::NEAR_BOUND) : p.z;

        pos[i] = p;
    }
}

__global__ void organizeBoidsByGrid(float4* dstPos, float4* dstVel, float4* srcPos, float4* srcVel, int* boidIndices) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Params::FLOCK_SIZE) {
        int src = boidIndices[i];
        dstPos[i] = srcPos[src];
        dstVel[i] = srcVel[src];
    }
}

__global__ void boidsTransfer(float4* dstPos, float4* dstVel, float4* srcPos, float4* srcVel) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Params::FLOCK_SIZE) {
        dstPos[i] = srcPos[i];
        dstVel[i] = srcVel[i];
    }
}

void Flock::step(float4* cudaPos, float4* cudaVel) {
    //organize boids into grids
    assignGrid<<<(Params::FLOCK_SIZE + Params::BLOCK_SIZE - 1 )/Params::BLOCK_SIZE,Params::BLOCK_SIZE>>>(cudaPos, mpd_gridIndices);

    thrust::device_ptr<int> d_thrustBoidIndices(mpd_boidIndices);
    thrust::device_ptr<int> d_thrustGridIndices(mpd_gridIndices);
    thrust::sequence(d_thrustBoidIndices, d_thrustBoidIndices + Params::FLOCK_SIZE);
    thrust::sort_by_key(d_thrustGridIndices, d_thrustGridIndices+Params::FLOCK_SIZE,d_thrustBoidIndices);

    //find start/end point of each grid
    thrust::device_ptr<int> d_thrustGridStarts(mpd_gridStarts);
    thrust::lower_bound(d_thrustGridIndices, d_thrustGridIndices + Params::FLOCK_SIZE,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(Params::X_GRIDS*Params::Y_GRIDS*Params::Z_GRIDS),
                    d_thrustGridStarts);
    
    thrust::device_ptr<int> d_thrustGridEnds(mpd_gridEnds);
    thrust::upper_bound(d_thrustGridIndices, d_thrustGridIndices + Params::FLOCK_SIZE,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(Params::X_GRIDS*Params::Y_GRIDS*Params::Z_GRIDS),
                    d_thrustGridEnds);

    //reorganize boids;
    organizeBoidsByGrid<<<(Params::FLOCK_SIZE + Params::BLOCK_SIZE - 1 )/Params::BLOCK_SIZE,Params::BLOCK_SIZE>>>(mpd_posBuffer, mpd_velBuffer, cudaPos, cudaVel, mpd_boidIndices);
    boidsTransfer<<<(Params::FLOCK_SIZE + Params::BLOCK_SIZE - 1 )/Params::BLOCK_SIZE,Params::BLOCK_SIZE>>>(cudaPos, cudaVel, mpd_posBuffer, mpd_velBuffer);

    //run boid computations
    calcNewVeloc<<<(Params::FLOCK_SIZE + Params::BLOCK_SIZE - 1)/Params::BLOCK_SIZE,Params::BLOCK_SIZE>>>(cudaPos, cudaVel, mpd_newVels,mpd_gridIndices,mpd_gridStarts,mpd_gridEnds);
    updateBoid<<<(Params::FLOCK_SIZE + Params::BLOCK_SIZE - 1)/Params::BLOCK_SIZE,Params::BLOCK_SIZE>>>(cudaPos, cudaVel, mpd_newVels);
};

void Flock::genRand(float4* cudaPos, float4* cudaVel) {
    float4* initPos = (float4*)malloc(Params::FLOCK_SIZE*sizeof(float4));
    float4* initVel = (float4*)malloc(Params::FLOCK_SIZE*sizeof(float4));

    std::mt19937 rng(std::random_device{}());
    for(size_t i = 0; i < Params::FLOCK_SIZE; i++) {
        initPos[i] = randomPos(rng);
    }

    for(size_t i = 0; i < Params::FLOCK_SIZE; i++) {
        initVel[i] = randomVel(rng);
    }

    cudaMemcpy(cudaPos, initPos, Params::FLOCK_SIZE*sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaVel, initVel, Params::FLOCK_SIZE*sizeof(float4), cudaMemcpyHostToDevice);
    free(initPos);
    free(initVel);
};

Flock::Flock() {
    cudaMalloc(&mpd_newVels, Params::FLOCK_SIZE*sizeof(float4));
    cudaMalloc(&mpd_posBuffer, Params::FLOCK_SIZE*sizeof(float4));
    cudaMalloc(&mpd_velBuffer, Params::FLOCK_SIZE*sizeof(float4));
    cudaMalloc(&mpd_gridIndices, Params::FLOCK_SIZE*sizeof(int));

    cudaMalloc(&mpd_gridStarts, Params::AREA_GRIDS*sizeof(int));
    cudaMalloc(&mpd_gridEnds, Params::AREA_GRIDS*sizeof(int));

    cudaMalloc(&mpd_boidIndices, Params::FLOCK_SIZE*sizeof(int));
};

Flock::~Flock() {
    cudaFree(mpd_newVels);
    cudaFree(mpd_posBuffer);
    cudaFree(mpd_velBuffer);
    cudaFree(mpd_gridIndices);
    cudaFree(mpd_gridStarts);
    cudaFree(mpd_gridEnds);
    cudaFree(mpd_boidIndices);
};

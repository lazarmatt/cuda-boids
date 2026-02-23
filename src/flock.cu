#include "flock.cuh"
#include <cmath>
#include <cuda/std/cmath>
#include <cuda_fp16.h>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/binary_search.h>

std::array<float,3> Flock::randomVel(std::mt19937& rng) {
    
    std::normal_distribution<float> dist(0.0f, 1.0f);

    float vxUnscaled = dist(rng);
    float vyUnscaled = dist(rng);
    float vzUnscaled = dist(rng);

    float invLen = 1.0f / std::sqrt(vxUnscaled*vxUnscaled + vyUnscaled*vyUnscaled + vzUnscaled*vzUnscaled);
    float magnitude = (HostParams::MIN_SPEED + HostParams::MAX_SPEED) * .5f;
    float scale = magnitude * invLen;

    return {vxUnscaled*scale,vyUnscaled*scale,vzUnscaled*scale};
};

std::array<float,3> Flock::randomPos(std::mt19937& rng) {

    std::uniform_real_distribution<float> distx(HostParams::LEFT_BOUND,HostParams::RIGHT_BOUND);
    std::uniform_real_distribution<float> disty(HostParams::BOTTOM_BOUND,HostParams::TOP_BOUND);
    std::uniform_real_distribution<float> distz(HostParams::NEAR_BOUND,HostParams::FAR_BOUND);

    return {distx(rng),disty(rng),distz(rng)};;
}

__global__ void calcNewVeloc(__half2* srcXvx, __half2* srcYvy, __half2* srcZvz,  __half2* dstXvx, __half2* dstYvy, __half2* dstZvz,
    const int* __restrict__ grids, const int* __restrict__ gridStarts, int* __restrict__ gridEnds) {

    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < HostParams::FLOCK_SIZE) {
        //get data for later
        float2 boidXvx = __half22float2(__ldg(&srcXvx[idx]));
        float2 boidYvy = __half22float2(__ldg(&srcYvy[idx]));
        float2 boidZvz = __half22float2(__ldg(&srcZvz[idx]));
        int gridIdx = __ldg(&grids[idx]);
        
        // Get 3D grid space
        int gx = gridIdx % HostParams::X_GRIDS;
        int gy = (gridIdx / HostParams::X_GRIDS) % HostParams::Y_GRIDS;
        int gz = gridIdx / (HostParams::X_GRIDS * HostParams::Y_GRIDS);

        //accumulator
        float closeX = 0, closeY = 0, closeZ = 0;
        float posAvgX = 0, posAvgY = 0, posAvgZ = 0;
        float velAvgX = 0, velAvgY = 0, velAvgZ = 0;
        int neighboringBoids = 0;
        
        //surrounding 9 grids
        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
                for (int k = -1; k <= 1; k++)
                {   
                    int nx = (gx + i + HostParams::X_GRIDS) % HostParams::X_GRIDS;
                    int ny = (gy + j + HostParams::Y_GRIDS) % HostParams::Y_GRIDS;
                    int nz = (gz + k + HostParams::Z_GRIDS) % HostParams::Z_GRIDS;
                    int neighborGridIdx = nx + ny * HostParams::X_GRIDS + nz * HostParams::X_GRIDS * HostParams::Y_GRIDS;

                    int neighborStart = __ldg(&gridStarts[neighborGridIdx]);
                    int neighborEnd = __ldg(&gridEnds[neighborGridIdx]);
                    //empty cell
                    if(neighborStart == neighborEnd)
                        continue;
                    
                    
                    
                    for(int neighborIdx = neighborStart;neighborIdx < neighborEnd;neighborIdx++)
                    {
                        if(neighborIdx == idx)
                            continue;

                        float2 neighborXvx = __half22float2(__ldg(&srcXvx[neighborIdx]));
                        float2 neighborYvy = __half22float2(__ldg(&srcYvy[neighborIdx]));
                        float2 neighborZvz = __half22float2(__ldg(&srcZvz[neighborIdx]));

                       
                        float dx = boidXvx.x - neighborXvx.x;
                        float dy = boidYvy.x - neighborYvy.x;
                        float dz = boidZvz.x - neighborZvz.x;

                        // for each axis: if the gap is more than half the world, the short path is through the wrap
                        if (dx >  HostParams::WORLD_WIDTH * 0.5f) dx -= HostParams::WORLD_WIDTH;
                        if (dx < -HostParams::WORLD_WIDTH * 0.5f) dx += HostParams::WORLD_WIDTH;
                        if (dy >  HostParams::WORLD_HEIGHT * 0.5f) dy -= HostParams::WORLD_HEIGHT;
                        if (dy < -HostParams::WORLD_HEIGHT * 0.5f) dy += HostParams::WORLD_HEIGHT;
                        if (dz >  HostParams::WORLD_DEPTH * 0.5f) dz -= HostParams::WORLD_DEPTH;
                        if (dz < -HostParams::WORLD_DEPTH * 0.5f) dz += HostParams::WORLD_DEPTH;

                        float sqDist = dx*dx + dy*dy + dz*dz;

                        if (sqDist < HostParams::AVOID_DISTANCE*HostParams::AVOID_DISTANCE) {
                            //Avoiding
                            closeX += dx;
                            closeY += dy;
                            closeZ += dz;
                        } else if (sqDist < HostParams::VISION_DISTANCE*HostParams::VISION_DISTANCE) {
                            // Centering/Matching
                            posAvgX += boidXvx.x - dx;
                            posAvgY += boidYvy.x - dy;
                            posAvgZ += boidZvz.x - dz;
                            velAvgX += neighborXvx.y;
                            velAvgY += neighborYvy.y;
                            velAvgZ += neighborZvz.y;
                            
                            neighboringBoids += 1;
                        }
                    }
                }
        

        if (neighboringBoids > 0) {
            //add centering/matching
            float invNeighbors = 1.0f/neighboringBoids;
            posAvgX *= invNeighbors;
            posAvgY *= invNeighbors;
            posAvgZ *= invNeighbors;
            velAvgX *= invNeighbors;
            velAvgY *= invNeighbors;
            velAvgZ *= invNeighbors;

            boidXvx.y = boidXvx.y + 
                (posAvgX - boidXvx.x) * HostParams::CENTERING_FACTOR +
                (velAvgX - boidXvx.y) * HostParams::MATCHING_FACTOR;
            boidYvy.y = boidYvy.y + 
                (posAvgY - boidYvy.x) * HostParams::CENTERING_FACTOR +
                (velAvgY - boidYvy.y) * HostParams::MATCHING_FACTOR;
            boidZvz.y = boidZvz.y + 
                (posAvgZ - boidZvz.x) * HostParams::CENTERING_FACTOR +
                (velAvgZ - boidZvz.y) * HostParams::MATCHING_FACTOR;
        }

        // add avoiding
        boidXvx.y = boidXvx.y + closeX*HostParams::AVOID_FACTOR;
        boidYvy.y = boidYvy.y + closeY*HostParams::AVOID_FACTOR;
        boidZvz.y = boidZvz.y + closeZ*HostParams::AVOID_FACTOR;

        float invSpeed = rsqrt(boidXvx.y*boidXvx.y + boidYvy.y*boidYvy.y + boidZvz.y*boidZvz.y);
        float speed = 1.0f / invSpeed;
        if (speed < HostParams::MIN_SPEED) {
            boidXvx.y = boidXvx.y * invSpeed * HostParams::MIN_SPEED;
            boidYvy.y = boidYvy.y * invSpeed * HostParams::MIN_SPEED;
            boidZvz.y = boidZvz.y * invSpeed * HostParams::MIN_SPEED;
        }
        else if (speed > HostParams::MAX_SPEED) {
            boidXvx.y = boidXvx.y * invSpeed * HostParams::MAX_SPEED;
            boidYvy.y = boidYvy.y * invSpeed * HostParams::MAX_SPEED;
            boidZvz.y = boidZvz.y * invSpeed * HostParams::MAX_SPEED;
        }

        // move boid
        boidXvx.x = boidXvx.x + boidXvx.y;
        boidYvy.x = boidYvy.x + boidYvy.y;
        boidZvz.x = boidZvz.x + boidZvz.y;

        //keep bounded, just teleport to other side
        boidXvx.x = boidXvx.x > HostParams::RIGHT_BOUND ? boidXvx.x - (HostParams::RIGHT_BOUND - HostParams::LEFT_BOUND) : boidXvx.x;
        boidXvx.x = boidXvx.x < HostParams::LEFT_BOUND ? boidXvx.x + (HostParams::RIGHT_BOUND - HostParams::LEFT_BOUND) : boidXvx.x;
        boidYvy.x = boidYvy.x > HostParams::TOP_BOUND ? boidYvy.x - (HostParams::TOP_BOUND - HostParams::BOTTOM_BOUND) : boidYvy.x;
        boidYvy.x = boidYvy.x < HostParams::BOTTOM_BOUND ? boidYvy.x + (HostParams::TOP_BOUND - HostParams::BOTTOM_BOUND) : boidYvy.x;
        boidZvz.x = boidZvz.x > HostParams::FAR_BOUND ? boidZvz.x - (HostParams::FAR_BOUND - HostParams::NEAR_BOUND) : boidZvz.x;
        boidZvz.x = boidZvz.x < HostParams::NEAR_BOUND ? boidZvz.x + (HostParams::FAR_BOUND - HostParams::NEAR_BOUND) : boidZvz.x;

        dstXvx[idx] = __float22half2_rn(boidXvx);
        dstYvy[idx] = __float22half2_rn(boidYvy);
        dstZvz[idx] = __float22half2_rn(boidZvz);
    }
};

__global__ void assignGrid(const __half2* __restrict__ xvx, const __half2* __restrict__ yvy, const __half2* __restrict__ zvz, int* gridIndices) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < HostParams::FLOCK_SIZE) {
        float px = __half2float(__ldg(&xvx[i]).x);
        float py = __half2float(__ldg(&yvy[i]).x);
        float pz = __half2float(__ldg(&zvz[i]).x);
       
        int x = int((px - HostParams::LEFT_BOUND) / HostParams::VISION_DISTANCE);
        int y = int((py - HostParams::BOTTOM_BOUND) / HostParams::VISION_DISTANCE);
        int z = int((pz - HostParams::NEAR_BOUND) / HostParams::VISION_DISTANCE);

        gridIndices[i] = x + y * HostParams::X_GRIDS + z * HostParams::X_GRIDS * HostParams::Y_GRIDS;
    }
};

__global__ void organizeBoidsByGrid(__half2* dstXvx, __half2* dstYvy, __half2* dstZvz,  
    __half2* srcXvx, __half2* srcYvy, __half2* srcZvz, int* boidIndices) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HostParams::FLOCK_SIZE) {
        int src = boidIndices[i];
        dstXvx[i] = srcXvx[src];
        dstYvy[i] = srcYvy[src];
        dstZvz[i] = srcZvz[src];
    }
}

void Flock::step(__half2* cudaXvx, __half2* cudaYvy, __half2* cudaZvz) {
    //organize boids into grids
    assignGrid<<<(HostParams::FLOCK_SIZE + HostParams::BLOCK_SIZE - 1 )/HostParams::BLOCK_SIZE,HostParams::BLOCK_SIZE>>>(cudaXvx, cudaYvy, cudaZvz, mpd_gridIndices);

    thrust::device_ptr<int> d_thrustBoidIndices(mpd_boidIndices);
    thrust::device_ptr<int> d_thrustGridIndices(mpd_gridIndices);
    thrust::sequence(d_thrustBoidIndices, d_thrustBoidIndices + HostParams::FLOCK_SIZE);
    thrust::sort_by_key(d_thrustGridIndices, d_thrustGridIndices+HostParams::FLOCK_SIZE,d_thrustBoidIndices);

    //find start/end point of each grid
    thrust::device_ptr<int> d_thrustGridStarts(mpd_gridStarts);
    thrust::lower_bound(d_thrustGridIndices, d_thrustGridIndices + HostParams::FLOCK_SIZE,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(HostParams::X_GRIDS*HostParams::Y_GRIDS*HostParams::Z_GRIDS),
                    d_thrustGridStarts);
    
    thrust::device_ptr<int> d_thrustGridEnds(mpd_gridEnds);
    thrust::upper_bound(d_thrustGridIndices, d_thrustGridIndices + HostParams::FLOCK_SIZE,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(HostParams::X_GRIDS*HostParams::Y_GRIDS*HostParams::Z_GRIDS),
                    d_thrustGridEnds);

    //reorganize boids;
    organizeBoidsByGrid<<<(HostParams::FLOCK_SIZE + HostParams::BLOCK_SIZE - 1 )/HostParams::BLOCK_SIZE,HostParams::BLOCK_SIZE>>>(mpd_xBuffer, mpd_yBuffer, mpd_zBuffer, cudaXvx, cudaYvy, cudaZvz, mpd_boidIndices);

    //run boid computations
    calcNewVeloc<<<(HostParams::FLOCK_SIZE + HostParams::BLOCK_SIZE - 1)/HostParams::BLOCK_SIZE,HostParams::BLOCK_SIZE>>>(mpd_xBuffer, mpd_yBuffer, mpd_zBuffer, cudaXvx, cudaYvy, cudaZvz, mpd_gridIndices,mpd_gridStarts,mpd_gridEnds);
};

void Flock::genRand(__half2* cudaXvx, __half2* cudaYvy, __half2* cudaZvz) {
    __half2* initXvx = (__half2*)malloc(HostParams::FLOCK_SIZE*sizeof(__half2));
    __half2* initYvy = (__half2*)malloc(HostParams::FLOCK_SIZE*sizeof(__half2));
    __half2* initZvz = (__half2*)malloc(HostParams::FLOCK_SIZE*sizeof(__half2));

    std::mt19937 rng(std::random_device{}());
    for(size_t i = 0; i < HostParams::FLOCK_SIZE; i++) {
        std::array<float,3> randPos = randomPos(rng);
        std::array<float,3> randVel = randomVel(rng);
        initXvx[i].x = half(randPos[0]);
        initYvy[i].x = half(randPos[1]);
        initZvz[i].x = half(randPos[2]);
        initXvx[i].y = half(randVel[0]);
        initYvy[i].y = half(randVel[1]);
        initZvz[i].y = half(randVel[2]);
    }


    cudaMemcpy(cudaXvx, initXvx, HostParams::FLOCK_SIZE*sizeof(__half2), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaYvy, initYvy, HostParams::FLOCK_SIZE*sizeof(__half2), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaZvz, initZvz, HostParams::FLOCK_SIZE*sizeof(__half2), cudaMemcpyHostToDevice);
    free(initXvx);
    free(initYvy);
    free(initZvz);
};


Flock::Flock() {
    cudaMalloc(&mpd_gridIndices, HostParams::FLOCK_SIZE*sizeof(int));

    cudaMalloc(&mpd_gridStarts, HostParams::AREA_GRIDS*sizeof(int));
    cudaMalloc(&mpd_gridEnds, HostParams::AREA_GRIDS*sizeof(int));

    cudaMalloc(&mpd_boidIndices, HostParams::FLOCK_SIZE*sizeof(int));
    cudaMalloc(&mpd_xBuffer,HostParams::FLOCK_SIZE*sizeof(__half2));
    cudaMalloc(&mpd_yBuffer,HostParams::FLOCK_SIZE*sizeof(__half2));
    cudaMalloc(&mpd_zBuffer,HostParams::FLOCK_SIZE*sizeof(__half2));
};

Flock::~Flock() {
    cudaFree(mpd_gridIndices);
    cudaFree(mpd_gridStarts);
    cudaFree(mpd_gridEnds);
    cudaFree(mpd_boidIndices);
    cudaFree(mpd_xBuffer);
    cudaFree(mpd_yBuffer);
    cudaFree(mpd_zBuffer);
};

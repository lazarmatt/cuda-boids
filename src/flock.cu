#include "flock.cuh"
#include <cmath>
#include <cuda/std/cmath>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/binary_search.h>

Boid Flock::randomBoid(std::mt19937& rng) {
    
    std::normal_distribution<float> dist(0.0f, 1.0f);

    float vxUnscaled = dist(rng);
    float vyUnscaled = dist(rng);
    float vzUnscaled = dist(rng);

    float len = std::sqrt(vxUnscaled*vxUnscaled + vyUnscaled*vyUnscaled + vzUnscaled*vzUnscaled);
    float magnitude = (Params::MIN_SPEED + Params::MAX_SPEED) / 2;
    float scale = magnitude / len;

    float3 veloc(vxUnscaled*scale,vyUnscaled*scale,vzUnscaled*scale);

    std::uniform_real_distribution<float> distx(Params::LEFT_BOUND,Params::RIGHT_BOUND);
    std::uniform_real_distribution<float> disty(Params::BOTTOM_BOUND,Params::TOP_BOUND);
    std::uniform_real_distribution<float> distz(Params::NEAR_BOUND,Params::FAR_BOUND);

    float3 posit(distx(rng),disty(rng),distz(rng));

    return Boid(posit, veloc);
};

__global__ void updateBoid(Boid* boids, int* grids, int* gridStarts, int* boidIndices) {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < Params::FLOCK_SIZE) {
        Accumulator accum;

        int boidIdx = boidIndices[idx];
        Boid b = boids[boidIdx];
        
        // Get 3D grid space
        int gx = grids[idx] % Params::X_GRIDS;
        int gy = (grids[idx] / Params::X_GRIDS) % Params::Y_GRIDS;
        int gz = grids[idx] / (Params::X_GRIDS * Params::Y_GRIDS);
        
        //surrounding 9 grids
        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
                for (int k = -1; k <= 1; k++)
                {   
                    int nx = (gx + i + Params::X_GRIDS) % Params::X_GRIDS;
                    int ny = (gy + j + Params::Y_GRIDS) % Params::Y_GRIDS;
                    int nz = (gz + k + Params::Z_GRIDS) % Params::Z_GRIDS;
                    int neighborGridIdx = nx + ny * Params::X_GRIDS + nz * Params::X_GRIDS * Params::Y_GRIDS;

                    //empty cell
                    if(gridStarts[neighborGridIdx] >= Params::FLOCK_SIZE)
                        continue;
                    
                    for(int neighborIdx = gridStarts[neighborGridIdx];neighborIdx < Params::FLOCK_SIZE;neighborIdx++)
                    {
                        //if we're out of the grid
                        if(grids[neighborIdx] != neighborGridIdx)
                            break;

                        Boid neighorBoid = boids[boidIndices[neighborIdx]];

                        //get data for later
                        float3 neighborPos = neighorBoid.getPosit();
                        float3 myPos = b.getPosit();

                        float3 d = DeviceHelpers::sub(myPos, neighborPos);

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
                            float3 wrappedNeighborPos = { myPos.x - d.x, myPos.y - d.y, myPos.z - d.z };
                            accum.pos_avg = DeviceHelpers::add(accum.pos_avg, wrappedNeighborPos);
                            accum.vel_avg = DeviceHelpers::add(accum.vel_avg, neighorBoid.getVeloc());
                            accum.neighboring_boids += 1;
                        }
                    }
                }
        
        float3 newVeloc = b.getVeloc();

        if (accum.neighboring_boids > 0) {
            //add centering/matching
            accum.pos_avg = DeviceHelpers::scale(accum.pos_avg,1.0f/((float) accum.neighboring_boids));
            accum.vel_avg = DeviceHelpers::scale(accum.vel_avg,1.0f/((float) accum.neighboring_boids));

            newVeloc.x = newVeloc.x + 
                (accum.pos_avg.x - b.getPosit().x) * Params::CENTERING_FACTOR +
                (accum.vel_avg.x - b.getVeloc().x) * Params::MATCHING_FACTOR;
            newVeloc.y = newVeloc.y + 
                (accum.pos_avg.y - b.getPosit().y) * Params::CENTERING_FACTOR +
                (accum.vel_avg.y - b.getVeloc().y) * Params::MATCHING_FACTOR;
            newVeloc.z = newVeloc.z + 
                (accum.pos_avg.z - b.getPosit().z) * Params::CENTERING_FACTOR +
                (accum.vel_avg.z - b.getVeloc().z) * Params::MATCHING_FACTOR;
        }

        // add avoiding
        newVeloc.x = newVeloc.x + accum.close.x*Params::AVOID_FACTOR;
        newVeloc.y = newVeloc.y + accum.close.y*Params::AVOID_FACTOR;
        newVeloc.z = newVeloc.z + accum.close.z*Params::AVOID_FACTOR;

        float invSpeed = rnorm3df(newVeloc.x, newVeloc.y, newVeloc.z);
        float speed = 1.0f / invSpeed;
        if (speed < Params::MIN_SPEED) {
            newVeloc.x = newVeloc.x * invSpeed * Params::MIN_SPEED;
            newVeloc.y = newVeloc.y * invSpeed * Params::MIN_SPEED;
            newVeloc.z = newVeloc.z * invSpeed * Params::MIN_SPEED;
        }
        else if (speed > Params::MAX_SPEED) {
            newVeloc.x = newVeloc.x * invSpeed * Params::MAX_SPEED;
            newVeloc.y = newVeloc.y * invSpeed * Params::MAX_SPEED;
            newVeloc.z = newVeloc.z * invSpeed * Params::MAX_SPEED;
        }

        // move boid
        boids[boidIdx].setNewVeloc(newVeloc);
    }
};

__global__ void genTransform(Boid* boids, float4* transforms) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < Params::FLOCK_SIZE) {
        cuda::std::array<float4,4> toWrite = DeviceHelpers::transform(boids[i].getPosit(),boids[i].getVeloc());
        memcpy(&transforms[i*4],&toWrite,sizeof(toWrite));
    }
};

__global__ void assignGrid(Boid* boids, int* gridIndices) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < Params::FLOCK_SIZE) {
        unsigned int x = cuda::std::floor((boids[i].getPosit().x - Params::LEFT_BOUND) / Params::VISION_DISTANCE);
        unsigned int y = cuda::std::floor((boids[i].getPosit().y - Params::BOTTOM_BOUND) / Params::VISION_DISTANCE);
        unsigned int z = cuda::std::floor((boids[i].getPosit().z - Params::NEAR_BOUND) / Params::VISION_DISTANCE);

        gridIndices[i] = x + y * Params::X_GRIDS + z * Params::X_GRIDS * Params::Y_GRIDS;
    }
};

__global__ void organizeBoidsByGrid(Boid* dst, Boid* src, int* boidIndices) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < Params::FLOCK_SIZE) {
        dst[i] = src[boidIndices[i]];
    }
}

__global__ void updatePosit(Boid* boids) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < Params::FLOCK_SIZE) {
        boids[i].step();
    }
}

void Flock::step(float4* transforms) {
    //organize boids into grids
    
    assignGrid<<<(Params::FLOCK_SIZE + Params::BLOCK_SIZE - 1 )/Params::BLOCK_SIZE,Params::BLOCK_SIZE>>>(mpd_boids, mpd_gridIndices);

    thrust::device_ptr<int> d_thrustBoidIndices(mpd_boidIndices);
    thrust::device_ptr<int> d_thrustGridIndices(mpd_gridIndices);
    thrust::sequence(d_thrustBoidIndices, d_thrustBoidIndices + Params::FLOCK_SIZE);
    thrust::sort_by_key(d_thrustGridIndices, d_thrustGridIndices+Params::FLOCK_SIZE,d_thrustBoidIndices);

    //find starting point of each grid
    thrust::device_ptr<int> d_thrustGridStarts(mpd_gridStarts);
    thrust::lower_bound(d_thrustGridIndices, d_thrustGridIndices + Params::FLOCK_SIZE,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(Params::X_GRIDS*Params::Y_GRIDS*Params::Z_GRIDS),
                    d_thrustGridStarts);

    //run boid computations
    updateBoid<<<(Params::FLOCK_SIZE + Params::BLOCK_SIZE - 1)/Params::BLOCK_SIZE,Params::BLOCK_SIZE>>>(mpd_boids,mpd_gridIndices,mpd_gridStarts, mpd_boidIndices);
    updatePosit<<<(Params::FLOCK_SIZE + Params::BLOCK_SIZE - 1)/Params::BLOCK_SIZE,Params::BLOCK_SIZE>>>(mpd_boids);
    genTransform<<<(Params::FLOCK_SIZE + Params::BLOCK_SIZE - 1)/Params::BLOCK_SIZE,Params::BLOCK_SIZE>>>(mpd_boids,transforms);

};

Flock::Flock() {


    Boid* boids = (Boid*)malloc(Params::FLOCK_SIZE*sizeof(Boid));

    std::mt19937 rng(std::random_device{}());
    for(size_t i = 0; i < Params::FLOCK_SIZE; i++) {
        boids[i] = randomBoid(rng);
    }

    cudaMalloc(&mpd_boids, Params::FLOCK_SIZE*sizeof(Boid));
    cudaMemcpy(mpd_boids, boids, Params::FLOCK_SIZE*sizeof(Boid), cudaMemcpyHostToDevice);
    free(boids);

    cudaMalloc(&mpd_gridIndices, Params::FLOCK_SIZE*sizeof(int));

    cudaMalloc(&mpd_gridStarts, Params::X_GRIDS*Params::Y_GRIDS*Params::Z_GRIDS*sizeof(int));

    cudaMalloc(&mpd_boidIndices, Params::FLOCK_SIZE*sizeof(int));
};

Flock::~Flock() {
    cudaFree(mpd_boids);
    cudaFree(mpd_gridIndices);
    cudaFree(mpd_gridStarts);
    cudaFree(mpd_boidIndices);
};

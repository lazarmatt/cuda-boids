#include "flock.cuh"
#include <random>
#include <cmath>
#include <cuda/std/cmath>
#include <chrono>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/binary_search.h>

Boid Flock::randomBoid() {
    static std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    float vxUnscaled = dist(rng);
    float vyUnscaled = dist(rng);
    float vzUnscaled = dist(rng);

    float len = std::sqrt(vxUnscaled*vxUnscaled + vyUnscaled*vyUnscaled + vzUnscaled*vzUnscaled);
    float magnitude = (Hyperparams::MIN_SPEED + Hyperparams::MAX_SPEED) / 2;
    float scale = magnitude / len;

    float3 veloc(vxUnscaled*scale,vyUnscaled*scale,vzUnscaled*scale);

    std::uniform_real_distribution<float> distx(Hyperparams::LEFT_BOUND,Hyperparams::RIGHT_BOUND);
    std::uniform_real_distribution<float> disty(Hyperparams::BOTTOM_BOUND,Hyperparams::TOP_BOUND);
    std::uniform_real_distribution<float> distz(Hyperparams::NEAR_BOUND,Hyperparams::FAR_BOUND);

    float3 posit(distx(rng),disty(rng),distz(rng));

    return Boid(posit, veloc);
};

__global__ void updateVeloc(Boid* boids, int* grids, int* gridStarts, int* boidIndices) {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < Hyperparams::FLOCK_SIZE) {
        Accumulator accum;
        
        //surrounding 9 grids
        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
                for (int k = -1; k <= 1; k++)
                {
                    unsigned int gx = cuda::std::floor((boids[idx].getPosit().x - Hyperparams::LEFT_BOUND) / Hyperparams::VISION_DISTANCE);
                    unsigned int gy = cuda::std::floor((boids[idx].getPosit().y - Hyperparams::BOTTOM_BOUND) / Hyperparams::VISION_DISTANCE);
                    unsigned int gz = cuda::std::floor((boids[idx].getPosit().z - Hyperparams::NEAR_BOUND) / Hyperparams::VISION_DISTANCE);
                    int myGrid = gx + gy * Hyperparams::X_GRIDS + gz * Hyperparams::X_GRIDS * Hyperparams::Y_GRIDS;

                    int neighborGridIdx = myGrid + i + j * Hyperparams::X_GRIDS + k * Hyperparams::X_GRIDS * Hyperparams::Y_GRIDS;
                    if(neighborGridIdx < 0 || neighborGridIdx > Hyperparams::X_GRIDS * Hyperparams::Y_GRIDS * Hyperparams::Z_GRIDS)
                        continue;
                    
                    //empty cell
                    if(gridStarts[neighborGridIdx] >= Hyperparams::FLOCK_SIZE)
                        continue;
                        
                    for(int neighborIdx = gridStarts[neighborGridIdx];neighborIdx < Hyperparams::FLOCK_SIZE;neighborIdx++)
                    {
                        //if we're out of the grid
                        if(grids[neighborIdx] != neighborGridIdx)
                            break;

                        //get data for later
                        float3 d = DeviceHelpers::sub(boids[idx].getPosit(), boids[boidIndices[neighborIdx]].getPosit());
                        float sqDist = d.x*d.x + d.y*d.y + d.z*d.z;

                        if (sqDist < Hyperparams::AVOID_DISTANCE*Hyperparams::AVOID_DISTANCE) {
                            //Avoiding
                            accum.close = DeviceHelpers::add(accum.close, DeviceHelpers::sub(boids[idx].getPosit(),boids[boidIndices[neighborIdx]].getPosit()));
                        } else if (sqDist < Hyperparams::VISION_DISTANCE*Hyperparams::VISION_DISTANCE) {
                            // Centering/Matching
                            accum.pos_avg = DeviceHelpers::add(accum.pos_avg, boids[boidIndices[neighborIdx]].getPosit());
                            accum.vel_avg = DeviceHelpers::add(accum.vel_avg, boids[boidIndices[neighborIdx]].getVeloc());
                            accum.neighboring_boids += 1;
                        }
                    }
                }
        
        float3 newVeloc = boids[idx].getVeloc();

        if (accum.neighboring_boids > 0) {
            //add centering/matching
            accum.pos_avg = DeviceHelpers::scale(accum.pos_avg,1.0f/((float) accum.neighboring_boids));
            accum.vel_avg = DeviceHelpers::scale(accum.vel_avg,1.0f/((float) accum.neighboring_boids));

            newVeloc.x = newVeloc.x + 
                (accum.pos_avg.x - boids[idx].getPosit().x) * Hyperparams::CENTERING_FACTOR +
                (accum.vel_avg.x - boids[idx].getVeloc().x) * Hyperparams::MATCHING_FACTOR;
            newVeloc.y = newVeloc.y + 
                (accum.pos_avg.y - boids[idx].getPosit().y) * Hyperparams::CENTERING_FACTOR +
                (accum.vel_avg.y - boids[idx].getVeloc().y) * Hyperparams::MATCHING_FACTOR;
            newVeloc.z = newVeloc.z + 
                (accum.pos_avg.z - boids[idx].getPosit().z) * Hyperparams::CENTERING_FACTOR +
                (accum.vel_avg.z - boids[idx].getVeloc().z) * Hyperparams::MATCHING_FACTOR;
        }

        // add avoiding
        newVeloc.x = newVeloc.x + accum.close.x*Hyperparams::AVOID_FACTOR;
        newVeloc.y = newVeloc.y + accum.close.y*Hyperparams::AVOID_FACTOR;
        newVeloc.z = newVeloc.z + accum.close.z*Hyperparams::AVOID_FACTOR;

        float invSpeed = rnorm3df(newVeloc.x, newVeloc.y, newVeloc.z);
        if ((1.0f / invSpeed) < Hyperparams::MIN_SPEED) {
            newVeloc.x = newVeloc.x * invSpeed * Hyperparams::MIN_SPEED;
            newVeloc.y = newVeloc.y * invSpeed * Hyperparams::MIN_SPEED;
            newVeloc.z = newVeloc.z * invSpeed * Hyperparams::MIN_SPEED;
        }
        if ((1.0f / invSpeed) > Hyperparams::MAX_SPEED) {
            newVeloc.x = newVeloc.x * invSpeed * Hyperparams::MAX_SPEED;
            newVeloc.y = newVeloc.y * invSpeed * Hyperparams::MAX_SPEED;
            newVeloc.z = newVeloc.z * invSpeed * Hyperparams::MAX_SPEED;
        }

        boids[idx].setNewVeloc(newVeloc);
    }
};

__global__ void updatePosit(Boid* boids) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < Hyperparams::FLOCK_SIZE) {
        boids[i].step();
        
    }
};

__global__ void genTransform(Boid* boids, float4* transforms) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < Hyperparams::FLOCK_SIZE) {
        cuda::std::array<float4,4> toWrite = DeviceHelpers::transform(boids[i].getPosit(),boids[i].getVeloc());
        memcpy(&transforms[i*4],&toWrite,sizeof(toWrite));
    }
};

__global__ void assignGrid(Boid* boids, int* gridIndices) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < Hyperparams::FLOCK_SIZE) {
        unsigned int x = cuda::std::floor((boids[i].getPosit().x - Hyperparams::LEFT_BOUND) / Hyperparams::VISION_DISTANCE);
        unsigned int y = cuda::std::floor((boids[i].getPosit().y - Hyperparams::BOTTOM_BOUND) / Hyperparams::VISION_DISTANCE);
        unsigned int z = cuda::std::floor((boids[i].getPosit().z - Hyperparams::NEAR_BOUND) / Hyperparams::VISION_DISTANCE);

        gridIndices[i] = x + y * Hyperparams::X_GRIDS + z * Hyperparams::X_GRIDS * Hyperparams::Z_GRIDS;
    }
};

void Flock::step(float4* transforms) {
    
    //auto start = std::chrono::high_resolution_clock::now()//;

    //organize boids into grids
    assignGrid<<<(Hyperparams::FLOCK_SIZE + 255)/256,256>>>(mpd_boids, mpd_gridIndices);
    
    //cudaDeviceSynchronize();
    //auto flag1 = std::chrono::high_resolution_clock::now();

    thrust::device_ptr<int> d_thrustBoidIndices(mpd_boidIndices);
    thrust::device_ptr<int> d_thrustGridIndices(mpd_gridIndices);
    thrust::sequence(d_thrustBoidIndices, d_thrustBoidIndices + Hyperparams::FLOCK_SIZE);
    thrust::sort_by_key(d_thrustGridIndices, d_thrustGridIndices+Hyperparams::FLOCK_SIZE,d_thrustBoidIndices);

    //cudaDeviceSynchronize();
    //auto flag2 = std::chrono::high_resolution_clock::now();

    thrust::device_ptr<int> d_thrustGridStarts(mpd_gridStarts);
    thrust::lower_bound(d_thrustGridIndices, d_thrustGridIndices + Hyperparams::FLOCK_SIZE,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(Hyperparams::X_GRIDS*Hyperparams::Y_GRIDS*Hyperparams::Z_GRIDS),
                    d_thrustGridStarts);

    //cudaDeviceSynchronize();
    //auto flag3 = std::chrono::high_resolution_clock::now();


    //run boid computations
    updateVeloc<<<(Hyperparams::FLOCK_SIZE + 255)/256,256>>>(mpd_boids,mpd_gridIndices,mpd_gridStarts,mpd_boidIndices);
    updatePosit<<<(Hyperparams::FLOCK_SIZE + 255)/256,256>>>(mpd_boids);
    genTransform<<<(Hyperparams::FLOCK_SIZE + 255)/256,256>>>(mpd_boids,transforms);

    //cudaDeviceSynchronize();
    //auto flag4 = std::chrono::high_resolution_clock::now();

    //std::chrono::duration<double,std::milli> gridDur = flag1 - start;
    //std::chrono::duration<double,std::milli> thrustDur = flag2 - flag1;
    //std::chrono::duration<double,std::milli> startDur = flag3 - flag2;
    //std::chrono::duration<double,std::milli> updateDur = flag4 - flag3;

    //printf("grid: %f, thrust: %f, start: %f, update: %f milliseconds\n", gridDur.count(), thrustDur.count(), startDur.count(), updateDur.count());
};

Flock::Flock() {
    Boid* boids = (Boid*)malloc(Hyperparams::FLOCK_SIZE*sizeof(Boid));
    for(size_t i = 0; i < Hyperparams::FLOCK_SIZE; i++) {
        boids[i] = randomBoid();
    }

    cudaMalloc(&mpd_boids, Hyperparams::FLOCK_SIZE*sizeof(Boid));
    cudaMemcpy(mpd_boids, boids, Hyperparams::FLOCK_SIZE*sizeof(Boid), cudaMemcpyHostToDevice);
    free(boids);

    cudaMalloc(&mpd_gridIndices, Hyperparams::FLOCK_SIZE*sizeof(int));

    cudaMalloc(&mpd_gridStarts, Hyperparams::X_GRIDS*Hyperparams::Y_GRIDS*Hyperparams::Z_GRIDS*sizeof(int));

    cudaMalloc(&mpd_boidIndices, Hyperparams::FLOCK_SIZE*sizeof(int));
};

Flock::~Flock() {
    cudaFree(mpd_boids);
    cudaFree(mpd_gridIndices);
    cudaFree(mpd_gridStarts);
    cudaFree(mpd_boidIndices);
};

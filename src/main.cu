#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <iostream>

#include "flock.cuh"

const char* vertSrc = "#version 330 core\n"
    "layout(location = 0) in vec3 aPos;\n"
    "layout(location = 1) in vec3 pos;\n"
    "layout(location = 2) in vec3 vel;\n"
    "uniform mat4 view;\n"
    "uniform mat4 proj;\n"
    "void main() { \n"
    "   vec3 normRotate = normalize(vel);\n"
    "   vec3 yAxis = vec3(0.0,1.0,0.0);\n"
    "   vec3 right;\n"
    "   if (abs(dot(vel, yAxis)) > 0.9999) {\n"
    "       right = normalize(cross(vel, vec3(1.0,0.0,0.0)));\n"
    "   } else {\n"
    "       right = normalize(cross(yAxis,vel));\n"
    "   }\n"
    "   vec3 up = normRotate;\n"
    "   vec3 fwd = normalize(cross(right,up));"
    "   mat4 instance = mat4(\n"
    "       vec4(right, 0.0),\n"
    "       vec4(up, 0.0),\n"
    "       vec4(fwd, 0.0),\n"
    "       vec4(pos, 1.0)\n"
    "   );\n"
    "   gl_Position = proj * view * instance * vec4(aPos*50, 1.0);\n"
    "}\n";

const char* fragSrc = "#version 330 core\n"
    "out vec4 FragColor;\n"
    "void main() { FragColor = vec4(1.0, 0.5, 0.2, 1.0); }\n";

// -- settings --
const int    WINDOW_WIDTH  = 1280;
const int    WINDOW_HEIGHT = 720;

void onKeyPress(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

int main() {

    // Set up window
    if (!glfwInit()) {
        std::cerr << "Failed to initialise GLFW\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

    GLFWwindow* window = glfwCreateWindow(
        WINDOW_WIDTH, WINDOW_HEIGHT, "Boids", nullptr, nullptr
    );

    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    glfwSetKeyCallback(window, onKeyPress);

    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    
    // Set up camera
    struct CamState {
        float yaw = 0.f, pitch = 0.0f, dist = 500.f;
        double lastX = 0, lastY = 0;
        bool dragging = false;
    };
    CamState cam;
    glfwSetWindowUserPointer(window, &cam);

    glfwSetMouseButtonCallback(window, [](GLFWwindow* w, int btn, int action, int) {
        auto& c = *(CamState*)glfwGetWindowUserPointer(w);
        if (btn == GLFW_MOUSE_BUTTON_LEFT) {
            c.dragging = (action == GLFW_PRESS);
            glfwGetCursorPos(w, &c.lastX, &c.lastY);
        }
    });

    glfwSetCursorPosCallback(window, [](GLFWwindow* w, double x, double y) {
        auto& c = *(CamState*)glfwGetWindowUserPointer(w);
        if (!c.dragging) return;
        c.yaw   += (float)(x - c.lastX) * 0.005f;
        c.pitch += (float)(y - c.lastY) * 0.005f;
        c.pitch  = std::clamp(c.pitch, -1.5f, 1.5f); // don't flip over the poles
        c.lastX = x; c.lastY = y;
    });

    glfwSetScrollCallback(window, [](GLFWwindow* w, double, double dy) {
        auto& c = *(CamState*)glfwGetWindowUserPointer(w);
        c.dist *= (float)std::pow(0.9, dy); // scroll up = zoom in
        c.dist  = std::clamp(c.dist, 0.5f, 700.f);
    });

    // Compile shaders
    unsigned int vert = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vert, 1, &vertSrc, NULL);
    glCompileShader(vert);

    unsigned int frag = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(frag, 1, &fragSrc, NULL);
    glCompileShader(frag);

    unsigned int program = glCreateProgram();
    glAttachShader(program, vert);
    glAttachShader(program, frag);
    glLinkProgram(program);
    glDeleteShader(vert);
    glDeleteShader(frag);

    int uView = glGetUniformLocation(program, "view");
    int uProj = glGetUniformLocation(program, "proj");

    // boid structure
    float boidVerts[] = {
        0.0f, 0.005f, 0.0f,
        -0.002598f, 0.0f, -0.0015f,
        0.0f, 0.0f, 0.003f,

        0.0f, 0.005f, 0.0f,
        0.002598f, 0.0f, -0.0015f,
        -0.002598f, 0.0f, -0.0015f,

        0.0f, 0.005f, 0.0f,
        0.0f, 0.0f, 0.003f,
        0.002598f, 0.0f, -0.0015f,

        0.0f, 0.0f, 0.003f,
        0.002598f, 0.0f, -0.0015f,
        -0.002598f, 0.0f, -0.0015f,
    };

    // Allocate vertex data
    unsigned int boidVAO, boidVBO;
    glGenVertexArrays(1, &boidVAO);
    glGenBuffers(1, &boidVBO);

    glBindVertexArray(boidVAO);
    glBindBuffer(GL_ARRAY_BUFFER, boidVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*3*12, boidVerts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);

    // Allocate instance data
    size_t vecSize = sizeof(float4);
    
    unsigned int posVBO[2];
    glGenBuffers(2, posVBO);
    glBindBuffer(GL_ARRAY_BUFFER, posVBO[0]);
    glBufferData(GL_ARRAY_BUFFER, vecSize*Params::FLOCK_SIZE,nullptr,GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, posVBO[1]);
    glBufferData(GL_ARRAY_BUFFER, vecSize*Params::FLOCK_SIZE,nullptr,GL_DYNAMIC_DRAW);
    
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vecSize , nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribDivisor(1,1);

    unsigned int velVBO[2];
    glGenBuffers(2, velVBO);
    glBindBuffer(GL_ARRAY_BUFFER, velVBO[0]);
    glBufferData(GL_ARRAY_BUFFER, vecSize*Params::FLOCK_SIZE,nullptr,GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, velVBO[1]);
    glBufferData(GL_ARRAY_BUFFER, vecSize*Params::FLOCK_SIZE,nullptr,GL_DYNAMIC_DRAW);
    
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, vecSize , nullptr);
    glEnableVertexAttribArray(2);
    glVertexAttribDivisor(2,1);

    //unbind for now
    glBindVertexArray(0);
    
    // create boid data
    Flock flock;
    
    // register with cuda
    cudaGraphicsResource* cudaPosVBO[2];
    cudaGraphicsGLRegisterBuffer(&cudaPosVBO[0], posVBO[0], cudaGraphicsMapFlagsNone);
    cudaGraphicsGLRegisterBuffer(&cudaPosVBO[1], posVBO[1], cudaGraphicsMapFlagsNone);
    
    cudaGraphicsResource* cudaVelVBO[2];
    cudaGraphicsGLRegisterBuffer(&cudaVelVBO[0], velVBO[0], cudaGraphicsMapFlagsNone);
    cudaGraphicsGLRegisterBuffer(&cudaVelVBO[1], velVBO[1], cudaGraphicsMapFlagsNone);

    int frontBuffer = 0;
    int backBuffer = 1;
    size_t size;
    float4* poss;
    float4* vels;
    float4* poss2;
    float4* vels2;

    cudaGraphicsMapResources(1,cudaPosVBO,0);
    cudaGraphicsResourceGetMappedPointer((void**)&poss,&size,cudaPosVBO[frontBuffer]);
    
    cudaGraphicsMapResources(1,cudaVelVBO,0);
    cudaGraphicsResourceGetMappedPointer((void**)&vels,&size,cudaVelVBO[frontBuffer]);
    
    flock.genRand(poss, vels);
    
    cudaGraphicsUnmapResources(1, cudaPosVBO, 0);
    cudaGraphicsUnmapResources(1, cudaVelVBO, 0);


    double lastTime = glfwGetTime();
    int frameCount = 0;

    while (!glfwWindowShouldClose(window)) {
        // run calculations
        
        cudaGraphicsMapResources(1,&cudaPosVBO[frontBuffer],0);
        cudaGraphicsResourceGetMappedPointer((void**)&poss,&size,cudaPosVBO[frontBuffer]);
        
        cudaGraphicsMapResources(1,&cudaVelVBO[frontBuffer],0);
        cudaGraphicsResourceGetMappedPointer((void**)&vels,&size,cudaVelVBO[frontBuffer]);

        cudaGraphicsMapResources(1,&cudaPosVBO[backBuffer],0);
        cudaGraphicsResourceGetMappedPointer((void**)&poss2,&size,cudaPosVBO[backBuffer]);
        
        cudaGraphicsMapResources(1,&cudaVelVBO[backBuffer],0);
        cudaGraphicsResourceGetMappedPointer((void**)&vels2,&size,cudaVelVBO[backBuffer]);
        
        flock.step(poss, vels, poss2, vels2);
        
        cudaGraphicsUnmapResources(1, &cudaPosVBO[frontBuffer], 0);
        cudaGraphicsUnmapResources(1, &cudaPosVBO[frontBuffer], 0);
        cudaGraphicsUnmapResources(1, &cudaVelVBO[backBuffer], 0);
        cudaGraphicsUnmapResources(1, &cudaVelVBO[backBuffer], 0);

        //rebind
        glBindVertexArray(boidVAO);
        glBindBuffer(GL_ARRAY_BUFFER, posVBO[backBuffer]);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float4), nullptr);
        glBindBuffer(GL_ARRAY_BUFFER, velVBO[backBuffer]);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(float4), nullptr);
        
        // Move camera
        glm::vec3 eye(
            cam.dist * cosf(cam.pitch) * sinf(cam.yaw),
            cam.dist * sinf(cam.pitch),
            cam.dist * cosf(cam.pitch) * cosf(cam.yaw)
        );
        glm::mat4 view = glm::lookAt(eye, glm::vec3(0), glm::vec3(0,1,0));
        glm::mat4 proj = glm::perspective(
            glm::radians(45.f),
            (float)WINDOW_WIDTH / WINDOW_HEIGHT,
            0.01f, 1000.f
        );

        glUseProgram(program);
        glUniformMatrix4fv(uView, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(uProj, 1, GL_FALSE, glm::value_ptr(proj));
        
        // draw
        glClear(GL_COLOR_BUFFER_BIT);
        glBindVertexArray(boidVAO);
        glDrawArraysInstanced(GL_TRIANGLES, 0, 12, Params::FLOCK_SIZE);

        glfwSwapBuffers(window);

        //fps
        double currentTime = glfwGetTime();
        frameCount++;
        if (currentTime - lastTime >= 1.0) {
            double fps = frameCount / (currentTime - lastTime);
            char title[64];
            snprintf(title,64,"Boids | FPS: %.1f", fps);
            glfwSetWindowTitle(window, title);
            frameCount = 0;
            lastTime = currentTime;
        }

        std::swap(frontBuffer, backBuffer);

        glfwPollEvents();
    }

    
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <string>
#include <iterator>

// --- ARCHITECTURAL TOGGLES ---
#define INTEL_LAPTOP_MODE 
#define EPOCH_SIZE 100    // Architect 04: Stay under 1.0s total per loop

const size_t P = 80281262;
const size_t N_ELEMENTS = 4683072; // Adjusted for 18-bit limbs

int main() {
    try {
        // 1. Platform & Device Setup
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) throw std::runtime_error("No OpenCL platforms found.");

        cl::Device device;
        std::vector<cl::Device> devices;
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) throw std::runtime_error("No GPU devices found.");
        device = devices[0];

        std::cout << " [DEVICE] " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        cl::Context context(device);
        cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

        // 2. Load and Build kernels.cl
        std::ifstream k_file("src/kernels.cl");
        if (!k_file.is_open()) throw std::runtime_error("Could not find src/kernels.cl");
        
        std::string src((std::istreambuf_iterator<char>(k_file)), std::istreambuf_iterator<char>());
        cl::Program::Sources sources;
        sources.push_back({src.c_str(), src.length()});
        cl::Program program(context, sources);

        try {
            program.build({device});
        } catch (cl::Error& e) {
            std::cerr << "Build Log:\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
            throw;
        }

        // 3. Memory Allocation (Architect 03 & 05)
        cl::Buffer buffer_limbs;
        double* svm_ptr = nullptr;

        #ifdef INTEL_LAPTOP_MODE
            // Coarse-Grained SVM for Intel Integrated GPU
            svm_ptr = (double*)clSVMAlloc(context(), CL_MEM_READ_WRITE, N_ELEMENTS * sizeof(double), 0);
            if (!svm_ptr) throw std::runtime_error("SVM Allocation Failed");
            std::cout << " [MEM] Initialized Coarse-Grained SVM Mode" << std::endl;
        #else
            // Explicit VRAM Buffer for Desktop
            buffer_limbs = cl::Buffer(context, CL_MEM_READ_WRITE, N_ELEMENTS * sizeof(double));
            std::cout << " [MEM] Initialized Explicit VRAM Mode" << std::endl;
        #endif

        cl::Buffer b_group_carries(context, CL_MEM_READ_WRITE, (N_ELEMENTS / 64) * sizeof(uint32_t));
        cl::Kernel k_sq(program, "dwt_squaring");
        cl::Kernel k_carry(program, "local_carry");

        // 4. THE HUNT LOOP (Architect 04: Sub-division)
        std::cout << " [STATUS] Active Hunt: M" << P << " Started..." << std::endl;

        for (int epoch = 0; epoch < 1000; ++epoch) {
            auto start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < EPOCH_SIZE; ++i) {
                // Set Kernel Arguments
                #ifdef INTEL_LAPTOP_MODE
                    clSetKernelArgSVMPointer(k_sq(), 0, svm_ptr);
                    clSetKernelArgSVMPointer(k_carry(), 0, svm_ptr);
                #else
                    k_sq.setArg(0, buffer_limbs);
                    k_carry.setArg(0, buffer_limbs);
                #endif
                
                k_carry.setArg(1, b_group_carries);

                // Enqueue Squaring
                queue.enqueueNDRangeKernel(k_sq, cl::NullRange, cl::NDRange(N_ELEMENTS), cl::NDRange(64));
                
                // Enqueue Carry
                queue.enqueueNDRangeKernel(k_carry, cl::NullRange, cl::NDRange(N_ELEMENTS), cl::NDRange(64));
            }
            queue.finish();

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            
            std::cout << " [EPOCH " << epoch << "] Time: " << diff.count() << "s | " 
                      << (diff.count() / EPOCH_SIZE) * 1000 << " ms/iter" << std::endl;
            
            if (diff.count() > 3.5) {
                std::cout << " [CRITICAL] Hangcheck risk detected. Reduce EPOCH_SIZE." << std::endl;
            }
        }

        #ifdef INTEL_LAPTOP_MODE
            clSVMFree(context(), svm_ptr);
        #endif

    } catch (const cl::Error& e) {
        std::cerr << "OpenCL Error: " << e.what() << " (" << e.err() << ")" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "System Error: " << e.what() << std::endl;
    }

    return 0;
}

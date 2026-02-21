#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iterator>
#include <sys/stat.h>

const std::string VERSION = "1.1.0-INTEL";
const std::string CODENAME = "Arithmancy";

// --- Persistence Helpers ---
bool file_exists(const std::string& name) {
    struct stat buffer;   
    return (stat(name.c_str(), &buffer) == 0); 
}

void save_checkpoint(cl::CommandQueue& queue, cl::Buffer& buffer, size_t size, int iteration) {
    std::vector<double> host_data(size / sizeof(double));
    queue.enqueueReadBuffer(buffer, CL_TRUE, 0, size, host_data.data());
    
    std::string filename = "checkpoints/m80m_latest.bin";
    std::ofstream ofs(filename, std::ios::binary);
    ofs.write(reinterpret_cast<char*>(host_data.data()), size);
    ofs.close();
    std::cout << " [SAVE]   Checkpoint verified at iteration " << iteration << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "==================================================" << std::endl;
    std::cout << " ὀρθός (ORTHOS) - " << CODENAME << " [" << VERSION << "]" << std::endl;
    std::cout << "==================================================" << std::endl;

    mkdir("checkpoints", 0777); // Ensure directory exists

    try {
        // 1. Setup OpenCL
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        cl::Device device;
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
        device = devices[0];
        cl::Context context(device);
        cl::CommandQueue queue(context, device);

        // 2. Compile
        std::ifstream file("src/kernels.cl");
        std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        cl::Program::Sources sources;
        sources.push_back({source.c_str(), source.length()});
        cl::Program program(context, sources);
        program.build({device});

        // 3. Memory (M80281262)
        size_t n_elements = ((80281262 / 18 + 1024) + 15) / 16 * 16;
        size_t buffer_size = n_elements * sizeof(double);
        cl::Buffer buffer_limbs(context, CL_MEM_READ_WRITE, buffer_size);
        cl::Buffer buffer_carries(context, CL_MEM_READ_WRITE, 4096 * sizeof(uint32_t));

        // 4. RESUME LOGIC
        if (file_exists("checkpoints/m80m_latest.bin")) {
            std::cout << " [LOAD]   Found existing checkpoint. Resuming..." << std::endl;
            std::vector<double> host_data(n_elements);
            std::ifstream ifs("checkpoints/m80m_latest.bin", std::ios::binary);
            ifs.read(reinterpret_cast<char*>(host_data.data()), buffer_size);
            queue.enqueueWriteBuffer(buffer_limbs, CL_TRUE, 0, buffer_size, host_data.data());
        }

        // 5. Hunt Loop
        cl::Kernel carry_kernel(program, "parallel_carry");
        std::cout << " [STATUS] Hunting for M80281262..." << std::endl;

        for (int i = 0; i <= 1000000; ++i) {
            carry_kernel.setArg(0, buffer_limbs);
            carry_kernel.setArg(1, buffer_carries);
            carry_kernel.setArg(2, 16);
            queue.enqueueNDRangeKernel(carry_kernel, cl::NullRange, cl::NDRange(n_elements), cl::NDRange(16));
            
            // Checkpoint every 5000 iterations (approx every 15-20 mins on HD 620)
            if (i > 0 && i % 5000 == 0) {
                save_checkpoint(queue, buffer_limbs, buffer_size, i);
            }
        }
        queue.finish();

    } catch (const std::exception& e) {
        std::cerr << "ERR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

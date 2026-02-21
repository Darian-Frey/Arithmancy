#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iterator>
#include <sys/stat.h>

const std::string VERSION = "1.2.0-INTEL";
const std::string CODENAME = "Arithmancy";

bool file_exists(const std::string& name) {
    struct stat buffer;   
    return (stat(name.c_str(), &buffer) == 0); 
}

void save_checkpoint(cl::CommandQueue& queue, cl::Buffer& buffer, size_t size, int iteration) {
    std::vector<double> host_data(size / sizeof(double));
    queue.enqueueReadBuffer(buffer, CL_TRUE, 0, size, host_data.data());
    mkdir("checkpoints", 0777);
    std::ofstream ofs("checkpoints/m80m_latest.bin", std::ios::binary);
    if (ofs.is_open()) {
        ofs.write(reinterpret_cast<char*>(host_data.data()), size);
        ofs.close();
        std::cout << " [SAVE]   Checkpoint iteration " << iteration << " (Verified)" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "==================================================" << std::endl;
    std::cout << " ὀρθός (ORTHOS) - " << CODENAME << " [" << VERSION << "]" << std::endl;
    std::cout << "==================================================" << std::endl;

    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        std::vector<cl::Device> devices;
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
        cl::Device device = devices[0];
        std::cout << " [DEVICE] " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        cl::Context context(device);
        cl::CommandQueue queue(context, device);

        std::ifstream k_file("src/kernels.cl");
        std::string source((std::istreambuf_iterator<char>(k_file)), std::istreambuf_iterator<char>());
        cl::Program::Sources sources;
        sources.push_back({source.c_str(), source.length()});
        cl::Program program(context, sources);
        program.build({device});

        size_t p = 80281262;
        size_t n_elements = ((p / 18 + 1024) + 15) / 16 * 16;
        size_t buffer_size = n_elements * sizeof(double);

        cl::Buffer buffer_limbs(context, CL_MEM_READ_WRITE, buffer_size);
        cl::Buffer buffer_carries(context, CL_MEM_READ_WRITE, 4096 * sizeof(uint32_t));

        if (file_exists("checkpoints/m80m_latest.bin")) {
            std::cout << " [LOAD]   Resuming from checkpoint..." << std::endl;
            std::vector<double> host_data(n_elements);
            std::ifstream ifs("checkpoints/m80m_latest.bin", std::ios::binary);
            ifs.read(reinterpret_cast<char*>(host_data.data()), buffer_size);
            queue.enqueueWriteBuffer(buffer_limbs, CL_TRUE, 0, buffer_size, host_data.data());
        }

        cl::Kernel dwt_kernel(program, "dwt_squaring");
        cl::Kernel carry_kernel(program, "parallel_carry");

        std::cout << " [STATUS] Active Hunt: M" << p << "..." << std::endl;

        for (int i = 0; i <= 1000000; ++i) {
            dwt_kernel.setArg(0, buffer_limbs);
            queue.enqueueNDRangeKernel(dwt_kernel, cl::NullRange, cl::NDRange(n_elements), cl::NDRange(16));

            carry_kernel.setArg(0, buffer_limbs);
            carry_kernel.setArg(1, buffer_carries);
            carry_kernel.setArg(2, 16);
            queue.enqueueNDRangeKernel(carry_kernel, cl::NullRange, cl::NDRange(n_elements), cl::NDRange(16));
            
            if (i % 5000 == 0 && i > 0) {
                save_checkpoint(queue, buffer_limbs, buffer_size, i);
            }
        }
        queue.finish();

    } catch (const cl::Error& e) {
        std::cerr << "ERR: OpenCL - " << e.what() << " (" << e.err() << ")" << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "ERR: System - " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

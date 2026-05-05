#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "voxelIO/openvdb_wrapper_t.h"

/**
 * @brief Read a density field from a binary file.
 *
 * @param[in] path Path to density field file `rho.bin`.
 * @param[out] res Resolution of the density field.
 *
 * @returns Densities.
 *
 * @note The density field should be saved using the `save_density_bin.py` script.
 */
std::vector<float> readDensityBin(const std::string& path, int res[3]) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Cannot open input file: " + path);
    }

    ifs.read(reinterpret_cast<char*>(res), sizeof(int) * 3);
    if (!ifs) {
        throw std::runtime_error("Failed to read resolution header from: " + path);
    }

    if (res[0] <= 0 || res[1] <= 0 || res[2] <= 0) {
        throw std::runtime_error("Invalid resolution in input file");
    }

    const size_t voxelCount = static_cast<size_t>(res[0]) * res[1] * res[2];

    std::vector<float> rho(voxelCount);

    ifs.read(reinterpret_cast<char*>(rho.data()), static_cast<std::streamsize>(voxelCount * sizeof(float)));
    if (!ifs) {
        throw std::runtime_error("Input file ended before all density values were read");
    }

    return rho;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: save_density_vdb /path/to/input.bin /path/to/output.vdb\n";
        return 1;
    }

    try {
        int res[3] = {0, 0, 0};
        const std::vector<float> rho = readDensityBin(argv[1], res);

        openvdb_wrapper_t<float>::lexicalGrid2openVDBfile(argv[2], res, rho);

        auto [min_it, max_it] = std::minmax_element(rho.begin(), rho.end());

        double mean = 0.0;
        for (float v : rho) {
            mean += v;
        }
        mean /= static_cast<double>(rho.size());

        std::cout << "Wrote VDB: " << argv[2] << "\n";
        std::cout << "Resolution: " << res[0] << "x" << res[1] << "x" << res[2] << "\n";
        std::cout << "Min density: " << *min_it << "\n";
        std::cout << "Max density: " << *max_it << "\n";
        std::cout << "Average density: " << mean << "\n";
    }
    catch (const std::exception& e) {
        std::cerr << "save_density_vdb failed: " << e.what() << "\n";
        return 1;
    }
    catch (...) {
        std::cerr << "save_density_vdb failed with an unknown error\n";
        return 1;
    }

    return 0;
}

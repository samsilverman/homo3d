#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "voxelIO/openvdb_wrapper_t.h"

/**
 * @brief Computes the normal vector for a triangle.
 *
 * @param[in] a First vertex.
 * @param[in] b Second vertex.
 * @param[in] c Third vertex.
 *
 * @returns Normalized normal vector.
 */
static std::array<float, 3> computeNormal(const std::array<float, 3>& a,
                                          const std::array<float, 3>& b,
                                          const std::array<float, 3>& c) {
    const std::array<float, 3> u = {
        b[0] - a[0],
        b[1] - a[1],
        b[2] - a[2]
    };
    const std::array<float, 3> v = {
        c[0] - a[0],
        c[1] - a[1],
        c[2] - a[2]
    };

    std::array<float, 3> n = {
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0]
    };

    float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (len > 1e-12f) {
        n[0] /= len;
        n[1] /= len;
        n[2] /= len;
    } else {
        n = {0.0f, 0.0f, 0.0f};
    }

    return n;
}

/**
 * @brief Write a binary STL file.
 *
 * @param[in] path Path to saved STL file.
 * @param[in] vertices Vertices.
 * @param[in] triangles Triangles.
 *
 * @note STL file format: https://en.wikipedia.org/wiki/STL_(file_format)#Binary
 */
static void writeBinarySTL(const std::string& path,
                           const std::vector<std::array<float, 3>>& vertices,
                           const std::vector<std::array<int, 3>>& triangles) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Cannot open output STL file: " + path);
    }

    const char header[80] = {};
    ofs.write(header, sizeof(header));

    const uint32_t numTriangles = static_cast<uint32_t>(triangles.size());
    ofs.write(reinterpret_cast<const char*>(&numTriangles), sizeof(numTriangles));

    const uint16_t attribute = 0;

    for (const auto& triangle : triangles) {
        const auto& v0 = vertices[triangle[0]];
        const auto& v1 = vertices[triangle[1]];
        const auto& v2 = vertices[triangle[2]];

        const auto normal = computeNormal(v0, v1, v2);

        ofs.write(reinterpret_cast<const char*>(normal.data()), sizeof(normal));
        ofs.write(reinterpret_cast<const char*>(v0.data()), sizeof(v0));
        ofs.write(reinterpret_cast<const char*>(v1.data()), sizeof(v1));
        ofs.write(reinterpret_cast<const char*>(v2.data()), sizeof(v2));
        
        ofs.write(reinterpret_cast<const char*>(&attribute), sizeof(attribute));
    }

    if (!ofs) {
        throw std::runtime_error("Error occurred while writing STL file: " + path);
    }
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 6) {
        std::cerr << "Usage: save_density_stl /path/to/input.vdb /path/to/output.stl iso_value" << "\n";
        return 1;
    }

    try {
        const std::string inputPath = argv[1];
        const std::string outputPath = argv[2];
        const double isoValue = std::stod(argv[3]);

        std::vector<glm::vec3> points;
        std::vector<glm::vec<3, int>> tris;
        std::vector<glm::vec<4, int>> quads;

        openvdb_wrapper_t<float>::meshFromFile(inputPath, points, tris, quads, isoValue, true);

        if (points.empty() || (tris.empty() && quads.empty())) {
            throw std::runtime_error("No mesh triangles or quads were generated from the VDB file.");
        }

        std::vector<std::array<float, 3>> vertices;
        vertices.reserve(points.size());
        for (const auto& point : points) {
            vertices.push_back({point.x, point.y, point.z});
        }

        std::vector<std::array<int, 3>> triangles;
        triangles.reserve(tris.size() + quads.size() * 2);

        for (const auto& tri : tris) {
            triangles.push_back({tri.x, tri.y, tri.z});
        }
        for (const auto& quad : quads) {
            triangles.push_back({quad.x, quad.y, quad.z});
            triangles.push_back({quad.x, quad.z, quad.w});
        }

        writeBinarySTL(outputPath, vertices, triangles);

        std::cout << "Wrote STL: " << outputPath << "\n";
        std::cout << "Input VDB: " << inputPath << "\n";
        std::cout << "Triangles: " << triangles.size() << "\n";
        std::cout << "Vertices: " << vertices.size() << "\n";
        std::cout << "Iso-surface value: " << isoValue << "\n";
    }
    catch (const std::exception& ex) {
        std::cerr << "save_density_stl failed: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}

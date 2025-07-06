#include <array>
#include <cmath>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define PI 3.14159265358979323846

namespace py = pybind11;


void e2l_cosine_kernel_vector_3d(
    // to fill
    py::array_t<double> lagrangian_grid_field,
    // Given
    py::array_t<double> eulerian_grid_field, py::array_t<double> lag_positions,
    double dx, double eul_grid_coord_shift,
    std::array<int, 3> mpi_local_substart_coord_shift) {
  // For cosine interpolation kernel, width is always 2. (Technically 1 is also
  // acceptable but we only use 2)
  auto lag_field = lagrangian_grid_field.mutable_unchecked<2>();
  auto eul_field = eulerian_grid_field.unchecked<4>();
  auto lag_pos = lag_positions.unchecked<2>();

  double cosine_prefac{std::pow(0.25, 3)}, idx{1.0 / dx}, hpi{0.5 * PI};
  double interp_weight;  // interpolation weight
  double rx, ry, rz;     // arguments to the cosine kernel [-2, 2]
  double ax, ay, az;     // Increments to lagrangian field
  double lx, ly, lz;     // Rescaled, shifted Lagrangian grid point coordinates
  int ix, iy, iz;        // indices of the eulerian grid points
  for (unsigned lp = 0; lp < lag_pos.shape(1); ++lp) {
    // for each lagrangian grid point
    ax = 0.0;
    ay = 0.0;
    az = 0.0;
    lx = (lag_pos(0, lp) - eul_grid_coord_shift) * idx;
    ly = (lag_pos(1, lp) - eul_grid_coord_shift) * idx;
    lz = (lag_pos(2, lp) - eul_grid_coord_shift) * idx;

    for (int k1 = -1; k1 < 3; ++k1) {
      for (int k2 = -1; k2 < 3; ++k2) {
        for (int k3 = -1; k3 < 3; ++k3) {
          ix = static_cast<int>(lx) + k1;
          iy = static_cast<int>(ly) + k2;
          iz = static_cast<int>(lz) + k3;
          rx = ix - lx;
          ry = iy - ly;
          rz = iz - lz;
          ix -= mpi_local_substart_coord_shift[0];
          iy -= mpi_local_substart_coord_shift[1];
          iz -= mpi_local_substart_coord_shift[2];

          interp_weight = cosine_prefac * (1 + std::cos(hpi * rx)) *
                          (1 + std::cos(hpi * ry)) * (1 + std::cos(hpi * rz));
          ax += eul_field(0, iz, iy, ix) * interp_weight;
          ay += eul_field(1, iz, iy, ix) * interp_weight;
          az += eul_field(2, iz, iy, ix) * interp_weight;
        }
      }
    }
    lag_field(0, lp) = ax;
    lag_field(1, lp) = ay;
    lag_field(2, lp) = az;
  }
}

void l2e_cosine_kernel_vector_3d(
    // to fill
    py::array_t<double> eulerian_grid_field,
    // Given
    py::array_t<double> lagrangian_grid_field,
    py::array_t<double> lag_positions, double dx, double eul_grid_coord_shift,
    std::array<int, 3> mpi_local_substart_coord_shift) {
  // For cosine interpolation kernel, width is always 2. (Technically 1 is also
  // acceptable but we only use 2)
  auto eul_field = eulerian_grid_field.mutable_unchecked<4>();
  auto lag_field = lagrangian_grid_field.unchecked<2>();
  auto lag_pos = lag_positions.unchecked<2>();

  double cosine_prefac{std::pow(0.25 / dx, 3)}, idx{1.0 / dx}, hpi{0.5 * PI};
  double interp_weight;  // interpolation weight
  double rx, ry, rz;     // arguments to the cosine kernel [-2, 2]
  double ax, ay, az;     // Increments to lagrangian field
  double lx, ly, lz;     // Rescaled, shifted Lagrangian grid point coordinates
  int ix, iy, iz;        // indices of the eulerian grid points
  for (unsigned lp = 0; lp < lag_pos.shape(1); ++lp) {
    // for each lagrangian grid point
    ax = 0.0;
    ay = 0.0;
    az = 0.0;
    lx = (lag_pos(0, lp) - eul_grid_coord_shift) * idx;
    ly = (lag_pos(1, lp) - eul_grid_coord_shift) * idx;
    lz = (lag_pos(2, lp) - eul_grid_coord_shift) * idx;

    for (int k1 = -1; k1 < 3; ++k1) {
      for (int k2 = -1; k2 < 3; ++k2) {
        for (int k3 = -1; k3 < 3; ++k3) {
          ix = static_cast<int>(lx) + k1;
          iy = static_cast<int>(ly) + k2;
          iz = static_cast<int>(lz) + k3;
          rx = ix - lx;
          ry = iy - ly;
          rz = iz - lz;
          ix -= mpi_local_substart_coord_shift[0];
          iy -= mpi_local_substart_coord_shift[1];
          iz -= mpi_local_substart_coord_shift[2];

          interp_weight = cosine_prefac * (1 + std::cos(hpi * rx)) *
                          (1 + std::cos(hpi * ry)) * (1 + std::cos(hpi * rz));
          eul_field(0, iz, iy, ix) += lag_field(0, lp) * interp_weight;
          eul_field(1, iz, iy, ix) += lag_field(1, lp) * interp_weight;
          eul_field(2, iz, iy, ix) += lag_field(2, lp) * interp_weight;
        }
      }
    }
  }
}

PYBIND11_MODULE(_cosine_kernel_helper, m) {
  m.def("e2l_cosine_kernel_vector_3d", &e2l_cosine_kernel_vector_3d);
  m.def("l2e_cosine_kernel_vector_3d", &l2e_cosine_kernel_vector_3d);
}
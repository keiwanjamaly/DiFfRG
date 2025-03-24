#pragma once

// external libraries
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/fe_field_function.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_double.h>
#include <iostream>
#include <sys/types.h>
#include <tbb/parallel_for.h>

// standard library
#include <algorithm>
#include <iterator>
#include <mutex>

namespace DiFfRG
{
  namespace internal
  {
    template <int dim> int gsl_unwrap(const gsl_vector *gsl_x, void *params, gsl_vector *gsl_f)
    {
      const int subdim = gsl_x->size;

      dealii::Point<dim> x{};
      for (int i = 0; i < subdim; ++i)
        x[i] = gsl_vector_get(gsl_x, i);

      auto fp = static_cast<std::function<std::array<double, dim>(const dealii::Point<dim> &)> *>(params);
      const auto f = (*fp)(x);

      for (int i = 0; i < subdim; ++i)
        gsl_vector_set(gsl_f, i, f[i]);

      return GSL_SUCCESS;
    }

    using namespace dealii;
    template <int dim, typename Cell>
    bool is_in_cell(const Cell &cell, const Point<dim> &point, const Mapping<dim> &mapping)
    {
      try {
        Point<dim> qp = mapping.transform_real_to_unit_cell(cell, point);
        if (GeometryInfo<dim>::is_inside_unit_cell(qp))
          return true;
        else
          return false;
      } catch (const typename Mapping<dim>::ExcTransformationFailed &) {
        // transformation failed, so assume the point is outside
        return false;
      }
    }

    template <int dim> double l1_norm(const Point<dim> &p)
    {
      double norm = 0.;
      for (uint d = 0; d < dim; ++d)
        norm += std::abs(p[d]);
      return norm;
    }

    template <int dim> bool point_comperator(const Point<dim> &p1, const Point<dim> &p2)
    {
      return l1_norm(p1) < l1_norm(p2);
    }
  } // namespace internal

  /**
   * @brief Get the EoM point for a given solution and model in 1D. This is done by first checking the origin, and
   * then checking all cell borders in order to find a zero crossing. Then, the EoM point is found by bisection within
   * the cell.
   *
   * @tparam VectorType type of the solution vector.
   * @tparam Model type of the model.
   * @param EoM_cell the cell where the EoM point is located, will be set by the function. Is also used as a starting
   * point for the search.
   * @param sol the solution vector.
   * @param dof_handler a DoFHandler object associated with the solution vector.
   * @param mapping a Mapping object associated with the solution vector.
   * @param model numerical model providing a method EoM(const VectorType &)->double which we use to find a zero
   * crossing.
   * @param EoM_abs_tol the relative tolerance for the bisection method.
   * @return Point<dim> the point where the EoM is zero.
   */
  template <typename VectorType, typename EoMFUN, typename EoMPFUN>
  dealii::Point<1> get_EoM_point_1D(
      typename dealii::DoFHandler<1>::cell_iterator &EoM_cell, const VectorType &sol,
      const dealii::DoFHandler<1> &dof_handler, const dealii::Mapping<1> &mapping, const EoMFUN &get_EoM,
      const EoMPFUN &EoM_postprocess = [](const auto &p, const auto & /* values */) { return p; },
      const double EoM_abs_tol = 1e-8, const uint max_iter = 100)
  {
    constexpr uint dim = 1;

    using namespace dealii;
    Functions::FEFieldFunction<dim, VectorType> fe_function(dof_handler, sol, mapping);

    auto get_vertex_minimum_of_cell = [](const ReferenceCell &cell) -> Point<1> {
      const auto vertex_number = cell.n_vertices();
      Point<dim> min_vertex = cell.vertex<dim>(0);
      for (uint i = 1; i < vertex_number; i++) {
        Point<dim> vertex_i = cell.vertex<dim>(i);
        min_vertex = std::min(min_vertex, vertex_i, internal::point_comperator<dim>);
      }
      return min_vertex;
    };

    auto check_zero_crossing_in_cell = [&fe_function](const ReferenceCell &reference_cell,
                                                      dealii::DoFHandler<dim>::cell_iterator cell) -> bool {
      const auto vertex_number = reference_cell.n_vertices();
      for (uint i = 0; i < vertex_number; i++)
        for (uint j = 0; j < i; j++) {
          Point<dim> point_1 = cell->vertex(i);
          Point<dim> point_2 = cell->vertex(j);
          double value_1 = fe_function.value(point_1);
          double value_2 = fe_function.value(point_2);
          if (value_1 * value_2 < 0.0) {
            return true;
          }
        }
      return false;
    };

    auto compute_cell_minimum = [&fe_function, EoM_abs_tol,
                                 max_iter](const ReferenceCell &referece_cell,
                                           dealii::DoFHandler<dim>::cell_iterator cell) -> Point<dim> {
      // apply a divide a conquer method to find the minimum.
      Point<dim> point_1 = cell->vertex(0);
      Point<dim> point_2 = cell->vertex(1);
      for (uint i = 0; i < max_iter; i++) {
        Point<dim> midpoint = (point_1 + point_2) / 2;
        double EoM_val = fe_function.value(midpoint);
        if (EoM_val < 0.0) {
          point_1 = midpoint;
        } else {
          point_2 = midpoint;
        }
        if ((point_1 - point_2).norm() < EoM_abs_tol) {
          return midpoint;
        }
      }
      throw std::runtime_error("could not find EoM");
    };

    DoFHandler<dim>::cell_iterator origin_Cell = dof_handler.begin_active();
    Point<dim> origin_point = get_vertex_minimum_of_cell(origin_Cell->reference_cell());

    for (auto &cell : dof_handler.active_cell_iterators()) {
      fe_function.set_active_cell(cell); // setting the current cell, enhances performance
      auto reference_cell = cell->reference_cell();
      // compute the origin in each iteration
      Point<dim> cell_minimum = get_vertex_minimum_of_cell(reference_cell);
      origin_point = std::min(origin_point, cell_minimum, internal::point_comperator<dim>);
      // check if the current cell contains a minimum (zero crossing)
      if (check_zero_crossing_in_cell(reference_cell, cell)) {
        return compute_cell_minimum(reference_cell, cell);
      }
    }

    // if nothing was found, return the origin
    EoM_cell = GridTools::find_active_cell_around_point(dof_handler, origin_point);
    return origin_point;
  }

  // /**
  //  * @brief Get the EoM point for a given solution and model in 2D. This is done by first checking the origin, and
  //  * then checking all cell borders in order to find a zero crossing. Then, the EoM point is found by bisection
  //  within
  //  * the cell.
  //  *
  //  * @tparam VectorType type of the solution vector.
  //  * @tparam Model type of the model.
  //  * @param EoM_cell the cell where the EoM point is located, will be set by the function. Is also used as a
  //  starting
  //  * point for the search.
  //  * @param sol the solution vector.
  //  * @param dof_handler a DoFHandler object associated with the solution vector.
  //  * @param mapping a Mapping object associated with the solution vector.
  //  * @param model numerical model providing a method EoM(const VectorType &)->double which we use to find a zero
  //  * crossing.
  //  * @param EoM_abs_tol the relative tolerance for the bisection method.
  //  * @return Point<dim> the point where the EoM is zero.
  //  */
  // template <int dim, typename VectorType, typename EoMFUN, typename EoMPFUN>
  // dealii::Point<dim> get_EoM_point_ND(
  //     typename dealii::DoFHandler<dim>::cell_iterator &EoM_cell, const VectorType &sol,
  //     const dealii::DoFHandler<dim> &dof_handler, const dealii::Mapping<dim> &mapping, const EoMFUN &get_EoM,
  //     const EoMPFUN &EoM_postprocess = [](const auto &p, const auto & /* values */) { return p; },
  //     const double EoM_abs_tol = 1e-8, const uint max_iter = 100)
  // {
  //   using namespace dealii;
  //   using CellIterator = typename dealii::DoFHandler<dim>::cell_iterator;
  //   using EoMType = std::array<double, dim>;
  //
  //   Vector<typename VectorType::value_type> values(dof_handler.get_fe().n_components());
  //   Functions::FEFieldFunction<dim, VectorType> fe_function(dof_handler, sol, mapping);
  //
  //   // We start by investigating the origin.
  //   const auto origin = internal::get_origin(dof_handler, EoM_cell);
  //   fe_function.set_active_cell(EoM_cell);
  //   fe_function.vector_value(origin, values);
  //   const auto origin_val = get_EoM(origin, values);
  //
  //   const auto secondary_point = (origin + EoM_cell->center()) / 2.;
  //   fe_function.vector_value(secondary_point, values);
  //   const auto secondary_val = get_EoM(secondary_point, values);
  //
  //   // this actually constrains the EoM to one axis, possibly.
  //   // in a future version, we should exploit this to reduce the dimensionality of the problem.
  //   std::array<bool, dim> axis_restrictions{{}};
  //   for (uint d = 0; d < dim; ++d)
  //     axis_restrictions[d] = (origin_val[d] >= 0) && (secondary_val[d] >= 0);
  //
  //   const bool any_axis_restricted =
  //       std::any_of(std::begin(axis_restrictions), std::end(axis_restrictions), [](bool i) { return i; });
  //   const bool all_axis_restricted =
  //       std::all_of(std::begin(axis_restrictions), std::end(axis_restrictions), [](bool i) { return i; });
  //
  //   auto EoM = origin;
  //   EoM_cell = GridTools::find_active_cell_around_point(dof_handler, EoM);
  //   if (all_axis_restricted) {
  //     EoM = origin;
  //     // std::cout << "All axis restricted" << std::endl;
  //     return EoM;
  //   }
  //
  //   // std::cout << "Restrictions: ";
  //   // for (uint d = 0; d < dim; ++d)
  //   //   std::cout << axis_restrictions[d] << " ";
  //   // std::cout << std::endl;
  //
  //   auto check_cell = [&](const CellIterator &cell) -> bool {
  //     if (any_axis_restricted && !cell->has_boundary_lines()) return false;
  //
  //     // Obtain the values at the vertices of the cell.
  //     std::array<EoMType, GeometryInfo<dim>::vertices_per_cell> EoM_vals;
  //     std::array<Point<dim>, GeometryInfo<dim>::vertices_per_cell> vertices;
  //     Vector<typename VectorType::value_type> values(dof_handler.get_fe().n_components());
  //     Functions::FEFieldFunction<dim, VectorType> fe_function(dof_handler, sol, mapping);
  //
  //     fe_function.set_active_cell(cell);
  //     for (uint i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i) {
  //       vertices[i] = cell->vertex(i);
  //       fe_function.vector_value(vertices[i], values);
  //       EoM_vals[i] = get_EoM(vertices[i], values);
  //     }
  //
  //     if (any_axis_restricted)
  //       for (uint d = 0; d < dim; ++d)
  //         if (axis_restrictions[d]) {
  //           bool has_zero_boundary = false;
  //           for (uint i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i)
  //             if (is_close(origin[d], vertices[i][d], 1e-12)) {
  //               has_zero_boundary = true;
  //               break;
  //             }
  //           if (!has_zero_boundary) {
  //             // std::cout << "No zero boundary for cell with vertices: ";
  //             // for (uint i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i) {
  //             // std::cout << "vertex " << i << ": ";
  //             // for (uint d = 0; d < dim; ++d)
  //             // std::cout << vertices[i][d] << " ";
  //             // std::cout << std::endl;
  //             // }
  //
  //             return false;
  //           }
  //         }
  //
  //     std::array<bool, dim> cell_has_EoM{{}};
  //     // Find if the cell has an EoM point, i.e. if the values at some vertices have different signs.
  //     if (!any_axis_restricted) {
  //       for (uint i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i)
  //         for (uint j = 0; j < i; ++j)
  //           for (uint d = 0; d < dim; ++d)
  //             cell_has_EoM[d] = cell_has_EoM[d] || (EoM_vals[i][d] * EoM_vals[j][d] < 0.);
  //     } else {
  //       std::vector<bool> valid_vertices(GeometryInfo<dim>::vertices_per_cell, true);
  //
  //       for (uint d = 0; d < dim; ++d)
  //         if (axis_restrictions[d])
  //           for (uint i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i)
  //             // if there is a vertex with smaller coordinate in direction d, we set the vertex to invalid
  //             for (uint j = 0; j < i; ++j) {
  //               if (vertices[i][d] > vertices[j][d]) {
  //                 valid_vertices[i] = false;
  //                 break;
  //               }
  //             }
  //
  //       for (uint i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i)
  //         if (valid_vertices[i])
  //           for (uint j = 0; j < i; ++j)
  //             if (valid_vertices[j])
  //               for (uint d = 0; d < dim; ++d)
  //                 cell_has_EoM[d] = cell_has_EoM[d] || (EoM_vals[i][d] * EoM_vals[j][d] < 0.);
  //     }
  //
  //     bool has_EoM = true;
  //     for (uint d = 0; d < dim; ++d)
  //       if (!axis_restrictions[d]) has_EoM = has_EoM && cell_has_EoM[d];
  //     // if all components have a potential crossing, we return true
  //     return has_EoM;
  //   };
  //
  //   gsl_set_error_handler_off();
  //
  //   auto find_EoM = [&](const CellIterator &cell, CellIterator &m_EoM_cell, Point<dim> &m_EoM,
  //                       double &EoM_val) -> bool {
  //     Functions::FEFieldFunction<dim, VectorType> fe_function(dof_handler, sol, mapping);
  //     Vector<typename VectorType::value_type> values(dof_handler.get_fe().n_components());
  //     fe_function.set_active_cell(cell);
  //
  //     // find the cell boundaries
  //     std::array<std::array<double, 2>, dim> cell_boundaries{{}};
  //     for (uint d = 0; d < dim; ++d) {
  //       cell_boundaries[d][0] = cell->center()[d];
  //       cell_boundaries[d][1] = cell->center()[d];
  //     }
  //     for (uint d = 0; d < dim; ++d) {
  //       for (uint i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i) {
  //         cell_boundaries[d][0] = std::min(cell_boundaries[d][0], cell->vertex(i)[d]);
  //         cell_boundaries[d][1] = std::max(cell_boundaries[d][1], cell->vertex(i)[d]);
  //       }
  //     }
  //
  //     int subdim = dim;
  //     for (uint d = 0; d < dim; ++d)
  //       if (axis_restrictions[d]) subdim--;
  //
  //     if (subdim == 1) {
  //       uint dir = 0;
  //       for (uint d = 0; d < dim; ++d)
  //         if (!axis_restrictions[d]) dir = d;
  //
  //       // utlize the fact that we have a 1D problem and we can just do a bisection
  //       Point<dim> p1{};
  //       Point<dim> p2{};
  //       for (uint d = 0; d < dim; ++d) {
  //         p1[d] = cell_boundaries[d][0];
  //         p2[d] = cell_boundaries[d][0];
  //       }
  //       p2[dir] = cell_boundaries[dir][1];
  //       auto p = (p1 + p2) / 2.;
  //
  //       fe_function.vector_value(p, values);
  //       EoM_val = get_EoM(p, values)[dir];
  //       double err = 1e100;
  //       uint iter = 0;
  //
  //       while (err > EoM_abs_tol) {
  //         if (EoM_val < 0.)
  //           p1 = p;
  //         else
  //           p2 = p;
  //         p = (p1 + p2) / 2.;
  //
  //         fe_function.vector_value(p, values);
  //         EoM_val = get_EoM(p, values)[dir];
  //         err = std::abs(EoM_val);
  //
  //         if (iter > max_iter) {
  //           m_EoM = EoM_postprocess(p, values);
  //           m_EoM_cell = cell;
  //           return false;
  //         }
  //         iter++;
  //       }
  //
  //       m_EoM = EoM_postprocess(p, values);
  //       m_EoM_cell = cell;
  //
  //       return true;
  //     }
  //
  //     std::function<std::array<double, dim>(const dealii::Point<dim> &)> eval_on_point =
  //         [&](const Point<dim> &p) -> std::array<double, dim> {
  //       Point<dim> p_proj = p;
  //       for (uint d = 0, i = 0; d < dim; ++d)
  //         p_proj[d] = axis_restrictions[d] ? origin[d] : p[i++];
  //
  //       // check if the point is inside the cell
  //       std::array<double, dim> out_distance{{}};
  //       for (uint d = 0; d < dim; ++d) {
  //         if (p_proj[d] < cell_boundaries[d][0]) {
  //           out_distance[d] = std::abs(p_proj[d] - cell_boundaries[d][0]);
  //           p_proj[d] = cell_boundaries[d][0];
  //           // std::cout << "Point outside cell" << std::endl;
  //           // std::cout << "Point: " << p << std::endl;
  //         } else if (p_proj[d] > cell_boundaries[d][1]) {
  //           out_distance[d] = std::abs(p_proj[d] - cell_boundaries[d][1]);
  //           p_proj[d] = cell_boundaries[d][1];
  //           // std::cout << "Point outside cell" << std::endl;
  //           // std::cout << "Point: " << p << std::endl;
  //         }
  //       }
  //
  //       try {
  //         fe_function.vector_value(p_proj, values);
  //       } catch (...) {
  //         // if p is outside the triangulation, give a default value
  //         return std::array<double, dim>{{
  //             std::numeric_limits<double>::quiet_NaN(),
  //         }};
  //       }
  //
  //       auto EoM = get_EoM(p, values);
  //
  //       // if (out_distance[d] > 0), linearly extrapolate the value
  //       for (uint d = 0; d < dim; ++d)
  //         if (out_distance[d] > 0) {
  //           // std::cout << "Extrapolating value from " << EoM[d];
  //           EoM[d] = is_close(EoM[d], 0.) ? out_distance[d] : EoM[d] * (1 + out_distance[d]);
  //           // std::cout << " to " << EoM[d] << std::endl;
  //         }
  //
  //       // reshuflle the values to the correct order
  //       std::array<double, dim> EoM_out{{}};
  //       for (uint d = 0, i = 0; d < dim; ++d)
  //         if (!axis_restrictions[d]) EoM_out[d] = EoM[i++];
  //
  //       return EoM_out;
  //     };
  //
  //     const auto cell_center = cell->center();
  //     std::vector<double> subdim_center(subdim);
  //     for (uint d = 0, i = 0; d < dim; ++d)
  //       if (!axis_restrictions[d]) subdim_center[i++] = cell_center[d];
  //
  //     // Create GSL multiroot solver
  //     const gsl_multiroot_fsolver_type *T = gsl_multiroot_fsolver_hybrids;
  //     gsl_multiroot_fsolver *s = gsl_multiroot_fsolver_alloc(T, subdim);
  //
  //     // Create GSL function
  //     gsl_multiroot_function f = {&internal::gsl_unwrap<dim>, (size_t)subdim, &eval_on_point};
  //
  //     // Create initial guess
  //     gsl_vector *x = gsl_vector_alloc(subdim);
  //     for (int d = 0; d < subdim; ++d)
  //       gsl_vector_set(x, d, subdim_center[d]);
  //
  //     // Set the solver with the function and initial guess
  //     gsl_multiroot_fsolver_set(s, &f, x);
  //
  //     // start the iteration
  //     uint iter = 0;
  //     int status;
  //     do {
  //       iter++;
  //       status = gsl_multiroot_fsolver_iterate(s);
  //
  //       if (status) break;
  //
  //       status = gsl_multiroot_test_residual(s->f, EoM_abs_tol);
  //
  //     } while (status == GSL_CONTINUE && iter < max_iter);
  //
  //     EoM_val = 0.;
  //     for (uint d = 0, sd = 0; d < dim; ++d)
  //       m_EoM[d] = axis_restrictions[d] ? origin[d] : gsl_vector_get(s->x, sd++);
  //
  //     // don't leak memory. Stupid C
  //     gsl_multiroot_fsolver_free(s);
  //     gsl_vector_free(x);
  //
  //     if (status != GSL_SUCCESS) return false;
  //
  //     m_EoM = EoM_postprocess(m_EoM, values);
  //     m_EoM_cell = internal::is_in_cell(cell, m_EoM, mapping)
  //                      ? cell
  //                      : GridTools::find_active_cell_around_point(dof_handler, m_EoM);
  //
  //     return true;
  //   };
  //
  //   std::vector<typename dealii::DoFHandler<dim>::cell_iterator> cell_candidates;
  //   std::vector<double> value_candidates;
  //   std::vector<Point<dim>> EoM_candidates;
  //
  //   // mutex for the EoM cell
  //   // std::mutex EoM_cell_mutex;
  //
  //   const uint n_active_cells = dof_handler.get_triangulation().n_active_cells();
  //
  //   // this needs to be done smarter, but for now we just iterate over all cells
  //   // preferrably, we would divide the full domain into smaller subdomains and then traverse a kind of tree.
  //   for (uint index = 0; index < n_active_cells; ++index) {
  //     auto cell = dof_handler.begin_active();
  //     std::advance(cell, index);
  //     if (check_cell(cell)) {
  //       Point<dim> t_EoM = cell->center();
  //       CellIterator t_EoM_cell = cell;
  //       double t_EoM_value = 0.;
  //
  //       // lock the mutex
  //       // std::lock_guard<std::mutex> lock(EoM_cell_mutex);
  //
  //       if (find_EoM(cell, t_EoM_cell, t_EoM, t_EoM_value)) {
  //         // std::cout << "Found EoM point in cell " << index << std::endl;
  //         // std::cout << "EoM: " << t_EoM << std::endl;
  //         EoM = t_EoM;
  //         EoM_cell = dealii::GridTools::find_active_cell_around_point(dof_handler, EoM);
  //         return EoM;
  //       }
  //
  //       cell_candidates.push_back(t_EoM_cell);
  //       EoM_candidates.push_back(t_EoM);
  //       value_candidates.push_back(std::abs(t_EoM_value));
  //     }
  //   }
  //
  //   // std::cout << "Found " << cell_candidates.size() << " candidates." << std::endl;
  //
  //   double EoM_value = 0.;
  //
  //   if (cell_candidates.size() == 1) {
  //     if (find_EoM(cell_candidates[0], EoM_cell, EoM, EoM_value)) return EoM;
  //   } else if (cell_candidates.size() == 0) {
  //     EoM_cell = GridTools::find_active_cell_around_point(dof_handler, origin);
  //     return origin;
  //   } else if (cell_candidates.size() > 1) {
  //     // If we have more than one candidate, we choose the one with the smallest EoM value.
  //     auto min_it = std::min_element(value_candidates.begin(), value_candidates.end());
  //     auto min_idx = std::distance(value_candidates.begin(), min_it);
  //     EoM_cell = cell_candidates[min_idx];
  //     EoM = EoM_candidates[min_idx];
  //     return EoM;
  //   }
  //
  //   return EoM;
  // }

  /**
   * @brief Get the EoM point for a given solution and model.
   *
   * @tparam dim dimension of the problem.
   * @tparam VectorType type of the solution vector.
   * @tparam Model type of the model.
   * @param EoM_cell the cell where the EoM point is located, will be set by the function. Is also used as a starting
   * point for the search.
   * @param sol the solution vector.
   * @param dof_handler a DoFHandler object associated with the solution vector.
   * @param mapping a Mapping object associated with the solution vector.
   * @param model numerical model providing a method EoM(const VectorType &)->double which we use to find a zero
   * crossing.
   * @param EoM_abs_tol the relative tolerance for the bisection method.
   * @return Point<dim> the point where the EoM is zero.
   */
  template <int dim, typename VectorType, typename EoMFUN, typename EoMPFUN>
  dealii::Point<dim> get_EoM_point(
      typename dealii::DoFHandler<dim>::cell_iterator &EoM_cell, const VectorType &sol,
      const dealii::DoFHandler<dim> &dof_handler, const dealii::Mapping<dim> &mapping, const EoMFUN &get_EoM,
      const EoMPFUN &EoM_postprocess = [](const auto &p, const auto & /* values */) { return p; },
      const double EoM_abs_tol = 1e-5, const uint max_iter = 100)
  {
    // if (max_iter == 0) return internal::get_origin(dof_handler, EoM_cell);

    if constexpr (dim == 1) {
      return get_EoM_point_1D(EoM_cell, sol, dof_handler, mapping, get_EoM, EoM_postprocess, EoM_abs_tol, max_iter);
    } else if constexpr (dim > 1) {
      throw std::runtime_error("get_EoM_point is not yet implemented for dim > 1. Please set EoM_max_iter = 0.");
      // return get_EoM_point_ND(EoM_cell, sol, dof_handler, mapping, get_EoM, EoM_postprocess, EoM_abs_tol, max_iter);
    } else
      throw std::runtime_error("get_EoM_point is not yet implemented for dim > 1. Please set EoM_max_iter = 0.");
  }
} // namespace DiFfRG

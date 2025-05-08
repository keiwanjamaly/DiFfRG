#pragma once

// external libraries
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/fe_field_function.h>
#include <gsl/gsl_blas.h> // For printing matrix/vector
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multifit.h> // Header for fdjacobian
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_double.h>

namespace DiFfRG
{
  namespace internal
  {
    using namespace dealii;

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

    template <int dim>
    using EoMPointCellFunction =
        std::function<std::array<double, dim>(dealii::Point<dim>, typename dealii::DoFHandler<dim>::cell_iterator)>;
    template <int dim> double l1_norm(const Point<dim> &p)
    {
      double norm = 0.;
      for (uint d = 0; d < dim; ++d)
        norm += std::abs(p[d]);
      return norm;
    }

    template <int dim> bool point_comparator(const Point<dim> &p1, const Point<dim> &p2)
    {
      return l1_norm(p1) < l1_norm(p2);
    }

    template <int dim> auto get_one_active_cell_from_cell(const typename DoFHandler<dim>::cell_iterator &cell)
    {
      if (cell->is_active()) {
        return cell;
      }
      return get_one_active_cell_from_cell<dim>(cell->child(0));
    }

    enum Direction { up = 1, down = 0 };
    template <int dim, typename VectorType>
    Direction get_direction(const typename DoFHandler<dim>::cell_iterator &cell, EoMPointCellFunction<dim> &compute_EoM,
                            const uint component)
    {
      const Point<dim> center = cell->center();
      if (compute_EoM(center, cell)[component] < 0) return up;
      return down;
    }

    template <int dim, typename VectorType>
    bool has_EoM_in_one_component(const typename DoFHandler<dim>::cell_iterator &cell,
                                  EoMPointCellFunction<dim> &compute_EoM, uint component)
    {
      const uint n_vertices = cell->n_vertices();
      for (uint i = 0; i < n_vertices; i++) {
        for (uint j = 0; j < i; j++) {
          Point<dim> point_1 = cell->vertex(i);
          Point<dim> point_2 = cell->vertex(j);
          auto value_1 = compute_EoM(point_1, cell)[component];
          auto value_2 = compute_EoM(point_2, cell)[component];
          // check for a sign change from negative to bositiv. This is important to remove zero crossings, which are
          // created due to numerical accuracy
          if (point_1[component] > point_2[component]) {
            std::swap(value_1, value_2);
            std::swap(point_1, point_2);
          }
          if (value_1 < 0 and value_2 >= 0) return true;
        }
      }
      return false;
    }

    template <int dim> uint get_face_from_direction(Direction direction, uint component)
    {
      return 2 * component + direction;
    }

    template <int dim>
    bool face_not_at_boundary(const typename DoFHandler<dim>::cell_iterator &cell, const uint face_id)
    {
      return !cell->at_boundary(face_id);
    }

    template <int dim, typename VectorType>
    bool check_if_neighbouring_cells_contain_EoM(const typename DoFHandler<dim>::cell_iterator &cell,
                                                 EoMPointCellFunction<dim> compute_EoM)
    {
      bool neighbour_contains_EoM = true;
      for (unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no) {
        if (!cell->at_boundary(face_no)) {
          const auto neighbor = cell->neighbor(face_no);
          for (uint component = 0; component < dim; component++) {
            neighbour_contains_EoM =
                neighbour_contains_EoM && has_EoM_in_one_component<dim, VectorType>(neighbor, compute_EoM, component);
          }
        }
      }
      return neighbour_contains_EoM;
    }

    template <int dim, typename VectorType>
    std::tuple<bool, uint> get_walking_direction(const typename DoFHandler<dim>::cell_iterator &cell,
                                                 EoMPointCellFunction<dim> &compute_EoM)
    {
      for (uint component = 0; component < dim; component++) {
        if (!has_EoM_in_one_component<dim, VectorType>(cell, compute_EoM, component)) {
          Direction direction = get_direction<dim, VectorType>(cell, compute_EoM, component);
          uint face_direction = get_face_from_direction<dim>(direction, component);
          if (face_not_at_boundary<dim>(cell, face_direction)) return std::make_tuple(false, face_direction);
        }
      }
      if (check_if_neighbouring_cells_contain_EoM<dim, VectorType>(cell, compute_EoM)) {
        std::string error_message = "So you reached a point,";
        error_message += " where you need to implement some feature.Apparently, there are cells,";
        error_message +=
            " which can contain an EoM.At the moment, the EoM handler is not able to handle such conditions,";
        error_message +=
            " since they are thought to be unlikely.So now you know, they are not!Have fun implementing that feature ";
        error_message += ":)\n";
        error_message += "Cheers Keiwan";
        throw std::runtime_error(error_message);
      }
      return std::make_tuple(true, 666); // the last number has in the case of "true" no meaning
    }

    template <int dim, typename VectorType>
    auto walk_in_direction(const typename DoFHandler<dim>::cell_iterator &cell, EoMPointCellFunction<dim> &compute_EoM)
    {
      auto [has_EoM, walking_direction] = get_walking_direction<dim, VectorType>(cell, compute_EoM);
      if (has_EoM) return cell;
      auto new_cell = cell->neighbor(walking_direction);
      auto active_new_cell = get_one_active_cell_from_cell<dim>(new_cell);
      return walk_in_direction<dim, VectorType>(active_new_cell, compute_EoM);
    }

    // ################################################################################################
    //                          The code above handles the cell finding
    //                          Below is the code, which handels the EoM finding inside the cell
    // ################################################################################################

    template <int dim>
    std::array<std::array<double, 2>, dim>
    compute_cell_boundaries(const typename dealii::DoFHandler<dim>::cell_iterator &cell)
    {
      using namespace dealii;

      // Define the type for the bounding box for clarity
      using CellBoundingBox = std::array<std::array<double, 2>, dim>;

      // Initialize boundaries using the coordinates of the first vertex
      Point<dim> first_vertex = cell->vertex(0);
      CellBoundingBox boundaries;
      for (unsigned int d = 0; d < dim; ++d) {
        boundaries[d][0] = first_vertex[d]; // Initialize min
        boundaries[d][1] = first_vertex[d]; // Initialize max
      }

      // Iterate through the rest of the vertices to find the min/max coordinates
      for (unsigned int i = 1; i < GeometryInfo<dim>::vertices_per_cell; ++i) {
        const Point<dim> vertex = cell->vertex(i);
        for (unsigned int d = 0; d < dim; ++d) {
          boundaries[d][0] = std::min(boundaries[d][0], vertex[d]);
          boundaries[d][1] = std::max(boundaries[d][1], vertex[d]);
        }
      }

      return boundaries;
    }

    template <int dim>
    std::array<double, dim> eval_on_point(const dealii::Point<dim> &p,
                                          const typename dealii::DoFHandler<dim>::cell_iterator &cell,
                                          const std::array<std::array<double, 2>, dim> &cell_boundaries,
                                          EoMPointCellFunction<dim> &compute_EoM_for_cell)
    {
      Point<dim> p_proj = p;

      // check if the point is inside the cell
      std::array<double, dim> out_distance{{}};
      for (uint d = 0; d < dim; ++d) {
        if (p_proj[d] < cell_boundaries[d][0]) {
          out_distance[d] = std::abs(p_proj[d] - cell_boundaries[d][0]);
          p_proj[d] = cell_boundaries[d][0];
          // std::cout << "Point outside cell" << std::endl;
          // std::cout << "Point: " << p << std::endl;
        } else if (p_proj[d] > cell_boundaries[d][1]) {
          out_distance[d] = std::abs(p_proj[d] - cell_boundaries[d][1]);
          p_proj[d] = cell_boundaries[d][1];
          // std::cout << "Point outside cell" << std::endl;
          // std::cout << "Point: " << p << std::endl;
        }
      }

      auto EoM = compute_EoM_for_cell(p_proj, cell);

      // Linearly extrapolate the value if the original point 'p' was outside
      for (uint d = 0; d < dim; ++d)
        if (out_distance[d] > 0) {
          if (!std::isnan(EoM[d])) {
            EoM[d] = is_close(EoM[d], 0.) ? out_distance[d] : EoM[d] * (1 + out_distance[d]);
          }
        }

      return EoM;
    };

    template <int dim>
    Point<dim> perform_1D_bisection(const typename DoFHandler<dim>::cell_iterator &cell, Point<dim> point_1,
                                    Point<dim> point_2, EoMPointCellFunction<dim> &compute_EoM,
                                    const double EoM_abs_tol, const uint max_iter, uint component = 0)
    {
      for (uint iter = 0; iter <= max_iter; iter++) {
        Point<dim> point_mid = 0.5 * (point_1 + point_2);
        if ((point_1 - point_2).norm() > EoM_abs_tol) {
          auto value_mid = compute_EoM(point_mid, cell)[component];
          if (value_mid > 0) {
            point_2 = point_mid;
          } else {
            point_1 = point_mid;
          }
        } else {
          return point_mid;
        }
      }
    }

    template <int dim>
    std::tuple<bool, Point<dim>> check_if_EoM_lies_on_vertex(const typename DoFHandler<dim>::cell_iterator &cell,
                                                             Point<dim> point_1, Point<dim> point_2,
                                                             EoMPointCellFunction<dim> &compute_EoM, uint component = 0)
    {
      auto value_1 = compute_EoM(point_1, cell)[component];
      auto value_2 = compute_EoM(point_2, cell)[component];
      std::cout << "point_1: " << point_1 << " point_2: " << point_2 << " value_1: " << value_1
                << " value_2: " << value_2 << " component " << component << std::endl;
      if (l1_norm(point_1) > l1_norm(point_2)) {
        std::swap(point_1, point_2);
        std::swap(value_1, value_2);
      }
      if (value_2 >= value_1 && value_1 * value_2 >= 0) {
        return std::make_tuple(true, point_1);
      }
      return std::make_tuple(false, cell->vertex(0));
    }

    template <int dim, typename VectorType>
    std::tuple<bool, Point<dim>, Point<dim>, uint>
    check_faces_if_they_contain_root(const typename DoFHandler<dim>::cell_iterator &cell,
                                     EoMPointCellFunction<dim> &compute_EoM)
    {
      for (uint component = 0; component < dim; component++) {
        if (!has_EoM_in_one_component<dim, VectorType>(cell, compute_EoM, component)) {
          Direction direction = get_direction<dim, VectorType>(cell, compute_EoM, component);
          uint face_direction = get_face_from_direction<dim>(direction, component);
          auto face = cell->face(face_direction);
          auto vertex_0 = face->vertex(0);
          auto vertex_1 = face->vertex(1);
          // std::cout << "wallah ich bin hier face_direction: " << face_direction << " center: " << cell->center()
          //           << std::endl;
          // std::cout << "vertex1: " << vertex_0 << " vertex2: " << vertex_1 << std::endl;
          uint component = 0;
          if (face_direction == 0) {
            component = 1;
          } else {
            component = 0;
          }
          return std::make_tuple(true, vertex_0, vertex_1, component);
        }
      }
      return std::make_tuple(false, cell->vertex(0), cell->vertex(0), 0);
    }

    template <int dim, typename VectorType>
    Point<dim> perform_in_cell_newton_rapson(const typename DoFHandler<dim>::cell_iterator &cell,
                                             EoMPointCellFunction<dim> &compute_EoM, const double EoM_abs_tol,
                                             const uint max_iter)
    {
      const auto cell_boundaries = compute_cell_boundaries<dim>(cell);

      std::function<std::array<double, dim>(const dealii::Point<dim> &)> gsl_eval_wrapper =
          [&](const Point<dim> &p) -> std::array<double, dim> {
        return eval_on_point<dim>(p, cell, cell_boundaries, compute_EoM);
      };

      // Create GSL multiroot solver
      const gsl_multiroot_fsolver_type *T = gsl_multiroot_fsolver_hybrids;
      gsl_multiroot_fsolver *s = gsl_multiroot_fsolver_alloc(T, dim);

      // Create GSL function
      gsl_multiroot_function f = {&internal::gsl_unwrap<dim>, (size_t)dim, &gsl_eval_wrapper};

      // Create initial guess
      Point<dim> center = cell->center();
      gsl_vector *x = gsl_vector_alloc(dim);
      for (int d = 0; d < dim; ++d)
        gsl_vector_set(x, d, center[d]);

      // Set the solver with the function and initial guess
      gsl_multiroot_fsolver_set(s, &f, x);

      // start the iteration
      uint iter = 0;
      int status;
      int status_delta;
      do {
        iter++;
        status = gsl_multiroot_fsolver_iterate(s);

        if (status) break;

        status = gsl_multiroot_test_residual(s->f, EoM_abs_tol);
        status_delta = gsl_multiroot_test_delta(s->dx, s->x, EoM_abs_tol, EoM_abs_tol);
        if (true) { // Or use a verbosity flag
          std::cout << "Newton Iter: " << iter << " X: [";
          for (int d = 0; d < dim; ++d)
            std::cout << gsl_vector_get(s->x, d) << (d == dim - 1 ? "" : ", ");
          std::cout << "] F: [";
          for (int d = 0; d < dim; ++d)
            std::cout << gsl_vector_get(s->f, d) << (d == dim - 1 ? "" : ", ");
          std::cout << "] Status: " << status << std::endl; // Status before iteration
        }

      } while ((status == GSL_CONTINUE || status_delta == GSL_CONTINUE) && iter < max_iter);

      Point<dim> result_point;
      if (status == GSL_SUCCESS) {
        for (uint d = 0; d < dim; ++d)
          result_point[d] = gsl_vector_get(s->x, d);
      } else {
        std::cerr << "Warning: Newton-Raphson failed to converge in cell centered at " << cell->center()
                  << " Status: " << gsl_strerror(status) << std::endl;
        result_point = cell->center(); // Fallback value
      }

      // don't leak memory. Stupid C
      gsl_multiroot_fsolver_free(s);
      gsl_vector_free(x);

      return result_point;
    }

    template <int dim, typename VectorType>
    Point<dim> comopute_in_cell_EoM(const typename DoFHandler<dim>::cell_iterator &cell,
                                    EoMPointCellFunction<dim> &compute_EoM, const double EoM_abs_tol,
                                    const uint max_iter)
    {
      // static_assert(dim <= 2, "in cell root finding is not implemented for dim > 2");
      if constexpr (dim == 1) {
        auto vertex_1 = cell->vertex(0);
        auto vertex_2 = cell->vertex(1);
        auto [EoM_is_on_boundary, vertex] = check_if_EoM_lies_on_vertex<dim>(cell, vertex_1, vertex_2, compute_EoM);
        if (EoM_is_on_boundary) {
          return vertex;
        } else {
          return perform_1D_bisection<dim>(cell, cell->vertex(0), cell->vertex(1), compute_EoM, EoM_abs_tol, max_iter);
        }
      }
      if constexpr (dim == 2) {
        auto [EoM_is_on_face, vertex_1, vertex_2, component] =
            check_faces_if_they_contain_root<dim, VectorType>(cell, compute_EoM);
        if (EoM_is_on_face) {
          auto [EoM_is_on_vertex, vertex] =
              check_if_EoM_lies_on_vertex<dim>(cell, vertex_1, vertex_2, compute_EoM, component);
          if (EoM_is_on_vertex) {
            return vertex;
          }
          std::cout << "any bisection happening?" << std::endl;
          return perform_1D_bisection<dim>(cell, vertex_1, vertex_2, compute_EoM, EoM_abs_tol, max_iter, component);
        }
        // return cell->center();
        return perform_in_cell_newton_rapson<dim, VectorType>(cell, compute_EoM, EoM_abs_tol, max_iter);
      }
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
  template <int dim, typename VectorType, typename EoMFUN, typename EoMPFUN>
  dealii::Point<dim> get_EoM_point(
      typename dealii::DoFHandler<dim>::cell_iterator &EoM_cell, const VectorType &sol,
      const dealii::DoFHandler<dim> &dof_handler, const dealii::Mapping<dim> &mapping, const EoMFUN &get_EoM,
      const EoMPFUN &EoM_postprocess = [](const auto &p, const auto & /* values */) { return p; },
      const double EoM_abs_tol = 1e-10, const uint max_iter = 100)
  {
    if (EoM_cell.state() != dealii::IteratorState::valid) {
      EoM_cell = dof_handler.begin_active();
    }
    dealii::Functions::FEFieldFunction<dim, VectorType> fe_function(dof_handler, sol, mapping);

    internal::EoMPointCellFunction<dim> compute_EoM =
        [&](dealii::Point<dim> point, typename dealii::DoFHandler<dim>::cell_iterator cell) -> std::array<double, dim> {
      VectorType sol(fe_function.n_components);
      fe_function.set_active_cell(cell);
      fe_function.vector_value(point, sol);
      return get_EoM(point, sol);
    };
    EoM_cell = internal::walk_in_direction<dim, VectorType>(EoM_cell, compute_EoM);
    dealii::Point<dim> EoM =
        internal::comopute_in_cell_EoM<dim, VectorType>(EoM_cell, compute_EoM, EoM_abs_tol, max_iter);

    // return EoM_cell->center();
    return EoM;
    // auto EoM = internal::find_EoM_in_cell(EoM_cell, fe_function);
    // return *EoM;
  }
} // namespace DiFfRG

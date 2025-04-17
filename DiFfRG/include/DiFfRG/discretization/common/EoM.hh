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
#include <gsl/gsl_vector.h>

namespace DiFfRG
{
  namespace internal
  {
    using namespace dealii;
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
    Point<dim> perform_1D_bisection(const typename DoFHandler<dim>::cell_iterator &cell,
                                    EoMPointCellFunction<dim> &compute_EoM, const double EoM_abs_tol,
                                    const uint max_iter)
    {
      Point<dim> point_1 = cell->vertex(0);
      Point<dim> point_2 = cell->vertex(1);
      for (uint iter = 0; iter <= max_iter; iter++) {
        Point<dim> point_mid = 0.5 * (point_1 + point_2);
        if ((point_1 - point_2).norm() > EoM_abs_tol) {
          auto value_mid = compute_EoM(point_mid, cell)[0];
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
                                                             EoMPointCellFunction<dim> &compute_EoM)
    {
      Point<dim> point_1 = cell->vertex(0);
      Point<dim> point_2 = cell->vertex(1);
      auto value_1 = compute_EoM(point_1, cell);
      std::cout << "here the error 1 occures" << value_1[0] << std::endl;
      auto value_2 = compute_EoM(point_2, cell);
      std::cout << "here the error 2 occures" << value_2[0] << std::endl;
      if (value_1[0] * value_2[0] >= 0) {
        std::cout << "here the error 3 occures" << std::endl;
        if (value_1[0] > value_2[0]) {
          return std::make_tuple(true, point_2);
        }
        return std::make_tuple(true, point_1);
      }
      return std::make_tuple(false, cell->vertex(0));
    }

    template <int dim, typename VectorType>
    Point<dim> comopute_in_cell_EoM(const typename DoFHandler<dim>::cell_iterator &cell,
                                    EoMPointCellFunction<dim> &compute_EoM, const double EoM_abs_tol,
                                    const uint max_iter)
    {
      std::cout << "hello" << std::endl;
      auto [EoM_is_on_boundary, vertex] = check_if_EoM_lies_on_vertex<dim>(cell, compute_EoM);
      if (EoM_is_on_boundary) {
        return vertex;
      } else {
        return perform_1D_bisection<dim>(cell, compute_EoM, EoM_abs_tol, max_iter);
      }
      // return cell->center();
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
      const double EoM_abs_tol = 1e-5, const uint max_iter = 100)
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

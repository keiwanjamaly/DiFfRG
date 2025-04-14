#pragma once

// external libraries
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/fe_field_function.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

// standard library
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
    template <int dim, typename VectorType> struct powell_params {
      Functions::FEFieldFunction<dim, VectorType> &fe_functions_as_param;
    };
    template <int dim, typename VectorType> int f_function(const gsl_vector *x, void *params, gsl_vector *f)
    {
      Point<dim> eval_point;
      for (uint component = 0; component < dim; component++) {
        eval_point[component] = gsl_vector_get(x, component);
      }
      for (uint component = 0; component < dim; component++) {
        double y0 = static_cast<struct powell_params<dim, VectorType> *>(params)->fe_functions_as_param.value(
            eval_point, component);
        gsl_vector_set(f, component, y0);
      }
      return GSL_SUCCESS;
    };
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

    template <int dim> auto get_one_active_cell_from_cell(const typename DoFHandler<dim>::cell_iterator &cell)
    {
      if (cell->is_active()) {
        return cell;
      }
      return get_one_active_cell_from_cell<dim>(cell->child(0));
    }

    enum Direction { up = 1, down = 0 };
    template <int dim, typename VectorType>
    Direction get_direction(const typename DoFHandler<dim>::cell_iterator &cell,
                            const Functions::FEFieldFunction<dim, VectorType> &fe_functions, const uint component)
    {
      const Point<dim> center = cell->center();
      // std::cout << "fe_value = " << fe_functions.value(center, component) << " at " << center << std::endl;
      if (fe_functions.value(center, component) < 0) return up;
      return down;
    }

    template <int dim, typename VectorType>
    bool has_EoM_in_one_component(const typename DoFHandler<dim>::cell_iterator &cell,
                                  Functions::FEFieldFunction<dim, VectorType> &fe_functions, uint component)
    {
      fe_functions.set_active_cell(cell);
      const uint n_vertices = cell->n_vertices();
      for (uint i = 0; i < n_vertices; i++) {
        for (uint j = 0; j < i; j++) {
          Point<dim> point_1 = cell->vertex(i);
          Point<dim> point_2 = cell->vertex(j);
          auto value_1 = fe_functions.value(point_1, component);
          auto value_2 = fe_functions.value(point_2, component);
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
      return dim * component + direction;
    }

    template <int dim>
    bool validate_face_direction(const typename DoFHandler<dim>::cell_iterator &cell, const uint face_id)
    {
      return !cell->at_boundary(face_id);
    }

    template <int dim, typename VectorType>
    bool check_if_neighbouring_cells_contain_EoM(const typename DoFHandler<dim>::cell_iterator &cell,
                                                 Functions::FEFieldFunction<dim, VectorType> &fe_functions)
    {
      bool neighbour_contains_EoM = true;
      for (unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no) {
        if (!cell->at_boundary(face_no)) {
          const auto neighbor = cell->neighbor(face_no);
          for (uint component = 0; component < dim; component++) {
            neighbour_contains_EoM =
                neighbour_contains_EoM && has_EoM_in_one_component<dim, VectorType>(neighbor, fe_functions, component);
          }
        }
      }
      return neighbour_contains_EoM;
    }

    template <int dim, typename VectorType>
    std::tuple<bool, uint> get_walking_direction(const typename DoFHandler<dim>::cell_iterator &cell,
                                                 Functions::FEFieldFunction<dim, VectorType> &fe_functions)
    {
      for (uint component = 0; component < dim; component++) {
        if (!has_EoM_in_one_component<dim, VectorType>(cell, fe_functions, component)) {
          Direction direction = get_direction(cell, fe_functions, component);
          uint face_direction = get_face_from_direction<dim>(direction, component);
          if (validate_face_direction<dim>(cell, face_direction)) return std::make_tuple(false, face_direction);
        }
      }
      if (check_if_neighbouring_cells_contain_EoM(cell, fe_functions)) {
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
    auto walk_in_direction(const typename DoFHandler<dim>::cell_iterator &cell,
                           Functions::FEFieldFunction<dim, VectorType> &fe_functions)
    {
      auto [has_EoM, walking_direction] = get_walking_direction(cell, fe_functions);
      if (has_EoM) return cell;
      // std::cout << "walking_direction = " << walking_direction << std::endl;
      auto new_cell = cell->neighbor(walking_direction);
      auto active_new_cell = get_one_active_cell_from_cell<dim>(new_cell);
      return walk_in_direction(active_new_cell, fe_functions);
    }

    template <int dim, typename VectorType>
    std::unique_ptr<VectorType> get_fe_values(Functions::FEFieldFunction<dim, VectorType> &fe_functions,
                                              Point<dim> &x_0)
    {
      auto value_vector = std::make_unique<VectorType>(dim);
      fe_functions.vector_value(x_0, *value_vector);
      return value_vector;
    }

    template <int dim, typename VectorType>
    auto get_fe_jacobian(Functions::FEFieldFunction<dim, VectorType> &fe_functions, Point<dim> &x_0)
    {
      auto jacobian = std::make_unique<std::vector<Tensor<1, dim, typename VectorType::value_type>>>(
          dim, Tensor<1, dim, typename VectorType::value_type>());
      fe_functions.vector_gradient(x_0, *(jacobian.get()));
      return jacobian;
    }

    template <int dim, typename VectorType>
    std::unique_ptr<Point<dim>>
    invert_linear_function(std::unique_ptr<std::vector<Tensor<1, dim, typename VectorType::value_type>>> &jacobian,
                           std::unique_ptr<VectorType> &values)
    {
      gsl_matrix *gsl_mat = gsl_matrix_alloc(dim, dim);
      gsl_vector *value_vector = gsl_vector_alloc(dim);
      gsl_vector *x = gsl_vector_alloc(dim);

      // Copy data from jac[i][j] into the GSL matrix
      for (size_t i = 0; i < dim; i++) {
        gsl_vector_set(value_vector, i, (*values)[i]);
        for (size_t j = 0; j < dim; j++) {
          gsl_matrix_set(gsl_mat, i, j, (*jacobian)[i][j]);
        }
      }

      // Create GSL matrix and vector views
      // gsl_matrix_view A = gsl_matrix_view_array(gsl_mat, dim, dim);
      // gsl_vector_view b = gsl_vector_view_array(values->data(), dim);

      // Allocate memory for solution vector x and permutation
      gsl_permutation *p = gsl_permutation_alloc(dim);
      int s;

      // Perform LU decomposition
      gsl_linalg_LU_decomp(gsl_mat, p, &s);

      // Solve the linear system
      gsl_linalg_LU_solve(gsl_mat, p, value_vector, x);

      auto result = std::make_unique<Point<dim>>();
      for (uint i = 0; i < dim; i++)
        (*result)[i] = gsl_vector_get(x, i);

      // Free allocated memory
      gsl_permutation_free(p);
      gsl_vector_free(x);
      gsl_vector_free(value_vector);
      gsl_matrix_free(gsl_mat);
      return result;
    }

    template <int dim, typename VectorType>
    std::unique_ptr<Point<dim>> compute_new_x_predition(Point<dim> &x_0, std::unique_ptr<Point<dim>> &offset,
                                                        VectorType t)
    {
      auto x_new = std::make_unique<Point<dim>>(x_0 - t * *offset.get());
      return x_new;
    }

    template <int dim>
    bool point_outside_cell(const typename DoFHandler<dim>::cell_iterator &cell, std::unique_ptr<Point<dim>> &x_new)
    {
      return !cell->point_inside(*(x_new.get()));
    }

    template <int dim, typename VectorType>
    std::unique_ptr<Point<dim>> find_EoM_in_cell(const typename DoFHandler<dim>::cell_iterator &cell,
                                                 Functions::FEFieldFunction<dim, VectorType> &fe_functions,
                                                 uint max_iter = 100, typename VectorType::value_type err = 1e-14)
    {

      fe_functions.set_active_cell(cell);
      Point<dim> x_0 = cell->center();
      for (uint i = 0; i < max_iter; i++) {
        typename VectorType::value_type t = 1.0;
        auto values = get_fe_values(fe_functions, x_0);
        auto jacobian = get_fe_jacobian(fe_functions, x_0);
        auto offset = invert_linear_function<dim, VectorType>(jacobian, values);
        auto x_new = compute_new_x_predition(x_0, offset, t);
        while (point_outside_cell(cell, x_new)) {
          t /= 2.0; // reduce the stepsize for the predicted point iteratively, until the prediction is inside the cell
          x_new = compute_new_x_predition(x_0, offset, t);
        }
        if ((*(x_new.get()) - x_0).norm() < err) {
          return x_new;
        }
        x_0 = *(x_new.get());
      }
      throw std::runtime_error("Root finding did not converged!");
    }

    template <int dim, typename VectorType, typename EoMFUN>
    std::array<double, dim> compute_EoM(Point<dim> point, Functions::FEFieldFunction<dim, VectorType> &fe_functions,
                                        const EoMFUN &get_EoM)
    {
      VectorType value_vector(fe_functions.n_components());
      fe_functions.vector_value(point, value_vector);
      return get_EoM(point, value_vector);
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
    if (EoM_cell.state() == dealii::IteratorState::invalid) {
      EoM_cell = dof_handler.begin_active();
    }
    dealii::Functions::FEFieldFunction<dim, VectorType> fe_function(dof_handler, sol, mapping);
    EoM_cell = internal::walk_in_direction(EoM_cell, fe_function);
    auto EoM = internal::find_EoM_in_cell(EoM_cell, fe_function);
    return *EoM;
  }
} // namespace DiFfRG

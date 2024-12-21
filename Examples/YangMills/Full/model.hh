#pragma once

#include "flows/flows.hh"
#include <DiFfRG/model/model.hh>
#include <DiFfRG/physics/utils.hh>
#include <cmath>

using namespace dealii;
using namespace DiFfRG;

struct Parameters {
  Parameters(const JSONValue &json)
  {
    try {
      Lambda = json.get_double("/physical/Lambda");

      alphaA3 = json.get_double("/physical/alphaA3");
      alphaA4 = json.get_double("/physical/alphaA4");
      alphaAcbc = json.get_double("/physical/alphaAcbc");

      tilt_A3 = json.get_double("/physical/tilt_A3");
      tilt_A4 = json.get_double("/physical/tilt_A4");
      tilt_Acbc = json.get_double("/physical/tilt_Acbc");

      m2A = json.get_double("/physical/m2A");

      p_grid_min = json.get_double("/discretization/p_grid_min");
      p_grid_max = json.get_double("/discretization/p_grid_max");
      p_grid_bias = json.get_double("/discretization/p_grid_bias");
    } catch (std::exception &e) {
      std::cout << "Error in reading parameters: " << e.what() << std::endl;
    }
  }

  double Lambda;
  double alphaA3, alphaA4, alphaAcbc;
  double tilt_A3, tilt_A4, tilt_Acbc;
  double m2A;

  double p_grid_min, p_grid_max, p_grid_bias;
};

// Size of the momentum grid
static constexpr uint p_grid_size = 96;
static constexpr uint S1_size = 8;
static constexpr uint SPhi_size = 7;
// As for components, we have one FE function (u) and several extractors.
using VariableDesc =
    VariableDescriptor<Scalar<"m2A">,

                       FunctionND<"ZA3", p_grid_size, S1_size, SPhi_size>,
                       FunctionND<"ZAcbc", p_grid_size, S1_size, SPhi_size>,
                       FunctionND<"ZA4tadpole", p_grid_size, S1_size, SPhi_size>, FunctionND<"ZA4SP", p_grid_size>,

                       FunctionND<"ZA", p_grid_size>, FunctionND<"Zc", p_grid_size>>;
using Components = ComponentDescriptor<FEFunctionDescriptor<>, VariableDesc, ExtractorDescriptor<>>;

constexpr auto idxv = VariableDesc{};

/**
 * @brief This class implements the numerical model for the quark-meson model at finite temperature and chemical
 * potential.
 */
class YangMills : public def::AbstractModel<YangMills, Components>,
                  public def::fRG,        // this handles the fRG time
                  public def::NoJacobians // define all jacobians per AD
{
  const Parameters prm;

  using Coordinates1D = LogarithmicCoordinates1D<float>;
  using CoordinatesAng = LinearCoordinates1D<float>;
  using Coordinates3D =
      CoordinatePackND<LogarithmicCoordinates1D<float>, LinearCoordinates1D<float>, LinearCoordinates1D<float>>;
  const Coordinates1D coordinates1D;
  const CoordinatesAng S1_coordinates;
  const CoordinatesAng SPhi_coordinates;
  const Coordinates3D coordinates3D;

  const std::vector<float> grid1D;
  const std::vector<float> S1_grid;
  const std::vector<float> SPhi_grid;
  const std::vector<std::array<float, 3>> grid3D;

  mutable YangMillsFlowEquations flow_equations;

  mutable TexLinearInterpolator1D<double, Coordinates1D> dtZc, dtZA, ZA, Zc, ZA4SP;
  mutable TexLinearInterpolator3D<double, Coordinates3D> ZAcbc, ZA3, ZA4tadpole;

public:
  YangMills(const JSONValue &json)
      : def::fRG(json.get_double("/physical/Lambda")), prm(json),
        coordinates1D(p_grid_size, prm.p_grid_min, prm.p_grid_max, prm.p_grid_bias),
        S1_coordinates(S1_size, 0.0, 0.9999), SPhi_coordinates(SPhi_size, 0.0, 2. * M_PI),
        coordinates3D(coordinates1D, S1_coordinates, SPhi_coordinates), grid1D(make_grid(coordinates1D)),
        S1_grid(make_grid(S1_coordinates)), SPhi_grid(make_grid(SPhi_coordinates)), grid3D(make_grid(coordinates3D)),
        flow_equations(json), dtZc(coordinates1D), dtZA(coordinates1D), ZA(coordinates1D),
        Zc(coordinates1D),                                                  // propagators
        ZA4SP(coordinates1D),                                               // 1D couplings
        ZAcbc(coordinates3D), ZA3(coordinates3D), ZA4tadpole(coordinates3D) // 3D couplings
  {
    flow_equations.set_k(prm.Lambda);
    flow_equations.print_parameters("log");
  }

  template <typename Vector> void initial_condition_variables(Vector &values) const
  {
    for (uint i = 0; i < p_grid_size; ++i) {
      for (uint j = 0; j < S1_size; ++j) {
        for (uint k = 0; k < SPhi_size; ++k) {
          values[idxv("ZA3") + i * S1_size * SPhi_size + j * SPhi_size + k] =
              std::sqrt(4. * M_PI * prm.alphaA3) + prm.tilt_A3 * std::log(grid1D[i] / prm.p_grid_max);
          values[idxv("ZAcbc") + i * S1_size * SPhi_size + j * SPhi_size + k] =
              std::sqrt(4. * M_PI * prm.alphaAcbc) + prm.tilt_Acbc * std::log(grid1D[i] / prm.p_grid_max);
          values[idxv("ZA4tadpole") + i * S1_size * SPhi_size + j * SPhi_size + k] =
              4. * M_PI * prm.alphaA4 + prm.tilt_A4 * std::log(grid1D[i] / prm.p_grid_max);
        }
      }

      values[idxv("ZA4SP") + i] = 4. * M_PI * prm.alphaA4;

      values[idxv("ZA") + i] = (powr<2>(grid1D[i]) + prm.m2A) / powr<2>(grid1D[i]);
      values[idxv("Zc") + i] = 1.;
    }
    // glue mass
    values[idxv("m2A")] = prm.m2A;
  }

  void set_time(double t_)
  {
    t = t_;
    k = std::exp(-t) * Lambda;
    flow_equations.set_k(k);
  }

  template <typename Vector, typename Solution> void dt_variables(Vector &residual, const Solution &data) const
  {
    const auto &variables = get<"variables">(data);

    const auto m2A = variables[idxv("m2A")];

    ZA3.update(&variables.data()[idxv("ZA3")]);
    ZAcbc.update(&variables.data()[idxv("ZAcbc")]);
    ZA4SP.update(&variables.data()[idxv("ZA4SP")]);
    ZA4tadpole.update(&variables.data()[idxv("ZA4tadpole")]);

    ZA.update(&variables.data()[idxv("ZA")]);
    Zc.update(&variables.data()[idxv("Zc")]);

    // set up arguments for the integrators
    const auto arguments = std::tie(ZA3, ZAcbc, ZA4SP, ZA4tadpole, dtZc, Zc, dtZA, ZA, m2A);

    // copy the propagators for comparison
    std::vector<double> old_dtZA(p_grid_size);
    std::vector<double> old_dtZc(p_grid_size);

    // start by solving the equations for propagators
    bool eta_converged = false;
    uint n_iter = 0;
    while (!eta_converged) {
      // copy
      for (uint i = 0; i < p_grid_size; ++i) {
        old_dtZA[i] = dtZA[i];
        old_dtZc[i] = dtZc[i];
      }

      auto futures_dtZA = request_data<double>(flow_equations.ZA_integrator, grid1D, k, arguments);
      auto futures_dtZc = request_data<double>(flow_equations.Zc_integrator, grid1D, k, arguments);

      for (uint i = 0; i < p_grid_size; ++i) {
        residual[idxv("ZA") + i] = (*futures_dtZA)[i].get();
        residual[idxv("Zc") + i] = (*futures_dtZc)[i].get();
        dtZA[i] = residual[idxv("ZA") + i];
        dtZc[i] = residual[idxv("Zc") + i];
      }
      dtZA.update();
      dtZc.update();

      // check distance
      double dist = 0.;
      for (uint i = 0; i < p_grid_size; ++i) {
        dist = std::max(dist, std::abs(dtZA[i] - old_dtZA[i]) / std::abs(dtZA[i]));
        dist = std::max(dist, std::abs(dtZc[i] - old_dtZc[i]) / std::abs(dtZc[i]));
      }
      if (dist < 1e-4 || n_iter > 10) eta_converged = true;
      n_iter++;
    }
    std::cout << "Converged after " << n_iter << " iterations." << std::endl;

    // call all other integrators
    auto futures_ZA3 = request_data<double>(flow_equations.ZA3_integrator, grid3D, k, arguments);
    auto futures_ZAcbc = request_data<double>(flow_equations.ZAcbc_integrator, grid3D, k, arguments);
    auto futures_ZA4tadpole = request_data<double>(flow_equations.ZA4tadpole_integrator, grid3D, k, arguments);
    auto futures_ZA4SP = request_data<double>(flow_equations.ZA4SP_integrator, grid1D, k, arguments);

    // get all results and write them to residual
    update_data(futures_ZA3, &residual[idxv("ZA3")]);
    update_data(futures_ZAcbc, &residual[idxv("ZAcbc")]);
    update_data(futures_ZA4tadpole, &residual[idxv("ZA4tadpole")]);
    update_data(futures_ZA4SP, &residual[idxv("ZA4SP")]);

    for (uint i = 0; i < p_grid_size; ++i) {
      residual[idxv("ZA") + i] = dtZA.data()[i];
      residual[idxv("Zc") + i] = dtZc.data()[i];
    }

    std::apply(
        [&](const auto &...args) { residual[idxv("m2A")] = flow_equations.m2A_integrator.get<double>(k, 0., args...); },
        arguments);

    if (Zc[0] < 0) throw std::runtime_error("Zc < 0");

    /*for (uint i = 0; i < p_grid_size; ++i) {
      if(!std::isfinite(residual[idxv("ZA") + i])) std::cout << "res ZA not finite" << std::endl;
      if(!std::isfinite(residual[idxv("Zc") + i])) std::cout << "res Zc not finite" << std::endl;

      for (uint j = 0; j < S1_size; ++j) {
        for (uint k = 0; k < SPhi_size; ++k) {
          if(!std::isfinite(residual[idxv("ZA3") + i * S1_size * SPhi_size + j * SPhi_size + k])) std::cout << "res ZA3
    not finite" << std::endl; if(!std::isfinite(residual[idxv("ZAcbc") + i * S1_size * SPhi_size + j * SPhi_size + k]))
    std::cout << "res ZAcbc not finite" << std::endl;
        }
      }

      if(!std::isfinite(residual[idxv("ZA4") + i])) std::cout << "res ZA4 not finite" << std::endl;
    }*/
  }

  template <int dim, typename DataOut, typename Solutions>
  void readouts(DataOut &output, const Point<dim> &, const Solutions &sol) const
  {
    const auto &variables = get<"variables">(sol);
    const auto m2A = variables[idxv("m2A")];

    this->Zc.update(&variables.data()[idxv("Zc")]);

    auto &out_file = output.csv_file("data.csv");
    out_file.set_Lambda(Lambda);
    out_file.value("m2A", m2A);

    const auto *ZA = &variables.data()[idxv("ZA")];
    const auto *Zc = &variables.data()[idxv("Zc")];
    std::vector<std::vector<double>> Zs_data(grid1D.size(), std::vector<double>(5, 0.));
    for (uint i = 0; i < p_grid_size; ++i) {
      Zs_data[i][0] = k;
      Zs_data[i][1] = grid1D[i];
      Zs_data[i][2] = ZA[i];
      Zs_data[i][3] = (ZA[i]); // * powr<2>(grid1D[i]) + m2A) / powr<2>(grid1D[i]);
      Zs_data[i][4] = Zc[i];
    }

    const auto *ZA3 = &variables.data()[idxv("ZA3")];
    const auto *ZAcbc = &variables.data()[idxv("ZAcbc")];
    const auto *ZA4SP = &variables.data()[idxv("ZA4SP")];
    std::vector<std::vector<double>> strong_couplings_data(grid1D.size(), std::vector<double>(5, 0.));
    for (uint i = 0; i < p_grid_size; ++i) {
      const double ZA_p = (ZA[i]); // * powr<2>(grid1D[i]) + m2A) / powr<2>(grid1D[i]);
      const double Zc_p = Zc[i];
      const size_t cur_idx = i * S1_size * SPhi_size;
      strong_couplings_data[i][0] = k;
      strong_couplings_data[i][1] = grid1D[i];
      strong_couplings_data[i][2] = powr<2>(ZAcbc[cur_idx]) / (4. * M_PI * ZA_p * powr<2>(Zc_p));
      strong_couplings_data[i][3] = powr<2>(ZA3[cur_idx]) / (4. * M_PI * powr<3>(ZA_p));
      strong_couplings_data[i][4] = ZA4SP[i] / (4. * M_PI * powr<2>(ZA_p));
    }

    const auto *ZA4tadpole = &variables.data()[idxv("ZA4tadpole")];
    std::vector<std::vector<double>> strong_couplings_3D_data(coordinates3D.size(), std::vector<double>(9, 0.));
    for (uint i = 0; i < p_grid_size; ++i) {
      for (uint j = 0; j < S1_size; ++j) {
        for (uint k = 0; k < SPhi_size; ++k) {
          const size_t cur_idx = i * S1_size * SPhi_size + j * SPhi_size + k;

          const auto S0 = grid1D[i];
          const auto S1 = S1_grid[j];
          const auto SPhi = SPhi_grid[k];

          const auto p = S0 * (sqrt(1.f - 1.f * S1 * sin(SPhi)));
          const auto r = S0 * (sqrt(1.f + 0.5f * S1 * (1.7320508075688772f * cos(SPhi) + sin(SPhi))));
          const auto s = 0.7071067811865475f * S0 * (sqrt(2.f - 1.7320508075688772f * S1 * cos(SPhi) + S1 * sin(SPhi)));

          const double Zc_r = this->Zc(r);
          const double Zc_s = this->Zc(s);

          const double Z_p = (this->ZA(p)); // * powr<2>(p) + m2A) / powr<2>(p);
          const double Z_r = (this->ZA(r)); // * powr<2>(r) + m2A) / powr<2>(r);
          const double Z_s = (this->ZA(s)); // * powr<2>(s) + m2A) / powr<2>(s);

          strong_couplings_3D_data[cur_idx][0] = this->k;
          strong_couplings_3D_data[cur_idx][1] = grid3D[cur_idx][0];
          strong_couplings_3D_data[cur_idx][2] = grid3D[cur_idx][1];
          strong_couplings_3D_data[cur_idx][3] = grid3D[cur_idx][2];
          strong_couplings_3D_data[cur_idx][4] = ZAcbc[cur_idx];
          strong_couplings_3D_data[cur_idx][5] = ZA3[cur_idx];
          strong_couplings_3D_data[cur_idx][6] = ZA4tadpole[cur_idx];
          strong_couplings_3D_data[cur_idx][7] = powr<2>(ZA3[cur_idx]) / (4. * M_PI * Z_p * Z_r * Z_s);
          strong_couplings_3D_data[cur_idx][8] = powr<2>(ZAcbc[cur_idx]) / (4. * M_PI * Z_p * Zc_r * Zc_s);
        }
      }
    }

    if (is_close(t, 0.)) {
      const std::vector<std::string> strong_couplings_header =
          std::vector<std::string>{"k [GeV]", "p [GeV]", "alphaAcbc", "alphaA3", "alphaA4"};
      const std::vector<std::string> strong_couplings_3D_header = std::vector<std::string>{
          "k [GeV]", "S0 [GeV]", "S1", "SPhi", "ZAcbc", "ZA3", "ZA4tadpole", "alphaA3", "alphaAcbc"};
      const std::vector<std::string> Zs_header = std::vector<std::string>{"k [GeV]", "p [GeV]", "ZAbar", "ZA", "Zc"};

      output.dump_to_csv("strong_couplings.csv", strong_couplings_data, false, strong_couplings_header);
      output.dump_to_csv("strong_couplings_3D.csv", strong_couplings_3D_data, false, strong_couplings_3D_header);
      output.dump_to_csv("Zs.csv", Zs_data, false, Zs_header);
    } else {
      output.dump_to_csv("strong_couplings.csv", strong_couplings_data, true);
      output.dump_to_csv("strong_couplings_3D.csv", strong_couplings_3D_data, true);
      output.dump_to_csv("Zs.csv", Zs_data, true);
    }
  }
};

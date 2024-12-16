#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/physics/integration/integrator_4D_cpu.hh>
#include <DiFfRG/physics/integration/integrator_4D_gpu.hh>
#include <DiFfRG/physics/integration/integrator_4D_qmc.hh>
#include <DiFfRG/physics/integration/quadrature_provider.hh>

using namespace DiFfRG;

//--------------------------------------------
// Quadrature integration

class PolyIntegrand
{
public:
  static __forceinline__ __host__ __device__ auto
  kernel(const double q, const double cos1, const double cos2, const double phi, const double /*k*/, const double /*c*/,
         const double x0, const double x1, const double x2, const double x3, const double x4, const double x5,
         const double cos1_x0, const double cos1_x1, const double cos1_x2, const double cos1_x3, const double cos2_x0,
         const double cos2_x1, const double cos2_x2, const double cos2_x3, const double phi_x0, const double phi_x1,
         const double phi_x2, const double phi_x3)
  {
    return (x0 + x1 * powr<1>(q) + x2 * powr<2>(q) + x3 * powr<3>(q) + x4 * powr<4>(q) + x5 * powr<5>(q)) *
           (cos1_x0 + cos1_x1 * powr<1>(cos1) + cos1_x2 * powr<2>(cos1) + cos1_x3 * powr<3>(cos1)) *
           (cos2_x0 + cos2_x1 * powr<1>(cos2) + cos2_x2 * powr<2>(cos2) + cos2_x3 * powr<3>(cos2)) *
           (phi_x0 + phi_x1 * powr<1>(phi) + phi_x2 * powr<2>(phi) + phi_x3 * powr<3>(phi));
  }

  static __forceinline__ __host__ __device__ auto
  constant(const double /*k*/, const double c, const double /*x0*/, const double /*x1*/, const double /*x2*/,
           const double /*x3*/, const double /*x4*/, const double /*x5*/, const double /*cos1_x0*/,
           const double /*cos1_x1*/, const double /*cos1_x2*/, const double /*cos1_x3*/, const double /*cos2_x0*/,
           const double /*cos2_x1*/, const double /*cos2_x2*/, const double /*cos2_x3*/, const double /*phi_x0*/,
           const double /*phi_x1*/, const double /*phi_x2*/, const double /*phi_x3*/)
  {
    return c;
  }
};

TEST_CASE("Benchmark 4D momentum integrals", "[4D integration][quadrature integration]")
{
  constexpr int dim = 4;

  const double x_extent = GENERATE(take(1, random(1., 2.)));
  QuadratureProvider quadrature_provider;

  constexpr uint take_n = 1;
  const auto poly = Polynomial({
      dim == 1 ? 0. : GENERATE(take(take_n, random(-1., 1.))), // x0
      GENERATE(take(take_n, random(-1., 1.))),                 // x1
      GENERATE(take(take_n, random(-1., 1.))),                 // x2
      GENERATE(take(1, random(-1., 1.))),                      // x3
      GENERATE(take(1, random(-1., 1.))),                      // x4
      GENERATE(take(1, random(-1., 1.))),                      // x5
  });
  const auto cos1_poly = Polynomial({
      GENERATE(take(take_n, random(-1., 1.))), // x0
      GENERATE(take(1, random(-1., 1.))),      // x1
      GENERATE(take(1, random(-1., 1.))),      // x2
      GENERATE(take(1, random(-1., 1.)))       // x3
  });
  const auto cos2_poly = Polynomial({
      GENERATE(take(take_n, random(-1., 1.))), // x0
      GENERATE(take(1, random(-1., 1.))),      // x1
      GENERATE(take(1, random(-1., 1.))),      // x2
      GENERATE(take(1, random(-1., 1.)))       // x3
  });
  const auto phi_poly = Polynomial({
      GENERATE(take(take_n, random(-1., 1.))), // x0
      GENERATE(take(1, random(-1., 1.))),      // x1
      GENERATE(take(1, random(-1., 1.))),      // x2
      GENERATE(take(1, random(-1., 1.)))       // x3
  });

  const double k = GENERATE(take(take_n, random(0., 1.)));
  const double q_extent = std::sqrt(x_extent * powr<2>(k));
  const double constant = GENERATE(take(take_n, random(-1., 1.)));

  auto int_poly = poly;
  std::vector<double> coeff_integrand(dim, 0.);
  coeff_integrand[dim - 1] = 1.;
  int_poly *= Polynomial(coeff_integrand);
  const double reference_integral =
      constant + int_poly.integral(0., q_extent) * ((4 * cos1_poly[0] + cos1_poly[2]) * M_PI / 8.) *
                     cos2_poly.integral(-1., 1.) * phi_poly.integral(0., 2. * M_PI) / powr<dim>(2. * M_PI);
  {
    Integrator4DGPU<double, PolyIntegrand> integrator(quadrature_provider, {{32, 12, 12, 12}}, x_extent);
    BENCHMARK_ADVANCED("GPU")(Catch::Benchmark::Chronometer meter)
    {
      meter.measure([&] {
        integrator
            .request(k, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], cos1_poly[0], cos1_poly[1],
                     cos1_poly[2], cos1_poly[3], cos2_poly[0], cos2_poly[1], cos2_poly[2], cos2_poly[3], phi_poly[0],
                     phi_poly[1], phi_poly[2], phi_poly[3])
            .get();
      });
    };
    BENCHMARK_ADVANCED("get GPU")(Catch::Benchmark::Chronometer meter)
    {
      meter.measure([&] {
        integrator.get(k, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], cos1_poly[0], cos1_poly[1],
                       cos1_poly[2], cos1_poly[3], cos2_poly[0], cos2_poly[1], cos2_poly[2], cos2_poly[3], phi_poly[0],
                       phi_poly[1], phi_poly[2], phi_poly[3]);
      });
    };
    BENCHMARK_ADVANCED("GPU 128x")(Catch::Benchmark::Chronometer meter)
    {
      meter.measure([&] {
        std::vector<std::future<double>> futures;
        for (int i = 0; i < 128; ++i)
          futures.emplace_back(std::move(
              integrator.request(k, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], cos1_poly[0],
                                 cos1_poly[1], cos1_poly[2], cos1_poly[3], cos2_poly[0], cos2_poly[1], cos2_poly[2],
                                 cos2_poly[3], phi_poly[0], phi_poly[1], phi_poly[2], phi_poly[3])));
        for (auto &f : futures)
          f.get();
      });
    };
    BENCHMARK_ADVANCED("get GPU 128x")(Catch::Benchmark::Chronometer meter)
    {
      meter.measure([&] {
        for (int i = 0; i < 128; ++i)
          integrator.get(k, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], cos1_poly[0], cos1_poly[1],
                         cos1_poly[2], cos1_poly[3], cos2_poly[0], cos2_poly[1], cos2_poly[2], cos2_poly[3],
                         phi_poly[0], phi_poly[1], phi_poly[2], phi_poly[3]);
      });
    };
  }
  {
    Integrator4DTBB<double, PolyIntegrand> integrator(quadrature_provider, {{32, 12, 12, 12}}, x_extent);
    BENCHMARK_ADVANCED("CPU")(Catch::Benchmark::Chronometer meter)
    {
      meter.measure([&] {
        integrator
            .request(k, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], cos1_poly[0], cos1_poly[1],
                     cos1_poly[2], cos1_poly[3], cos2_poly[0], cos2_poly[1], cos2_poly[2], cos2_poly[3], phi_poly[0],
                     phi_poly[1], phi_poly[2], phi_poly[3])
            .get();
      });
    };
    BENCHMARK_ADVANCED("get CPU")(Catch::Benchmark::Chronometer meter)
    {
      meter.measure([&] {
        integrator.get(k, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], cos1_poly[0], cos1_poly[1],
                       cos1_poly[2], cos1_poly[3], cos2_poly[0], cos2_poly[1], cos2_poly[2], cos2_poly[3], phi_poly[0],
                       phi_poly[1], phi_poly[2], phi_poly[3]);
      });
    };
    BENCHMARK_ADVANCED("CPU 128x")(Catch::Benchmark::Chronometer meter)
    {
      meter.measure([&] {
        std::vector<std::future<double>> futures;
        for (int i = 0; i < 128; ++i)
          futures.emplace_back(std::move(
              integrator.request(k, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], cos1_poly[0],
                                 cos1_poly[1], cos1_poly[2], cos1_poly[3], cos2_poly[0], cos2_poly[1], cos2_poly[2],
                                 cos2_poly[3], phi_poly[0], phi_poly[1], phi_poly[2], phi_poly[3])));
        for (auto &f : futures)
          f.get();
      });
    };
    BENCHMARK_ADVANCED("get CPU 128x")(Catch::Benchmark::Chronometer meter)
    {
      meter.measure([&] {
        for (int i = 0; i < 128; ++i)
          integrator.get(k, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], cos1_poly[0], cos1_poly[1],
                         cos1_poly[2], cos1_poly[3], cos2_poly[0], cos2_poly[1], cos2_poly[2], cos2_poly[3],
                         phi_poly[0], phi_poly[1], phi_poly[2], phi_poly[3]);
      });
    };
  }
  {
    Integrator4DQMC<double, PolyIntegrand> integrator(x_extent);
    BENCHMARK_ADVANCED("GPU adaptive")(Catch::Benchmark::Chronometer meter)
    {
      meter.measure([&] {
        integrator
            .request(k, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], cos1_poly[0], cos1_poly[1],
                     cos1_poly[2], cos1_poly[3], cos2_poly[0], cos2_poly[1], cos2_poly[2], cos2_poly[3], phi_poly[0],
                     phi_poly[1], phi_poly[2], phi_poly[3])
            .get();
      });
    };
    BENCHMARK_ADVANCED("get GPU adaptive")(Catch::Benchmark::Chronometer meter)
    {
      meter.measure([&] {
        integrator.get(k, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], cos1_poly[0], cos1_poly[1],
                       cos1_poly[2], cos1_poly[3], cos2_poly[0], cos2_poly[1], cos2_poly[2], cos2_poly[3], phi_poly[0],
                       phi_poly[1], phi_poly[2], phi_poly[3]);
      });
    };
    BENCHMARK_ADVANCED("GPU adaptive 128x")(Catch::Benchmark::Chronometer meter)
    {
      meter.measure([&] {
        std::vector<std::future<double>> futures;
        for (int i = 0; i < 128; ++i)
          futures.emplace_back(std::move(
              integrator.request(k, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], cos1_poly[0],
                                 cos1_poly[1], cos1_poly[2], cos1_poly[3], cos2_poly[0], cos2_poly[1], cos2_poly[2],
                                 cos2_poly[3], phi_poly[0], phi_poly[1], phi_poly[2], phi_poly[3])));
        for (auto &f : futures)
          f.get();
      });
    };
    BENCHMARK_ADVANCED("get GPU adaptive 128x")(Catch::Benchmark::Chronometer meter)
    {
      meter.measure([&] {
        for (int i = 0; i < 128; ++i)
          integrator.get(k, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], cos1_poly[0], cos1_poly[1],
                         cos1_poly[2], cos1_poly[3], cos2_poly[0], cos2_poly[1], cos2_poly[2], cos2_poly[3],
                         phi_poly[0], phi_poly[1], phi_poly[2], phi_poly[3]);
      });
    };
  }
}
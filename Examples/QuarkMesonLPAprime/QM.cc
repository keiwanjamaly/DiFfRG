#include <DiFfRG/common/configuration_helper.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/timestepping/timestepping.hh>

#include "model.hh"

using namespace dealii;
using namespace DiFfRG;

// Choices for types
using Model = QuarkMesonLPAprime;
constexpr uint dim = Model::dim;
using Discretization = CG::Discretization<Model::Components, double, RectangularMesh<dim>>;
using VectorType = typename Discretization::VectorType;
using SparseMatrixType = typename Discretization::SparseMatrixType;
using Assembler = CG::Assembler<Discretization, Model>;
using TimeStepper = TimeStepperSUNDIALS_IDA_BoostABM<VectorType, SparseMatrixType, dim, UMFPack>;

int main(int argc, char *argv[])
{
  // declare/get all needed parameters and parse from the CLI
  ConfigurationHelper config_helper(argc, argv);
  const auto json = config_helper.get_json();

  // Define the objects needed to run the simulation
  Model model(json);
  RectangularMesh<dim> mesh(json);
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);
  DataOutput<dim, VectorType> data_out(json);
  TimeStepper time_stepper(json, &assembler, &data_out);

  // Set up the initial condition
  FE::FlowingVariables initial_condition(discretization);
  initial_condition.interpolate(model);

  // Now we start the timestepping
  Timer timer;
  try {
    time_stepper.run(&initial_condition, 0., json.get_double("/timestepping/final_time"));
  } catch (std::exception &e) {
    spdlog::get("log")->error("Simulation finished with exception {}", e.what());
    return -1;
  }

  // We print a bit of exit information.
  assembler.log("log");
  const auto time = timer.wall_time();
  spdlog::get("log")->info("Simulation finished after " + time_format(time));
  return 0;
}

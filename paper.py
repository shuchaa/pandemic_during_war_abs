# project imports
from pandemic_during_war.Simulator import Simulator
from pandemic_during_war.Parameters import parameters

import importlib
import os
import numpy as np
import argparse

# Constants
DOMAIN_RANDOMIZATION_ROUNDS = 5
SIMULATION_STEPS = 100
EVAL_N_RUNS = 3
TRAIN_N_RUNS = 3
POLICIES = ['no_policy', 'random', 'dqn', 'prioritize_soldiers', 'prioritize_civilians']
NO_MODELS_POLICIS = ['no_policy', 'random']
OPTIMIZATION_HORIZONS = [1, 7, 14, 28] # Days

def load_static_parameters(file_path):
    module = importlib.machinery.SourceFileLoader('static_parameters',file_path).load_module()
    return module.static_parameters

def train_and_save_model(model_name, policy, simulation_steps, domain_randomization_rounds, n_runs, static_parameters=None):
    """
    Train the simulator using domain randomization and multiple repetitions.
    """
    templates = []
    simulator = Simulator(number_of_steps=simulation_steps, train=True, policy=policy)
    
    # Generate templates through domain randomization
    for _ in range(domain_randomization_rounds):
        simulator.create_location_graph_template(parameters, static_parameters=static_parameters, save_template=True)
        templates.append(simulator.location_graph_template_uuid)
    
    # Load each template and run simulations
    for template in templates:
        simulator.load_location_graph_template(template)
        simulator.run_multiple_simulations(n_runs=n_runs, make_plots=False)
        
    # Save the trained DQN model
    simulator.dqn_agent.save_model(filename=f'{model_name}.pth')

def train_and_save_required_models(static_parameters=None):
    # Train policies models
    for policy in POLICIES:
        if policy not in NO_MODELS_POLICIS:
            train_and_save_model(model_name = f'{policy}_model', policy = policy,
                                simulation_steps = SIMULATION_STEPS,
                                domain_randomization_rounds = DOMAIN_RANDOMIZATION_ROUNDS,
                                n_runs=TRAIN_N_RUNS, static_parameters=static_parameters
                                )
    # Train opimization horizons models
    for optimization_horizon in OPTIMIZATION_HORIZONS:
        if static_parameters:
            static_parameters_modified = static_parameters.copy()
        else:
            static_parameters_modified = {}

        static_parameters_modified['DISCOUNT_FACTOR'] = round(1 - (1/optimization_horizon),2)

        train_and_save_model(model_name = f'dqn_optimization_horizon_{optimization_horizon}_model', policy = 'dqn',
                            simulation_steps = SIMULATION_STEPS,
                            domain_randomization_rounds = DOMAIN_RANDOMIZATION_ROUNDS,
                            n_runs=TRAIN_N_RUNS, static_parameters=static_parameters
                            )

def evaluate_performance(simulator, n_runs):
    # Execution using random policy
    simulator.policy = 'random'
    simulator.run_multiple_simulations(1, make_plots=False)
    soldier_deaths_random  = [step['soldier_deaths_A'] for step in simulator.simulation_runs_data]
    civilian_deaths_random = [step['civilian_deaths'] for step in simulator.simulation_runs_data]
    # Sum the corresponding elements from the two lists to keep the same dimensions
    avg_random_performance =  np.array([s + c for s, c in zip(soldier_deaths_random, civilian_deaths_random)])

    # Execution using DRL
    simulator.policy = 'dqn'
    simulator.dqn_agent.load_model('dqn_model.pth')
    simulator.dqn_agent.epsilon = 0
    simulator.run_multiple_simulations(n_runs, make_plots=False)
    soldier_deaths_dqn = [step['soldier_deaths_A'] for step in simulator.simulation_runs_data]
    civilian_deaths_dqn = [step['civilian_deaths'] for step in simulator.simulation_runs_data]

    # Sum the corresponding elements from the two lists to keep the same dimensions
    avg_drl_performance = np.array([s + c for s, c in zip(soldier_deaths_dqn, civilian_deaths_dqn)])
    
    # Calculate min and max of array2 (baseline)
    min_baseline = np.min(avg_random_performance)
    max_baseline = np.max(avg_random_performance)
    normalized_performance = (avg_drl_performance - min_baseline) / (max_baseline - min_baseline)

    simulator.plotter.plot_pref_evals(avg_drl_performance, avg_random_performance)
    simulator.plotter.plot_pref_evals_norm(normalized_performance, avg_drl_performance, avg_random_performance)

def evaluate_administration_policies(simulator, n_runs):
    averages_per_policy_mean = {}
    averages_per_policy_std = {}
    
    for policy in POLICIES:
        simulator.policy = policy
        if policy not in NO_MODELS_POLICIS:
            simulator.dqn_agent.load_model(f'{policy}_model.pth')
        simulator.run_multiple_simulations(n_runs, make_plots=False)
        averages_per_policy_mean[simulator.policy] = simulator.simulation_runs_data.copy()
        averages_per_policy_std[simulator.policy] = simulator.simulation_runs_data_std.copy()

    simulator.plotter.plot_deaths_by_patient_administration_policies(averages_per_policy_mean, averages_per_policy_std)

def evaluate_optimization_horizon(simulator, n_runs):
    simulator.policy = 'dqn'
    averages_per_policy_mean = {}
    averages_per_policy_std = {}
    for optimization_horizon in OPTIMIZATION_HORIZONS:
        simulator.dqn_agent.load_model(f'dqn_optimization_horizon_{optimization_horizon}_model.pth')
        simulator.run_multiple_simulations(n_runs, make_plots=False)

        averages_per_policy_mean[optimization_horizon] = simulator.simulation_runs_data.copy()
        averages_per_policy_std[optimization_horizon] = simulator.simulation_runs_data_std.copy()
    simulator.plotter.plot_deaths_by_optimizaion_horizons(averages_per_policy_mean, averages_per_policy_std)

def paper_run(train_models = True,
              location_graph_template_uuid=None,
              static_parameters_file_path = None,
              perform_evaluate_performance=True,
              perform_evaluate_administration_policies=True,
              perform_evaluate_optimization_horizon=True):
    # Load static parameters from file
    if static_parameters_file_path:
        static_parameters = load_static_parameters(static_parameters_file_path)
    else:
        static_parameters = None
    # Train Required models
    if train_models:
        print("Training models")
        train_and_save_required_models(static_parameters)
    # Define Simulator
    print("Creating simulator")
    simulator = Simulator(number_of_steps=SIMULATION_STEPS, train=False)
    if location_graph_template_uuid:
        simulator.load_location_graph_template(location_graph_template_uuid)
    else:
        simulator.create_location_graph_template(parameters, static_parameters=static_parameters,
                                                 save_template=True)
    location_graph_template_uuid = simulator.location_graph_template_uuid
    # Execute simulations
    if perform_evaluate_performance:
        print("Performing performance evaluation.")
        evaluate_performance(simulator, n_runs=EVAL_N_RUNS)
    if perform_evaluate_administration_policies:
        print("Performing administration policies evaluation.")
        evaluate_administration_policies(simulator, n_runs=EVAL_N_RUNS)
    if perform_evaluate_optimization_horizon:
        print("Performing optimization horizons evaluation.")
        evaluate_optimization_horizon(simulator, n_runs=EVAL_N_RUNS)

    print("Plots saved to plots folders.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run the paper with customizable parameters."
    )

    # Boolean flags that invert the default True behavior
    parser.add_argument("--skip-train-models",
                        action="store_true",
                        help="Skip training models. By default, models are trained.")
    parser.add_argument("--skip-evaluate-performance",
                        action="store_true",
                        help="Skip evaluating performance. By default, performance is evaluated.")
    parser.add_argument("--skip-evaluate-administration-policies",
                        action="store_true",
                        help="Skip evaluating administration policies. By default, they are evaluated.")
    parser.add_argument("--skip-evaluate-optimization-horizon",
                        action="store_true",
                        help="Skip evaluating optimization horizon. By default, it is evaluated.")

    # Optional string arguments
    parser.add_argument("--location_graph_template_uuid",
                        type=str,
                        default=None,
                        help="UUID string for the location graph template (default: None).")
    parser.add_argument("--static_parameters_file_path",
                        type=str,
                        default=None,
                        help="Path to the static parameters file (default: None).")

    args = parser.parse_args()

    paper_run(
        train_models=not args.skip_train_models,
        location_graph_template_uuid=args.location_graph_template_uuid,
        static_parameters_file_path=args.static_parameters_file_path,
        perform_evaluate_performance=not args.skip_evaluate_performance,
        perform_evaluate_administration_policies=not args.skip_evaluate_administration_policies,
        perform_evaluate_optimization_horizon=not args.skip_evaluate_optimization_horizon
    )
# library imports
import os
import sys
import time
import uuid
import pickle
import random
import logging
import threading
import numpy as np
from tqdm import tqdm
from colorama import Fore, Style, init

# project imports
from .Agents import State
from .Plotter import Plotter
from .DQNAgent import DQAgent
from .RandomWalk import move_agents_parallel
from .LocationGraph import LocationGraph
from .WarDynamics import run_war_dynamics
from .EpidemicDynamics import update_epidemiological_state

lock = threading.Lock()
# Initialize colorama
init(autoreset=True)

# Configure logging to write only to a file
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('simulation.log')]) # Add stream if you want it to print to the screen for faster debugging , logging.StreamHandler(sys.stdout)

logger = logging.getLogger(__name__)

class Simulator:
    """
    The main class of the project with the simulation logic
    """
    def __init__(self, number_of_steps, train, policy="random"):
        self.number_of_steps = number_of_steps
        # technical members #
        self.simulation_data = [] # To store data for summary plots
        self.dqn_agent = None
        self.location_graph = None
        self.location_graph_template_uuid = None
        self.plotter = None
        self.running_multiple = False
        self.policy = policy
        self.training = train
        self.location_graph_templates_dir = './location_graph_templates'

    def create_location_graph_template(self, parameters, static_parameters=None, save_template=True):
        """create_environment
            returns the template_uuid
        Args:
            static_parameters (dict, optional): Decide on static parameters to create an from).
            save_enviroment (bool, optional): Decide if to save the environement to a file. Defaults to True.
        """
        self.location_graph_template_uuid = uuid.uuid1()
        logger.info(f"Creating simulation template uuid: {self.location_graph_template_uuid}")
        # Choose for each parameter choose the value or keep as a range.
        ranged_parameters = ['POPULATION_SIZE','SOLDIER_PORTION','HOSPITAL_NODES_COUNT','WAR_ZONE_NODES_COUNT','CIVILIAN_ZONE_NODES_COUNT', 'GAMMA_A','GAMMA_S','PSI','BETA','RHO','LAMBDA','BASELINE_RECOVERY_RATE','OVERFLOW_RECOVERY_RATE','epsilon_a','epsilon_b','theta_a','theta_b']
        integers_parameters = ['POPULATION_SIZE', 'HOSPITAL_NODES_COUNT', 'WAR_ZONE_NODES_COUNT', 'CIVILIAN_ZONE_NODES_COUNT', 'GAMMA_A','GAMMA_S','PSI', 'BATCH_SIZE', 'SEQUENCE_LENGTH']
        location_graph_template_parameters = parameters.copy()
        
        for param in ranged_parameters:
            if param in integers_parameters:
                location_graph_template_parameters[param] = random.randint(*parameters[param])
            else:
                location_graph_template_parameters[param] = random.uniform(*parameters[param])
                
        if static_parameters:
            for key, value in static_parameters.items():
                location_graph_template_parameters[key] = value
        
        self.location_graph = None
        max_retries = 3
        for i in range(max_retries):
            try:
                self.location_graph = LocationGraph(location_graph_template_parameters)
                break
            except RuntimeError as e:
                print(f"Error creting location graph, retry attempt {i} : {e}")
        if self.location_graph is None:
            raise RuntimeError(f"Failed to create loacation graph after {max_retries} retries.")
            
        self.location_graph.create_and_allocate_agents()
        
        if save_template:
            path = os.path.join(self.location_graph_templates_dir,f'{self.location_graph_template_uuid}.pkl')
            try:
                os.makedirs(self.location_graph_templates_dir, exist_ok=True) # Create the directory if it doesn't exist
                with open(path, 'wb') as f:
                    pickle.dump(self.location_graph, f)
                logger.info(f"Location graph saved successfully to path: {path}")
            except Exception as error:
                logger.error(f"Was not able to save location graph to path {path}, saying: {error}")
                
        self.plotter = Plotter(simulator=self)
        if self.dqn_agent is None:
            self.dqn_agent = DQAgent(self.location_graph, logger=logger)
        else:
            self.dqn_agent.set_location_graph(self.location_graph)
        
    def load_location_graph_template(self, template_uuid):
        logger.info(f"Loading location graph template uuid: {template_uuid}")
        path = os.path.join(self.location_graph_templates_dir,f'{template_uuid}.pkl')
        with open(path, 'rb') as f:
            self.location_graph = pickle.load(f)
            self.location_graph_template_uuid = template_uuid
            self.plotter = Plotter(simulator = self)
            if self.dqn_agent is None:
                self.dqn_agent = DQAgent(self.location_graph, logger=logger)
            else:
                self.dqn_agent.set_location_graph(self.location_graph)
        
    def reset_environment(self):
        self.load_location_graph_template(self.location_graph_template_uuid)

    def run_simulation(self, simulation_id=0):
        for step in tqdm(range(self.number_of_steps), desc=f"{Fore.GREEN}Simulation Progress{Style.RESET_ALL}", unit="step", colour="green"):
            self.run_step(step, simulation_id)

        # Check if running multiple simulations; if not, plot immediately
        if not hasattr(self, 'running_multiple') or not self.running_multiple:
            self.plotter.simulation_data = self.simulation_data # Pass collected data to plotter
            self.plotter.plot_summary()
    
    def run_multiple_simulations(self, n_runs, make_plots = True):
        """Runs the simulation multiple times to gather data for summary plots."""
        self.simulation_runs_data = []
        self.simulation_runs_data_std = []
        self.running_multiple = True
        self.all_runs_data = [[] for _ in range(self.number_of_steps+1)]
        for simulation_id in tqdm(range(n_runs), desc=f"{Fore.BLUE}Multiple Simulations Progress{Style.RESET_ALL}", unit="run", colour="blue"):
            self.simulation_data = [] # Reset step data for each run
            self.run_simulation(simulation_id)
            # self.collect_step_data()
            for step_index, step_data in enumerate(self.simulation_data):
                self.all_runs_data[step_index].append(step_data)
            self.reset_environment()
            
        # Calculate averages across all runs
        self.simulation_runs_data = Simulator.calculate_averages(self.all_runs_data, avg_type='mean')
        if n_runs > 1:
            self.simulation_runs_data_std = Simulator.calculate_averages(self.all_runs_data, avg_type='std')
        
        if make_plots:
            # After all runs are complete, pass the aggregated data to the plotter and plot summary
            self.plotter.simulation_data = self.simulation_runs_data
            self.plotter.raw_data = self.all_runs_data
            self.plotter.plot_summary()
        self.running_multiple = False
        
    def run_step(self, step, simulation_id):
        if step == 0:
            # collect the step data for initial graph allocations     
            step_data = self.collect_step_data()
            self.simulation_data.append(step_data)
            
        logger.info(f"Starting step {step + 1} of {self.number_of_steps} "
                f"({round((step + 1) * 100 / self.number_of_steps, 2)}%)")
        
        # Update agent location in parallel
        move_agents_parallel(self.location_graph, logger)
        logger.info(f"Step {step + 1}: Agents moved")
        
        # Update epidemiological state of agents in parallel
        update_epidemiological_state(self.location_graph) 
        logger.info(f"Step {step + 1}: pandemic spread")
        
        # Update war dynamics in parallel
        run_war_dynamics(self.location_graph)
        logger.info(f"Step {step + 1}: war dynamics")
        
        if self.policy != 'no_policy':
            # Update hospital dynamics 
            self.dqn_agent.allocate_to_hospitals(self.policy, self.training)
            logger.info(f"Step {step + 1}: hospitaization")
        
        # collect the step data for later plotting       
        step_data = self.collect_step_data()
        self.simulation_data.append(step_data)
        logger.info(f"Simulation data collected, dumping...")
        # Save all simulation data for plots
        if step == self.number_of_steps:
            self.dump_simulation_data(simulation_id, step)
        
        logger.info(f"Step {step + 1} completed")
            
    def dump_simulation_data(self, simulation_id, step):
        time_as_int = int(time.time())
        path = f'./simulation_data/{time_as_int}_{self.policy}_{simulation_id}_{step}.pkl'
        directory = os.path.dirname(path) # Extract the directory path from the file path
        try:
            os.makedirs(directory, exist_ok=True) # Create the directory if it doesn't exist
            with open(path, 'wb') as f:
                pickle.dump(self.simulation_data, f)
            logger.info(f"Simulation data saved successfully to path: {path}")
        except Exception as error:
            logger.error(f"Was not able to save location Simulation data to path {path}, saying: {error}")    
        
    def animate_simulation(self):
        if self.location_graph.parameters['WAR_ZONE_NODES_COUNT'] > 50 or self.location_graph.parameters['CIVILIAN_ZONE_NODES_COUNT'] > 50:
            logger.error("You can't use animate_simulation with more than 50 nodes per zone type")
            sys.exit(1)
        self.plotter.animate_simulation(self.number_of_steps)
        self.plotter.simulation_data = self.simulation_data # Pass collected data to plotter
        self.plotter.plot_summary()
        
    def collect_step_data(self):
        """Collects data at each step for summary plots."""
        data = {
            'alive_civilians': 0,
            'alive_soldiers': 0,
            'hospitalized_civilians': 0,
            'hospitalized_soldiers': 0,
            'civilian_deaths': 0,
            'soldier_deaths': 0,
            'hospital_capacity': 0,
            'commute_rate_c': 0,
            'commute_rate_w': 0,
            'army_A_size': 0,
            'army_A_injured': 0,
            'army_B_size': 0,
            'army_B_injured': 0,
            'reward': 0,
            'hospital_occupancy_total': 0,
            'hospital_deaths': 0,
            'hospital_recovered': 0,
            'overflow_capacity': 0,
            'loss':0
        }
        data['loss'] += self.dqn_agent.loss 
        for state in State:
            data[state.name.lower()] = 0
        graph = self.location_graph.graph
        for node in graph.nodes:
            population = graph.nodes[node]['population']
            data['alive_civilians'] += population.get_agent_counts_by_criteria(agent_types='civilian')
            data['alive_soldiers'] += population.get_agent_counts_by_criteria(agent_types='soldier')
            for state in State:
                data[state.name.lower()] += population.get_agent_counts_by_criteria(states=state)
                        
            data['army_A_size'] += population.get_agent_counts_by_criteria(armies='A')
            data['army_B_size'] += population.get_agent_counts_by_criteria(armies='B')
            data['army_A_injured'] += population.get_agent_counts_by_criteria(armies='A', is_injured=True)
            data['army_B_injured'] += population.get_agent_counts_by_criteria(armies='B', is_injured=True)
                
            if graph.nodes[node]['is_hospital']:
                data['hospitalized_civilians'] += population.get_agent_counts_by_criteria(agent_types='civilian')
                
                data['hospitalized_soldiers'] += population.get_agent_counts_by_criteria(agent_types='soldier', armies='A')
                data['hospital_capacity'] += graph.nodes[node]['hospital_capacity']
                data['overflow_capacity'] += graph.nodes[node]['overflow_capacity']
                
                data['hospital_occupancy_total'] += population.get_agent_counts_by_criteria()
                if 'total_dead' in graph.nodes[node]:
                    data['hospital_deaths'] += graph.nodes[node]['total_dead']
                if 'total_recovered' in graph.nodes[node]:
                    data['hospital_recovered'] += graph.nodes[node]['total_recovered']
                    
                data['commute_rate_w'] = sum(graph[node][neighbor]['cw'] for neighbor in graph.neighbors(node))
                data['commute_rate_c'] = sum(graph[node][neighbor]['cc'] for neighbor in graph.neighbors(node))
                    
        # Since DEAD agents are being removed from every node's population, in order to calculate the death
        # we need to substract the alive ones from the initial agents amount when the simulation started.
        entire_population_size = self.location_graph.parameters['POPULATION_SIZE']
        entire_population_soldiers_amount = int(entire_population_size * self.location_graph.parameters['SOLDIER_PORTION'])
        
        data['civilian_deaths'] = entire_population_size - entire_population_soldiers_amount - data['alive_civilians']
        data['soldier_deaths'] = entire_population_soldiers_amount - data['alive_soldiers']
        data['soldier_deaths_A'] = self.location_graph.initialized_armies_sizes['A'] - data['army_A_size']
        data['soldier_deaths_B'] = self.location_graph.initialized_armies_sizes['B'] - data['army_B_size']
        data['deaths'] = data['civilian_deaths'] + data['soldier_deaths_A'] # We don't care about army B
        data['total_deaths_with_army_B'] = data['civilian_deaths'] + data['soldier_deaths']
        data['reward'] = self.dqn_agent.avg_reward
        return data
    
    @staticmethod
    def calculate_averages(all_runs_data, avg_type):
        """Calculate average values across multiple simulation runs."""
        avg_data = []

        # Loop through each list of step data (one list per step)
        for step_data in all_runs_data:
            step_avg = {}
            # Calculate the mean for each key across all runs for the current step
            for key in step_data[0].keys():
                if avg_type == 'mean':
                    step_avg[key] = np.mean([data[key] for data in step_data])
                if avg_type == 'std':
                    step_avg[key] = np.std([data[key] for data in step_data])
            avg_data.append(step_avg)

        return avg_data

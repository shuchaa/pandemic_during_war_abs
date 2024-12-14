# library imports
import random
from concurrent.futures import ThreadPoolExecutor

# project imports
from .Agents import State

def update_epidemiological_state(location_graph, num_threads=120):
    # Extract parameters
    beta = location_graph.parameters['BETA']
    rho = location_graph.parameters['RHO']
    lambda_ = location_graph.parameters['LAMBDA']
    psi = location_graph.parameters['PSI'] # In order to convert days to hours multiply by 24
    gamma_a = location_graph.parameters['GAMMA_A'] # In order to convert days to hours multiply by 24
    gamma_s = location_graph.parameters['GAMMA_S'] # In order to convert days to hours multiply by 24
    time_step = location_graph.parameters['TIME_STEP']
    
    def process_node(node):
        population = location_graph.graph.nodes[node]['population']
        population_size = population.get_agent_counts_by_criteria()
        infected_count = population.get_agent_counts_by_criteria(states=[State.ASYMPTOMATIC_INFECTED, 
                                                                         State.SYMPTOMATIC_INFECTED])
    
        # Update the states of susceptible agents
        for agent in population.get_agents_by_criteria():
            should_reset_inner_clock = False
            if agent.state == State.SUSCEPTIBLE and infected_count > 0 and random.random() < beta * infected_count / population_size:
                agent.state = State.EXPOSED
                should_reset_inner_clock = True
            elif agent.state == State.EXPOSED and agent.inner_clock >= psi:
                if random.random() > rho:
                    agent.state = State.SYMPTOMATIC_INFECTED
                else:
                    agent.state = State.ASYMPTOMATIC_INFECTED
                should_reset_inner_clock = True
            elif agent.state == State.ASYMPTOMATIC_INFECTED and agent.inner_clock >= gamma_a:
                agent.state = State.RECOVERED
                should_reset_inner_clock = True
            elif agent.state == State.SYMPTOMATIC_INFECTED and agent.inner_clock >= gamma_s:
                if random.random() < lambda_:
                    agent.state = State.DEAD
                    should_reset_inner_clock = True
                else:
                    agent.state = State.RECOVERED
                    should_reset_inner_clock = True
                
            if should_reset_inner_clock:
                agent.inner_clock = 0
            else:
                agent.inner_clock += time_step
                
    # Use ThreadPoolExecutor to run the process_node function in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(process_node, location_graph.non_hospital_nodes)

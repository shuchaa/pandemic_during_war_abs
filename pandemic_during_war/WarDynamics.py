# project imports
import random
from concurrent.futures import ThreadPoolExecutor

# library imports
from .Agents import State

def process_war_node(node, location_graph):
    population = location_graph.graph.nodes[node]['population']
    
    # Get army sizes and wounded counts for the node's population
    A = population.get_agent_counts_by_criteria(armies='A')
    B = population.get_agent_counts_by_criteria(armies='B')
    injured_A = population.get_agent_counts_by_criteria(armies='A', is_injured=True)
    injured_B = population.get_agent_counts_by_criteria(armies='B', is_injured=True)
    
    # Check if both army sizes are greater than zero
    if A > 0 and B > 0:
        theta_a = location_graph.parameters['theta_a']
        theta_b = location_graph.parameters['theta_b']
        epsilon_a = location_graph.parameters['epsilon_a']
        epsilon_b = location_graph.parameters['epsilon_b']
        d = location_graph.parameters['d']
        
        # Calculate the casualties inflicted by each army
        casualties_A = int(epsilon_b * B)
        casualties_B = int(epsilon_a * A)
        
        # Calculate the number of soldiers who are wounded rather than killed
        wounded_A = int(theta_a * casualties_A)
        wounded_B = int(theta_b * casualties_B)
        
        # The remaining casualties are soldiers who are killed directly
        dead_A_initial = casualties_A - wounded_A
        dead_B_initial = casualties_B - wounded_B
        
        # Process deaths due to lack of treatment for wounded soldiers
        dead_A_from_wounds = int(d * injured_A)
        dead_B_from_wounds = int(d * injured_B)
        
        injured_soldiers_A = population.get_agents_by_criteria(armies='A', is_injured=True)
        if dead_A_from_wounds > 0:
            for agent in random.sample(injured_soldiers_A, min(dead_A_from_wounds, len(injured_soldiers_A))):
                agent.state = State.DEAD

        injured_soldiers_B = population.get_agents_by_criteria(armies='B', is_injured=True)
        if dead_B_from_wounds > 0:
            for agent in random.sample(injured_soldiers_B, min(dead_B_from_wounds, len(injured_soldiers_B))):
                agent.state = State.DEAD

        # Update injured soldiers for Army A
        soldiers_A = population.get_agents_by_criteria(armies='A', is_injured=False)
        if wounded_A > 0:
            for agent in random.sample(soldiers_A, min(wounded_A, len(soldiers_A))):
                agent.is_injured = True
        
        # Update dead soldiers for Army A directly from battle
        if dead_A_initial > 0:
            for agent in random.sample(soldiers_A, min(dead_A_initial, len(soldiers_A))):
                agent.state = State.DEAD
        
        # Update injured soldiers for Army B
        soldiers_B = population.get_agents_by_criteria(armies='B', is_injured=False)
        if wounded_B > 0:
            for agent in random.sample(soldiers_B, min(wounded_B, len(soldiers_B))):
                agent.is_injured = True
        
        # Update dead soldiers for Army B directly from battle
        if dead_B_initial > 0:
            for agent in random.sample(soldiers_B, min(dead_B_initial, len(soldiers_B))):
                agent.state = State.DEAD

def run_war_dynamics(location_graph, num_threads=120):
    war_nodes = [node for node in location_graph.non_hospital_nodes if location_graph.graph.nodes[node]['zone'] == 'war']
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit each node processing to the thread pool
        executor.map(lambda node: process_war_node(node, location_graph), war_nodes)
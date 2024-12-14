# library imports
import sys
import random
from threading import Lock
import math
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

def normalize_probabilities(probabilities):
    """Normalize a list of probabilities."""
    total_prob = sum(probabilities) or 1.0 # Avoid division by zero
    return [p / total_prob for p in probabilities]

def allocate_movement(probabilities, total_agents):
    """Calculate movement counts based on normalized probabilities and total agents."""
    move_counts = [int(math.floor(p * total_agents)) for p in probabilities]
    adjustment = total_agents - sum(move_counts)

    # Adjust the last move count to ensure the total matches the expected count
    if move_counts:
        move_counts[-1] += adjustment
    assert sum(move_counts) <= total_agents, "move_counts higher than total agents."
    return move_counts

def calculate_movement_decisions(location_graph, num_threads=120):
    # Prepare a dictionary to keep track of movement plans
    movement_decisions = {node: {'civilian': {}, 'soldier': {}} for node in location_graph.graph.nodes}
    lock = Lock() # Lock to ensure thread safety when updating movement_decisions

    def process_node(node):
        # Get the lists of civilian and soldier agents once with a deep copy to not modify the original list
        node_civilians = location_graph.graph.nodes[node]['population'].get_agents_by_criteria(agent_types='civilian').copy()
        node_soldiers = location_graph.graph.nodes[node]['population'].get_agents_by_criteria(agent_types='soldier').copy()
        neighbors = list(location_graph.graph.neighbors(node))
        
        # Initialize lists for neighbors and commute rates
        neighbors_civilian = []
        neighbors_soldiers = []
        civilian_probs = []
        soldier_probs = []
        
        # Iterate over neighbors once
        for neighbor in neighbors:
            neighbor_data = location_graph.graph.nodes[neighbor]
            
            # Check for civilians: not a hospital node and not in a war zone
            if not neighbor_data.get('is_hospital', False) and neighbor_data.get('zone', '') != 'war':
                neighbors_civilian.append(neighbor)
                civilian_probs.append(location_graph.graph[node][neighbor]['cc'])
            
            # Check for soldiers: not a hospital node
            if not neighbor_data.get('is_hospital', False):
                neighbors_soldiers.append(neighbor)
                soldier_probs.append(location_graph.graph[node][neighbor]['cw'])
        
        # Normalize probabilities for civilians and soldiers
        civilian_probs = normalize_probabilities(civilian_probs)
        soldier_probs = normalize_probabilities(soldier_probs)

        # Get the current population counts at the node
        civilian_count = len(node_civilians)
        soldier_count = len(node_soldiers)
        
        # Calculate movement decisions for civilians
        civilian_moves = allocate_movement(civilian_probs, civilian_count)
        with lock:
            for neighbor, move_count in zip(neighbors_civilian, civilian_moves):
                movement_decisions[node]['civilian'][neighbor] = []
                for _ in range(move_count):
                    agent = node_civilians.pop(random.randint(0, len(node_civilians) - 1)) 
                    movement_decisions[node]['civilian'][neighbor].append(agent)

        # Calculate movement decisions for soldiers
        soldier_moves = allocate_movement(soldier_probs, soldier_count)
        with lock:
            for neighbor, move_count in zip(neighbors_soldiers, soldier_moves):
                movement_decisions[node]['soldier'][neighbor] = []
                for _ in range(move_count):
                    agent = node_soldiers.pop(random.randint(0, len(node_soldiers) - 1))  
                    movement_decisions[node]['soldier'][neighbor].append(agent)
                
        # Debugging output for mismatches
        total_moved_civilians = sum(civilian_moves)
        total_moved_soldiers = sum(soldier_moves)
        
        if total_moved_civilians != civilian_count:
            print(f"Warning: Mismatch in civilian movement at node {node}. Total moved: {total_moved_civilians}, Expected: {civilian_count}")

        if total_moved_soldiers != soldier_count:
            print(f"Warning: Mismatch in soldier movement at node {node}. Total moved: {total_moved_soldiers}, Expected: {soldier_count}")

    # Use ThreadPoolExecutor to process nodes in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(process_node, location_graph.graph.nodes)
    
    return movement_decisions

def move_agents_serial(location_graph, logger):
    """Perform the movement of agents based on movement decisions."""
    # Calculate movement decisions for all nodes
    logger.info('calculate_movement_decisions start')
    movement_decisions = calculate_movement_decisions(location_graph)
    # Set to track agents that have already moved
    moved_agents = set()
    # Iterate over each node's movement decisions
    logger.info('moving agents start')
    for current_node, agent_types in movement_decisions.items():
        num_moved = 0
        for agent_type, destinations in agent_types.items():
            for destination_node, agents in destinations.items():
                if current_node == destination_node:
                    continue
                # Move each agent to the new location
                for agent in agents:
                    if agent in moved_agents:
                        print(f"Error: Agent {agent} has already moved once in this move_agents call.")
                        sys.exit(1) # Raising exception in thread won't kill the program
                    else: 
                        agent.move_agent(destination_node) # Move the agent to the new node
                        num_moved += 1
    logger.info('moving agents end')
     
def move_agents_parallel(location_graph, logger, num_threads = 120):
    """Perform the movement of agents based on movement decisions in parallel per destination node."""
    logger.info('[move_agents_parallel] calculate_movement_decisions start')
    movement_decisions = calculate_movement_decisions(location_graph)
    moved_agents = set() # Set to track agents that have already moved
    node_locks = defaultdict(Lock) # Use defaultdict to create locks on demand

    def move_agents_batch(current_node, destination_node, agents):
        """Move agents from current_node to destination_node with batch processing and reduced locking."""
        # Ensure locks are always acquired in a consistent order to avoid deadlock
        first_node, second_node = sorted([current_node, destination_node])
        first_lock = node_locks[first_node]
        second_lock = node_locks[second_node]

        # Batch move all agents with a single lock acquisition
        with first_lock, second_lock:
            for agent in agents:
                if agent in moved_agents:
                    logger.error(f"Error: Agent {agent} has already moved once in this move_agents call.")
                    continue # Skip this agent to prevent duplicates
                agent.move_agent(destination_node) # Efficient move operation
                moved_agents.add(agent)

    logger.info('[move_agents_parallel] moving agents start')

    # Collect all movement tasks to shuffle
    tasks = []
    for current_node, agent_types in movement_decisions.items():
        for agent_type, destinations in agent_types.items():
            for destination_node, agents in destinations.items():
                if current_node != destination_node:
                    tasks.append((current_node, destination_node, agents))

    # Shuffle tasks to reduce lock contention on the same source node
    random.shuffle(tasks)

    # Create a ThreadPoolExecutor to manage parallel execution with a specified max number of threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit shuffled tasks to the executor
        futures = [executor.submit(move_agents_batch, current_node, destination_node, agents) 
                   for current_node, destination_node, agents in tasks]

        # Wait for all tasks to complete
        for future in futures:
            future.result() # Ensures that exceptions are raised if they occur in the threads

    logger.info('[move_agents_parallel] moving agents end')
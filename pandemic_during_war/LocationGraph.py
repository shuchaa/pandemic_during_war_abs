# library imports
import math
import random
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
import sys
# project imports
from .Agents import State, Agent
from .Population import Population

class LocationGraph:
    def __init__(self, parameters):
        self.parameters = parameters
        self.graph = nx.Graph()
        self.num_locations = self.parameters['CIVILIAN_ZONE_NODES_COUNT'] + self.parameters['WAR_ZONE_NODES_COUNT']
        # Define zones
        self.nodes_zones_allocation = ['civilian'] * self.parameters['CIVILIAN_ZONE_NODES_COUNT'] + ['war'] * self.parameters['WAR_ZONE_NODES_COUNT']
        self.initialized_armies_sizes = {'A':0, 'B':0}
        hospital_nodes_count = self.parameters['HOSPITAL_NODES_COUNT']
        # Choose random nodes to be a hospitals based on the proportion
        self.hospital_nodes = random.sample(list(range(len(self.nodes_zones_allocation))), int(hospital_nodes_count))
        
        # Define non-hospital locations
        self.non_hospital_nodes = [i for i in range(self.num_locations) if i not in self.hospital_nodes]
        
        self.valid_civilian_allocation_nodes = []
        self.valid_soldier_allocation_nodes = []
        
        for i in range(self.num_locations):
            self.graph.add_node(i,                                 
                                zone=self.nodes_zones_allocation[i], 
                                is_hospital=(i in self.hospital_nodes),
                                population=Population() # should match on agent.population_str
                                )
            # If it's a hospital, add the hospital_capacity attribute
            if i in self.hospital_nodes:
                hospital_capacity = int(random.uniform(*self.parameters['HOSPITAL_CAPACITY_RANGE']))
                self.graph.nodes[i]['hospital_capacity'] = hospital_capacity
                overflow_percent = random.uniform(*self.parameters['OVERFLOW_PERCENT'])
                self.graph.nodes[i]['overflow_capacity'] = int(hospital_capacity*overflow_percent)

            if i in self.non_hospital_nodes:
                if self.nodes_zones_allocation[i] == 'civilian':
                    self.valid_civilian_allocation_nodes.append(i)
                else:
                    self.valid_soldier_allocation_nodes.append(i)
        self.add_edges()
    
    def add_edges(self):
        num_edges = int((self.num_locations * math.sqrt(self.num_locations)) / 2)
        
        # Add self-edges for each node
        for i in range(self.num_locations):
            self.graph.add_edge(i, i, cw=random.uniform(*self.parameters['COMMUTE_RATE_W']), 
                                cc=random.uniform(*self.parameters['COMMUTE_RATE_C']))
        
        # Initialize a set to keep track of added edges to avoid duplicates
        added_edges = set((i, i) for i in range(self.num_locations))
        
        # Separate hospital and non-hospital nodes
        hospital_nodes = [i for i in range(self.num_locations) if self.graph.nodes[i].get('is_hospital')]
        non_hospital_nodes = [i for i in range(self.num_locations) if not self.graph.nodes[i].get('is_hospital')]
        
        # Check if the number of hospital nodes is greater than the number of non-hospital nodes
        if len(hospital_nodes) > len(non_hospital_nodes):
            raise ValueError("The number of hospital nodes cannot be greater than the number of non-hospital nodes.")
        
        # Ensure all non-hospital nodes are connected
        connected_nodes = set()
        if non_hospital_nodes:
            connected_nodes.add(non_hospital_nodes[0])
            stack = [non_hospital_nodes[0]]
            
            while stack:
                u = stack.pop()
                for v in non_hospital_nodes:
                    if v not in connected_nodes and self._is_valid_edge(u, v):
                        self.graph.add_edge(u, v, cw=random.uniform(*self.parameters['COMMUTE_RATE_W']), 
                                            cc=random.uniform(*self.parameters['COMMUTE_RATE_C']))
                        added_edges.add((u, v))
                        added_edges.add((v, u))
                        connected_nodes.add(v)
                        stack.append(v)
        
        # Connect hospital nodes to non-hospital nodes without restricting to one connection
        max_retries = 100 # Set a maximum number of retries to avoid infinite loops
        for h in hospital_nodes:
            retries = 0
            while retries < max_retries:
                u = random.choice(non_hospital_nodes)
                if self._is_valid_edge(h, u, hospital_connection=True):
                    self.graph.add_edge(h, u, cw=random.uniform(*self.parameters['COMMUTE_RATE_W']), 
                                        cc=random.uniform(*self.parameters['COMMUTE_RATE_C']))
                    added_edges.add((h, u))
                    added_edges.add((u, h))
                retries += 1

        # Randomly add edges until the desired number of edges is reached
        while len(added_edges) // 2 < num_edges:
            u = random.randint(0, self.num_locations - 1)
            v = random.randint(0, self.num_locations - 1)
            if u != v and (u, v) not in added_edges and (v, u) not in added_edges:
                if self._is_valid_edge(u, v):
                    self.graph.add_edge(u, v, cw=random.uniform(*self.parameters['COMMUTE_RATE_W']), 
                                        cc=random.uniform(*self.parameters['COMMUTE_RATE_C']))
                    added_edges.add((u, v))
                    added_edges.add((v, u))

    def _is_valid_edge(self, u, v, hospital_connection=False):
        # Check if both nodes are in the same war zone if one of them is a hospital
        node_u = self.graph.nodes[u]
        node_v = self.graph.nodes[v]
        
        # If the edge involves a hospital, ensure the connection rules
        if hospital_connection:
            if node_u['is_hospital'] and node_v['is_hospital']:
                return False
            if node_u['is_hospital'] and node_u['zone'] == 'war':
                return node_v['zone'] == 'war'
            if node_v['is_hospital'] and node_v['zone'] == 'war':
                return node_u['zone'] == 'war'
        
        return True

    def create_and_allocate_agents(self, num_threads=32):
        num_soldiers = int(self.parameters['POPULATION_SIZE'] * self.parameters['SOLDIER_PORTION'])
        infected_indices = set(random.sample(
            range(self.parameters['POPULATION_SIZE']), 
            int(self.parameters['POPULATION_SIZE'] * self.parameters['INFECTED_PROPORTION'])))
        
        # Split infected indices into symptomatic and asymptomatic
        infected_indices = list(infected_indices)
        half_length = len(infected_indices) // 2
        symptomatic_indices = set(infected_indices[:half_length])
        asymptomatic_indices = set(infected_indices[half_length:])
        
        def create_agent(i):
            if i < num_soldiers:
                agent_type = 'soldier'
                # The nodes are ordered so that civilian zone is first and after the nodes in warzone
                army = random.choice(['A', 'B'])
                self.initialized_armies_sizes[army] += 1
                location = random.choice(self.valid_soldier_allocation_nodes)
            else:
                agent_type = 'civilian'
                army = None
                location = random.choice(self.valid_civilian_allocation_nodes)
            # Randomly set 10% of agents to be SYMPTOMATIC_INFECTED
            # Set state based on infection status
            if i in symptomatic_indices:
                state = State.SYMPTOMATIC_INFECTED
            elif i in asymptomatic_indices:
                state = State.ASYMPTOMATIC_INFECTED
            else:
                state = State.SUSCEPTIBLE
            agent = Agent(agent_type=agent_type, location=location, 
                          state=state, army=army, location_graph=self)
            if self.graph.nodes[location]['is_hospital']:
                sys.exit(1) # Raising exception in a thread won't kill the program
                
            self.graph.nodes[location]['population'].add_agent(agent)
            
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            executor.map(create_agent, range(self.parameters['POPULATION_SIZE']))

# library imports
import random
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
# project imports
from .Agents import State

# Replay buffer to store experience
class ReplayBuffer:
    def __init__(self, buffer_size, sequence_length, batch_size, state_size):
        self.buffer = deque(maxlen=buffer_size) # Stores individual transitions
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.state_size = state_size

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample_sequences(self):
        # Sample sequences of transitions from the replay buffer
        sequences = []
        for _ in range(self.batch_size):
            if len(self.buffer) < self.sequence_length:
                break # Not enough data to form a sequence
            # Randomly select a starting index for the sequence
            start_idx = random.randint(0, len(self.buffer) - self.sequence_length)
            # Extract the sequence of transitions of length `sequence_length`
            sequence = list(self.buffer)[start_idx:start_idx + self.sequence_length]
            # Unpack the sequence into its components
            states, actions, rewards, next_states, dones = zip(*sequence)
            # Convert lists of states and next_states into tensors
            states = torch.stack(states) 
            next_states = torch.stack(next_states)
            
            # Remove the extra dimensions by squeezing
            states = states.squeeze(1).squeeze(1)
            next_states = next_states.squeeze(1).squeeze(1) 
            
            new_sequence_length = self.sequence_length // self.batch_size
            assert states.size(0) % new_sequence_length == 0, "The new sequence length must evenly divide the total number of sequences."

            # Reshape the tensor to [batch_size, sequence_length, input_size]
            states = states.view(self.batch_size, new_sequence_length, self.state_size) # required shape of [3,4,2400]
            next_states = next_states.view(self.batch_size, new_sequence_length, self.state_size) # required shape of [3,4,2400]

            # Convert actions to a tensor and keep the correct shape (sequence_length, num_nodes, num_hospitals, 2)
            actions_tensor = torch.stack(actions) # Shape: (sequence_length, num_nodes, num_hospitals, 2)
            # Extract the last action of the sequence to be used in replay
            actions_last = actions_tensor[-1] # Shape: (num_nodes, num_hospitals, 2)
            sequences.append((states, actions_last, rewards[-1], next_states, dones[-1]))
        return sequences
    
    def __len__(self):
        return len(self.buffer)

class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64, num_layers=2):
        super(DQNetwork, self).__init__()
        # Replace hidden layers with RNN layers
        self.lstm = nn.LSTM(state_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        # Output layer dimensions: num_nodes * num_hospitals * 2 (civilians and soldiers)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, state, hidden_state):
        # Forward pass through the LSTM
        lstm_out, hidden_state = self.lstm(state, hidden_state)

        # Take the output from the last time step of the sequence
        lstm_out = lstm_out[:, -1, :] # Shape becomes (7, 64)
        x = torch.relu(self.fc1(lstm_out))
        # Pass through the fully connected layer to match the action size
        logits = self.fc2(x) # Shape: (batch_size, num_nodes * num_hospitals * 2)
        
        return logits, hidden_state

class DQAgent:
    def __init__(self, location_graph, logger):
        self.location_graph = location_graph
        self.logger = logger
        parameters = self.location_graph.parameters
        self.parameters = parameters
        self.gamma = parameters['DISCOUNT_FACTOR']
        self.epsilon = parameters['EXPLORATION_RATE']
        self.epsilon_decay = parameters['EXPLORATION_DEDAY_RATE']
        self.epsilon_min = parameters['MIN_EXPLORATION_RATE']
        self.num_non_hospital_nodes = len(self.location_graph.non_hospital_nodes)
        self.num_hospital_nodes = len(self.location_graph.hospital_nodes)
        self.num_total_nodes = self.num_non_hospital_nodes + self.num_hospital_nodes
        self.sequence_length = parameters['SEQUENCE_LENGTH']
        self.batch_size = parameters['BATCH_SIZE']
        self.state_size = int(self.num_total_nodes * 20) # Number of nodes times the total features
        self.action_size = int(self.num_non_hospital_nodes * self.num_hospital_nodes * 2) # Number of hospital nodes times the non hospital nodes times 2 (civilian and soldier)
        self.replay_buffer = ReplayBuffer(parameters['REPLAY_BUFFER_SIZE'], self.sequence_length, self.batch_size, self.state_size)
        self.target_update_freq = parameters['TARGET_UPDATE_FREQ'] 
        self.episodes = parameters['EPISODES_SIZE']  
        self.hidden_size = parameters['HIDDEN_SIZE'] # Number of features in the hidden state of LSTM
        self.num_layers = parameters['NUM_LAYERS'] # Number of stacked LSTM layers
        self.lr = parameters['LEARNING_RATE']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.baseline_recovery_rate = parameters['BASELINE_RECOVERY_RATE']
        self.overflow_recovery_rate = parameters['OVERFLOW_RECOVERY_RATE']
        # Initialize the hidden state
        self.reset_hidden()
        self.avg_reward = 0
        self.policy_network = DQNetwork(self.state_size, self.action_size, self.hidden_size, self.num_layers).to(self.device)
        self.target_network = DQNetwork(self.state_size, self.action_size, self.hidden_size, self.num_layers).to(self.device)
        self.loss = 0
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.update_target_network()
        self.output_directory = './models'
        os.makedirs(self.output_directory, exist_ok=True) # Create the directory if it doesn't exist
        
    def save_model(self, filename="dqn_model.pth"):
        # Create a checkpoint dictionary with all necessary components
        checkpoint = {
            'policy_network_state_dict': self.policy_network.state_dict(),  # Save the policy network's weights
            'target_network_state_dict': self.target_network.state_dict(),  # Save the target network's weights
            'optimizer_state_dict': self.optimizer.state_dict(),            # Save the optimizer's state
            'epsilon': self.epsilon,                                        # Save the current epsilon value
            'avg_reward': self.avg_reward,                                  # Save the average reward (optional)
            'parameters': self.parameters,                                  # Save the training parameters
            'hidden_state': self.hidden_state                               # Save the hidden state if relevant (e.g., LSTM states)
        }
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        # Define the full path where the model will be saved
        filepath = os.path.join(self.output_directory, filename)
        # Save the checkpoint to the specified file path
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filename="dqn_model.pth"):
        # Define the full path where the model is saved
        filepath = os.path.join(self.output_directory, filename)
        
        # Load the checkpoint
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Restore the state dictionaries
            self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore additional parameters
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.avg_reward = checkpoint.get('avg_reward', self.avg_reward)
            self.parameters = checkpoint.get('parameters', self.parameters)
            self.hidden_state = checkpoint.get('hidden_state', self.hidden_state)
            print(f"Model loaded successfully from {filepath}")
        else:
            print(f"Checkpoint file not found at {filepath}. Ensure the path and filename are correct.")
        
    def set_location_graph(self, location_graph):
        # check that the given location graph is legit
        assert self.num_hospital_nodes == len(self.location_graph.hospital_nodes), 'new location graph is invalid (hospital nodes)'
        assert self.num_non_hospital_nodes == len(self.location_graph.non_hospital_nodes), 'new location graph is invalid (num nodes)'
        self.location_graph = location_graph
        
    def reset_hidden(self, batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size
        # Reset hidden state between episodes or sequences if needed
        self.hidden_state = self.init_hidden(batch_size)
        
    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device))
    
    def run_hospital_dynamics(self):
        rewards = []
        for hospital_index in self.location_graph.hospital_nodes:
            hospital_data = self.location_graph.graph.nodes[hospital_index]
            # Update current occupancy after moving agents
            current_occupancy = hospital_data['population'].get_agent_counts_by_criteria()
            hospital_capacity = hospital_data['hospital_capacity']
            overflow_capacity = hospital_data['overflow_capacity']
            
            # Parameters for recovery rate adjustment
            baseline_recovery_rate = self.location_graph.parameters['BASELINE_RECOVERY_RATE']
            overflow_recovery_rate = self.location_graph.parameters['OVERFLOW_RECOVERY_RATE']
            death_rate_pandemic = self.location_graph.parameters['LAMBDA']
            death_rate_war = self.location_graph.parameters['d']

            # Adjust the recovery rate based on occupancy
            if current_occupancy <= hospital_capacity:
                # Recovery rate is at its maximum when the hospital is not full
                recovery_rate = baseline_recovery_rate
            elif hospital_capacity < current_occupancy <= overflow_capacity:
                # Linearly decrease the recovery rate between hospital capacity and overflow capacity
                recovery_rate = baseline_recovery_rate - (
                    (baseline_recovery_rate - overflow_recovery_rate) * 
                    (current_occupancy - hospital_capacity) / 
                    (overflow_capacity - hospital_capacity)
                )
                
            else:
                # Set the recovery rate to 0 if the hospital occupancy exceeds the overflow capacity
                recovery_rate = overflow_recovery_rate   

            # Adjust death probabilities based on new recovery rate
            death_probability_pandemic = max(0, death_rate_pandemic * (1 - recovery_rate))
            death_probability_war = max(0, death_rate_war * (1 - recovery_rate))
            
            # Update agent states based on death probabilities
            sick_civilians = hospital_data['population'].get_agents_by_criteria(agent_types='civilian', states=State.SYMPTOMATIC_INFECTED)
            sick_soldiers_A = hospital_data['population'].get_agents_by_criteria(agent_types='soldier', states=State.SYMPTOMATIC_INFECTED, is_injured=False, armies='A') 
            sick_agents = sick_civilians + sick_soldiers_A
            injured_soldiers_A = hospital_data['population'].get_agents_by_criteria(is_injured = True, armies='A')
            
            # Initialize total dead count if not present
            if 'total_dead' not in hospital_data:
                hospital_data['total_dead'] = 0
            if 'total_recovered' not in hospital_data:
                hospital_data['total_recovered'] = 0
            
            # Update the state of symptomatic infected agents
            for agent in sick_agents:
                rand_value = random.random()
                if rand_value < recovery_rate:
                    agent.state = State.RECOVERED
                elif rand_value < recovery_rate + (1 - recovery_rate) * death_probability_pandemic:
                    agent.state = State.DEAD
                    hospital_data['total_dead'] += 1
                            
            # Update the state of injured soldiers
            for agent in injured_soldiers_A:
                rand_value = random.random()
                if rand_value < recovery_rate:
                    agent.state = State.RECOVERED
                elif rand_value < recovery_rate + (1 - recovery_rate) * death_probability_war:
                    agent.state = State.DEAD
                    hospital_data['total_dead'] += 1
                
            recovered_count = hospital_data['population'].get_agent_counts_by_criteria(states=State.RECOVERED)
            dead_count = hospital_data['total_dead']
            hospital_data['total_recovered'] = recovered_count
            
            # Calculate the reward based on the updated hospital dynamics
            reward = self.get_reward(current_occupancy,
                                     overflow_capacity,
                                     recovered_count,
                                     dead_count,
                                     recovery_rate)
            
            rewards.append(reward)
        self.avg_reward = sum(rewards) / len(rewards) if rewards else 0 # Avoid division by zero
    
        return rewards
    
    def get_reward(self, current_occupancy, hospital_overflow_capacity, recovered_count, 
                dead_count, recovery_rate):
        # Initialize reward
        reward = 0.0

        # Maximum penalty if hospital occupancy exceeds overflow capacity
        if current_occupancy > hospital_overflow_capacity:
            return -100 # Maximum penalty

        # Dynamic thresholds based on current occupancy
        high_threshold = 0.2 * current_occupancy # Set as 20% of current occupancy
        moderate_threshold = 0.05 * current_occupancy # Set as 5% of current occupancy

        # Calculate the difference between recovered and dead counts
        recovery_difference = recovered_count - dead_count

        # Check for high recovery difference separately from the recovery rate condition
        if recovery_difference > high_threshold:
            if self.overflow_recovery_rate < recovery_rate <= self.baseline_recovery_rate:
                reward += 90 # High reward scaling when recovery rate is within preferred range
            else:
                reward += 80 # Moderate reward even if recovery rate condition is not met

        # Moderate reward when slightly more recovered than dead but recovery rate is low
        elif moderate_threshold < recovery_difference <= high_threshold:
            if self.overflow_recovery_rate < recovery_rate <= self.baseline_recovery_rate:
                reward += 60 # Moderate reward scaling
            else:
                reward += 40 # Moderate reward even if recovery rate condition is not met

        # Moderate penalty when slightly more dead than recovered
        elif -moderate_threshold < recovery_difference < 0:
            if self.overflow_recovery_rate < recovery_rate <= self.baseline_recovery_rate:
                reward -= 20 # Moderate penalty scaling
            else:
                reward -= 40 # Moderate reward even if recovery rate condition is not met


        # High penalty when significantly more dead than recovered relative to occupancy
        elif recovery_difference <= -high_threshold:
            if self.overflow_recovery_rate < recovery_rate <= self.baseline_recovery_rate:
                reward -= 60 # High penalty scaling
            else:
                reward -= 90

        # Clip reward to ensure it remains within a reasonable range
        reward = max(-100, min(reward, 100)) # Clip values to reduce extreme feedback loops

        # Return the rounded reward to maintain consistency
        return round(reward, 2)

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def remember(self, state, action, reward, next_state, done=False):
        self.replay_buffer.add(state, action, reward, next_state, done)
        
    def decode_action(self, actions):
        allocations = []
        for node_idx, node_index in enumerate(self.location_graph.non_hospital_nodes):
            max_civilians_at_node  = self.location_graph.graph.nodes[node_index]['population'].get_agent_counts_by_criteria(agent_types='civilian', states=State.SYMPTOMATIC_INFECTED)
            symptomatic_soldiers = self.location_graph.graph.nodes[node_index]['population'].get_agent_counts_by_criteria(agent_types='soldier', armies='A', states=State.SYMPTOMATIC_INFECTED)
            injured_soldiers = self.location_graph.graph.nodes[node_index]['population'].get_agent_counts_by_criteria(agent_types='soldier', armies='A', is_injured=True, states=[State.ASYMPTOMATIC_INFECTED, State.EXPOSED, State.RECOVERED, State.SUSCEPTIBLE])
            max_soldiers_at_node = symptomatic_soldiers + injured_soldiers
            # Initialize counters for the total allocations
            civilians_allocated = 0
            soldiers_allocated = 0
            
            for hospital_idx, hospital_index in enumerate(self.location_graph.hospital_nodes):
                # Get the allocations from the action tensor
                num_civilians = actions[node_idx, hospital_idx, 0].item()
                num_soldiers = actions[node_idx, hospital_idx, 1].item()

                # Only consider the allocation if there's a non-zero allocation
                if num_civilians > 0 or num_soldiers > 0:
                    # Accumulate the allocations for this node
                    civilians_allocated += num_civilians
                    soldiers_allocated += num_soldiers
                    
                    # Add the allocation to the list
                    allocations.append((node_index, hospital_index, num_civilians, num_soldiers))
            # Check if the total allocations exceed the available resources at the node
            if civilians_allocated > max_civilians_at_node:
                raise ValueError(f"Allocation of civilians exceeds available amount at node {node_idx}. "
                                f"Allocated: {civilians_allocated}, Available: {max_civilians_at_node}")

            if soldiers_allocated > max_soldiers_at_node:
                raise ValueError(f"Allocation of soldiers exceeds available amount at node {node_idx}. "
                                f"Allocated: {soldiers_allocated}, Available: {max_soldiers_at_node}")

        return allocations

    def act(self, state_space, action_space, action_mask, policy, training):
        """
        Selects actions based on the current state using a specified policy. 
        If 'dqn' is passed, it performs exploration vs. exploitation using the Q-network.
        If 'random' is passed, it performs only exploration.

        Args:
            state_space (torch.Tensor): The current state of the environment, shaped as (sequence_length, input_size).
            action_space (torch.Tensor): The available action space for each node-hospital pair, shaped as (num_nodes, num_hospitals, 2).
            action_mask (torch.Tensor): A mask to filter valid actions, shaped as (num_nodes, num_hospitals, 2).
            policy (str): The policy type ('dqn' or 'random') to determine the behavior of the function.

        Returns:
            torch.Tensor: Selected actions for each node-hospital pair, shaped as (num_nodes, num_hospitals, 2).
        """
        # Ensure the state tensor is moved to the correct device
        state = state_space.to(self.device)
        action_space = action_space.to(self.device)
        action_mask = action_mask.to(self.device)

        # Initialize a tensor to store allocation decisions for all node-hospital pairs
        actions = torch.zeros((self.num_non_hospital_nodes, self.num_hospital_nodes, 2), dtype=torch.int32).to(self.device)

        # Exploration vs. Exploitation for all node-hospital combinations
        # Determine policy behavior
        if policy == 'random' or (training and policy in ('dqn', 'prioritize_soldiers', 'prioritize_civilians') and np.random.rand() <= self.epsilon):
            self.logger.info(f"dqagent is exploring.....")
            # Extract the actual count of civilians and soldiers available at each node by taking the maximum value across all hospitals
            max_civilians_per_node = action_space[:, :, 0].max(dim=1).values # Max civilians across hospitals for each node
            max_soldiers_per_node = action_space[:, :, 1].max(dim=1).values # Max soldiers across hospitals for each node

            for node_index in range(self.num_non_hospital_nodes):
                # Initialize remaining counts for each node
                remaining_civilians = max_civilians_per_node[node_index].item() # Reset for each node
                remaining_soldiers = max_soldiers_per_node[node_index].item() # Reset for each node

                for hospital_index in range(self.num_hospital_nodes):
                    # Only attempt to allocate if the action mask allows it
                    actual_graph_hospital_index = self.location_graph.hospital_nodes[hospital_index]
                    hospital_capacity = self.location_graph.graph.nodes[actual_graph_hospital_index]['overflow_capacity']
                    if action_mask[node_index, hospital_index].sum() > 0: # Check if any valid action exists
                        # Check and update the allocation for civilians
                        if action_mask[node_index, hospital_index, 0] == 1 and remaining_civilians > 0:
                            # Allocate civilians up to the remaining available count
                            max_civilians_to_move = min(remaining_civilians, action_space[node_index, hospital_index, 0].item())
                                                        
                            civilians_to_move = random.randint(0, max_civilians_to_move)
                            actions[node_index, hospital_index, 0] = civilians_to_move
                            remaining_civilians -= civilians_to_move # Update remaining count
                        else:
                            # Set allocation to 0 if the condition is not met
                            actions[node_index, hospital_index, 0] = 0

                        # Check and update the allocation for soldiers
                        if action_mask[node_index, hospital_index, 1] == 1 and remaining_soldiers > 0:
                            # Allocate soldiers up to the remaining available count
                            max_soldiers_to_move = min(remaining_soldiers, action_space[node_index, hospital_index, 1].item())
                            soldiers_to_move = random.randint(0, max_soldiers_to_move)
                            actions[node_index, hospital_index, 1] = soldiers_to_move
                            remaining_soldiers -= soldiers_to_move # Update remaining count
                        else:
                            # Set allocation to 0 if the condition is not met
                            actions[node_index, hospital_index, 1] = 0
                 
        else: # Exploitation: Use the policy network for the best action
            self.logger.info(f"dqagent is exploiting")
            # Exploitation: Use the policy network to decide on the best actions
            with torch.no_grad(): # Inference mode, no gradient computation needed
                # Determine the batch size of the current input state
                if state.dim() == 2: # If the state shape is (batch_size, feature_size), e.g., (9, 20)
                    state = state.unsqueeze(1) # Add sequence length dimension, now (9, 1, 20)
                self.reset_hidden(batch_size=1)
                logits, _ = self.policy_network(state, self.hidden_state) # Forward pass through the Q-network

            # Determine the batch size, number of nodes, and hospitals from logits
            batch_size = logits.size(0) # Should be 9
            num_nodes = self.num_non_hospital_nodes # Replace with appropriate variable for the number of nodes
            num_hospitals = self.num_hospital_nodes # Replace with appropriate variable for the number of hospitals

            # Reshape logits to match the action space: (batch_size, num_nodes, num_hospitals, 2)
            logits = logits.view(batch_size, num_nodes, num_hospitals, 2)

            logits = logits.max(dim=0)[0] # Shape will now be (num_nodes, num_hospitals, 2)

            # Apply a mask to logits to ensure invalid actions are ignored
            masked_logits = logits.clone()
            masked_logits[action_mask == 0] = -1e9 # Replace -inf with a large negative value to avoid NaNs in softmax
            
            # Check masked logits to ensure no issues
            if torch.isnan(masked_logits).any():
                print("Warning: NaNs detected in masked logits before softmax")
            # Use softmax safely, ignoring masked positions effectively
            civilians_logits = masked_logits[:, :, 0] # Extract logits for civilians
            soldiers_logits = masked_logits[:, :, 1] # Extract logits for soldiers

            # Replace logits corresponding to masked actions with a large negative value
            civilians_logits[action_mask[:, :, 0] == 0] = -1e9
            soldiers_logits[action_mask[:, :, 1] == 0] = -1e9
            
            # Apply softmax along the hospital axis (dim=1) for each node
            civilians_probs = F.softmax(civilians_logits, dim=1)
            soldiers_probs = F.softmax(soldiers_logits, dim=1)
            
            # Multiply by action_space to zero out probabilities for masked actions
            civilians_probs *= action_mask[:, :, 0]
            soldiers_probs *= action_mask[:, :, 1]
            
            # Ensure no NaNs remain after masking
            civilians_probs = torch.nan_to_num(civilians_probs, nan=0.0)
            soldiers_probs = torch.nan_to_num(soldiers_probs, nan=0.0)
            
            # Determine the total number of civilians and soldiers that can be allocated from each node
            max_civilians = action_space[:, :, 0].max(dim=1).values # Corrected: max available civilians per node
            max_soldiers = action_space[:, :, 1].max(dim=1).values # Corrected: max available soldiers per node

            # Iterate over each node and hospital to determine specific allocations
            for node_index in range(self.num_non_hospital_nodes):
                civilians_total = 0
                soldiers_total = 0
                
                total_civilians_at_node = max_civilians[node_index].item() # Maximum civilians at the node
                total_soldiers_at_node = max_soldiers[node_index].item() # Maximum soldiers at the node

                for hospital_index in range(self.num_hospital_nodes):
                    # Calculate initial allocations based on scaled probabilities
                    civilians_alloc = (civilians_probs[node_index, hospital_index] * total_civilians_at_node).round().item()
                    soldiers_alloc = (soldiers_probs[node_index, hospital_index] * total_soldiers_at_node).round().item()

                    # Adjust allocations if they exceed the remaining available resources
                    civilians_alloc = min(civilians_alloc, total_civilians_at_node - civilians_total)
                    soldiers_alloc = min(soldiers_alloc, total_soldiers_at_node - soldiers_total)

                    # Accumulate allocations to ensure total stays within limits
                    civilians_total += civilians_alloc
                    soldiers_total += soldiers_alloc

                    # Update action tensor with the computed allocations
                    actions[node_index, hospital_index, 0] = civilians_alloc
                    actions[node_index, hospital_index, 1] = soldiers_alloc
        if training:
            self.adjust_epsilon() # Decay epsilon during training# Decay epsilon after the action is taken

        return actions

    def create_action_space_and_mask(self):  
        """
        Each element in the tensor represents the number of civilians and soldiers that can be allocated from a specific node to a specific healthcare center.
        Initialize the mask with ones (all actions initially valid)
        Initialize the hidden 
        Mask Node-Hospital Pairs Without Edges:
        Mask Civilians Allocated to War Zone Hospitals:
        Account for Injured/Infected Agents Only:
        Exclude Army B:
        exclude hospital at maximum capacity
        1 (Unmasked / Valid Action)
        0 (Masked / Invalid Action)
        """
            
        # Initialize the action space tensor: nodes × hospitals × 2 (civilians, soldiers)
        action_tensor = torch.zeros((self.num_non_hospital_nodes, self.num_hospital_nodes, 2), dtype=torch.int32)
        
        # Initialize the action mask tensor with ones: nodes × hospitals × 2 (civilians, soldiers)
        action_mask = torch.ones((self.num_non_hospital_nodes, self.num_hospital_nodes, 2), dtype=torch.int32)
        
        # Loop over non-hospital nodes to consider civilian and soldier allocations
        for node_idx, node_index in enumerate(self.location_graph.non_hospital_nodes):
            node_data = self.location_graph.graph.nodes[node_index]
            population = node_data['population']
            
            # Get counts of infected civilians and soldiers (Army A), and injured soldiers (Army A)
            infected_civilians = population.get_agent_counts_by_criteria(agent_types='civilian', states=State.SYMPTOMATIC_INFECTED)
            symptomatic_soldiers = population.get_agent_counts_by_criteria(agent_types='soldier', armies='A', states=State.SYMPTOMATIC_INFECTED)
            injured_soldiers = population.get_agent_counts_by_criteria(agent_types='soldier', armies='A', is_injured=True, states=[State.ASYMPTOMATIC_INFECTED, State.EXPOSED, State.RECOVERED, State.SUSCEPTIBLE])
            
            # Loop over all hospital nodes to determine possible allocations
            for hospital_idx, hospital_index in enumerate(self.location_graph.hospital_nodes):
                hospital_data = self.location_graph.graph.nodes[hospital_index]
                hospital_occupancy = hospital_data['population'].get_agent_counts_by_criteria()
                hospital_capacity = hospital_data['hospital_capacity']
                # Calculate available space in the hospital
                available_space = hospital_capacity - hospital_occupancy
                is_war_zone = hospital_data['zone'] == 'war'
                
                # Check if the hospital is a neighbor (has an edge with the node)
                if hospital_index not in list(self.location_graph.graph.neighbors(node_index)):
                    # No edge, mask both civilians and soldiers
                    action_mask[node_idx, hospital_idx, :] = 0
                    continue
                
                # If the hospital is in a war zone, mask civilians
                if is_war_zone:
                    action_mask[node_idx, hospital_idx, 0] = 0 # Mask civilians
                
                # If the hospital is at maximum capacity, mask both civilians and soldiers
                if available_space <= 0:
                    action_mask[node_idx, hospital_idx, :] = 0
                    continue
                
                # Determine how many civilians and soldiers can actually be allocated
                max_civilians_to_allocate = min(infected_civilians, available_space)
                max_soldiers_to_allocate = min((symptomatic_soldiers + injured_soldiers), (available_space - max_civilians_to_allocate))
                # Set the action space with the available injured/infected agents
                action_tensor[node_idx, hospital_idx, 0] = max_civilians_to_allocate
                action_tensor[node_idx, hospital_idx, 1] = max_soldiers_to_allocate
                if (max_soldiers_to_allocate) != (symptomatic_soldiers + injured_soldiers):
                    pass

        return action_tensor, action_mask
    
    def adjust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        # Sample a minibatch of sequences from the replay buffer
        # replay buffer is experience replay buffer or memory
        minibatch = self.replay_buffer.sample_sequences()
        # Separate sequences into their components
        for states, actions_last, reward, next_states, done in minibatch:
            # Forward pass through the Q-network
            self.reset_hidden(self.batch_size)
            logits, _ = self.policy_network(states, self.hidden_state)
            
            # Dynamically determine the batch size, number of nodes, and hospitals from logits and actions
            batch_size = logits.size(0) # Get batch size dynamically
            num_nodes = actions_last.size(0) # Should match num_nodes from actions_last, e.g., 7
            num_hospitals = actions_last.size(1) # Should match num_hospitals from actions_last
            # Reshape logits to match the action space: (batch_size, num_nodes, num_hospitals, 2)
            logits = logits.view(batch_size, num_nodes, num_hospitals, 2)
            # Ensure indices are within valid range [0, 1]
            civilians_indices = torch.clamp(actions_last[:, :, 0], 0, logits.size(-1) - 1) # Shape: (num_nodes, num_hospitals)
            soldiers_indices = torch.clamp(actions_last[:, :, 1], 0, logits.size(-1) - 1) # Shape: (num_nodes, num_hospitals)

            # Gather Q-values for the actions taken
            batch_indices = torch.arange(batch_size).unsqueeze(-1).unsqueeze(-1)
            node_indices = torch.arange(num_nodes).unsqueeze(-1)
            batch_indices = batch_indices.long()
            node_indices = node_indices.long()
            civilians_indices = civilians_indices.long()
            soldiers_indices = soldiers_indices.long()
            # Gather Q-values for civilians and soldiers, ensuring the indices are within range
            civilians_q_values = logits[batch_indices, node_indices, torch.arange(num_hospitals), civilians_indices]
            soldiers_q_values = logits[batch_indices, node_indices, torch.arange(num_hospitals), soldiers_indices]
            
            max_civilians_q_values = civilians_q_values.max(dim=-1)[0] # Shape: (batch_size, num_nodes)
            max_soldiers_q_values = soldiers_q_values.max(dim=-1)[0] # Shape: (batch_size, num_nodes)

            # Calculate target Q-values using the target network
            with torch.no_grad():
                next_logits, _ = self.target_network(next_states, self.hidden_state)
                next_logits = next_logits.view(batch_size, num_nodes, num_hospitals, 2)
                next_civilians_q_values = next_logits[:, :, :, 0].max(dim=-1)[0] # Max over hospitals, dim=-1
                next_soldiers_q_values = next_logits[:, :, :, 1].max(dim=-1)[0] # Max over hospitals, dim=-1

            
            # Compute target Q-values
            target_civilians_q_values = reward + (1 - done) * self.gamma * next_civilians_q_values
            target_soldiers_q_values = reward + (1 - done) * self.gamma * next_soldiers_q_values
            
            if max_civilians_q_values.shape != target_civilians_q_values.shape:
                print(f"Shape mismatch: q_values: {max_civilians_q_values.shape}, target_q_values: {target_civilians_q_values.shape}")

            if max_soldiers_q_values.shape != target_soldiers_q_values.shape:
                print(f"Shape mismatch: q_values: {max_soldiers_q_values.shape}, target_q_values: {target_soldiers_q_values.shape}")

            # Compute loss and backpropagate
            # Compute separate losses for civilians and soldiers
            civilians_loss = self.criterion(max_civilians_q_values.view(-1), target_civilians_q_values.view(-1))
            soldiers_loss = self.criterion(max_soldiers_q_values.view(-1), target_soldiers_q_values.view(-1))

            # Combine the losses (if needed) and backpropagate
            total_loss = civilians_loss + soldiers_loss
            self.loss = total_loss.item()
            total_loss.backward() # Backpropagate the combined loss
            self.optimizer.step() # Update the network weights
            self.optimizer.zero_grad() # Clear gradients for the next pass            
    
        self.adjust_epsilon()
            
    def save(self, filepath):
        torch.save(self.policy_network.state_dict(), filepath)

    def load(self, filepath):
        self.policy_network.load_state_dict(torch.load(filepath, map_location=self.device))
        self.policy_network.eval()
        
    def get_current_state(self):
        """
        Constructs the current state tensor from the location graph.
        The state tensor includes node (location) information, indicators for whether a node is a war zone,
        whether it is a hospital, hospital status, the counts of agents by their epidemiological status, 
        sociological status, and army affiliation.
        """
        all_features = []
        
        for node in self.location_graph.graph.nodes:
            node_data = self.location_graph.graph.nodes[node]
            population = node_data['population']
            # Aggregate features for each node
            features = []
            # Epidemic Dynamics
            S = population.get_agent_counts_by_criteria(states=[State.SUSCEPTIBLE])
            E = population.get_agent_counts_by_criteria(states=[State.EXPOSED])
            Is = population.get_agent_counts_by_criteria(states=[State.SYMPTOMATIC_INFECTED])
            Ia = population.get_agent_counts_by_criteria(states=[State.ASYMPTOMATIC_INFECTED])
            R = population.get_agent_counts_by_criteria(states=[State.RECOVERED])
            D = population.get_agent_counts_by_criteria(states=[State.DEAD])
            features.extend([S, E, Is, Ia, R, D])
            
            # Location Information
            W = 1 if node_data['zone'] == 'war' else 0
            H = 1 if node in self.location_graph.hospital_nodes else 0
            # Initialize occupancy and capacity to zero for non-hospital nodes
            occupancy = 0
            capacity = 0
            overlow = 0
            if H == 1:
                occupancy = population.get_agent_counts_by_criteria()
                capacity = node_data['hospital_capacity'] 
                overlow = node_data['overflow_capacity']
                
            degree = len(list(self.location_graph.graph.neighbors(node)))
            total_cw = sum(self.location_graph.graph[node][neighbor]['cw'] for neighbor in self.location_graph.graph.neighbors(node))
            total_cc = sum(self.location_graph.graph[node][neighbor]['cc'] for neighbor in self.location_graph.graph.neighbors(node))
            features.extend([W, H, occupancy, capacity, overlow, degree, total_cw, total_cc])
            
            # Sociological Status
            soldiers = population.get_agent_counts_by_criteria(agent_types='soldier')
            civilians = population.get_agent_counts_by_criteria(agent_types='civilian')
            features.extend([soldiers, civilians])
            
            # War Dynamics
            injured_A = population.get_agent_counts_by_criteria(armies='A', is_injured=True)
            injured_B = population.get_agent_counts_by_criteria(armies='B', is_injured=True)
            A = population.get_agent_counts_by_criteria(armies='A')
            B = population.get_agent_counts_by_criteria(armies='B')
            features.extend([injured_A, injured_B, A, B]) 
            
            # Append the aggregated features for this node to all_features
            all_features.append(features)
        # Convert the aggregated feature lists to a tensor
        state_tensor = torch.tensor(all_features, dtype=torch.float32)
        state_tensor = state_tensor.view(1, 1, self.state_size)

        return state_tensor # Should match (batch_size, input_size)
    
    def move_agents_to_hospital(self, allocations, policy):
        for allocation in allocations:
            node_index, hospital_index, civ_count, soldier_count = allocation
            
            node = self.location_graph.graph.nodes[node_index]
            hospital = self.location_graph.graph.nodes[hospital_index]
            
            # Ensure there is an edge between the node and the hospital
            if hospital_index not in self.location_graph.graph.neighbors(node_index):
                raise ValueError(f"No direct connection between node {node} and hospital {hospital}.")

            # Get the populations at the node
            current_occupancy = hospital['population'].get_agent_counts_by_criteria()
            civilians = node['population'].get_agents_by_criteria(agent_types='civilian', states=State.SYMPTOMATIC_INFECTED)
            sick_soldiers_A = node['population'].get_agents_by_criteria(agent_types='soldier', armies='A', states=State.SYMPTOMATIC_INFECTED) 
            injured_soldiers_A = node['population'].get_agents_by_criteria(agent_types='soldier', armies='A', is_injured=True, states=[State.ASYMPTOMATIC_INFECTED, State.EXPOSED, State.RECOVERED, State.SUSCEPTIBLE])
            soldiers = list(set(sick_soldiers_A + injured_soldiers_A))
            
            # Update current occupancy after moving agents            
            overflow_capacity = hospital['overflow_capacity']
            available_space = overflow_capacity - current_occupancy
            
            if civ_count + soldier_count > available_space:
                self.logger.info(f'adjusting counters, civ_count: {civ_count}, soldier_count: {soldier_count}, available_space: {available_space}')
                if policy == 'prioritize_soldiers':
                    civ_count, soldier_count = self.adjust_counts_prioritize_soldiers(civ_count, soldier_count, available_space)
                elif policy == 'prioritize_civilians':
                    civ_count, soldier_count = self.adjust_counts_prioritize_civilians(civ_count, soldier_count, available_space)
                else:
                    civ_count, soldier_count = self.adjust_counts_keep_ratio(civ_count, soldier_count, available_space)
                self.logger.info(f'counters adjusted, civ_count: {civ_count}, soldier_count: {soldier_count}, available_space: {available_space}')
            
            # Move civilians from the node to the hospital
            if civ_count > 0:
                for agent in random.sample(civilians, civ_count):
                    agent.move_agent(hospital_index)

            # Move soldiers from the node to the hospital
            if soldier_count > 0:
                for agent in random.sample(soldiers, soldier_count):
                    agent.move_agent(hospital_index)

            assert overflow_capacity >= hospital['population'].get_agent_counts_by_criteria(), "too many agents in hospital"
    
    def adjust_counts_prioritize_soldiers(self, civ_count, soldier_count, available_space):
        new_soldier_count = min(available_space, soldier_count)
        new_available_space = available_space - new_soldier_count
        new_civ_count = min(new_available_space, civ_count)
        return new_civ_count, new_soldier_count
    
    def adjust_counts_prioritize_civilians(self, civ_count, soldier_count, available_space):
        new_civ_count = min(available_space, civ_count)
        new_available_space = available_space - new_civ_count
        new_soldier_count = min(new_available_space, soldier_count)
        return new_civ_count, new_soldier_count
    
    def adjust_counts_keep_ratio(self, civ_count, soldier_count, available_space):
        # Calculate the total ratio sum
        total_count = civ_count + soldier_count
        
        # Calculate the scaling factor
        if total_count == 0: # Handle case where both counts are zero
            return 0, 0
        
        scaling_factor = available_space / total_count
        
        # Adjust the counts
        new_civ_count = int(civ_count * scaling_factor)
        new_soldier_count = int(soldier_count * scaling_factor)
        
        # Check if the total count needs adjustment
        total_new_count = new_civ_count + new_soldier_count
        if total_new_count < available_space:
            # Distribute the remaining space
            diff = available_space - total_new_count
            if civ_count >= soldier_count:
                new_civ_count += diff
            else:
                new_soldier_count += diff
        
        return new_civ_count, new_soldier_count
    
    def allocate_to_hospitals(self, policy, training=True):
        # Step 1: Extract the current state from the environment
        state_space = self.get_current_state() 
        # Step 2: Create the action mask
        action_space, action_mask = self.create_action_space_and_mask() 
        # Step 3: Decide on the allocation based on current state
        actions = self.act(state_space, action_space, action_mask, policy, training)
        # Step 4: Decode the global action into multiple allocations
        allocations = self.decode_action(actions)
        # Step 5: Move agents to hospitals
        self.move_agents_to_hospital(allocations, policy)
        # Step 6: Run hospital dynamics after all agents have been moved
        rewards = self.run_hospital_dynamics()
        # Step 7: Determine the final state after all dynamics have been processed
        next_state = self.get_current_state()
        done = self.episodes <= 1 # Set done to True if this is the last episode
        if training:
            # Training mode: Perform experience replay and network updates
            if policy != 'random':
                # Step 8: Store the experience (state, action, reward, next_state, done) and train
                for reward in rewards:
                    self.remember(state_space, actions, reward, next_state, done)
                
            # Step 9: Perform experience replay to train the Q-network
            self.replay()

            # Step 10: Manage the episode count
            if self.episodes > 0:
                self.episodes -= 1 # Decrease the episode counter
                if self.episodes % (self.target_update_freq // 1000) == 0: # Update target network at intervals
                    self.update_target_network()
                    
                    self.logger.info(f"dqagent has updated the network")
            else:
                print("Training complete. No more episodes remaining.")
                self.save_model("dq_model.pth") # Save the model after training is finished
                print("Model saved successfully.")
        
        return next_state

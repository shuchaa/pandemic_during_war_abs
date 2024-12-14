# library imports
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# project imports
from .Agents import State

class Plotter:
    def __init__(self, simulator, output_directory='plots'):
        self.simulator = simulator
        self.output_directory = output_directory 
        self.create_output_directory()       
        self.simulation_data = []  # To store data for summary plots
        self.raw_data = []  # To store data for multiple runs
        self.pos = nx.spring_layout(self.simulator.location_graph.graph)  # positions for all nodes
        self.color_map = {
            State.SUSCEPTIBLE: 'blue',
            State.EXPOSED: 'yellow',
            State.ASYMPTOMATIC_INFECTED: 'orange',
            State.SYMPTOMATIC_INFECTED: 'red',
            State.RECOVERED: 'green',
            State.DEAD: 'black'
        }
        self.non_hospital_edges = [(u, v) for u, v in self.simulator.location_graph.graph.edges if v not in self.simulator.location_graph.hospital_nodes and u not in self.simulator.location_graph.hospital_nodes]
        self.hospital_edges = [(u, v) for u, v in self.simulator.location_graph.graph.edges if v not in self.simulator.location_graph.hospital_nodes or u not in self.simulator.location_graph.hospital_nodes]
        # Filter out self-edges
        self.non_hospital_edges = [(u, v) for u, v in self.non_hospital_edges if u != v]
        self.hospital_edges = [(u, v) for u, v in self.hospital_edges if u != v]
            
    def create_output_directory(self):
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

    def save_plot(self, plot_name, show_plot=False):
        # Create the output directory if it doesn't exist
        self.create_output_directory()
        # Ensure the plot is rendered
        plt.draw()
        # Construct the full path for the plot
        output_path = os.path.join(self.output_directory, plot_name)
        # Save the plot with the specified settings
        plt.savefig(output_path, dpi=300, format='pdf')
        # Clear the figure to avoid overlap with subsequent plots
        if show_plot:
            plt.show()

    def adjust_positions(self):
        """ Adjust positions so 'war' nodes are on the left and 'civilian' nodes are on the right """
        # These numbers were selected for better visualization.
        warzone_positions_x = [-0.12, -0.86, -0.74, -0.33, -0.71, -0.24, -0.1, -0.59, -0.88, -0.7, -0.98, -0.64, -0.19, -0.44, -0.25, -0.49, -0.39, -0.45, -0.16, -0.14, -0.57, -0.97, -0.48, -0.79, -0.94, -0.92, -0.81, -0.62, -0.76, -0.66, -0.8, -0.38, -0.28, -0.54, -0.82, -0.21, -0.36, -0.56, -0.61, -0.83, -0.46, -0.41, -0.67, -0.87, -0.29, -0.78, -0.69, -1.2, -0.6, -0.91]
        warzone_positions_y = [0.28, -0.02, -0.32, 0.42, 0.8, -0.61, -0.65, -0.16, -0.45, 0.74, -0.22, -0.28, 0.78, 0.51, 0.16, -0.35, 0.58, 0.29, 0.33, 0.43, 0.11, -0.41, 0.18, 0.49, -0.47, 0.38, -0.06, -0.73, 0.57, 0.72, 0.67, -0.31, -0.59, -0.89, 0.66, -0.55, 0.85, -0.75, 0.39, 0.27, 0.22, 0.09, -0.18, -0.03, 0.59, 0.37, 0.43, 0.32, -0.62, -0.77]
        civilian_positions_x = [0.67, 0.28, 0.2, 0.31, 0.62, 0.36, 0.93, 0.37, 0.76, 0.95, 0.17, 0.79, 0.92, 0.26, 0.81, 0.41, 0.68, 0.98, 0.46, 0.29, 0.34, 0.66, 0.87, 0.83, 0.64, 0.3, 0.44, 0.69, 0.53, 0.32, 0.72, 0.1, 0.97, 0.35, 0.54, 0.12, 0.7, 0.24, 0.13, 0.84, 0.19, 0.61, 0.99, 1, 0.9, 0.33, 0.49, 0.63, 0.18, 0.94]
        civilian_positions_y = [-0.33, 0.69, -0.76, 0.19, -0.06, -0.25, 0.36, -0.69, 0.12, 0.67, 0.59, -0.9, -0.14, -0.15, 0.77, -0.72, 0.23, -0.05, 0.83, 0.14, 0.45, -0.54, -0.59, 0.89, 0.32, -0.85, -0.77, 0.85, -0.58, -0.11, -0.1, -0.37, -0.67, 0.5, -0.82, 0.26, 0.35, -0.52, -0.61, -0.23, -0.16, 0.25, 0.51, -0.38, -0.79, 0.49, -0.41, 0.07, 0.57, -0.6]
        
        # choose random location and make sure no two nodes are on the same line
        for node, data in self.simulator.location_graph.graph.nodes(data=True):
            if data.get('zone') == 'war':
                self.pos[node][0] = warzone_positions_x.pop()  # Move war zones to the right
                self.pos[node][1] = warzone_positions_y.pop()  # Keep the same y location between iterations
            else:
                self.pos[node][0] = civilian_positions_x.pop()  # Move civilian zones to the left
                self.pos[node][1] = civilian_positions_y.pop()  # Keep the same y location between iterations

    def plot_population(self, step):
        plt.figure(figsize=(12, 8))
        node_colors = []
        node_labels = {}
        for node in self.simulator.location_graph.graph.nodes():
            population = self.simulator.location_graph[node]['population']
            civilians = population.get_agent_counts_by_criteria(agent_types='civilian')
            soldiers = population.get_agent_counts_by_criteria(agent_types='soldier')
            if civilians or soldiers:
                node_colors.append('red' if self.simulator.location_graph.graph.nodes[node]['is_hospital'] else 'gray')
                node_labels[node] = f"C:{civilians}\nS:{soldiers}"
            else:
                node_colors.append('gray')
                node_labels[node] = f"C:0\nS:0"

        nx.draw(self.simulator.location_graph.graph, self.pos, node_color=node_colors, with_labels=True, node_size=500, font_weight='bold')
        nx.draw_networkx_labels(self.simulator.location_graph.graph, self.pos, labels=node_labels, font_size=12, bbox=dict(facecolor='white', alpha=0.5))

        plt.title(f'Population at step {step}')
        plt.show()

    def update_plot(self, frame):
        self.simulator.run_step(frame, simulation_id=0)
        self.ax.clear()
        
        node_colors = []
        node_labels = {}
        
        for node in self.simulator.location_graph.graph.nodes():
            population = self.simulator.location_graph.graph.nodes[node]['population']
            civilians = population.get_agent_counts_by_criteria(agent_types='civilian')
            soldiers = population.get_agent_counts_by_criteria(agent_types='soldier')
            node_colors.append('red' if self.simulator.location_graph.graph.nodes[node]['is_hospital'] else 'gray')
            node_labels[node] = f"NID: {node} | C:{civilians} | S:{soldiers}"
            if self.simulator.location_graph.graph.nodes[node]['zone'] == 'war':
                node_labels[node] += ' | WARZONE'
            if self.simulator.location_graph.graph.nodes[node]['is_hospital']:
                node_labels[node] += ' | HOSPITAL'
        
        # Draw non-hospital nodes as circles
        nx.draw_networkx_nodes(self.simulator.location_graph.graph, self.pos, nodelist=self.simulator.location_graph.non_hospital_nodes, node_color='grey', edgecolors='black', node_size=500, ax=self.ax)
        # Draw hospital nodes as squares
        nx.draw_networkx_nodes(self.simulator.location_graph.graph, self.pos, nodelist=self.simulator.location_graph.hospital_nodes, node_color='green', edgecolors='black', node_shape='s', node_size=500, ax=self.ax)
        nx.draw_networkx_edges(self.simulator.location_graph.graph, self.pos, edgelist=self.hospital_edges, ax=self.ax, style='dashed', arrows=True, arrowstyle='-', width=3)
        nx.draw_networkx_edges(self.simulator.location_graph.graph, self.pos, edgelist=self.non_hospital_edges, ax=self.ax, arrows=True, width=3)

        # Adjust label positions to be above the nodes
        label_pos = {node: (self.pos[node][0], self.pos[node][1] + 0.1) for node in self.simulator.location_graph.graph.nodes()}
        nx.draw_networkx_labels(self.simulator.location_graph.graph, label_pos, labels=node_labels, font_size=12, ax=self.ax, bbox=dict(facecolor='white', alpha=0.5))
        self._shade_background()
        self._add_titles()
        plt.title(f'Population at step {frame}')

    def _shade_background(self):
        """ Shades the background based on zones """
        self.ax.axvspan(-1.5, 0, facecolor='lightcoral', alpha=0.5)  # Shade left side light blue
        self.ax.axvspan(0, 1.5, facecolor='lightblue', alpha=0.5)  # Shade right side light red

    def _add_titles(self):
        """ Adds titles to the zones """
        self.ax.text(0.25, 1, 'Warzone', transform=self.ax.transAxes, fontsize=14, verticalalignment='center', horizontalalignment='center', color='white', bbox=dict(facecolor='black', alpha=1))
        self.ax.text(0.75, 1, 'Civilian Zone', transform=self.ax.transAxes, fontsize=14, verticalalignment='center', horizontalalignment='center', color='white', bbox=dict(facecolor='black', alpha=1))
        
    def animate_simulation(self, num_steps):
        self.adjust_positions()
        fig, self.ax = plt.subplots(figsize=(12, 8))
        ani = animation.FuncAnimation(fig, self.update_plot, frames=range(num_steps), repeat=False)
        plt.show()
        
    def plot_pref_evals(self, avg_drl_performance, avg_random_performance):
        # Check if the lengths of avg_drl_performance and avg_random_performance are the same
        assert len(avg_drl_performance) == len(avg_random_performance), "Performance lists must be of the same length"
        
        # Calculate the number of simulations
        num_simulations = range(1, len(avg_drl_performance) + 1)
        
        # Calculate the difference in performance between consecutive simulations
        performance_diff = np.diff(avg_drl_performance)
        # Set a threshold for convergence (e.g., 0.5 improvement)
        threshold = 0.5
        # Find the index where the change becomes less than the threshold
        convergence_index = np.where(np.abs(performance_diff) < threshold)[0]
        
        if len(convergence_index) > 0:
            # The convergence point is the first instance where the change is below the threshold
            convergence_point = num_simulations[convergence_index[0] + 1]
        else:
            # If no convergence point is found, set it to the last simulation
            convergence_point = num_simulations[-1]

        # Plotting the graph
        plt.figure(figsize=(12, 6))

        # Plot the DRL Model Performance Curve
        plt.plot(num_simulations, avg_drl_performance, label='DRL Model Performance', color='blue', linewidth=2)

        # Plot the Random Policy Baseline
        plt.plot(num_simulations, avg_random_performance, color='red', linestyle='--', label='Random Policy Baseline (100%)')

        # Mark the convergence point
        plt.axvline(x=convergence_point, color='green', linestyle='--', label=f'Convergence Point at {convergence_point}')

        # Adding labels
        plt.xlabel('Number of Training Simulations', fontsize=14)
        plt.ylabel('Performance of DRL Model', fontsize=14)

        # Adding a legend
        plt.legend()
        # Display the grid
        plt.grid(True)
        # Save and show the plot
        self.save_plot('DRL_Model_Performance.pdf')
        
        
    def plot_pref_evals_norm(self, normalized_performance, avg_drl_performance, avg_random_performance):
        # Check if the lengths of avg_drl_performance and avg_random_performance are the same
        assert len(avg_drl_performance) == len(avg_random_performance), "Performance lists must be of the same length"
        
        # Calculate the number of simulations
        num_simulations = range(1, len(avg_drl_performance) + 1)
        
        # Calculate the difference in performance between consecutive simulations
        performance_diff = np.diff(normalized_performance)
        # Set a threshold for convergence (e.g., 0.5 improvement)
        threshold = 0.5
        # Find the index where the change becomes less than the threshold
        convergence_index = np.where(np.abs(performance_diff) < threshold)[0]
        
        if len(convergence_index) > 0:
            # The convergence point is the first instance where the change is below the threshold
            convergence_point = num_simulations[convergence_index[0] + 1]
        else:
            # If no convergence point is found, set it to the last simulation
            convergence_point = num_simulations[-1]

        # Plotting the graph
        plt.figure(figsize=(12, 6))

        # Plot the DRL Model Performance Curve
        plt.plot(num_simulations, normalized_performance, label='DRL Model Performance', color='blue', linewidth=2)

        # Plot the Random Policy Baseline
        plt.plot(num_simulations, avg_random_performance, color='red', linestyle='--', label='Random Policy Baseline (100%)')

        # Mark the convergence point
        plt.axvline(x=convergence_point, color='green', linestyle='--', label=f'Convergence Point at {convergence_point}')

        # Adding labels
        plt.xlabel('Number of Training Simulations', fontsize=14)
        plt.ylabel('Performance of DRL Model (% of Random Policy)', fontsize=14)

        # Adding a legend
        plt.legend()
        # Display the grid
        plt.grid(True)
        # Save and show the plot
        self.save_plot('DRL_Model_Norm_Performance.pdf')

    def plot_summary(self):
        """ Generates summary plots after the simulation run """
        self.plot_epidemic_curve()
        if self.raw_data:
            self.plot_wounded_and_deaths_over_time_std()
        else:
            self.plot_wounded_and_deaths_over_time()
        self.plot_epidemiological_states_over_time()
        self.plot_civilian_vs_soldier_deaths_correlation()
        self.plot_hospital_dynamics()
        self.plot_rewards_and_hospital_deaths()
    
    def plot_epidemic_curve(self):
        states = ['recovered','deaths','susceptible','exposed','asymptomatic_infected','symptomatic_infected']
        state_counts = {state: [] for state in states}

        for step_data in self.simulation_data:
            for state in states:
                state_counts[state].append(step_data[state.lower()])

        plt.figure()
        for state in states:
            plt.plot(state_counts[state], label=state)
        plt.xlabel('Steps', fontsize=14)
        plt.ylabel('Number of Agents', fontsize=14)
        plt.legend()
        self.save_plot('epidemic_curve.pdf', show_plot=True)
    
    def plot_hospital_dynamics(self):
        # Define the metrics to plot
        metrics = ['hospitalized_civilians', 'hospitalized_soldiers', 'hospital_occupancy_total', 
                'hospital_capacity', 'overflow_capacity']
        
        # Initialize a dictionary to store the metrics data
        metric_counts = {metric: [] for metric in metrics}

        # Collect data for each metric from the simulation data
        for step_data in self.simulation_data:
            for metric in metrics:
                metric_counts[metric].append(step_data[metric])

        # Plot each metric
        plt.figure()
        for metric in metrics:
            plt.plot(metric_counts[metric], label=metric)

        # Label axes and add a legend
        plt.xlabel('Steps', fontsize=14)
        plt.ylabel('Count / Rate', fontsize=14)
        plt.legend()
        
        # Save and show the plot
        self.save_plot('hospital_dynamics.pdf', show_plot=True)
        plt.show()
    
    def plot_rewards_and_hospital_deaths(self):
        # Define the metrics to plot
        metrics = ['reward', 'hospital_deaths', 'hospital_recovered']
        
        # Initialize a dictionary to store the metrics data
        metric_counts = {metric: [] for metric in metrics}

        # Collect data for each metric from the simulation data
        for step_data in self.simulation_data:
            for metric in metrics:
                metric_counts[metric].append(step_data[metric])

        # Plot each metric
        plt.figure(figsize=(10, 6))
        for metric in metrics:
            plt.plot(metric_counts[metric], label=metric)

        # Label axes and add a legend
        plt.xlabel('Steps', fontsize=14)
        plt.ylabel('Values', fontsize=14)
        plt.title('Reward and Hospital Deaths Over Time', fontsize=16)
        plt.legend()
        plt.grid(True)
        
        # Save and show the plot
        self.save_plot('rewards_vs_hospital_deaths.pdf', show_plot=True)
        plt.show()
     
    def plot_wounded_and_deaths_over_time_std(self):
        number_of_steps = len(self.raw_data[0])
        steps = np.arange(number_of_steps)

        # Initialize lists to store mean and std data for the keys
        mean_injured = []
        std_injured = []
        mean_deaths = []
        std_deaths = []

        # Calculate mean and std for each step and each key
        for step in range(number_of_steps):
            injured_values = [run[step]['army_A_injured'] for run in self.raw_data]
            deaths_values = [run[step]['soldier_deaths_A'] for run in self.raw_data]

            mean_injured.append(np.mean(injured_values))
            std_injured.append(np.std(injured_values))

            mean_deaths.append(np.mean(deaths_values))
            std_deaths.append(np.std(deaths_values))

        # Plotting the mean with standard deviation
        plt.figure(figsize=(8, 6))
         
        plt.plot(steps, mean_injured, 'b-', marker='o', label='Injured Soldiers (Mean)')
        plt.fill_between(steps, np.array(mean_injured) - np.array(std_injured), 
                        np.array(mean_injured) + np.array(std_injured), color='blue', alpha=0.2)

        plt.plot(steps, mean_deaths, 'r-', marker='o', label='Soldier Deaths (Mean)')
        plt.fill_between(steps, np.array(mean_deaths) - np.array(std_deaths), 
                        np.array(mean_deaths) + np.array(std_deaths), color='red', alpha=0.2)

        # Customizing the plot to match the provided style
        plt.xlabel('Steps', fontsize=14)
        plt.ylabel('Portion of Population', fontsize=14)
        plt.legend()
        plt.grid(True)
        self.save_plot('wounded_and_deaths_over_time_with_std.pdf', show_plot=True)

    def plot_wounded_and_deaths_over_time(self):
        states = ['army_A_injured','soldier_deaths_A']
        state_counts = {state: [] for state in states}

        for step_data in self.simulation_data:
            for state in states:
                state_counts[state].append(step_data[state])

        plt.figure()
        for state in states:
            plt.plot(state_counts[state], label=state)
        plt.xlabel('Steps', fontsize=14)
        plt.ylabel('Portion of Population', fontsize=14)
        plt.legend()
        self.save_plot('wounded_and_deaths_over_time.pdf', show_plot=True)
        plt.show()
    
    def plot_epidemiological_states_over_time(self):
        # Define the states with their corresponding labels
        states = ['deaths', 'army_A_injured', 'hospital_occupancy_total', 
                'susceptible', 'exposed', 'asymptomatic_infected', 'symptomatic_infected']
        
        # Create a dictionary to store data for each state
        state_counts = {state: [] for state in states}
        steps = range(len(self.simulation_data))
        
        # Total population (assuming it is the same across all steps)
        total_population = self.simulator.location_graph.parameters['POPULATION_SIZE']
        
        # Custom labels for plotting
        custom_labels = {
            'deaths': 'Dead',
            'army_A_injured': 'Wounded',
            'hospital_occupancy_total': 'Hospitalized',
            'susceptible': 'Susceptible',
            'exposed': 'Exposed',
            'asymptomatic_infected': 'Asymptomatic Infected',
            'symptomatic_infected': 'Symptomatic Infected'
        }

        # Populate state_counts with data from simulation_data
        for step_data in self.simulation_data:
            for state in states:
                if state in step_data:  # Ensure the state exists in step_data
                    state_counts[state].append(step_data[state] / total_population * 100)  # Normalize by total population
                else:
                    state_counts[state].append(0)  # Append 0 if the state data is missing

        # Check that all lists have the correct length
        for state, counts in state_counts.items():
            if len(counts) != len(steps):
                raise ValueError(f"Length mismatch in state '{state}': expected {len(steps)}, got {len(counts)}")

        # Plot each state individually to allow overlaps with transparency
        plt.figure(figsize=(8, 6))
        for state, counts in state_counts.items():
            plt.fill_between(steps, counts, label=custom_labels[state], alpha=0.3)  # Lower alpha to show overlap
        plt.xlabel('Steps', fontsize=14)
        plt.ylabel('Proportion of Population (%)', fontsize=14)  # Set y-axis label as percentage
        plt.yticks(range(0, 101, 10))  # Set y-axis ticks from 0% to 100%
        plt.legend(loc='upper right')
        
        plt.tight_layout()

        # Save and show the plot
        self.save_plot('epidemiological_states_over_time.pdf', show_plot=True)
        plt.show()

    def plot_civilian_vs_soldier_deaths_correlation(self):
        civilian_deaths = [step_data['civilian_deaths'] for step_data in self.simulation_data]
        soldier_deaths = [step_data['soldier_deaths'] for step_data in self.simulation_data]
    
        plt.figure()
        plt.scatter(civilian_deaths, soldier_deaths, alpha=0.6)
        m, b = np.polyfit(civilian_deaths, soldier_deaths, 1)
        plt.plot(civilian_deaths, m*np.array(civilian_deaths) + b, color='red')
        plt.xlabel('Number of Civilian Deaths', fontsize=14)
        plt.ylabel('Number of Soldier Deaths', fontsize=14)
        self.save_plot('civilian_vs_soldier_deaths_correlation.pdf')
        plt.show()
        
    def plot_death_toll_vs_hospital_capacity_and_commute_rate(self):
        hospital_capacity = sorted(set(run_data['hospital_capacity'] for run_data in self.simulation_data))
        
        # Separate commute rates for civilian and war zones
        commute_rate_c = sorted(set(run_data['commute_rate_c'] for run_data in self.simulation_data))
        commute_rate_w = sorted(set(run_data['commute_rate_w'] for run_data in self.simulation_data))

        # Initialize death toll matrices
        death_toll_c = np.zeros((len(commute_rate_c), len(hospital_capacity)))
        death_toll_w = np.zeros((len(commute_rate_w), len(hospital_capacity)))

        for run_data in self.simulation_data:
            # Update death toll matrix for civilian commute rates
            i_c = commute_rate_c.index(run_data['commute_rate_c'])
            j_c = hospital_capacity.index(run_data['hospital_capacity'])
            death_toll_c[i_c, j_c] += run_data['deaths']

            # Update death toll matrix for war commute rates
            i_w = commute_rate_w.index(run_data['commute_rate_w'])
            j_w = hospital_capacity.index(run_data['hospital_capacity'])
            death_toll_w[i_w, j_w] += run_data['deaths']

        # Normalize the death toll by the number of simulation runs
        death_toll_c /= len(self.simulation_data) / (len(commute_rate_c) * len(hospital_capacity))
        death_toll_w /= len(self.simulation_data) / (len(commute_rate_w) * len(hospital_capacity))

        # Plot for civilian commute rates
        plt.figure()
        plt.imshow(death_toll_c, cmap='hot', interpolation='nearest', aspect='auto',
                extent=[hospital_capacity[0], hospital_capacity[-1], commute_rate_c[0], commute_rate_c[-1]])
        plt.colorbar(label='Overall Death Toll')
        plt.xlabel('Hospital Capacity', fontsize=14)
        plt.ylabel('Civilian Commute Rate', fontsize=14)
        self.save_plot('death_toll_vs_hospital_capacity_and_civilian_commute_rate.pdf')
        plt.show()
     
        # Plot for war commute rates
        plt.figure()
        plt.imshow(death_toll_w, cmap='hot', interpolation='nearest', aspect='auto',
                extent=[hospital_capacity[0], hospital_capacity[-1], commute_rate_w[0], commute_rate_w[-1]])
        plt.colorbar(label='Overall Death Toll')
        plt.xlabel('Hospital Capacity', fontsize=14)
        plt.ylabel('War Commute Rate', fontsize=14)
        self.save_plot('death_toll_vs_hospital_capacity_and_war_commute_rate.pdf')
        plt.show()

    def plot_deaths_by_patient_administration_policies(self, averages_per_policy_mean, averages_per_policy_std):
        policies = {
            'no_policy':'blue',
            'random':'orange',
            'dqn':'green',
            'prioritize_soldiers':'purple',
            'prioritize_civilians':'yellow'
        }
        
        value_to_compare = 'deaths'
        number_of_steps = len(self.simulator.simulation_runs_data)

        # Precompute aggregate mean and std for each policy across all steps
        policy_means = []
        policy_stds = []
        policy_labels = []

        for policy, color in policies.items():
            # Extract mean values across all steps
            mean_values = [averages_per_policy_mean[policy][step][value_to_compare] for step in range(number_of_steps)]
            
            # For error bars, we can:
            # Option 1: Use the standard deviation of these mean values
            # (This represents variability of the mean across steps.)
            mean_of_means = np.mean(mean_values)
            std_of_means = np.std(mean_values)
            
            # Append results
            policy_means.append(mean_of_means)
            policy_stds.append(std_of_means)
            policy_labels.append(policy)

        # Now we only have one bar per policy
        
        x_positions = np.arange(len(policies))
        bar_width = 0.5

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(x_positions, policy_means, bar_width, yerr=policy_stds, capsize=5,
            color=[policies[p] for p in policy_labels], alpha=0.7)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(policy_labels, fontsize=10)
        ax.set_xlabel('Patient Administration Policies', fontsize=14)
        ax.set_ylabel('Overall Deaths', fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        self.save_plot('deaths_by_patient_administration_policies.pdf')
    
    def plot_deaths_by_optimizaion_horizons(self, averages_per_policy_mean, averages_per_policy_std):
        number_of_steps = len(self.simulator.simulation_runs_data)
        steps = np.arange(number_of_steps)

        policies = {1:'blue',
                    7:'orange',
                    14:'green',
                    28:'purple'}

        value_to_compare = 'deaths'
        plt.figure()
        for policy, color in policies.items():
            mean_values = []
            std_values = []
            for step in range(number_of_steps):
                mean_values.append(averages_per_policy_mean[policy][step][value_to_compare])
                std_values.append(averages_per_policy_std[policy][step][value_to_compare])
            plt.plot(steps, mean_values, color, marker='o', label=f'{policy} days')
            plt.fill_between(steps, np.array(mean_values) - np.array(std_values), 
                            np.array(mean_values) + np.array(std_values), color=color, alpha=0.2)

        # Customizing the plot to match the provided style
        plt.xlabel('Steps', fontsize=14)
        plt.ylabel('Number of deaths', fontsize=14)
        plt.legend()
        plt.grid(True)
        self.save_plot('deaths_by_optimizaion_horizons.pdf')
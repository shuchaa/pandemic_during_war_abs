# library imports
from collections import defaultdict

# project imports
from .Agents import Agent, State


class Population:
    def __init__(self):
        """
            For making the code more efficient and make every calculation accessible
            with ~O(1) complexity for lookups. We created a class that stores and manages
            agents for faster access and statistics calcuations per node.
        """
        self._agent_map = defaultdict(list) # Map (agent_type, army, is_injured, state) to agents

    def __repr__(self):
        return f"<Population - {len(self._agent_map)} agents>"

    def __str__(self):
        return f"<Population - {', '.join(self._agent_map[:min(5, len(self._agent_map))])}>"

    def add_agent(self, agent: Agent):
        """Adds an agent to the population."""
        key = (agent.agent_type, agent.army, agent.is_injured, agent.state)
        self._agent_map[key].append(agent)

    def remove_agent(self, agent: Agent):
        """Removes an agent from the population."""
        key = (agent.agent_type, agent.army, agent.is_injured, agent.state)
        self._agent_map[key].remove(agent)

    def get_agents_by_criteria(self, agent_types=None, armies=None, is_injured=None, states=None):
        """Retrieves agents based on given criteria.
        Each parameter can get either specific value or a list of strings.
        If a parameter is None, we'll iterate over every possible parameter.
        """
        # input validation
        if agent_types is None:
            agent_types = ('civilian', 'soldier')
        elif not isinstance(agent_types, list):
            agent_types = [agent_types]
        
        if armies is None:
            armies = ['A','B', None] # None is for civilians which doesn't have an army
        elif not isinstance(armies, list):
            armies = [armies]
                
        if is_injured is None:
            is_injured = [True, False, None] # None is for civilians which can't be injured
        elif not isinstance(is_injured, list):
            is_injured = [is_injured]
            
        if states is None:
            states = list(State)
        elif not isinstance(states, list):
            states = [states]
        
        result = []
        for agent_type in agent_types:
            for army in armies:
                for injury_state in is_injured:
                    for state in states:
                        key = (agent_type, army, injury_state, state)
                        result.extend(self._agent_map.get(key, []))
        return result
    
    def get_agent_counts_by_criteria(self, agent_types=None, armies=None, is_injured=None, states=None):
        """Calculates statistics for the population based on given criteria."""
        return len(self.get_agents_by_criteria(agent_types, armies, is_injured, states))
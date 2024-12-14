# library imports
from enum import Enum, auto

class State(Enum):
    SUSCEPTIBLE = auto()
    EXPOSED = auto()
    ASYMPTOMATIC_INFECTED = auto()
    SYMPTOMATIC_INFECTED = auto()
    RECOVERED = auto()
    DEAD = auto()

class Agent:
    def __init__(self, agent_type, location, state, army, location_graph):
        self.agent_type = agent_type # "civilian" or "soldier"
        self.location = location
        self.last_non_hospital_node = location
        self.location_graph = location_graph
        self.hospital = False
        self._state = state
        self.army = army # 'A' or 'B' for soldiers, None for civilians
        self.inner_clock = 0
        self._is_injured = False
        if army:
            self.population_str = f'soldiers_{self.army}'
        else:
            self.population_str = 'civilians'

    def __repr__(self):
        return f"<Agent({self.agent_type}): loc={self.location}, epi_state={self._state}>"

    def __str__(self):
        return self.__repr__()
        
    @property
    def is_injured(self) -> bool:
        """Getter for the is_injured attribute."""
        return self._is_injured

    @is_injured.setter
    def is_injured(self, value: bool) -> None:
        """Setter for the is_injured attribute.
        We have to use it to re-order the indexes in the Population class"""
        if not isinstance(value, bool):
            raise ValueError("is_injured must be a boolean value.")
        self.location_graph.graph.nodes[self.location]['population'].remove_agent(self)
        self._is_injured = value
        self.location_graph.graph.nodes[self.location]['population'].add_agent(self)
    
    @property
    def state(self) -> State:
        """Getter for the state attribute."""
        return self._state

    @state.setter
    def state(self, new_state: State) -> None:
        """Setter for the state attribute.
        We have to use it to re-order the indexes in the Population class"""
        if not isinstance(new_state, State):
            raise ValueError("is_injured must be a boolean value.")
        if new_state == self._state:
            return
        self.location_graph.graph.nodes[self.location]['population'].remove_agent(self)
        self._state = new_state
        if self._state != State.DEAD:
            self.location_graph.graph.nodes[self.location]['population'].add_agent(self)
        else:
            self.location = None

    def move_agent(self, new_location):
        """
            Function to move an agent to any node within the graph.
            Prior to calling this function, the caller should validate the movement is allowed.
            That the new location node is connected to the current agent's location node by an edge.
            This function updates the "last_non_hospital_node" parameter, and the location_graph.

        Parameters
        ----------
            new_location : int
                The new node to move the agent to.
        """
        self.location_graph.graph.nodes[self.location]['population'].remove_agent(self)
        self.location = new_location
        if not self.location_graph.graph.nodes[self.location]['is_hospital']:
            self.last_non_hospital_node = self.location
        self.location_graph.graph.nodes[new_location]['population'].add_agent(self)


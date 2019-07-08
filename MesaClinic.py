from mesa import Agent, Model
from mesa.time import RandomActivation
import networkx as nx

class ClinicAgent(Agent):
    """ An agent with fixed initial wealth."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.wealth = 1
        self.queue=[]
        self.edges_to = []

    def step(self):
        # The agent's step will go here.
        other_clinic = self.random.choice(self.model.schedule.agents)
        other_clinic.queue.append(1)

class ClinicModel(Model):
    """A model with some number of agents."""
    def __init__(self, N):
        self.G = nx.DiGraph()
        self.num_agents = N
        self.schedule = RandomActivation(self)
        # Create agents
        for i in range(self.num_agents):
            clinic = ClinicAgent(i, self)
            self.G.add_node(clinic.unique_id, queue = clinic.queue)
            self.schedule.add(clinic)
        self.set_network(0.5)

    def set_network(self,p):
        '''
        A method to setup the network consultation
        parameter:
        p -  parameter for attachement
        '''
        self.network = nx.generators.random_graphs.erdos_renyi_graph(self.num_agents,p)
        for  eO,eT in self.network.edges():
            self.G.add_edge(eO,eT)




    def step(self):
        '''Advance the model by one step.'''
        self.schedule.step()


#####
cm  = ClinicModel(20)
for i in range(10):
    cm.step()

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.batchrunner import BatchRunner
import networkx as nx
from numpy.random import exponential, randint,poisson, binomial
from numpy import mean, power
from math import ceil
from random import sample,uniform,seed
import json
from networkx.readwrite.json_graph import cytoscape_data
from mesa.datacollection import DataCollector
from matplotlib import pyplot as plt

class CytoScape():
    '''
    A simple testing of migrating nx to cytoscape
    '''
    def __init__(self, N, p):
        '''
        create a random graph
        '''
        self.G = nx.erdos_renyi_graph(N,p).to_directed()
        # return nx.readwrite.json_graph.cytoscape_data(self.G)

class NetObj():
    '''
    A set of methods to generate, read and write networks
    ''' 
    def __init__(self,n,p):
        '''
        A method to initialize the network, 
        params:
        n - number of  nodes
        p -  prob of edge between nodes
        '''
        iter =n*2
        self.g = nx.MultiGraph() # create a multi graph
        self.g.add_nodes_from(range(n))
        for i in range(iter):
          n1,n2 = sample(range(n),2)
          self.g.add_edge(n1,n2, key = round(exponential(float(randint(5,10)))))
          
                    


class ClinicAgent(Agent):
    """ An agent object with a set of attributes"""
    def __init__(self, unique_id, model):
        seed(123)
        super().__init__(unique_id, model)
        
        self.queue=[] # which stop code is in line
        self.queue_size = len(self.queue)
        self.queue_time = [] # time stamp from removal from queue of clinic at that location
        self.edges_to = [] # which edges the node hass
        self.stop_code = 0
        self.factor = 1.0
        self.theta = 7 #uniform(50,100) # time constant for processing of a request
        self.activation = 1.0 # set activation, the probability to recieve a consult from another clinic

    def set_activation_prob(self):
        '''
        A method to set the prob of a clinic to recieve a consult from another clinic
        '''
        self.activation = min(1.0, power(1.0/float(self.queue_size+0.1,),2))

    def link_func(self):
        '''
        A function to link the length of the queue with the time of processiong/probability of node removal
        '''
        k = uniform(0,1)       
        print('q is:',self.queue_size)
        return k*self.queue_size +10

    def step(self):
        # The agent's step will go here.
        print('----the step is ---------:', self.model.schedule.steps)
        print('calling consult')
        self.consult()
        print('*****calling remove_queue*****')
        self.resolve_consult()
        print('####updating ####')
        
    
  
    def consult(self):
        '''
        A method to create an edge between clinics where one clinic consults the other
        ie adding a consult
        '''

        source = self.random.choice(self.model.schedule.agents) # pick a random clinic 
        nbunch = list(self.model.G[source]) #  get the outedges...
       
        
        if (nbunch != []):
          target = sample(nbunch, 1)[0]  # pick a random edge from that node
         
          if (1):
        #   if (target.activation):
            
            
            self.model.G.add_edge(source,target,key = ceil(exponential(target.theta)) + self.model.schedule.steps)
            target.queue_size = self.model.G.in_degree(target)
            target.theta = target.link_func()
            print('added an edge')
            
            
          
        
              
    def resolve_consult(self):
        '''
        A method to resolve the consult and remove the edge from the network
        '''
        clinic = self.random.choice(self.model.schedule.agents)
        a = [[k,v ] for k,v in dict(self.model.G[clinic]).items()]
        
        remove_list = [{aa[0]:
                [f for f in filter(lambda x: x<= self.model.schedule.steps,list(dict(aa[1]).keys())) ]} for aa in a
                ]         
        for r in remove_list:
            for k in r.keys():
                for n in r[k]:
                    self.model.G.remove_edge(clinic,k, key=n)
                    k.queue_size = self.model.G.in_degree(k)
                    print('removed an edge')
       

class ClinicModel(Model):
    """A model with some number of agents."""
    def __init__(self,net):
        self.G =  nx.MultiDiGraph() # create an empy clinic graph
        
        
        self.schedule = RandomActivation(self)
        self.clinic_dict = {} 
        # Create agents and add to graph
        for i in range(net.g.number_of_nodes()):  # create clinic agents as there are clinics
            clinic = ClinicAgent(i, self) 
            clinic.stop_code = i+500
            self.clinic_dict[i] = clinic
            self.G.add_node(clinic, queue = clinic.queue)   
            self.schedule.add(clinic)
        #  add edges
        for e1,e2 in net.g.edges():
            for key in dict(net.g[e1][e2]).keys(): 
               self.G.add_edge(self.clinic_dict[e1], self.clinic_dict[e2],key=key) 

        self.datacollector = DataCollector(
            model_reporters={"load": get_load_count},  # `compute_gini` defined above
            agent_reporters = {'queue_size':"queue_size"})

    def step(self):
        '''Advance the model by one step.'''
        self.datacollector.collect(self)
        self.schedule.step()

        print('cm step is:',self.schedule.time)
   



class Utils():
    '''
    A class of utilities for the ABM model
    '''
    def __init__(self,cm):
        '''
        Intializing the model with a MESA object
        '''
        self.agents = cm.schedule.agents
        self.show = [a for a in cm.schedule.agents]
        

    def get_edge_id(self):
        '''
        A method to retunr the nodes id connected by an edge in the clinic graph
        '''
        return {(e1.unique_id,e2.unique_id): cm.G[e1][e2]['weight'] for e1,e2 in cm.G.edges()}
    
    
   

    
    def cyto_2_json(self):
        '''
        Dump the cytoscape graph to a json file
        '''
        nodes = [n.unique_id for n in cm.G.nodes()]
        edges = [(e0.unique_id,e1.unique_id) for e0,e1 in cm.G.edges()]
        g = nx.DiGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        graph = cytoscape_data(g)
        with open('../cy/cytoABM.json','w') as f:
            json.dump(graph['elements'], f, indent = 4)
        return graph

def get_load_count(model):
    '''
    A method to return the  number of active edges with each clinic
    '''
   
    load = [len(list(dict(model.G[clinic]).keys())) for clinic in model.schedule.agents]
    return load

def get_clinic_times(model):
    '''
    A method to return the mean time for processing
    '''
    step0 = [v  for v in [list(dict(model.G[clinic]).values()) for clinic in [c for c in model.schedule.agents]]] 
    return step0

#####

net = NetObj(10,0.5) # construct the network object
cm  = ClinicModel(net)
u = Utils(cm)
#
steps_number = 300
for i in range(steps_number): 
    cm.step()
# plot results
d = [cm.datacollector.get_agent_vars_dataframe().xs(n).queue_size.quantile([0.25,0.5,0.75]).tolist() for n in range(steps_number)]
v = [cm.datacollector.get_agent_vars_dataframe().xs(n).std() for n in range(steps_number)]
plt.plot(d)
plt.show()
# cm.datacollector.get_agent_vars_dataframe()

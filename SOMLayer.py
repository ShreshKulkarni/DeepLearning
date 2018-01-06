"""
Created on Sun Nov  5 06:40:19 2017

@author: Shreshtha Kulkarni
"""
    
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min

#Defines a particular node of the SOM.
#members: codebook_vec or weights, position assuming rectangular topography
#         input_vec_len: dimensions/features in input_data
#         lattice_type: shape of neighborhood area: rectangle/hexagon
class Node:
     #initialize node   
    def __init__(self,pos_x,pos_y,input_vec_len,lattice_type):
        self.codebook_vec = np.random.rand(1,input_vec_len)
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.lattice_type = lattice_type
    
             
    #evaluate if the passed node is a neighbor
    def isMyNeighbor(self,node,infl_radius):
        dist = np.linalg.norm(np.array([self.pos_x,self.pos_y]) - np.array([node.pos_x,node.pos_y]))
        if((self.lattice_type == "rect") and (pow(dist,2) < pow(infl_radius,2))):
            return True
        
        if(self.lattice_type == "hex"):
            R = 1.5*np.sqrt(3)*infl_radius*infl_radius
            if(pow(dist,2) <= R):
                return True
        
        return False

#Defines the SOM network. This will take care of training, updating the learning
#parameters and wts/codebook_vecs
class SOMNetwork:
    
    #Initialise the network. Internally initializes each node individually.
    #params: no_neurons = no. of nodes
    #        input_vec_len = features in input data
    #        topo = topology of network: line if 1D, rect if 2D
    #        topo_x, topo_y = length,width of the rectangle
    #        eta_0 = Initial learning rate
    #        eta_decay_method = method to decay eta: lin <linear>, exp, inv (inverse)
    #        dist_method = distance metric to be used. default is euclidean
    #        lattice_type = as used in node distance computation. default:rect
    def __init__(self,no_neurons,input_vec_len,topo,topo_x,topo_y,eta_0,eta_decay_method
                 ,dist_method,neighborhood_fun,lattice_type):
        if lattice_type is None or lattice_type == "":
            lattice_type = "rect"
        
        if(topo == "line"):
            self.neuron_mat = [Node(0,x,input_vec_len,lattice_type) for x in range(no_neurons)]
        elif(topo == "rect"):
            self.neuron_mat = []
            if(topo_x*topo_y != no_neurons):
                raise ValueError("topo_x*topo_y should be equal to no_neurons")
            for x in range(topo_x):
                for y in range(topo_y):
                    self.neuron_mat.append(Node(x,y,input_vec_len,lattice_type))
        else:
            raise ValueError("Unsupported topology type. Valid type: line,rect")
            
        self.no_neurons = no_neurons
        self.topo = topo
        self.tau_1 = 1000 #some constants to decay influence radius
        self.tau_2 = 1000 #constant to decay learning rate
        self.eta = eta_0
        self.eta_0 = eta_0
        
        self.infl_radius_0 = 1 if topo== "line" else np.linalg.norm([topo_x,topo_y])
        self.infl_radius = self.infl_radius_0
        
        if(dist_method  == "" or dist_method == None):
            self.dist_method = "euclidean"
        else:
            self.dist_method = dist_method
        
        if(eta_decay_method in ["exp","lin","inv"]):
            self.eta_decay_method = eta_decay_method
        elif(eta_decay_method == "" or eta_decay_method == None):
            self.eta_decay_method = "exp"
        else:
            raise ValueError("Invalid value %s for eta_decay_method.Valid are exp, lin or inv" %(eta_decay_method))
            
        if(neighborhood_fun in ["gaussian","bubble"]):
            self.neighborhood_func = neighborhood_fun
        elif(neighborhood_fun == "" or neighborhood_fun == None):
            self.neighborhood_func = "gaussian"
        else:
            raise ValueError("Invalid value %s for neighborhood_fun.Valid are gaussian or bubble" %(neighborhood_fun))
    
    #Get the best matching unit
    def getBMU(self,input_vec):       
        wt_mat = []
        for i in range(self.no_neurons):
            wt_mat.append(self.neuron_mat[i].codebook_vec[0])
        #from IPython.core.debugger import Tracer; Tracer()()
        res = pairwise_distances_argmin_min(np.array(input_vec).reshape(1,-1),wt_mat,metric = self.dist_method)
            
        return (self.neuron_mat[res[0][0]],res[1][0])
    
    #Decay learning rate
    def setDecayedEta(self,epoch):
        if(self.eta_decay_method == "exp"):
            self.eta = self.eta_0 * np.exp(-epoch/self.tau_2) #Exponential decay
        elif(self.eta_decay_method == "lin"):
            self.eta = self.eta_0 *(1- epoch/self.tau_2) #linear decay
        else:
            self.eta = self.eta_0/(epoch+1) #inverse decay
    
    #Decay influence radius
    def setDecayedInfluenceRadius(self,epoch):
        self.infl_radius = self.infl_radius_0 * np.exp(-epoch/self.tau_1)
    
    #Calculate influence value as per a guassian distribution function
    def calculateInfluenceValue(self,neighbor_node,bmu_node):
        #calculate the positional distance in the topographic map of SOM.
        #Its not the distance between wt vectors
        dist = np.linalg.norm(np.array([bmu_node.pos_x,bmu_node.pos_y])-
                                  np.array([neighbor_node.pos_x,neighbor_node.pos_y]))
        #dist=1
        return np.exp(-pow(dist,2)/(2*pow(self.infl_radius,2)))
    
    #Update the codebook vectors
    def updateWeights(self,bmu_node,neighbor_node,input_vec):
        #from IPython.core.debugger import Tracer; Tracer()()
        if(self.neighborhood_func == "gaussian"):
            infl_val = self.calculateInfluenceValue(neighbor_node,bmu_node)
            neighbor_node.codebook_vec += self.eta*infl_val*(input_vec - neighbor_node.codebook_vec)
        else:
            neighbor_node.codebook_vec += self.eta*(input_vec - neighbor_node.codebook_vec) #bubble update

    def updateParameters(self,epoch):
        if(self.eta > 0.01):
            self.setDecayedEta(epoch)
        #Do not update influence radius when it is very small.Assuming that
        #neighboorhood has already reduce to single neuron
        if(self.infl_radius > 0.001):
            self.setDecayedInfluenceRadius(epoch)
    
    #Finp the neighbors of BMU.
    def findNeighbors(self,bmu_node):
        bmu_node_ind = self.neuron_mat.index(bmu_node)
        neighbor_coords = []
                
        if(self.topo == "line"):
            if(self.infl_radius < 1):
                return neighbor_coords #return empty neighbors as neighborhood has shrunk to single neuron
                                       #else infl_val becomes too small producing NaNs in wt update.
            if(bmu_node_ind == 0): #if end neuron
                neighbor_coords.append(1) 
            elif(bmu_node_ind == (len(self.neuron_mat)-1)): #if other end neuron
                neighbor_coords.append(len(self.neuron_mat) - 2)
            else:
                neighbor_coords.append(bmu_node_ind - 1)
                neighbor_coords.append(bmu_node_ind + 1)
        elif(self.topo == "rect"):
            for i in range(len(self.neuron_mat)):
                #Do not append bmu_node as infl_value for itself is 1
                if((i!=bmu_node_ind) and (bmu_node.isMyNeighbor(self.neuron_mat[i],self.infl_radius))):
                    neighbor_coords.append(i)
        return neighbor_coords
            
    #Function to train the network
    def train(self,input_data,num_epochs):
        #Set the decay constants to equal to num_epochs
        self.tau_1 = num_epochs/np.log(self.infl_radius_0)
        self.tau_2 = num_epochs
        train_history = dict()
        net_output = dict()
        row_ind = 0
        bmu_list=[]
        eta_history =[]
        mse_history=[]
        infl_rad_history =[]
        repeatInputSet = True
    
        
        for k in range(num_epochs):
            #This loop is to ensure all the training instances are used once atleast.
            #Once all are used they are repeated.
            while((repeatInputSet == True) or 
                  ((row_ind in net_output.keys()) and (len(net_output.keys()) < len(input_data)))):
                row_ind = np.random.randint(0,len(input_data))
                repeatInputSet = False
            
            bmu_node,bmu_dist = self.getBMU(input_data[row_ind]) #get best matching unit
            if(bmu_node == None):
                print("row:%d epoch:%d" %(row_ind,k)) #troubleshooting
            
            neighbor_coords = self.findNeighbors(bmu_node) #find neighbors
            #to update the weights of bmu_node. infl_val =1
            bmu_node.codebook_vec += self.eta*(input_data[row_ind] - bmu_node.codebook_vec) 
            
            #update the neighbor wts. 
            for coord in neighbor_coords:
                self.updateWeights(bmu_node,self.neuron_mat[coord],input_data[row_ind])
        
            #decay learning parameters, but record them prior to update for 
            #generating some graphs at the end
            eta_history.append(self.eta)
            infl_rad_history.append(self.infl_radius)
            self.updateParameters(k)
            
            #Return the output of network in form of one hot vector
            op = np.zeros(self.no_neurons)
            op[self.neuron_mat.index(bmu_node)] =1
            net_output[row_ind] = op
                      
            #Calculate and store the distance of input_vec from BMU for calculating mse 
            #once all the training samples are utilised.
            bmu_list.insert(row_ind,bmu_dist)
            if(len(bmu_list) == len(input_data)):
                #from IPython.core.debugger import Tracer; Tracer()()
                mse = sum(bmu_list)/len(input_data)
                print("MSE : %0.2f" %(mse))  
                mse_history.append(mse) #for future plots
                bmu_list=[]
                repeatInputSet = True
        #from IPython.core.debugger import Tracer; Tracer()()
        train_history['eta'] = eta_history
        train_history['mse'] = mse_history
        train_history['irad'] = infl_rad_history
        return(train_history,net_output)
    
    #Map the input data to network nodes and output the one hot vector representation
    #output where BMU =1 and other nodes =0
    def mapInput(self,input_data):
        node_output = [] #This will be the layer output in the form of one hot vector
        
        for i in range(len(input_data)):
            bmu_node,bmu_dist = self.getBMU(input_data[i])
            op = np.zeros(  self.no_neurons)
            op[self.neuron_mat.index(bmu_node)] =1
            node_output.insert(i,op)
        
        return np.array(node_output)
    
    '''def dotProdSim(self,input_vec):
        dp = []
        for i in range(self.no_neurons):
            dp[i] = np.dot(input_vec,self.neuron_mat[i].codebook_vec[0])
        
        return self.neuron_mat[dp.index(max(dp))]'''

#Extra layer to map the SOM output in the target format. For accuarcy calculation.
#it uses and outstar representation of the network.
class GrossbergLayer:
    def __init__(self,in_neurons,out_neurons,eta):
        self.in_neurons = in_neurons
        self.out_neurons = out_neurons
        self.eta = eta
        self.wts = np.random.rand(out_neurons,in_neurons)
        
    def generateResult(self,input_vec):
        y_hat = np.matmul(self.wts,input_vec)
        return y_hat
    
    def train(self,output,target,input_vec):
        self.wts += np.matmul(self.eta*(target - output), input_vec.T)
        
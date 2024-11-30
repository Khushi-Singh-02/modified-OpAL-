import numpy as np
from scipy.stats import beta
import random
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import csv
from scipy.optimize import minimize
from scipy.stats import expon


# The model OpAL*, modified for fitting
class OpAL:
	def __init__(self, n_trials, crit, v0, rat_choice, rat_reward, n_options=2, r_mag=[1,1], l_mag=[0,0], norm_type = None, disp=True):

		def init_crit(crit):

			if crit == "SA":  	
				return np.zeros((sz+1, n_options)) + v0
			else:
				err = 'Invalid value given for arg crit. %s given' %crit
				raise Exception(err)

		sz = n_trials
		self.n_trials = sz
		self.n_options = n_options					    # number of choices
		self.crit = crit 							    # critic type
		self.V = init_crit(crit)					    # critic
		self.QG = np.zeros((sz+1, n_options)) + 1	    # Go Actor Values
		self.QN = np.zeros((sz+1, n_options)) + 1	    # NoGo Actor Values
		self.alphags = np.zeros(sz)	    				# Go Actor Learning Rate
		self.alphans = np.zeros(sz)	    				# NoGo Actor Learning Rate
		self.beta_g = np.zeros(sz)	    				# Inverse temp, Go Actor
		self.beta_n = np.zeros(sz)	    				# Inverse temp, NoGo Actor
		self.Act = np.zeros((sz, n_options))		    # Act values for softmax
		self.H = np.zeros(sz) 						    # entropy
		self.gammas = np.zeros(sz+1)				    # decay to prior gammas
		self.SM  = np.zeros((sz, n_options))		    # Softmax values
		self.rho = np.zeros(sz)						    # DA at choice
		self.C = rat_choice			    ###################### pass choice data ######################
		self.R = rat_reward			    ###################### pass reward data ######################
		self.beta_dist = np.zeros((sz+1,2)) +1		    # beta distribution of avg reward
		self.r_mag = r_mag							    # magnitude of reward for choice
		self.l_mag = l_mag							    # magnitude of loss for choice
		self.PE  = np.zeros(sz)			    			# choice PE
		self.idx = 0						    		# idx of trial in state
		self.norm_type = norm_type				    	# anneal the learning rate
		self.disp = disp

	def policy (self):
		idx = self.idx
		beta_g = self.beta_g[idx]
		beta_n = self.beta_n[idx]
		crit  = self.crit

		Act = beta_g*self.QG[idx] - beta_n*self.QN[idx]
		self.Act[idx,:] = Act
		newAct = Act - np.max(Act)
		expAct = np.exp(newAct)
		ps = expAct/np.sum(expAct)
		self.SM[idx,:] = ps ############################### NEED TO RETRIEVE ###################################
		self.H[idx] = -np.sum(ps*np.log2(ps))
		if np.isnan(self.H[idx]):
			self.H[idx] = 0

		C= self.C[idx] ###################### Just fetches choice. ########################
		reward= self.R[idx] ###################### Just fetches reward. ########################
		if crit == "SA":
			PE = reward - self.V[idx,C]
		else:
			err = 'Invalid value given for arg crit. %s given' %crit
			raise Exception(err)
		self.PE[idx] = PE
		epsilon = 1e-10
		if ps[C]==0:
			return epsilon          # to avoid log(0) error if ps[C] is too small 
		return ps[C]

	def critic (self,alpha):
		idx = self.idx
		C = self.C[idx]
		PE = self.PE[idx]
		crit = self.crit
		if crit == "SA":
			self.V[idx+1] = self.V[idx]
			self.V[idx+1,C] = self.V[idx,C] + alpha*PE
		else:
			err = 'Invalid value given for arg crit. %s given' %crit
			raise Exception(err)

	def act (self,alphaGo, alphaNoGo,norm=False,mag=1, hebb=True):
		idx = self.idx
		PE = self.PE[idx]
		C = self.C[idx]
		self.QG[idx+1] = self.QG[idx]
		self.QN[idx+1] = self.QN[idx]

		if norm:
			if self.norm_type is None:
				PE = PE/mag
			else:
				err = 'Norm type %s is not supported' %(self.norm_type)
				raise Exception(err)

		rho = self.rho[idx]
		alphag = alphaGo*(1. + rho)
		alphan = alphaNoGo*(1. - rho)

		# constrain between [0,1]
		alphag = np.max([0,np.min([alphag,1])])
		alphan = np.max([0,np.min([alphan,1])])

		self.alphags[idx] = alphag
		self.alphans[idx] = alphan

		if hebb:
			updateG = alphag*self.QG[idx,C]*PE
			updateN = alphan*self.QN[idx,C]*-PE
		else:
			updateG = alphag*PE
			updateN = alphan*-PE

		self.QG[idx+1,C] = self.QG[idx,C] + updateG
		self.QN[idx+1,C] = self.QN[idx,C] + updateN

		# actor values should not be negative, represent activations
		self.QG[idx+1,C] = np.max([0,self.QG[idx+1,C]])
		self.QN[idx+1,C] = np.max([0,self.QN[idx+1,C]])

	def decay_to_prior (self,prior=1,gamma=1):
		idx = self.idx + 1
		gamma = gamma + gamma*self.H[idx-1]
		logit = 1/(1 + np.exp(-gamma))
		self.gammas[idx] = logit

		self.QG[idx] = self.QG[idx]*logit + prior*(1 - logit)
		self.QN[idx] = self.QN[idx]*logit + prior*(1 - logit)

# pass your data file here
data_file='dyna_CSV.csv'

# define negative log likelihood function
#set k in definition to change
def ll_func(params, rat_choice, rat_reward, v0=0.0, crit="SA", mod = "avg_value", k=1.5 , norm=False, mag=1, hebb=True, variant = None, decay_to_prior = True, decay_to_prior_gamma = 1):

	def calc_rho(state):
			chance = state.r_mag[0]*.5 + state.l_mag[0]*.5
			if mod == "avg_value":
				state.rho[t] = np.mean(state.V[t] - chance)*k
			else:
				err = 'Invalid value given for arg mod. \"%s\" given' %mod
				raise Exception(err)

			if variant is None:
				this_r = state.rho[t]
				state.beta_g[t] = np.max([0,beta_GO*(1 + this_r)])
				state.beta_n[t] = np.max([0,beta_NOGO*(1 - this_r)])
			else:
				err = 'Invalid value given for arg variant. \"%s\" given' %variant
				raise Exception(err)
			return state
    
	alpha_c=0.01    #change as needed
	beta_GO=2       #change as needed
	beta_NOGO=2     #change as needed
	n_trials=len(rat_choice)
	alpha_a_Go, alpha_a_NoGo = params
	ll=0

	agent = OpAL(n_trials, crit, v0, rat_choice, rat_reward)
	agent.params = params

	t_no=[]

	for t in range(n_trials):
		t_no.append(t)
		agent.idx = t
		agent = calc_rho(agent)
		cp=agent.policy()
		#print('cp:', cp)
		ll+=np.log(cp)
		#print('ll:', ll)
		agent.critic(alpha_c)
		agent.act(alpha_a_Go, alpha_a_NoGo,norm=norm,mag=mag,hebb=hebb)
		if decay_to_prior:
			agent.decay_to_prior(gamma=decay_to_prior_gamma)
	return ll*-1

# list all instnaces of each combination (id, grp, (session?)) in the simulated data
def find_all_occurrences(id, grp, sesh=None): #session optional, depending on if we want to filter by session 
    indexes = [] 
    with open(data_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        next(csv_reader, None)  
        
        for index, row in enumerate(csv_reader, start=1):
            if sesh:
                if row['ratID'] == str(id) and row['group'] == str(grp) and row['session']==str(sesh):
                    indexes.append(index)
            else:
                if row['ratID'] == str(id) and row['group'] == str(grp):
                    indexes.append(index)

    return indexes 

rat_ids = set()
grps=[1, 2] 
with open(data_file, 'r') as file:
    reader = csv.DictReader(file)  
    for row in reader:
        rat_id = row['ratID'] 
        rat_ids.add(rat_id) 
rat_ids_list = list(rat_ids)

PE=[]
ll_PE=[]
ids=[]
session=[]
sex=[]
AG=[]
AN=[]
G=[]

for id in rat_ids_list:
        for grp in grps:
            #print('id', id) #UNCOMMENT TO TRACK PROGRESS
            #print('grp', grp)
            indexes_ss = find_all_occurrences(str(id), str(grp))
            df = pd.read_csv(data_file, usecols=['session', 'sex'])
            sessions = set(df['session'].iloc[indexes_ss].tolist())
            sx=5 
            if indexes_ss:
                sx=df['sex'][indexes_ss[0]]
        
            for sesh in sessions:
                indexes= find_all_occurrences(str(id), str(grp), str(sesh))
                df = pd.read_csv(data_file, usecols=['choice', 'reward'])
                rat_choice = df['choice'].iloc[indexes].astype(int).to_numpy() - 1 
                rat_reward = df['reward'].iloc[indexes].to_numpy()
                fit_func = lambda x: ll_func(x, rat_choice, rat_reward)

                bounds = [(0, 1), (0, 1)]
                pe = []  
                ll = []
                for j in range(30): #30 initial guesses / session / rat			
                    X0 = [np.random.rand(), np.random.rand()]
                    result = minimize(fit_func, X0, bounds=bounds)
                    pe.append(result.x)
                    this_ll = ll_func(result.x, rat_choice, rat_reward)
                    ll.append(this_ll)
                ll_min = np.min(ll)
                ll_PE.append(ll_min)
                k = ll.index(ll_min)
                PE.append(pe[k])
                AG.append(pe[k][0])
                AN.append(pe[k][1])
                ids.append(id)
                session.append(sesh)
                sex.append(sx)
                G.append(grp)

# save results
results=pd.DataFrame({
	'ID': ids,
	'Session': session,
	'Sex': sex,
	'Group': G,
	'PE': PE,
	'll_PE':ll_PE,
	'ag': AG,
	'an': AN})

results.to_csv('Parameters_from_rat_data.csv', index=False)
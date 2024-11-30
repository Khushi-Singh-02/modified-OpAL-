# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank
import numpy as np
from opal_dash import OpAL
import Yifeng_task
#import os
#import sys
#cwd = os.getcwd()
#parent_dir = os.path.dirname(cwd)
#sys.path.insert(0, parent_dir)
#import opal
import random

#from environments import get_probs
from scipy.stats import beta as beta_rv
import matplotlib.pyplot as plt
import pandas as pd

import time

def simulate(params,n_states,n_trials, task, v0=0.0, crit="SA", mod = "constant", k=20., phi = 1, rho=0.0, baselinerho = 0.0, threshRho = 0, threshRand = 0, rnd_seeds = None, norm=False, mag=1, norm_type = None, hebb=True, variant = None,\
	anneal = True, T = 100.0, use_var=False, decay_to_prior = False, decay_to_prior_gamma = 1., save_warmup=False,disp=False):
	#task.set_task()
	def calc_rho(state):
		#print(rho)
		# use base rho for earlier trials
		if t < threshRho: 
			state.rho[t] = rho
		else:
			chance = state.r_mag[0]*.5 + state.l_mag[0]*.5 
			if mod == "constant": 		# artificially set rho to specific value, e.g. rho = 0
				state.rho[t] = rho + baselinerho
			elif mod == "avg_value":	# avg state critic value (if crit = SA)
				state.rho[t] = np.mean(state.V[t] - chance)*k
				if disp:
					print('state.rho[t], np.mean(state.V[t] - chance), k, state.V[t], chance')
					print(state.rho[t], np.mean(state.V[t] - chance), k, state.V[t], 'chance', chance)
			elif mod == "avg_gamble_value":
				if t == 0:
					state.rho[t] = 0
				else:
					uniform_dist = 0.5*((1+4/2)) + 0.5*(0.5*(2+8)/2 + 0.5*0) # EV if X, C were uniform distr (flat prior)
					state.rho[t] = k*(np.mean(.5*state.Cs[0:t] + .5*(0.5*state.Xs[0:t] +  0.5*0)) - uniform_dist) # EV of gambles til now relative to flat prior
			elif mod == "beta":
				# get the mean val of the environment
				# calculated in opal.py during critic update
				mean = state.mean[t]

				thresh = phi*state.std[t]
				cond1 = (mean - thresh) > .5 # lower bound above .5
				cond2 = (mean + thresh) < .5 # upper bound below .5
				sufficient = cond1 or cond2

				# is sufficiently above/below .5 
				if sufficient:
					# calc rho val direction
					# above p(reward) = .50 is positive, below negative
					# use mean of state
					chance_centered_val = (state.mean[t] - .5)
					state.rho[t] = chance_centered_val*k + baselinerho
				else:
					state.rho[t] = rho + baselinerho
			else:
				err = 'Invalid value given for arg mod. \"%s\" given' %mod
				raise Exception(err)

		# variants for comparison
		if variant is None:
			# standard value modulation
			this_r = state.rho[t]
			state.beta_g[t] = np.max([0,beta_GO*(1 + this_r)])
			#print('beta, this_r', beta, this_r)
			state.beta_n[t] = np.max([0,beta_NOGO*(1 - this_r)])
		'''
		elif variant == "flip":
			# flip the sign of modulation
			this_r = state.rho[t]
			this_r = -1.*this_r
			state.beta_g[t] = np.max([0,beta*(1 + this_r)])
			state.beta_n[t] = np.max([0,beta*(1 - this_r)])
		elif variant == "bmod":
			# rather than change bg and bg assymmetrically, 
			# changing rho only changes the overall choice temp.
			absrho = np.abs(state.rho[t])
			state.beta_g[t] = np.max([0,beta*(1 + absrho)])
			state.beta_n[t] = np.max([0,beta*(1 + absrho)])
		elif variant == "flipbmod":
			# rather than change bg and bg assymmetrically, 
			# changing rho only changes the overall choice temp.
			absrho = np.abs(state.rho[t])
			state.beta_g[t] = np.max([0,beta*(1 + absrho)])
			state.beta_n[t] = np.max([0,beta*(1 + absrho)])
		elif variant == "lrate":
			# rather than change bg and bg assymmetrically, 
			# changing rho only changes asymetry in learning rate
			state.beta_g[t] = beta  # only change alpha asym
			state.beta_n[t] = beta
		else:
			err = 'Invalid value given for arg variant. \"%s\" given' %variant
			raise Exception(err)
			'''

		return state
	
	
	def generate_state():
		probs = [0.45, 0.45]				
		n_options = len(probs)
		rmag=np.zeros(n_options) + 1 #+outcome?			#IF MANIPULATION, ADJUST	
		lmag=np.zeros(n_options) + 0 #+outcome?

		new_state = OpAL(n_trials,crit,v0,n_options,probs,rmag,lmag,anneal=anneal,use_var=use_var,T=T,norm_type=norm_type,disp=disp)	
		return new_state

	alpha_c, alpha_a_Go, alpha_a_NoGo, beta_GO, beta_NOGO = params
	#if disp:
		#print('alpha_c, alpha_a_Go, alpha_a_NoGo, beta', alpha_c, alpha_a_Go, alpha_a_NoGo, beta )
	states = []
	V=[]
	QG=[]
	QN=[]
	alphags=[]
	alphans=[]
	RPE=[]
	RHO=[]
	CHOICES=[]
	REWARDS=[]
	ACTIONS=[]
	SOFTMAX=[]
	ENTROPY=[]
	GAMMAS=[]
	BG=[]
	BN=[]
	
	df=pd.DataFrame()
	# let's do this thing (pls don't die again)
	for state in range(n_states):
		s=state
		task.set_task()
		# check if random seed provided for sim
		if rnd_seeds is not None:
				int_seed=int(rnd_seeds[state])
				random.seed(int_seed)
				np.random.seed(int_seed)

		state = generate_state()
		state.params = params

		reward_probs_left=[]
		reward_probs_right=[]
		good_choice=[]
		block_switch=[]
		t_no=[]
		RMAP=[]

		#warmup=random.random()	###############################################
		
		# begin simulation
		for t in range(n_trials):
			if disp:
				print('trial', t)
			t_no.append(t)

			if task.switch_tag[t]:
				block_switch.append(t)
			
			rmap=task.reward_maps[t]
			RMAP.append(rmap)
			state.probs = rmap
			if rmap[0]>rmap[1]:
				good_choice.append(0)
			elif rmap[0]==rmap[1]:
				good_choice.append(0.5)
			else:
				good_choice.append(1)
			if disp:
				print('rmap', rmap)

			reward_probs_left.append(rmap[0])
			reward_probs_right.append(rmap[1])

			state.idx = t
			# calculate rho, beta_g, beta_n
			state = calc_rho(state)	
			if disp:
				print('calculated rho, beta_g, beta_n')
			# pick an action and generate PE  modify, make model choose
			state.policy(thresh=0)
			if disp:
				print('made a choice, got PE')
			# update critic with PE
			state.critic(alpha_c)	
			if disp:
				print('updated critic')

			# update actors with PE 
			# with efficient coding, actors set explicitly, so no need to update
			state.act(alpha_a_Go, alpha_a_NoGo,norm=norm,mag=mag,hebb=hebb,var=variant) 	
			if disp:
				print('updated actor')


			# decay actors to prior?
			if decay_to_prior:
				state.decay_to_prior(gamma=decay_to_prior_gamma)
				if disp:
					print('decay_to_prior on')


			VALUES=state.V
			qg=state.QG
			qn=state.QN
			ag=state.alphags
			an=state.alphans
			pe=state.PE
			r=state.rho
			choices=state.C
			rewards=state.R
			actions=state.Act
			smax=state.SM
			entropy=state.H
			gammas=state.gammas
			bg=state.beta_g
			bn=state.beta_n

			
			if t<50:
				#print(t)
				if save_warmup:
					df=df._append({
					#'Session number': s+1,
        			'Trial number': t +1,
        			'Alpha_a_GO': alpha_a_Go,
        			'Alpha_a_NOGO': alpha_a_NoGo,
        			'Beta_GO': beta_GO,
					'Beta_NOGO': beta_NOGO,
					'Alpha_critic': alpha_c,
        			#'pleft': RMAP[t][0],
					#'pright': RMAP[t][1],
        			'Choice': choices[t]+1,
    				'Outcome': rewards[t]},
    				#'RPE': pe[t],
        			#'rho': r[t],
        			#'Alpha_GO': ag[t],
        			#'Alpha_NO_GO': an[t],
    				#'QG1': qg[t][0],
					#'QG2': qg[t][1],
    				#'QN1': qn[t][0],
					#'QN2': qn[t][1],
        			#'BG': bg[t],
        			#'BN': bn[t],
        			#'Q1': actions[t][0],
					#'Q2': actions[t][1],
    				#'pa1':smax[t][0],
					#'pa2':smax[t][1],
    				#'Entropy':entropy[t],
        			#'Gammas': gammas[t],
        			#'V1': VALUES[t][0],
					#'V2': VALUES[t][1]},
        			ignore_index=True)
			else:
				#print('saving now', t)
				df=df._append({
        			#'Session number': s+1,
        			'Trial number': t +1,
        			'Alpha_a_GO': alpha_a_Go,
        			'Alpha_a_NOGO': alpha_a_NoGo,
        			'Beta_GO': beta_GO,
					'Beta_NOGO': beta_NOGO,
					'Alpha_critic': alpha_c,
        			#'pleft': RMAP[t][0],
					#'pright': RMAP[t][1],
        			'Choice': choices[t]+1,
    				'Outcome': rewards[t]},
    				#'RPE': pe[t],
        			#'rho': r[t],
        			#'Alpha_GO': ag[t],
        			#'Alpha_NO_GO': an[t],
    				#'QG1': qg[t][0],
					#'QG2': qg[t][1],
    				#'QN1': qn[t][0],
					#'QN2': qn[t][1],
        			#'BG': bg[t],
        			#'BN': bn[t],
        			#'Q1': actions[t][0],
					#'Q2': actions[t][1],
    				#'pa1':smax[t][0],
					#'pa2':smax[t][1],
    				#'Entropy':entropy[t],
        			#'Gammas': gammas[t],
        			#'V1': VALUES[t][0],
					#'V2': VALUES[t][1]},
        			ignore_index=True)

			


		if disp:
			print('alpha_c, alpha_a, beta', alpha_c, alpha_a_Go, alpha_a_NoGo, beta)
		
			VALUES=state.V
			qg=state.QG
			qn=state.QN
			ag=state.alphags
			an=state.alphans
			pe=state.PE
			r=state.rho
			choices=state.C
			rewards=state.R
			actions=state.Act
			smax=state.SM
			entropy=state.H
			gammas=state.gammas
			bg=state.beta_g
			bn=state.beta_n

			window = np.ones(5) / 5
			moving_avg_choice = np.convolve(choices, window, 'valid')
			moving_avg_reward = np.convolve(rewards, window, 'valid')

			zone=np.zeros((len(block_switch), 20))
			made_good_choice=np.zeros(n_trials)
			for i in range(n_trials):
				if good_choice[i]==0.5:
					made_good_choice[i]=0.5
				else:
					if choices[i]==good_choice[i]:
						made_good_choice[i]=1
					else:
						made_good_choice[i]=0
			print(len(made_good_choice))

			for i in range(len(block_switch)):
				bs=block_switch[i]
				if bs+11<n_trials:
					zone[i, :]=made_good_choice[bs-9:bs+11]
				else:
					zone[i, :]=np.zeros(20)
				zone_avg=np.mean(zone, axis=0)

				#plt.plot(pre, post, zone)

			zone_choice=np.zeros((len(block_switch), 20))
			zone_good_choice=np.zeros((len(block_switch), 20))
			zone_reward=np.zeros((len(block_switch), 20))
			for i in range(len(block_switch)):
				bs=block_switch[i]
				if bs+11<n_trials:
					zone_choice[i, :]=choices[bs-9:bs+11]
					zone_good_choice[i, :]=good_choice[bs-9:bs+11]
					zone_reward[i, :]=rewards[bs-9:bs+11]
				else:
					zone_choice[i, :]=np.zeros(20)
					zone_good_choice[i, :]=np.zeros(20)
					zone_reward[i, :]=np.zeros(20)
				zone_choice_avg=np.mean(zone_choice, axis=0)
				zone_good_choice_avg=np.mean(zone_good_choice, axis=0)
				zone_reward_avg=np.mean(zone_reward, axis=0)

			
			
			fig, axs = plt.subplots(14, 1, figsize=(13, 20))


			axs[0].plot(VALUES)
			axs[0].set_title('VALUES', fontsize=8)
			axs[0].set_xlabel('Trial', fontsize=8)
			axs[0].set_ylabel('VALUES', fontsize=8)

			axs[1].plot(qg, label='qg')
			axs[1].plot(qn, label='qn')
			axs[1].legend(fontsize=8, loc='upper right')
			#axs[0].set_title('Reward Probabilities')
			#axs[0].set_xlabel('Trial')
			#axs[0].set_ylabel('Probability')
			#axs[0].set_yticks(np.arange(0, 1.1, 0.1))  
			#axs[0].set_yticklabels(['{:.1f}'.format(x) for x in np.arange(0, 1.1, 0.1)], fontsize=10)  


			axs[2].plot(ag, label='ag')
			axs[2].plot(an, label='an')
			axs[2].legend(fontsize=8, loc='upper right')

			# Choices
			axs[3].plot(pe)
			axs[3].set_title('pe',fontsize=8)
			#axs[1].set_xlabel('Trial')
			#axs[1].set_ylabel('Your Choices') #(0: No Reward, 1: Reward, 2: Double Reward)')

			axs[4].plot(r)
			axs[4].set_title('r', fontsize=8)

			axs[5].plot(choices, label='Choices')
			axs[5].plot(moving_avg_choice, color='black', label='Choice trace')
			axs[5].plot(good_choice, color='red', linestyle='dashdot', label='Optimal Choice')
			axs[5].set_title('Choices', fontsize=10)
			axs[5].legend(fontsize=8, loc='upper right')

			axs[6].plot(rewards)
			axs[6].plot(moving_avg_reward, color='black')
			axs[6].set_title('rewards', fontsize=8)

			axs[7].plot(actions)
			axs[7].set_title('actions', fontsize=8)

			axs[8].plot(smax)
			axs[8].set_title('smax', fontsize=8)

			axs[9].plot(entropy)
			axs[9].set_title('entropy', fontsize=8)

			axs[10].plot(gammas)
			axs[10].set_title('gammas', fontsize=8)

			axs[11].plot(reward_probs_left, label='Left')
			axs[11].plot(reward_probs_right, label='Right')
			axs[11].set_title('Reward Probabilities')
			axs[11].set_xlabel('Trial')
			axs[11].set_ylabel('Probability')
			axs[11].set_yticks(np.arange(0, 1.1, 0.1))  
			axs[11].set_yticklabels(['{:.1f}'.format(x) for x in np.arange(0, 1.1, 0.1)], fontsize=10)  
			axs[11].set_xticks(np.arange(0, n_trials, 40))  
			axs[11].set_xticklabels(['{:.1f}'.format(x) for x in np.arange(0, n_trials, 40)], fontsize=10)  
			axs[11].legend(fontsize=8, loc='upper right')

			axs[12].plot(np.arange(-10, 10, 1), zone_avg)
			axs[12].set_yticks(np.arange(0, 1.1, 0.25))  
			axs[12].set_yticklabels(['{:.1f}'.format(x) for x in np.arange(0, 1.1, 0.25)], fontsize=8)  

			axs[13].plot(np.arange(-10, 10, 1), zone_choice_avg, color='black', label='choice')
			axs[13].plot(np.arange(-10, 10, 1), zone_good_choice_avg, color='red', label='good choice')
			axs[13].plot(np.arange(-10, 10, 1), zone_reward_avg, color='blue', label='reward')
			axs[13].set_yticks(np.arange(0, 1.1, 0.25))  
			axs[13].set_yticklabels(['{:.1f}'.format(x) for x in np.arange(0, 1.1, 0.25)], fontsize=8)  
			axs[13].legend(fontsize=8, loc='upper right')

			plt.tight_layout()
			plt.show()

			fig, axs = plt.subplots(1, 1, figsize=(8, 10))
			axs.plot(np.arange(-5, 5, 1), zone_avg[5:15], color='green', label='made good choice')
			axs.set_yticks(np.arange(0, 0.8, 0.2))  
			axs.set_yticklabels(['{:.1f}'.format(x) for x in np.arange(0, 0.8, 0.2)], fontsize=8)
			axs.legend(fontsize=8, loc='upper right')
			plt.show()

			fig, ax = plt.subplots(len(block_switch), 1, figsize=(13, 20))

			for i in range(len(block_switch)):
				ax[i].plot(np.arange(-10, 10, 1), zone_choice[i], color='black', label='choice')
				ax[i].plot(np.arange(-10, 10, 1), zone_good_choice[i], color='red', label='good choice', linestyle='dotted')
				ax[i].plot(np.arange(-10, 10, 1), zone_reward[i], color='blue', label='reward', linestyle='dashdot')
				ax[i].plot(np.arange(-10, 10, 1), zone[i], color='green', label='made good choice', linestyle='dashdot')
				ax[i].legend(fontsize=8, loc='upper right')

			#print(choices)
			plt.tight_layout()
			plt.show()
			
		
			# save state learning
			states.append(state)
			V.append(VALUES)
			QG.append(qg)
			QN.append(qn)
			alphags.append(ag)
			alphans.append(an)
			RPE.append(pe)
			RHO.append(r)
			CHOICES.append(choices)
			REWARDS.append(rewards)
			ACTIONS.append(actions)
			SOFTMAX.append(smax)
			ENTROPY.append(entropy)
			GAMMAS.append(gammas)
			BG.append(bg)
			BN.append(bn)
		
	states.append(state)

	#return states, state_no, V, QG, QN, alphags, alphans, RPE, RHO, CHOICES, REWARDS, ACTIONS, SOFTMAX, ENTROPY, GAMMAS, t_no, RMAP, BG, BN
	return df


#import necessary libraries
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import itertools
import pandas as pd

#define the Q learning function
def Q_learning(alpha, beta, T=1000, mu=np.array([0.8, 0.2])):
    a=np.zeros(T)
    r=np.zeros(T)
    Q=np.array([0.5, 0.5])
    for t in range(T):
        p=np.exp(beta*Q)/sum(np.exp(beta*Q))
        if len(p) != 2:
            raise ValueError("The probability array 'p' must have exactly two elements.")
        a[t]= np.random.choice([0, 1], p=p)  
        r[t]= np.random.rand() < mu[int(a[t])]
        delta=r[t]-Q[int(a[t])]
        Q[int(a[t])]+=alpha*delta
    return a, r

#define the negative log likelihood function
def nll_func_RW(a, r, alpha, beta):
	T= len(a)
	Q = np.array([0.5, 0.5])
	ll=0
		
	for t in range(T):
		p = np.exp(beta * Q) / np.sum(np.exp(beta * Q)) ###### Q updated for every t
		if p.any==0:
			print('p=0 for', t, 'th trial, beta=', beta, 'Q=', Q)
			break
		ll+=np.log(p[int(a[t])])
		delta = r[t] - Q[int(a[t])]
		Q[int(a[t])] += alpha * delta
	return -1*ll

# make parameter combinations
alphas=np.arange(0, 1.1, 0.1)
betas=np.arange(1, 6, 1)
num_var_parameters=[alphas, betas]
num_combos=list(itertools.product(*num_var_parameters))

GT=[]
ll_GT=[]
IG=[]
PE=[]
ll_PE=[]
IG=[]

fit_func = lambda x: nll_func_RW(a, r, x[0], x[1])

# fit each parameter combination
for i in range(len(num_combos)):
	#print(i) #uncomment to view progress
	A=num_combos[i][0]
	B=num_combos[i][1]
	GT.append([A,B])
	a, r = Q_learning(A, B)
	nll = nll_func_RW(a, r, A, B)
	ll_GT.append(nll)
	bounds = [(0, 1), (1, 5)]
	xos = []
	pe = []
	ll = []
	for i in range(10):     #10 initial guesses / paramter combination 
		X0 = [np.random.uniform(0, 1), np.random.uniform(1, 5)] 
		xos.append(X0)
		result = minimize(fit_func, X0, bounds=bounds, method='SLSQP')
		pe.append(result.x)
		this_ll = nll_func_RW(a, r, result.x[0], result.x[1])
		ll.append(this_ll)
	ll_min = np.min(ll)
	ll_PE.append(ll_min)
	k = ll.index(ll_min)
	PE.append(pe[k])
	IG.append(xos[k])

para= ['alpha', 'beta']

#plot results
for i in range(2):
    y_array = [row[i] for row in GT]
    x_array = [row[i] for row in PE]
    
    lower, upper = bounds[i]
    ticks = np.arange(lower, upper*1.1, upper/10)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x_array, y_array, color='red')
    
    z = np.polyfit(x_array, y_array, 1)  
    p = np.poly1d(z)
    plt.plot(x_array, p(x_array), color="black", linestyle='-.')
    
    plt.xlabel('Recovered')
    plt.ylabel('True')
    plt.title(f'Recovery of {para[i]}')
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.xlim(lower, upper)
    plt.ylim(lower, upper)
    plt.plot([lower, upper], [lower, upper], color='gray', linestyle='--')
    plt.tight_layout()
    plt.savefig(f'Q_fit_plot_{para[i]}_recovery') 
    plt.show()

# store results to dataframe 
results=pd.DataFrame({
	'GT': GT,
	'll_GT':ll_GT,
    'IG': IG,
	'PE': PE,
	'll_PE':ll_PE})

results.to_csv('RW_model_fit_results.csv', index=False)
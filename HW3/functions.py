import numpy as np

######################
# Objective function #
######################

# initial objective
f = lambda l,s,t : lam*l + np.mean(s) + np.mean(t)

# self-concordant barrier from constraints
f_1 = lambda st : -np.sum(np.log(st))                                   # nu = nA/nB
f_3 = lambda h,c,s : -np.sum(np.log(-np.ones(nA)-a@h-c*np.ones(nA)+s))  # nu = nA
f_4 = lambda h,c,t : -np.sum(np.log(-np.ones(nB)+b@h+c*np.ones(nB)+t))  # nu = nB
f_5 = lambda h,l : -np.log(l-np.linalg.norm(h)**2)                             # nu = 2

# thus
f_mu = lambda h,c,s,t,l : f(l,s,t)/mu+f_1(s)+f_1(t)+f_3(h,c,s)+f_4(h,c,t)+f_5(h,l) # nu = 2*(nA+nB+1)
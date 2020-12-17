import numpy as np

n = 3
lam = 1
mu = 1
nA = 3
nB = 4
a = np.zeros((nA,n))
b = np.zeros((nB,n))

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

############
# Gradient #
############

# w.r.t. h
def f_mu_h(h,c,s,t,l):
    sumA = 0
    for i in range(nA):
        sumA += a[i]/(-1-np.dot(h,a[i])-c+s[i])
    sumB = 0
    for i in range(nB):
        sumB += b[i]/(-1+np.dot(h,b[i])+c+t[i])
    return sumA-sumB - 2*h/(l-np.linalg.norm(h)**2)

# w.r.t. c
f_mu_c = lambda h,c,s,t : np.sum(1/(-np.ones(nA)-a@h-c*np.ones(nA)+s)) - np.sum(1/(-np.ones(nB)+b@h+c*np.ones(nB)+t))

# w.r.t. s and t
f_mu_s = lambda h,c,s : 1/(nA*mu)*np.ones(nA) - 1/s - 1/(-np.ones(nA)-a@h-c*np.ones(nA)+s)
f_mu_t = lambda h,c,t : 1/(nB*mu)*np.ones(nB) - 1/t - 1/(-np.ones(nB)+b@h+c*np.ones(nB)+t)

# w.r.t. l
f_mu_l = lambda h,l : lam/mu - 1/(l-np.linalg.norm(h)**2)

# thus
f_mu_grad = lambda h,c,s,t,l : np.hstack([f_mu_h(h,c,s,t,l),
                                          f_mu_c(h,c,s,t),
                                          f_mu_s(h,c,s),
                                          f_mu_t(h,c,t),
                                          f_mu_l(h,l)])

###########
# Hessian #
###########




import numpy as np

n = 3
lam = 1
mu = 1
nA = 3
nB = 4
a = np.zeros((nA,n))
b = np.zeros((nB,n))

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
    return sumA-sumB + 2*h/(l-np.linalg.norm(h)**2)

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
# (n+nA+nB+2) x (n+nA+nB+2) matrix

# (n)x(n) matrix
def f_mu_h_h(h,c,s,t,l):
    mat = np.zeros((n,n))
    for i in range(nA):
        mat += np.outer(a[i],a[i])/(-1-np.dot(h,a[i])-c+s[i])**2
    for i in range(nB):
        mat += np.outer(b[i],b[i])/(-1+np.dot(h,b[i])+c+t[i])**2
    mat += np.outer(h,h)*4/(l-np.linalg.norm(h)**2)**2
    mat += 2/(l-np.linalg.norm(h)**2)*np.eye(n)
    return mat

# 1x(n) vector
def f_mu_c_h(h,c,s,t):
    mat = np.zeros(n)
    for i in range(nA):
        mat += a[i]/(-1-np.dot(h,a[i])-c+s[i])**2
    for i in range(nB):
        mat += b[i]/(-1+np.dot(h,b[i])+c+t[i])**2
    return np.reshape(mat,(1,n))

# (nA)x(n) matrix
def f_mu_s_h(h,c,s):
    mat = np.zeros((nA,n))
    for i in range(nA):
        mat[i,:] = -a[i]/(-1-np.dot(h,a[i])-c+s[i])**2
    return mat

# (nB)x(n) matrix
def f_mu_t_h(h,c,t):
    mat = np.zeros((nB,n))
    for i in range(nB):
        mat[i,:] = -b[i]/(-1+np.dot(h,b[i])+c+t[i])**2
    return mat

# 1x(n) vector
f_mu_l_h = lambda h,l : np.reshape(-2*h/(l-np.linalg.norm(h)**2)**2,(1,n))

# scalar
f_mu_c_c = lambda h,c,s,t : [[np.sum(1/(-np.ones(nA)-a@h-c*np.ones(nA)+s)**2) + np.sum(1/(-np.ones(nB)+b@h+c*np.ones(nB)+t)**2)]]

# (nA)x1 matrix
def f_mu_s_c(h,c,s):
    mat = np.zeros(nA)
    for i in range(nA):
        mat[i] = -1/(-1-np.dot(h,a[i])-c+s[i])**2
    return np.reshape(mat,(nA,1))

# (nB)x1 matrix
def f_mu_t_c(h,c,t):
    mat = np.zeros(nB)
    for i in range(nB):
        mat[i] = -1/(-1+np.dot(h,b[i])+c+t[i])**2
    return np.reshape(mat,(nB,1))

# (nA)x(nA) diagonal matrix
f_mu_s_s = lambda h,c,s : np.diag(1/s**2 + 1/(-np.ones(nA)-a@h-c*np.ones(nA)+s)**2)

# (nB)x(nB) diagonal matrix
f_mu_t_t = lambda h,c,t : np.diag(1/t**2 + 1/(-np.ones(nB)+b@h+c*np.ones(nB)+t)**2)

# scalar
f_mu_l_l = lambda h,l : [[1/(l-np.linalg.norm(h)**2)**2]]

def f_mu_hessian(h,c,s,t,l):
    h_row = np.hstack([f_mu_h_h(h,c,s,t,l),
                       np.transpose(f_mu_c_h(h,c,s,t)),
                       np.transpose(f_mu_s_h(h,c,s)),
                       np.transpose(f_mu_t_h(h,c,t)),
                       np.transpose(f_mu_l_h(h,l))])
    c_row = np.hstack([f_mu_c_h(h,c,s,t),
                       f_mu_c_c(h,c,s,t),
                       np.transpose(f_mu_s_c(h,c,s)),
                       np.transpose(f_mu_t_c(h,c,t)),
                       [[0]]])
    s_row = np.hstack([f_mu_s_h(h,c,s),
                       f_mu_s_c(h,c,s),
                       f_mu_s_s(h,c,s),
                       np.zeros((nA,nB)),
                       np.zeros((nA,1))])
    t_row = np.hstack([f_mu_t_h(h,c,t),
                       f_mu_t_c(h,c,t),
                       np.zeros((nB,nA)),
                       f_mu_t_t(h,c,t),
                       np.zeros((nB,1))])
    l_row = np.hstack([f_mu_l_h(h,l),
                       [[0]],
                       np.zeros((1,nA)),
                       np.zeros((1,nB)),
                       f_mu_l_l(h,l)])
    return np.vstack([h_row,c_row,s_row,t_row,l_row])

print(f_mu_hessian(np.ones(n),1,np.ones(nA),np.ones(nB),1))



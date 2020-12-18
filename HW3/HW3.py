import numpy as np
import pickle
import matplotlib.pyplot as plt

lam = 0.5

########
# Data #
########

#train_data = np.loadtxt("data/mnist_train.csv",delimiter=",")
#pickle.dump(train_data[:int(0.1*len(train_data))],open("train_data.p","wb"))

# train_data = pickle.load(open("train_data.p", "rb"))
# train_data=train_data[:4000]
#
# n = len(train_data[0])-1
#
# ind_a = np.where(train_data[:,0]==0)[0][:300]
# ind_b = np.where(train_data[:,0]!=0)[0][:300]
# train_data = train_data[np.hstack([ind_b,ind_a])]
# ind_a = np.where(train_data[:,0]==0)[0][:300]
# ind_b = np.where(train_data[:,0]!=0)[0][:300]
# nA = len(ind_a)
# nB = len(ind_b)
# print(len(train_data),nA,nB)
# a = train_data[ind_a,1:]
# b = train_data[ind_b,1:]
#
# true_labels=np.zeros(nA+nB)
# true_labels[ind_b]=1

a = np.array([[1,2]])
b = np.array([[2,3],[2,2]])
nA = 1
nB = 2
n = 2

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
    return sumA - sumB + 2*h/(l**2-np.dot(h,h))

# w.r.t. c
f_mu_c = lambda h,c,s,t : np.sum(1/(-np.ones(nA)-a@h-c*np.ones(nA)+s)) - np.sum(1/(-np.ones(nB)+b@h+c*np.ones(nB)+t))

# w.r.t. s and t
f_mu_s = lambda h,c,s,mu : 1/(nA*mu)*np.ones(nA) - 1/s - 1/(-np.ones(nA)-a@h-c*np.ones(nA)+s)
f_mu_t = lambda h,c,t,mu : 1/(nB*mu)*np.ones(nB) - 1/t - 1/(-np.ones(nB)+b@h+c*np.ones(nB)+t)

# w.r.t. l
f_mu_l = lambda h,l,mu: lam/mu - 2*l/(l**2-np.dot(h,h))

# thus
f_mu_grad = lambda h,c,s,t,l,mu : np.hstack([f_mu_h(h,c,s,t,l),
                                             f_mu_c(h,c,s,t),
                                             f_mu_s(h,c,s,mu),
                                             f_mu_t(h,c,t,mu),
                                             f_mu_l(h,l,mu)])

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
    mat += np.outer(h,h)*4/(l**2-np.dot(h,h))**2
    mat += 2/(l**2-np.dot(h,h))*np.eye(n)
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
        mat[i,:] = b[i]/(-1+np.dot(h,b[i])+c+t[i])**2
    return mat

# 1x(n) vector
f_mu_l_h = lambda h,l : np.reshape(-4*l*h/(l**2-np.dot(h,h))**2,(1,n))

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
        mat[i] = 1/(-1+np.dot(h,b[i])+c+t[i])**2
    return np.reshape(mat,(nB,1))

# (nA)x(nA) diagonal matrix
f_mu_s_s = lambda h,c,s : np.diag(1/s**2 + 1/(-np.ones(nA)-a@h-c*np.ones(nA)+s)**2)

# (nB)x(nB) diagonal matrix
f_mu_t_t = lambda h,c,t : np.diag(1/t**2 + 1/(-np.ones(nB)+b@h+c*np.ones(nB)+t)**2)

# scalar
f_mu_l_l = lambda h,l : [[(2*l**2 + 2*np.dot(h,h))/(l**2-np.dot(h,h))**2]]

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

#######
# IPM #
#######

def initial_point():
    len = n+2+nA+nB
    var = np.zeros(len)
    var[n+1:n+1+nA]= nA/(nA+nB)
    var[n+1+nA:n+1+nA+nB] = nB/(nA+nB)
    var[-1]=1
    return var

def n_mu(mu,x):
    hess = f_mu_hessian(x[:n],x[n],x[n+1:n+1+nA],x[n+1+nA:n+1+nA+nB],x[-1])
    grad = f_mu_grad(x[:n],x[n],x[n+1:n+1+nA],x[n+1+nA:n+1+nA+nB],x[-1],mu)
    n_x = -np.linalg.solve(hess,grad)
    return n_x

def fit(x,h,c):
    labels = -1*np.ones(len(x))
    for i in range(len(x)):
        if np.dot(h,x[i]) + c < 0:
            labels[i]=0
        elif np.dot(h,x[i]) + c > 0 :
            labels[i]=1
    return labels

def check(x):
    h = x[:n]
    c = x[n]
    current_labels = fit(train_data[:, 1:], h, c)
    correct_match_idx = np.where(current_labels == true_labels)[0]
    print(len(correct_match_idx)/len(train_data),
          len(np.where(true_labels[correct_match_idx] == 0)[0])/nA,
          len(np.where(true_labels[correct_match_idx] == 1)[0])/nB)

def delta(x, mu):
    grad = f_mu_grad(x[:n],x[n],x[n+1:n+1+nA],x[n+1+nA:n+1+nA+nB],x[-1],mu)
    step = n_mu(mu,x)
    return np.sqrt(-np.dot(grad,step))

# nu = 2*(nA+nB+1)
# mu_k = 1
# x_k = initial_point()
# theta = 1/(16*np.sqrt(nu))
# tau = 0.25
# epsilon = 1e-2
# mu_final = epsilon * (1-tau)/nu
#
# delt = delta(x_k,mu_k)
# while delt>tau:
#     print("delt : ",delt)
#     x_k += n_mu(mu_k,x_k)/(1+delt) # damped
#     delt = delta(x_k,mu_k)
# print("damped done")
#
# while mu_k > mu_final:
#     mu_k = mu_k * (1-theta)
#     x_k += n_mu(mu_k,x_k)
#     #check(x_k)
# print("following path done")
#
# plt.plot(a[0,0],a[0,1],'.r')
# plt.plot(b[0,0],b[0,1],'.g')
# plt.plot(b[1,0],b[1,1],'.g')
#
# x = np.linspace(1,3,100)
#
# print(x_k)
# plt.plot(x,(x_k[0]*x + x_k[2])/(-x_k[1]))
# plt.show()



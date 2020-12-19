import numpy as np
import pickle
import matplotlib.pyplot as plt

lam = 1

########
# Data #
########

# train_data = np.loadtxt("data/mnist_train.csv",delimiter=",")
# pickle.dump(train_data[:int(0.1*len(train_data))],open("train_data.p","wb"))

train_data = pickle.load(open("train_data.p", "rb"))
train_data = train_data[:4000]

n = len(train_data[0]) - 1

ind_a = np.where(train_data[:, 0] == 0)[0][:300]
ind_b = np.where(train_data[:, 0] != 0)[0][:300]
train_data = train_data[np.hstack([ind_b, ind_a])]
ind_a = np.where(train_data[:, 0] == 0)[0][:300]
ind_b = np.where(train_data[:, 0] != 0)[0][:300]
nA = len(ind_a)
nB = len(ind_b)
print(len(train_data), nA, nB)
a = train_data[ind_a, 1:]
b = train_data[ind_b, 1:]

true_labels = np.zeros(nA + nB)
true_labels[ind_b] = 1


# a = np.array([[1,2]])
# b = np.array([[2,3],[2,2]])
# nA = 1
# nB = 2
# n = 2

############
# Gradient #
############

def gradient_f(h, c, s, t, l, mu):
    den_A = 1 / (s - 1 - a @ h - c)
    den_B = 1 / (t - 1 + b @ h - c)
    den_L = 1 / (l * l - h.T @ h)

    dfdh = np.sum(a * den_A, axis=1) - np.sum(b * den_B, axis=1) + 2 * h * den_L
    dfdc = [np.sum(den_A) - np.sum(den_B)]
    dfds = 1 / (nA * mu) - 1 / s - den_A
    dfdt = 1 / (nB * mu) - 1 / t - den_B
    dfdl = [lam / mu - 2 * l * den_L]

    return np.hstack([dfdh, dfdc, dfds, dfdt, dfdl])

###########
# Hessian #     (n+nA+nB+2) x (n+nA+nB+2) matrix
###########

def hessian_f(h, c, s, t, l):

    den_A = 1 / (s - 1 - a @ h - c)
    den_A2 = den_A * den_A

    den_B = 1 / (t - 1 + b @ h + c)
    den_B2 = den_B * den_B

    den_L = 1 / (l * l - np.dot(h, h))
    den_L2 = den_L * den_L

    dfdhdh = (a.T*den_A2) @ a + (b.T * den_B2) @ b + np.outer(h, h) * 4 * den_L2 + 2 * den_L * np.eye(n)
    dfdhdc = np.sum(a.T * den_A2, axis=1) - np.sum(b.T * den_B2, axis=1)
    dfdhds = -a.T @ np.diag(den_A2)
    dfdhdt = b.T @ np.diag(den_B2)
    dfdhdl = -4 * l * h * den_L2

    dfdcdc = np.sum(den_A2) + np.sum(den_B2)
    dfdcds = -den_A2
    dfdcdt = den_B2

    dfdsds = np.diag(1/s*s +  den_A2)

    dfdtdt = np.diag(1/t*t +  den_B2)

    dfdldl = (2 * l * l + 2 * np.dot(h, h)) * den_L2

    hessian = np.zeros((n + nA + nB + 2, n + nA + nB + 2))
    # dfdhdh
    hessian[:n, :n] = dfdhdh

    # dfdhdc/dfdcdh
    hessian[:n, n] = dfdhdc
    hessian[n, :n] = dfdhdc

    # dfdhds/dfdsdh
    hessian[:n, n + 1:n + 1 + nA] = dfdhds
    hessian[n + 1:n + 1 + nA, :n] = dfdhds.T

    # dfdhdt/dfdtdh
    hessian[: n, n + 1 + nA: n + 1 + nA + nB] = dfdhdt
    hessian[n + 1 + nA: n + 1 + nA + nB, : n] = dfdhdt.T

    # dfdhdl/dfdldh
    hessian[: n, -1] = dfdhdl
    hessian[-1, : n] = dfdhdl

    # dfdcdc
    hessian[n, n] = dfdcdc

    # dfdcds/dfdsdc
    hessian[n + 1:n + 1 + nA, n] = dfdcds
    hessian[n, n + 1:n + 1 + nA] = dfdcds

    # dfdcdt/dfdtdc
    hessian[n + 1 + nA:n + 1 + nA + nB, n] = dfdcdt
    hessian[n, n + 1 + nA:n + 1 + nA + nB] = dfdcdt

    #dfdsds
    hessian[n+1:n+1+nA,n+1:n+1+nA] = dfdsds

    #dfdtdt
    hessian[n + 1 + nA:n + 1 + nA + nB, n + 1 + nA:n + 1 + nA + nB] = dfdtdt

    #dfdldl
    hessian[-1,-1]=dfdldl

    return hessian

#######
# IPM #
#######

def initial_point():
    len = n + 2 + nA + nB
    var = np.zeros(len)
    var[n + 1:n + 1 + nA] = 7 * (1 - nA / (nA + nB))
    var[n + 1 + nA:n + 1 + nA + nB] = 7 * (nA / nB) * (1 - nA / (nA + nB))
    var[-1] = 1
    return var


def n_mu(mu, x):
    hess = hessian_f(x[:n], x[n], x[n + 1:n + 1 + nA], x[n + 1 + nA:n + 1 + nA + nB], x[-1])
    grad = gradient_f(x[:n], x[n], x[n + 1:n + 1 + nA], x[n + 1 + nA:n + 1 + nA + nB], x[-1], mu)
    n_x = -np.linalg.solve(hess, grad)
    return n_x


def fit(x, h, c):
    labels = -1 * np.ones(len(x))
    for i in range(len(x)):
        if np.dot(h, x[i]) + c < 0:
            labels[i] = 0
        elif np.dot(h, x[i]) + c > 0:
            labels[i] = 1
    return labels


def check(x):
    h = x[:n]
    c = x[n]
    current_labels = fit(train_data[:, 1:], h, c)
    correct_match_idx = np.where(current_labels == true_labels)[0]
    print(len(correct_match_idx) / len(train_data),
          len(np.where(true_labels[correct_match_idx] == 0)[0]) / nA,
          len(np.where(true_labels[correct_match_idx] == 1)[0]) / nB)


def delta(x, mu):
    grad = gradient_f(x[:n], x[n], x[n + 1:n + 1 + nA], x[n + 1 + nA:n + 1 + nA + nB], x[-1], mu)
    step = n_mu(mu, x)
    return np.sqrt(-np.dot(grad, step))


nu = 2 * (nA + nB + 1)
mu_k = 1
x_k = initial_point()
theta = 1 / (16 * np.sqrt(nu))
tau = 0.25
epsilon = 1e-2
mu_final = epsilon * (1 - tau) / nu

delt = delta(x_k, mu_k)
while delt > tau:
    print("delt : ", delt)
    x_k += n_mu(mu_k, x_k) / (1 + delt)  # damped
    delt = delta(x_k, mu_k)
print("damped done")

while mu_k > mu_final:
    mu_k = mu_k * (1 - theta)
    x_k += n_mu(mu_k, x_k)
    check(x_k)
print("following path done")

plt.plot(a[0, 0], a[0, 1], '.r')
plt.plot(b[0, 0], b[0, 1], '.g')
plt.plot(b[1, 0], b[1, 1], '.g')

x = np.linspace(1, 3, 100)

print(x_k)
plt.plot(x, (x_k[0] * x + x_k[2]) / (-x_k[1]))
plt.show()



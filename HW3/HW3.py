import numpy as np
import pickle
from keras.datasets import mnist


def gradient_f(x_k, mu,lam,a,b):
    nA = len(a)
    nB = len(b)
    n = len(a[0])
    h, c, s, t, l = x_k[:n], x_k[n], x_k[n + 1:n + 1 + nA], x_k[n + 1 + nA:n + 1 + nA + nB], x_k[-1]
    den_A = 1 / (- 1 - a @ h - c + s)
    den_B = 1 / (- 1 + b @ h + c + t)
    den_L = 1 / (l * l - h.T @ h)

    dfdh = np.sum(a.T * den_A, axis=1) - np.sum(b.T * den_B, axis=1) + 2 * h * den_L
    dfdc = np.sum(den_A) - np.sum(den_B)
    dfds = 1 / (nA * mu) - 1 / s - den_A
    dfdt = 1 / (nB * mu) - 1 / t - den_B
    dfdl = lam / mu - 2 * l * den_L

    return np.hstack([dfdh, dfdc, dfds, dfdt, dfdl])

def hessian_f(x_k,a,b):
    nA = len(a)
    nB = len(b)
    n = len(a[0])

    h,c,s,t,l=x_k[:n], x_k[n], x_k[n + 1:n + 1 + nA], x_k[n + 1 + nA:n + 1 + nA + nB], x_k[-1]

    den_A = 1 / (- 1 - a @ h - c + s)
    den_A2 = den_A * den_A
    den_B = 1 / (- 1 + b @ h + c + t)
    den_B2 = den_B * den_B
    den_L = 1 / (l * l - np.dot(h, h))
    den_L2 = den_L * den_L

    dfdhdh = (a.T*den_A2) @ a + (b.T * den_B2) @ b + np.outer(h, h) * 4 * den_L2 + 2 * den_L * np.eye(n)
    dfdhdc = np.sum(a.T * den_A2, axis=1) + np.sum(b.T * den_B2, axis=1)
    dfdhds = -a.T @ np.diag(den_A2)
    dfdhdt = b.T @ np.diag(den_B2)
    dfdhdl = -4 * l * h * den_L2
    dfdcdc = np.sum(den_A2) + np.sum(den_B2)
    dfdcds = -den_A2
    dfdcdt = den_B2
    dfdsds = np.diag(1/(s*s) +  den_A2)
    dfdtdt = np.diag(1/(t*t) +  den_B2)
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

def initial_point(lam,nA,nB,n):
    var = np.zeros(n + 2 + nA + nB)
    var[n + 1:n + 1 + nA] = lam * (1 - nA / (nA + nB))
    var[n + 1 + nA:n + 1 + nA + nB] = lam * (nA / nB) * (1 - nA / (nA + nB))
    var[-1] = 1
    return var

def predict(x, h, c):
    labels = -1 * np.ones(len(x))
    for i in range(len(x)):
        if np.dot(h, x[i]) + c <= 0:
            labels[i] = 0
        elif np.dot(h, x[i]) + c >= 0:
            labels[i] = 1
    return labels

def accuracy(h,c,a,b):
    nA = len(a)
    nB = len(b)
    pred_labels_a = predict(a, h, c)
    pred_labels_b = predict(b, h, c)
    acc_a = len(np.where(pred_labels_a == 0)[0])
    acc_b = len(np.where(pred_labels_b == 1)[0])
    print('     Accuracy  : %1.4f (A : %1.4f,B : %1.4f)' %((acc_a+acc_b)/(nA+nB),acc_a/nA,acc_b/nB))

def newton(x_k,mu_k,lam,a,b,damped=False):
    hess = hessian_f(x_k,a,b)
    grad = gradient_f(x_k, mu_k,lam,a,b)
    n_x = -np.linalg.solve(hess, grad)
    if damped:
        delt = np.sqrt(-np.dot(grad, n_x))
        x_k += 1 / (1 + delt) * n_x
        return x_k,delt
    else:
        return x_k + n_x

def fit(lam,mu_0,a,b,epsilon,num_theta):
    tau = 0.25

    nA = len(a)
    nB = len(b)
    n = len(a[0])
    nu = 2 * (nA + nB + 1)

    theta = num_theta / (1 * np.sqrt(nu))
    x_k = initial_point(lam,nA,nB,n)

    delt = tau + 1
    k = 1
    while delt > tau:
        print('--------------- Iteration %3.f (damped in progress) ----------------' % k)
        x_k, delt = newton(x_k,mu_0,lam,a,b,True) # damped
        print('     Delta : %3.4f' %delt, '(threshold : %3.4f)' %tau)
        k+=1

    mu_final = epsilon * (1 - tau) / nu
    mu_k = mu_0
    while mu_k > mu_final:
        print('------------------- Iteration %3.f (damped done) -------------------' % k)
        print('     Mu : %1.9f' %mu_k,'(threshold : %1.9f)' %mu_final)
        mu_k *= (1 - theta)
        x_k = newton(x_k, mu_k,lam,a,b)
        accuracy(x_k[:n],x_k[n],a,b)
        k+=1

    return x_k[:n],x_k[n]

if __name__ == '__main__':
    #train_data = pickle.load(open("train_data.p", "rb"))

    (train_X, train_y), (test_X, test_y) = mnist.load_data()


    size = 500

    ind_a = np.where(train_y == 0)[0][:size]
    ind_b = np.where(train_y != 0)[0][:size]

    train_data = train_X[np.hstack([ind_a, ind_b])]

    train_data = [digit.ravel() for digit in train_data]

    a_train = train_data[:size, 1:]
    b_train = train_data[size:, 1:]

    lam = 10
    epsilon = 1e-4
    num_theta = 5
    label_a, label_b = 0, 1
    h,c = fit(lam, 1, a_train, b_train, epsilon, num_theta)
    print('-------------------------- Final Train Results --------------------------')
    accuracy(h, c, a_train, b_train)


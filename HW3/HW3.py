import numpy as np
import pickle
from keras.datasets import mnist
import itertools

printed = False

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
    print('     Accuracy  : %1.8f (A : %1.8f,B : %1.8f)' %((acc_a+acc_b)/(nA+nB),acc_a/nA,acc_b/nB))
    return (acc_a+acc_b)/(nA+nB)

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

        x_k, delt = newton(x_k,mu_0,lam,a,b,True) # damped
        if k%100 == 0 :
            print('--------------- Iteration %3.f (damped in progress) ----------------' % k)
            print('     Delta : %3.4f' %delt, '(threshold : %3.4f)' %tau)
            accuracy(x_k[:n], x_k[n], a, b)
        k+=1
    mu_final = epsilon * (1 - tau) / nu
    mu_k = mu_0
    k = 1
    while mu_k > mu_final:
        if k % 100 == 0 :
            print('------------------- Iteration %3.f (damped done) -------------------' % k)
            print('     Mu : %1.9f' %mu_k,'(threshold : %1.9f)' %mu_final)
            accuracy(x_k[:n], x_k[n], a, b)
        mu_k *= (1 - theta)
        x_k = newton(x_k, mu_k,lam,a,b)

        k+=1

    return x_k[:n],x_k[n],accuracy(x_k[:n], x_k[n], a, b)

if __name__ == '__main__':
    #train_data = pickle.load(open("train_data.p", "rb"))
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    size = 6000
    size_class = int(size/2)
    ind_a = np.where(train_y == 0)[0][:size_class]
    ind_b = np.where(train_y != 0)[0][:size_class]
    train_X = np.array(np.reshape(train_X[np.hstack([ind_a, ind_b])],(size,784)),dtype=float)
    a_train = train_X[:size_class]
    b_train = train_X[size_class:]

    ind_a_test = np.where(test_y == 0)[0]
    ind_b_test = np.where(test_y != 0)[0]
    test_X = np.array(np.reshape(test_X,(len(test_X),784)),dtype=float)
    a_test = test_X[ind_a_test]
    b_test = test_X[ind_b_test]

    epsilon = 1e-4
    num_theta = 1
    acc_lam = np.zeros(10)
    for i,lam in enumerate([10]):
        h,c,acc = fit(lam, 1, a_train, b_train, epsilon, num_theta)
        print("#####################################################################")
        print("#####################################################################")
        print(lam,acc)
        accuracy(h, c, a_test, b_test)
        #99,91 lambda=4 num_delta = 1
        #99,91 lambda=6 num_delta = 1
        #99,91 lambda=8 num_delta = 5
        #99,91 lambda=10 num_delta = 5
        #99,66 lambda=12 num_delta = 5
        #99,66 lambda=14 num_delta = 5
        #99,66 lambda=16 num_delta = 5
        print("#####################################################################")
        print("#####################################################################")
        acc_lam[i]=acc

    ### Multiclass classification ###
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    size = 6000
    train_X = np.array(np.reshape(train_X, (len(train_X), 784)), dtype=float)

    idx_X = [np.where(train_y == x)[0][:int(size/10)] for x in range(10)]
    classifiers_pairs = [(i,j) for i,j in itertools.combinations(range(10), 2)]
    classifiers_hyperplanes = np.zeros(45,2)

    test_y_pred = -np.ones(45,len(test_y))
    for idx in range(45):
        A,B = classifiers_pairs[idx]
        a_train = train_X[idx_X[A]]
        b_train = train_X[idx_X[B]]
        h, c, acc = fit(lam=10, mu_0=1, a=a_train, b=b_train, epsilon=1e-4, num_theta=1)
        pred_idx = predict(test_X,h,c) # 0 -> A , 1 -> B
        test_y_pred[idx] = A * (pred_idx==0) + B * (pred_idx==1)

    pred_y = np.zeros(len(test_y))
    for i in range(len(test_y)) :
        pred_y[i] = np.bincount(test_y_pred[:][i]).argmax()













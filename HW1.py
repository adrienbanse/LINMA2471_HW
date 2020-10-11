# LINMA2471 - Homework 1

import numpy  as np
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
def plot_result():
    data = np.loadtxt('HW1-pos.txt')
    data2 = np.loadtxt('HW1-vit.txt')
    data3 = np.loadtxt('HW1-acc.txt')

    #plt.grid()

    x = data[:, 0]
    y = data[:, 1]

    vx = data2[:, 0]
    vy = data2[:, 1]

    ax = data3[:, 0]
    ay = data3[:, 1]


    plt.arrow(x[0], y[0], vx[0], vy[0], color="blue", width=0.001,
                 head_width=2, head_length=4, overhang=0, length_includes_head = True, label = 'Speed')
    plt.arrow(x[0], y[0], ax[0], ay[0], color="red", width=0.001,
                 head_width=2, head_length=4, overhang=0, length_includes_head = True, label = 'Acceleration')

    for i in range(1,len(x)-1):
        plt.arrow(x[i], y[i], vx[i], vy[i], color="blue", width=0.001,
                 head_width=2, head_length=4, overhang=0, length_includes_head = True)
        plt.arrow(x[i], y[i], ax[i], ay[i], color="red", width=0.001,
                 head_width=2, head_length=4, overhang=0, length_includes_head = True)

    plt.plot(x, y, '.', color = "black", label='Position')
    plt.plot(0,0,'.', markersize= 15, color = 'yellow', label = 'Base')
    plt.plot(0,90,'.', markersize= 15, color = 'yellow')
    plt.plot(90,90,'.', markersize= 15, color = 'yellow')
    plt.plot(90,0,'.', markersize= 15, color = 'yellow')

    plt.xlabel('x [ft]')
    plt.ylabel('y [ft]')

    plt.legend()
    plt.axis('equal')

    plt.show()

def plot_error():
    fig, ax = plt.subplots()
    x = [16, 32, 64, 128, 256, 512, 1024]

    data = np.loadtxt('HW1-obj1.txt')
    error = data[:7] - 16.58911*np.ones(7)
    [a1, a0] = np.polyfit(np.log(x), np.log(error), 1)
    print(a1,a0)
    ax.plot(x, np.exp(a0) * (x**a1), color = 'orange', label='approximation method 1' )
    ax.plot(x, error, '.', color = 'orange', label='error method 1')

    data = np.loadtxt('HW1-obj2.txt')
    error = data[:7] - 16.58911 * np.ones(7)
    [a1, a0] = np.polyfit(np.log(x), np.log(error), 1)
    print(a1, a0)
    ax.plot(x, np.exp(a0) * (x ** a1), color='purple', label='approximation method 2')
    ax.plot(x, error, '.', color='purple', label='error method 2')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_xlabel("Discretized points (4K)")
    ax.set_ylabel("Error")
    plt.legend()
    plt.show()

def plot_error():
    fig, ax = plt.subplots()
    x = [16, 32, 64, 128, 256, 512, 1024, 2048]

    data = np.loadtxt('HW1-obj1.txt')
    ax.plot(x, data, color = 'orange', label='objective value method 1')

    data = np.loadtxt('HW1-obj2.txt')
    ax.plot(x, data, color='purple', label='objective value method 2')
    ax.set_xscale('log', base=2)
    #ax.set_yscale('log', base=2)
    ax.set_xlabel("Discretized points (4K)")
    ax.set_ylabel("Time [s]")



    plt.legend()
    plt.show()

def plot_etrange():
    data = np.loadtxt('HW1.txt')
    x = data[:, 0]
    y = data[:, 1]
    plt.plot(x,y)
    plt.plot(0, 0, '.', markersize=15, color='yellow', label='Base')
    plt.plot(0, 90, '.', markersize=15, color='yellow')
    plt.plot(90, 90, '.', markersize=15, color='yellow')
    plt.plot(90, 0, '.', markersize=15, color='yellow')
    plt.show()


def plot_acc():
    data = np.loadtxt('HW1-pos2.txt')
    data2 = np.loadtxt('HW1-pos3.txt')
    data3 = np.loadtxt('HW1-pos4.txt')

    x = data[:, 0]
    y = data[:, 1]

    x2 = data2[:, 0]
    y2 = data2[:, 1]

    x3 = data3[:, 0]
    y3 = data3[:, 1]

    plt.plot(x3, y3, label='$a_{max} = 40$')
    plt.plot(x,y,label='$a_{max} = 45$')
    plt.plot(x2,y2,label='$a_{max} = 50$')

    plt.xlabel('x [ft]')
    plt.ylabel('y [ft]')

    plt.axis('equal')


    plt.legend()
    plt.show()

def plot_acc():
    data = np.loadtxt('HW1-pos5.txt')
    data2 = np.loadtxt('HW1-pos6.txt')
    data3 = np.loadtxt('HW1-pos7.txt')
    data4 = np.loadtxt('HW1-pos8.txt')

    x = data[:, 0]
    y = data[:, 1]
    x2 = data2[:, 0]
    y2 = data2[:, 1]
    x3 = data3[:, 0]
    y3 = data3[:, 1]
    x4 = data4[:, 0]
    y4 = data4[:, 1]

    plt.plot(x3, y3, label='$K = 2^3$')
    plt.plot(x4, y4, label='$K = 2^6$')
    plt.plot(x2,y2,label='$K = 2^9$')
    plt.plot(x, y, label='$K = 2^{12}$')

    plt.xlabel('x [ft]')
    plt.ylabel('y [ft]')

    plt.axis('equal')


    plt.legend()
    plt.show()


#plot_error()
plot_acc()
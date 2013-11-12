'''
takes in X_out.npy, y_out.npy which are training datas as np arrays, produced by generate_X_y_self.py (for synthetic data) or generate_X_y.np for real data

Performs gradient ascend.

'''



from sigvisa.models.spatial_regression.SparseGP import SparseGP
import numpy as np
import sys
from math import exp

'''
if len(sys.argv) == 1:
    X = np.load('X_training.npy')
    y = np.load('y_training.npy')
elif len(sys.argv) ==2:
    X = np.load('X_training_' + sys.argv[1] + '.npy')
    y = np.load('y_training_' + sys.argv[1] + '.npy')
#X_star = np.load("X_tbp.npy")
#y_star = np.load("y_tbp.npy")
'''


def dist(point1, point2):
    assert len(point1) == len(point2)
    temp_sum = 0
    for i in range(len(point1)):
        temp_sum += (point1[i] - point2[i])**2
    return temp_sum**0.5


def grad_ascend(func, precision=0.01, step=0.001, initial_guess = [5, 5, 5, 5, 5, 5]):
    x_old = [0, 0, 0, 0, 0, 0]
    x_new = initial_guess
    while dist(x_new, x_old) > precision:
#        import pdb; pdb.set_trace()
        x_old = x_new
        ll, grad = func(x_old)
        ll, grad = -ll, -grad
        temp = [0, 0, 0, 0, 0, 0]
        #temp[0] = x_old[0] * exp(step*x_old[0]*grad[0])
        #temp[1] = x_old[1] * exp(step*x_old[1]*grad[1])
        for i in range(6):
            temp[i] = x_old[i] *exp(step*x_old[i]*grad[i])
        #temp[2:] = [sum(pair) for pair in zip(x_old[2:], [i*step for i in grad[2:]])]
        x_new = temp
        '''
        while not reduce(lambda x, y:x and y, [i >= 0 for i in x_new]):
            print 'negative value'
            counter_neg += 1
            eps = eps/2
            x_new = [sum(pair) for pair in zip(x_old, [i*eps for i in deriv])]
        '''
        #print "x_old = ", x_old
        print "x_new = ", x_new
        #print "step = ", step
        print "ll = ", ll
        print "grad = ", grad
        print
        #wait = raw_input("press enter to continue")
    return x_new, ll

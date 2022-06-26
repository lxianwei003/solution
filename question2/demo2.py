
from scipy import stats
import numpy as np


def fun1():
    mu = 0
    sigma=1
    X = np.arage(-5,5,0.1)

    pList = stats.norm.pdf(X,mu,sigma)

    print('pList {}'.format(pList))

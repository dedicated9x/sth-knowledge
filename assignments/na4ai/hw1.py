import numpy as np

def get_data():
    def line(a, b, x):
        return a*x + b

    a = 20*np.random.random()-10
    b = 20*np.random.random()-10
    nrpts = 100
    pts = np.zeros( (nrpts, 2))
    for i in range(nrpts):
        x = np.random.random() * 20 - 10
        y = line(a,b,x) + np.random.normal(0, 30/np.max([abs(x),2]))
        pts[i,0] = x
        pts[i,1] = y
    return pts
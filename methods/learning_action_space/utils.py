# @author: RongcongChen <chenrc@mail2.sysu.edu.cn>

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  
from matplotlib.pyplot import plot,savefig,legend, title, ylabel, xlabel, xlim, close 

def distribution(X_val, y_val , classifier): 
    action_space = {}
    for i in range(len(y_val)):
        p = classifier.path(X_val[i])
        if action_space.__contains__(p):
            action_space[p].append(y_val[i])
        else:
            action_space[p] = []
            action_space[p].append(y_val[i])
    return action_space

def plotfig(line, images_dir, filename, x_lim = [0.6, 0.9]):
    key = list(line)
    key.sort()
    for k in key:
        h = np.array(line[k]) 
        l = len(h)  
        b = 50
        h = h[(h >= x_lim[0]) + (h <= x_lim[1])] 
        y, x = np.histogram(h, bins = b)
        y = (y+0.0) / l * (1 / (x[1] - x[0]))
        x = [(x[i] + x[i+1])/2 for i in range(len(x)-1)]
        plot(x, y, label=k+'-'+str(l)+'-'+'{:.5f}'.format(np.mean(h)))
    xlabel('Accuracy')
    ylabel('Probability Distribution')
    xlim(x_lim[0],x_lim[1]) 
    title(filename) 
    legend()
    savefig(os.path.join(images_dir, filename+'.jpg'))
    close() 


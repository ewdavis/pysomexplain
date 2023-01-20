import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from minisom import MiniSom
from sklearn.preprocessing import minmax_scale, scale

def display_SOM(som, ex):
    fig, ax = plt.subplots(figsize=(ex.x_size, ex.y_size))
    ax.pcolor(som.distance_map().T, cmap='cividis', alpha=.8)
    ax.set_xticks(np.arange(ex.x_size+1))
    ax.set_yticks(np.arange(ex.y_size+1))
    ax.grid()

    eucl_map = som.labels_map(ex.X, ex.labels)

    for idx, loc in eucl_map.items():
        loc = list(loc)
        x = idx[0] + .1
        y = idx[1] - .3
        for i, c in enumerate(loc):
            off_set = (i+1)/len(loc) - 0.05
            map_name = str(c) + ": " + str(idx[0]) + ", " + str(idx[1])
            if (ex.label_key[c] == 1):
                ax.text(x, y+off_set, map_name, color="crimson", fontsize=8)
            else:
                ax.text(x, y+off_set, map_name, color="white", fontsize=8)
                
    return(fig, ax)

def classify(som, data, ex):
    labels_map = ex.eucl_map
    default_class = np.sum(list(labels_map.values())).most_common()[0][0]
    result = []
    for d in data:
        win_position = som.winner(d)
        if win_position in labels_map:
            result.append(ex.label_key[labels_map[win_position].most_common()[0][0]])
        else:
            result.append(ex.label_key[default_class])
    return(result)


class Explainer:
    def __init__(self, features, feature_cols, target, labels, test):
        self.labels = labels
        self.label_key = dict(zip(list(labels), list(target)))
        self.X = scale(features)
        self.x_size = -1
        self.y_size = -1
        self.test = test
        self.feature_cols = feature_cols
        
    def get_som(self, x_size=20, opt='occupancy', opt_threshold=0.05, opt_iters=100, opt_x_step=10, 
                som_iters=10000, class_error_threshold=0.1, verbose=False):
        self.x_size = x_size
        self.y_size = int(x_size/2)
        
        som = MiniSom(self.x_size, self.y_size, len(self.X[0]), neighborhood_function='gaussian', sigma=1.5)
        som.pca_weights_init(self.X)
        som.train_random(self.X, som_iters)
        
        self.eucl_map = som.labels_map(self.X, self.labels)
        
        q_error = som.quantization_error(self.X)
        t_error = som.topographic_error(self.X)
        
        occupied = 0
        
        for idx, loc in self.eucl_map.items():            
            occupied += 1
            
        occupancy = occupied / (self.x_size * self.y_size)
        
        iters = 0
        qerrs = []
        terrs = []
        cerrs = []
        occs = []
        
        c_error = 1.0
        
        while (((opt=='occupancy') and (occupancy > opt_threshold)) or (c_error > class_error_threshold)) and (iters < opt_iters):
            if (verbose):
                print("@"*5, ":\t", "Run {:03d}".format(iters), "/{:03d}".format(opt_iters))
                print("X"*5, ":\t", "  x {:04d}".format(self.x_size), ", {:04d}".format(self.y_size), "\n\n")
            self.x_size += opt_x_step
            
            self.y_size = int(self.x_size/2)
        
            som = MiniSom(self.x_size, self.y_size, len(self.X[0]), neighborhood_function='gaussian', sigma=1.5)
            som.pca_weights_init(self.X)
            som.train_random(self.X, som_iters)
        
            self.eucl_map = som.labels_map(self.X, self.labels)
        
            q_error = som.quantization_error(self.X)
            t_error = som.topographic_error(self.X)
        
            occupied = 0
        
            for idx, loc in self.eucl_map.items():            
                occupied += 1
            
            occupancy = occupied / (self.x_size * self.y_size)
            iters += 1
            
            raw_score = classify(som, np.array(self.test[self.feature_cols]), self)
            c_error = 1.0 - (sum(raw_score) / len(raw_score))
        
            qerrs.append(q_error)
            terrs.append(t_error)
            cerrs.append(c_error)
            occs.append(occupancy)
        
        return({'q_error': q_error, 't_error': t_error, 'c_error': c_error, 'occupancy': occupancy, 'iters': iters, 
                'history': (qerrs, terrs, occs, cerrs)}, som)
        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 20:23:28 2020

@author: kaikaneshina
"""

import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

path = r'/Users/kaikaneshina/Documents/GitHub/Colin_Eli_Kai_Math178_Final/results/motionSense/results.csv'

data = pd.read_csv(path)

matplotlib.rcParams.update({'font.size': 22})
ax = plt.subplot(111)
ax.bar(data['Users']-.2, data['SVC'], width=0.2, color='b', align='center', label = 'SVC')
ax.bar(data['Users']+.2, data['K-NN'], width=0.2, color='g', align='center', label = 'K-NN')
ax.bar(data['Users']+.4, data['extra_trees']*100, width=0.2, color='r', align='center', label = 'extraTrees')
plt.xlim((1))
plt.ylim((50,110))
plt.grid(alpha = .3)
plt.title('SVC vs K-NN for user cross validation')
plt.ylabel('Accuracy (%)')
plt.xlabel('Users')
plt.legend(loc = 'upper right')

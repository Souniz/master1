#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:17:20 2019

@author: berar
"""
import numpy as np
import matplotlib.pyplot as plt

from method_optim import gradient_descent
from step_optim import backtrack

def create_problem(m,n,scale,x0):
    #centered a
    A = scale*(np.random.rand(n,m)-.5)
    b = x0@A + 5*scale*np.random.rand(m)
    c = (np.random.rand(n)-.5)
    return A,b,c



# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 14:42:10 2014

@author: martin
"""

from numpy import zeros
from numpy import ones
import itertools
def calc_prob(current,prob,max_prob, clusters, last):    
    probs = [0]*clusters
    sum = 0
    if(current == last):
        probs[0]=1.0
        return probs
        
    for i in range(clusters-1,-1,-1):
        if(i > current):
            probs[i]=0.0
        elif(i == current):
            probs[i]=prob*max_prob
        elif(i > 0):
            probs[i]=(1-sum)*max_prob
        elif(i == 0):
            probs[i]=(1-sum)
            
        sum += probs[i]
            
    return probs
        
def create_line(dimension):
    return (zeros(dimension),ones(dimension)) 
    
def createBoundsList(min,max):
    result = []
    for x,y in itertools.izip(min,max):
        result.append(x)
        result.append(y)
    return result
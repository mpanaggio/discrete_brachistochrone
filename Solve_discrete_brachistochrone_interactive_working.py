#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
from brachistochrone_files import *
np.set_printoptions(precision=6, suppress=True) 
from matplotlib.widgets import Slider, Button, RadioButtons
from ipywidgets import IntSlider,FloatSlider,interact
import pandas as pd


# # Define solution parameters

# In[2]:


# make sure to run: conda install nodejs


# In[3]:


a=10 # x-coordinate for endpoint
b=-1  # y-coordinate for endpoint
N=4 # number of segments


# In[10]:


def norm(x):
    return np.sqrt(np.sum(x**2))
def angles_with_previous(x,y):
    vec_prev=np.array([x[1]-x[0],y[1]-y[0]])
    vec_next=np.array([1,0])
    vec_prev=vec_prev/norm(vec_prev)
    vec_next=vec_next/norm(vec_next)
    angles=[np.arccos(np.dot(vec_prev,vec_next))]
    for k in range(1,len(x)-1):
        vec_prev=np.array([x[k-1]-x[k],y[k-1]-y[k]])
        vec_next=np.array([x[k+1]-x[k],y[k+1]-y[k]])
        vec_prev=vec_prev/norm(vec_prev)
        vec_next=vec_next/norm(vec_next)
        angles.append(np.arccos(np.dot(vec_prev,vec_next)))
    return angles


# In[11]:


get_ipython().run_line_magic('matplotlib', 'auto')

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.3)

x,y=compute_optimal_path(a,b,N)

theta_end,r=get_optimal_cycloid(a,b)
theta=np.linspace(0,theta_end)
x_cyc=r*(theta-np.sin(theta))
y_cyc=-r*(1-np.cos(theta))


l1,=plt.plot([0,a],[0,0],'--k')
l2,=plt.plot(x,y,'.b')
l3,=plt.plot(x,y,'-b')
l4,=plt.plot(x_cyc,y_cyc,'r')
plt.scatter(0,0,c='k')
l5,=plt.plot(a,b,'.k')


def update(a=FloatSlider(description='a', step=0.1, min=0,max=10.0,value=10),
           b=FloatSlider(description='b', step=0.1, min=-10,max=0, value=-1),
           N= IntSlider(description='N', step=1, min=1,max=20,value=2)):
    x,y=compute_optimal_path(a,b,N)

    theta_end,r=get_optimal_cycloid(a,b)
    theta=np.linspace(0,theta_end)
    x_cyc=r*(theta-np.sin(theta))
    y_cyc=-r*(1-np.cos(theta))


    l1.set_xdata([0,a])
    l1.set_ydata([0,0])
    l2.set_xdata(x)
    l2.set_ydata(y)
    l3.set_xdata(x)
    l3.set_ydata(y)
    l4.set_xdata(x_cyc)
    l4.set_ydata(y_cyc)
    
    l5.set_xdata(a)
    l5.set_ydata(b)
    ax.set_xlim(x.min()-1,x.max()+1)
    ax.set_ylim(y.min()-1,y.max()+1)
    #print("Travel Times:",travel_times(x,y))
    #print("x-coordinates:",x)
    #print("y-coordinates:",y)
    res=pd.DataFrame()
    res['initial x']=x[:-1]
    res['initial y']=y[:-1]
    res['final x']=x[1:]
    res['final y']=y[1:]
    res['initial velocity']=np.sqrt(-2*9.8*y[:-1])
    res['final velocity']=np.sqrt(-2*9.8*y[1:])
    res['time-averaged velocity']=(res['initial velocity']+res['final velocity'])/2
    res['travel time']=travel_times(x,y)
    res['length']=np.sqrt(np.diff(x)**2+np.diff(y)**2)
    res['angle from vertical (deg)']=np.arctan(np.diff(x)/np.diff(y))*180/np.pi
    
    angle_w_prev = lambda x,y,k: np.arcos(np.array([x[k-1]-x[k],y[k-1]-y[k]]))
    res['angle with previous side (deg)']=angles_with_previous(x,y)
    display(res)
    fig.canvas.draw_idle()
    
interact(update)
plt.show()


# # Compare travel times


# In[ ]:





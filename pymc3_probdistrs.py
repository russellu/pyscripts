import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt

"""
The most fundamental step in building Bayesian models is the specification of a 
full probability model for the problem at hand. This primarily involves assigning 
parametric statistical distributions to unknown quantities in the model, in addition 
to appropriate functional forms for likelihoods to represent the information from the data.
To this end, PyMC3 includes a comprehensive set of pre-defined statistical distributions 
that can be used as model building blocks.
"""

# For example, if we wish to define a particular variable as having a normal prior,
# we can specify that using an instance of the Normal class.

with pm.Model():

    x = pm.Normal('x', mu=0, sigma=1)
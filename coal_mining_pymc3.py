import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt

disaster_data = pd.Series([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                           3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                           2, 2, 3, 4, 2, 1, 3, np.nan, 2, 1, 1, 1, 1, 3, 0, 0,
                           1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                           0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                           3, 3, 1, np.nan, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                           0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])
years = np.arange(1851, 1962)

plt.plot(years, disaster_data, 'o', markersize=8);
plt.ylabel("Disaster count")
plt.xlabel("Year");

"""
occurences in disasters in time series is thought to follow a poisson process 
with a large rate parameter in the early part of the time series, and a smaller
rate in the later part. we are interested in locating the change point in the 
series, perhaps related to changes in mining safety regulations
"""

# pm.Model - create a new model object, a container for the model's random variables
with pm.Model() as disaster_model:
    # with pm.MOdel() - creates a context manager, with disaster_model as the context,
    # all PYMC3 objects introduced below are added to the model behind the scenes

    switchpoint = pm.DiscreteUniform('switchpoint', lower=years.min(), upper=years.max(), testval=1900)


    # Priors for pre- and post-switch rates number of disasters
    early_rate = pm.Exponential('early_rate', 1)
    late_rate = pm.Exponential('late_rate', 1)

    # Allocate appropriate Poisson rates to years before and after current
    rate = pm.math.switch(switchpoint >= years, early_rate, late_rate)
    """
    switch is a theano function that works like an if statement, uses the first argument
    to switch between the next two arguments
    """

    disasters = pm.Poisson('disasters', rate, observed=disaster_data)
    
    
with disaster_model:
    trace = pm.sample(250,cores=1)
    
    
"""
in the trace plot, we can see there's a 10 year span thats plausible for a significant change in safety,
but a 5 year span that contains most of the probability mass
the distribution is jagged because of the jumpy relationship between the year switchpoint and the
likelihood, not due to sampling error
"""    

pm.traceplot(trace);


"""
following plot shows the switch point as an orange vertical line, together with its 
HPD (highest posterior density) as a semitransparent band. dashed black line shows the accident rate
"""

plt.figure(figsize=(10, 8))
plt.plot(years, disaster_data, '.')
plt.ylabel("Number of accidents", fontsize=16)
plt.xlabel("Year", fontsize=16)

plt.vlines(trace['switchpoint'].mean(), disaster_data.min(), disaster_data.max(), color='C1')
average_disasters = np.zeros_like(disaster_data, dtype='float')
for i, year in enumerate(years):
    idx = year < trace['switchpoint']
    average_disasters[i] = (trace['early_rate'][idx].sum() + trace['late_rate'][~idx].sum()) / (len(trace) * trace.nchains)

sp_hpd = pm.hpd(trace['switchpoint'])
plt.fill_betweenx(y=[disaster_data.min(), disaster_data.max()],
                  x1=sp_hpd[0], x2=sp_hpd[1], alpha=0.5, color='C1');
plt.plot(years, average_disasters,  'k--', lw=2);


















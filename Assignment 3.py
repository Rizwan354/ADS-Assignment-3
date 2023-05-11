# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Using required packages
import numpy as np
import pandas as pd
import wbgapi as wb
import sklearn
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

#Loading the file values
dtafrm=pd.read_csv(r"Climatic_Economic_Indicators.csv", low_memory=False)

#Visualising the loaded data
print(dtafrm.head(7))

#Visualising the loaded data in transpose form
print(dtafrm.T)

#Choosing countries, economic indicators and climatic indicators
import warnings
with warnings.catch_warnings(record=True):
    econmc_indc = ['NY.GDP.MKTP.PP.CD','NE.IMP.GNFS.ZS']
    nation = ["AUS","JPN",'LUX','CHE','PAK','IND','CHL','CHN','JAM','BGR']
    climtc_indc=['EN.ATM.GHGT.KT.CE','EN.ATM.CO2E.PC']
    df_econmc  = wb.data.DataFrame(econmc_indc, nation, mrv=6)
    df_climtc = wb.data.DataFrame(climtc_indc, nation, mrv=6)

#NY.GDP.MKTP.PP.CD: GDP on PPP basis of a nation
#NE.IMP.GNFS.ZS: Import of a nation
#EN.ATM.GHGT.KT.CE: Emission of Greenhouse gas in a nation
#EN.ATM.CO2E.PC: Emissions of CO2 in a nation


# Ecnminc Indicator of a nation
df_econmc.columns = [a.replace('YR','') for a in df_econmc.columns]      
df_econmc=df_econmc.stack().unstack(level=1)                             
df_econmc.index.names = ['Nation_Code', 'Year']                           
df_econmc.fillna(0)
df_econmc.columns                                                     
print(df_econmc.head(7))


# Climtic Indicator of a nation
df_climtc.columns = [a.replace('YR','') for a in df_climtc.columns]      
df_climtc=df_climtc.stack().unstack(level=1)                             
df_climtc.index.names = ['Nation_Code', 'Year']                           
df_climtc.fillna(0)
df_climtc.columns                                                     
print(df_climtc.head(7))


#Cleaning the dataset
e=df_econmc.reset_index()
c=df_climtc.reset_index()
e1=e.fillna(0)
c1=c.fillna(0)


#Joining the dataframes
fnl = pd.merge(e1, c1)
print(fnl.head(7))


#Standardising the values of the dataset
f1 = fnl.iloc[:,2:]
fnl.iloc[:,2:] = (f1-f1.min())/ (f1.max() - f1.min())
print(fnl.head(7))


#Clustering the data with K-means
fnl_num = fnl.drop('Nation_Code', axis = 1)
kmn = KMeans(n_clusters=3, init='k-means++', random_state=0).fit(fnl_num)


#Clustering on the basis of Emissions of CO2 in a nation
sns.scatterplot(data=fnl, x="Nation_Code", y="EN.ATM.CO2E.PC", hue=kmn.labels_)
plt.legend(loc='best')
plt.show()


#Scatter plot visualisation for Import vs Emission of greenshouse gas of a nation - Luxembourg
c1=fnl[(fnl['Nation_Code']=='LUX')]
c2 = c1.values
x, y = c2[:, 2], c2[:, 5]
plt.scatter(x, y,color="violet")
plt.ylabel('Total emission of greenhouse gas')
plt.xlabel('Total Import')
plt.show()


#Using the curve_fit function for Australia- High Emissions of CO2 for relationship between Total Import and Total emission of greenhouse gas
c3=fnl[(fnl['Nation_Code']=='AUS')]
c4 = c3.values
x, y = c4[:, 2], c4[:, 5]
def fit_func(x, a, b, c):
    return a*x**2+b*x+c
para, covar = curve_fit(fit_func, x, y)
print("Covariance is: ", covar)
print("Params is: ", para)
para, _ = curve_fit(fit_func, x, y)
a, b, c = para[0], para[1], para[2]
yfit_value = a*x**2+b*x+c

import warnings

with warnings.catch_warnings(record=True):
    plt.plot(x, yfit_value, label="y=a*x**2+b*x+c",color="violet")
    plt.grid(True)
    plt.plot(x, y, 'bo', label="Actual Y value",color="violet")
    plt.title("For Australia")
    plt.ylabel('Total emission of greenhouse gas')
    plt.xlabel('Total Import')
    plt.legend(loc='best', fancybox=True, shadow=True)

    plt.show() 


#Using the curve_fit function for Japan- Medium Emissions of CO2 for relationship between Total Import and Total emission of greenhouse gas
c5=fnl[(fnl['Nation_Code']=='JPN')]
c6 = c5.values
x, y = c6[:, 2], c6[:, 5]
def fit_func(x, a, b, c):
    return a*x**2+b*x+c
para, covar = curve_fit(fit_func, x, y)
print("Covariance is: ", covar)
print("Params is: ", para)
para, _ = curve_fit(fit_func, x, y)
a, b, c = para[0], para[1], para[2]
yfit_value = a*x**2+b*x+c

import warnings

with warnings.catch_warnings(record=True):
    plt.plot(x, yfit_value, label="y=a*x**2+b*x+c",color="violet")
    plt.grid(True)
    plt.plot(x, y, 'bo', label="Actual Y value",color="violet")
    plt.title("For Japan")
    plt.ylabel('Total emission of greenhouse gas')
    plt.xlabel('Total Import')
    plt.legend(loc='best', fancybox=True, shadow=True)

    plt.show() 
    

#Using the curve_fit function for Pakistan- Low Emissions of CO2 for relationship between Total Import and Total emission of greenhouse gas
c7=fnl[(fnl['Nation_Code']=='PAK')]
c8 = c7.values
x, y = c8[:, 2], c8[:, 5]
def fit_func(x, a, b, c):
    return a*x**2+b*x+c
para, covar = curve_fit(fit_func, x, y)
print("Covariance is: ", covar)
print("Params is: ", para)
para, _ = curve_fit(fit_func, x, y)
a, b, c = para[0], para[1], para[2]
yfit_value = a*x**2+b*x+c

import warnings

with warnings.catch_warnings(record=True):
    plt.plot(x, yfit_value, label="y=a*x**2+b*x+c",color="violet")
    plt.grid(True)
    plt.plot(x, y, 'bo', label="Actual Y value",color="violet")
    plt.title("For Pakistan")
    plt.ylabel('Total emission of greenhouse gas')
    plt.xlabel('Total Import')
    plt.legend(loc='best', fancybox=True, shadow=True)

    plt.show() 
    

def err_ranges(x, func, param, sigma):
    import itertools as iter
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper 
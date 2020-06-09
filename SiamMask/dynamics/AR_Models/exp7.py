import numpy as np
import statsmodels.api as sm
import pandas as pd


obj = 2

c = np.load('/Users/marinaalonsopoal/Desktop/Objects/target_pos_'+str(obj)+'.npy')


a = np.arange(1, 15)
df = pd.DataFrame(a)

print(df.head(5))
#%%
# data
import pandas as pd
import numpy as np
import os
from pathlib import Path
# %%
#Read data
row_data_rpath = os.path.join(Path.cwd().parent,"data/raw/credit_risk_dataset.csv")
print(f'loading {row_data_rpath}')
data =  pd.read_csv(row_data_rpath) 



# Print sample
def struc (df, nsample=3):
    '''Explores data structure by showing data types, number of unique values and a sample of the data.'''
    notnull_ptj = (df.notnull().sum()/len(df)).apply(lambda x: f'{x:.2%}')
    notnull_df = df.notnull().sum().apply(lambda x: f'{x:,}')       
    out_df =  pd.concat([df.dtypes, notnull_df, notnull_ptj , df.nunique(), df.sample(nsample).reset_index(drop=True).T], axis=1)
    out_df.columns = ['dtype', 'notnull', 'notnull_pct',     'nunique'] + [f'sample_{i}' for i in range(1,nsample+1)]
    return out_df
print(struc(data, nsample=1))

#%%
# Drop null values on loan int rate and person emp length

data2 = data.loc[~((data['loan_int_rate'].isnull()) | (data['person_emp_length'].isnull())),:]
struc(data2)



# %%
# Drop outliers on age, and emp length
emp_length_over50_II = data2.loc[data2['person_emp_length']>50,:]
age_over_94_II = data2.loc[data2['person_age']>94,:]
data3 = data2.loc[~((data2['person_emp_length']>50) | (data2['person_age']>94)),:]
struc(data3)
# %%
# Write out data
out_data_rpath = os.path.join(Path.cwd().parent,"data/processed/credit_risk_prepared.csv")
print(f'writing {out_data_rpath}')
data3.to_csv(out_data_rpath, index=False)
# %%

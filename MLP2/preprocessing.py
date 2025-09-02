# %%
import matplotlib.pyplot as plt

# %%
"""
## Contoh 1
"""

# %%
from os import listdir
Filenames = listdir("/home/jundial/Learning/ML_learning/codes/MLP2/Preprocessing/pidato")
print(Filenames)

# %%
import pandas as pd
df = pd.DataFrame(index=range(len(Filenames)), columns=['File Name', 'Content'])

# %%
print(df)

# %%
for i, f_name in enumerate(Filenames):
    
    with open('/home/jundial/Learning/ML_learning/codes/MLP2/Preprocessing/pidato/'+f_name,"r",encoding='utf-8') as f:
        f_content = f.readlines()

    df.loc[i,'File Name'] = f_name
    df.loc[i,'Content'] = f_content[0]

# df.columns=['File Name','Content']
df

# %%
df.describe()

# %%
"""
## Contoh 2
"""

# %%
temp_df = pd.read_csv(r"/home/jundial/Learning/ML_learning/codes/MLP2/Preprocessing/TempData.csv")
temp_df

# %%
temp_df.info()

# %%
temp_df.nunique()

# %%
temp_df_2016 = temp_df.drop(columns=['Year'])
temp_df_2016.set_index(['Month','Day', 'Time'], inplace=True)
temp_df_2016

# %%
"""
## Contoh 3
"""

# %%
response_df = pd.read_csv(r'/home/jundial/Learning/ML_learning/codes/MLP2/Preprocessing/OSMI Mental Health in Tech Survey 2019.csv')
response_df

# %%
keys = ['Q'+ str(i) for i in range(1,83)]
keys

# %%
response_df.columns = keys
response_df.columns

# %%
"""
## Contoh 4

"""

# %%
Months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Oct','Sep','Nov','Dec']

def seperate_city(v):
    for mon in Months:
        if mon in v:
            return v[:v.find(mon)]
        
df['City'] = df['File Name'].apply(seperate_city)
df.head(5)

def seperate_date(v):
    en = v['File Name'].find('.txt')
    city_len = len(v['City'])

    return v['File Name'][city_len:en]

df['Date'] = df.apply(seperate_date, axis=1)
df.Date = pd.to_datetime(df.Date, format='%b%d_%Y')

def extract_DMY(r):
    r['Day'] = r.Date.day
    r['Month'] = r.Date.month
    r['Year'] = r.Date.year
    return r

df = df.apply(extract_DMY, axis=1)
df.head()

# %%
Words = ['vote', 'tax','campaign', 'economy' ]

def find_word_ratio(row):
    total_word = len(row.Content.split(' '))
    for w in Words:
        row['r_{}'.format(w)] = 100*(row.Content.count(w)/total_word)
    return row

df = df.apply(find_word_ratio, axis=1)
df.head()

# %%
"""
## Contoh 5
"""

# %%
cust_df = pd.read_csv('/home/jundial/Learning/ML_learning/codes/MLP2/Preprocessing/Customer Churn.csv')
cust_df.columns = ['Call_Failure', 'Complains','Subscription_Length', 'Seconds_of_Use','Frequency_of_use', 'Frequency_of_SMS','Distinct_Called_Numbers','Status', 'Churn']
churn_possibilities = cust_df.Churn.unique()
box_sr = pd.Series('',index = churn_possibilities)
for poss in churn_possibilities:
    BM = cust_df.Churn == poss
    box_sr[poss] = cust_df[BM].Call_Failure.values#Plot data origin
print(box_sr)
plt.boxplot(box_sr,vert=False)
plt.yticks([1,2],['Not Churn','Churn'])
plt.show()

# %%
n_unique = cust_df.Churn.nunique()
n_unique

# %%
"""
## Contoh 6

"""

# %%
month_df = pd.read_csv('/home/jundial/Learning/ML_learning/codes/MLP2/Preprocessing/Electric_Production.csv')
month_df.info()

# %%
month_df.head()

# %%
month_df.columns=['Date', 'Demand']
month_df.set_index(pd.to_datetime(month_df.Date, format='%m/%d/%Y'),inplace=True)
month_df.drop(columns=['Date'], inplace=True)
month_df.head()

# %%
attributes_dic={'IA1':'Average demand of the month','IA2':'Slope of change for the demand of the month','IA3': 'Average demands of months t-2, t-3 and t-4','DA': 'Demand of month t'}
predict_df =pd.DataFrame(index=month_df.iloc[24:].index,columns=attributes_dic.keys())
predict_df

# %%
predict_df.DA = month_df.loc['1987-01-01':].Demand
predict_df

# %%
"""
## Contoh 7
"""

# %%
import pandas as pd
data2_df = pd.read_csv('/home/jundial/Learning/ML_learning/codes/MLP2/Preprocessing/data2.csv')
data2_df.dropna(inplace = True)
# print(data2_df.to_string())
data2_df.head()

# %%
"""
## Contoh 8
"""

# %%
data2_df.columns = ['Q1','Date','Q2','Q3','Q4']
data2_df.head()
# data2_df['Date'] = pd.to_datetime(data2_df['Date'])
# print(data2_df.to_string())

# %%
data2_df['Date'] = data2_df['Date'].str.replace("'", "")
data2_df['Date'] = pd.to_datetime(data2_df['Date'], format="%Y/%m/%d")
print(data2_df.to_string())

# %%
"""
## Contoh 9
"""

# %%
print(data2_df.duplicated())# Remove all duplicates
data2_df.drop_duplicates(inplace = True)

# %%
data2_df.reset_index(drop=True, inplace=True)
data2_df

# %%
"""
## Contoh 10
"""

# %%
cl_df = pd.read_csv('/home/jundial/Learning/ML_learning/codes/MLP2/Preprocessing/columns.csv')
cl_df.info()

# %%
rsp_df = pd.read_csv('/home/jundial/Learning/ML_learning/codes/MLP2/Preprocessing/responses.csv')
rsp_df.info()

# %%
fig = plt.boxplot(rsp_df.Weight.dropna(),vert=False)
rsp_df[rsp_df.Weight>105]

# %%
"""
## Contoh 11
"""

# %%
data2_df.columns = ['Q1','Date','Q2','Q3','Calories']
data2_df.head()

# %%
data2_df.columns

# %%
data2_df.fillna(100,inplace=True)
data2_df

# %%
"""
## Contoh 12
"""

# %%

data2_df["Calories"].fillna(100, inplace=True)
print(data2_df.to_string())

# %%
"""
### Contoh 13

"""

# %%
data2_df.Calories.fillna(data2_df.mean(), inplace=True)
data2_df
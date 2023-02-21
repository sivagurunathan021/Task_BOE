# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:54:30 2023

@author: 91995
"""

import pandas as pd
import matplotlib as plt
import dtale as dt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

from sklearn.preprocessing import scale
from sklearn.cluster import KMeans

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df_General=pd.read_excel(r'C:\Users\91995\OneDrive\Desktop\BOI\General_dataset.xlsx')
df_Underwriting=pd.read_excel(r'C:\Users\91995\OneDrive\Desktop\BOI\Underwriting_dataset.xlsx')
#df3= pd.read_excel(r'C:\Users\91995\OneDrive\Desktop\BOI\Under.xlsx')

df_General.columns

cols=df_General.columns
cols2=df_Underwriting.columns

# add column names from the first row
df_General.columns = df_General.iloc[0]
df_General = df_General.rename(columns=lambda x: f'{x}_')

df_Underwriting.columns = df_Underwriting.iloc[0]
df_Underwriting = df_Underwriting.rename(columns=lambda x: f'{x}_')

# merge the first row with the existing column name
df_General.set_axis([f'{col}{df_General.iloc[0][i]}' for i, col in enumerate(cols)], axis=1, inplace=True)
df_Underwriting.set_axis([f'{col}{df_Underwriting.iloc[0][i]}' for i, col in enumerate(cols2)], axis=1, inplace=True)

df_General.drop(df_General.index[0], inplace=True)
df_Underwriting.drop(df_Underwriting.index[0], inplace=True)

df_General.rename( columns={'Unnamed: 0nan':'Firms'}, inplace=True )
df_Underwriting.rename( columns={'Unnamed: 0nan':'Firms'}, inplace=True )


df_final1=df_General.copy()



#df_final1=df_final1.iloc[1:]

###################################Data Prep:###################
##NWP
df_final1.columns
df_final1.shape
# drop rows when all 5 column values of NWP and GWP are equal to zero
#NWP
count = ((df_final1[['NWP (£m) 2016YE', 'NWP (£m) .12017YE', 'NWP (£m) .22018YE', 'NWP (£m) .32019YE','NWP (£m) .42020YE']] == 0).sum(axis=1) >= 5).sum()

df_final1 = df_final1.loc[~(df_final1[['NWP (£m) 2016YE', 'NWP (£m) .12017YE', 'NWP (£m) .22018YE', 'NWP (£m) .32019YE','NWP (£m) .42020YE']] == 0).all(axis=1)]
########GWP

count = ((df_final1[['GWP (£m)2016YE', 'GWP (£m).12017YE', 'GWP (£m).22018YE','GWP (£m).22018YE', 'GWP (£m).32019YE','GWP (£m).42020YE']] == 0).sum(axis=1) >= 5).sum()
rows = df_final1.loc[(df_final1[['GWP (£m)2016YE', 'GWP (£m).12017YE', 'GWP (£m).22018YE','GWP (£m).22018YE', 'GWP (£m).32019YE','GWP (£m).42020YE']] == 0).sum(axis=1) >= 3]
rows_1=df_final1.loc[(df_final1[['NWP (£m) 2016YE', 'NWP (£m) .12017YE', 'NWP (£m) .22018YE', 'NWP (£m) .32019YE','NWP (£m) .42020YE']] == 0).sum(axis=1) >= 3]

df_final1 = df_final1.loc[~(df_General[['GWP (£m)2016YE', 'GWP (£m).12017YE', 'GWP (£m).22018YE', 'GWP (£m).32019YE','GWP (£m).42020YE']] == 0).all(axis=1)]

### drop rows when all 3 column values of NWP and GWP are equal to zero
#GWP
df_final1 = df_final1.drop(df_final1[(df_final1[['GWP (£m)2016YE', 'GWP (£m).12017YE', 'GWP (£m).22018YE', 'GWP (£m).32019YE','GWP (£m).42020YE']] == 0).sum(axis=1) >= 3].index)
#NWP
df_final1 = df_final1.drop(df_final1[(df_final1[['NWP (£m) 2016YE', 'NWP (£m) .12017YE', 'NWP (£m) .22018YE', 'NWP (£m) .32019YE','NWP (£m) .42020YE']] == 0).sum(axis=1) >= 3].index)

################First analysis################
##Firm Size
# group data by firm and calculate average GWP over 5 years
#avg_gwp = df_final1.groupby('Firms')['GWP (£m)2016YE', 'GWP (£m).12017YE', 'GWP (£m).22018YE', 'GWP (£m).32019YE','GWP (£m).42020YE'].mean()

##sum of GWP over 5 years.
df_final1 = df_final1.assign(GWP_Total=df_final1[['GWP (£m)2016YE', 'GWP (£m).12017YE', 'GWP (£m).22018YE', 'GWP (£m).32019YE','GWP (£m).42020YE']].sum(axis=1))
#df_final1=df_final1.drop('Total', axis=1)

##avg of GWP over 5 years.
df_final1 = df_final1.assign(GWP_Avg=df_final1[['GWP (£m)2016YE', 'GWP (£m).12017YE', 'GWP (£m).22018YE', 'GWP (£m).32019YE','GWP (£m).42020YE']].mean(axis=1))

##std of GWP over 5 years
df_final1 = df_final1.assign(GWP_STD=df_final1[['GWP (£m)2016YE', 'GWP (£m).12017YE', 'GWP (£m).22018YE', 'GWP (£m).32019YE','GWP (£m).42020YE']].std(axis=1))

##sum of NWP over 5 years.
df_final1 = df_final1.assign(NWP_Total=df_final1[['NWP (£m) 2016YE', 'NWP (£m) .12017YE', 'NWP (£m) .22018YE', 'NWP (£m) .32019YE','NWP (£m) .42020YE']].sum(axis=1))
#df_final1=df_final1.drop('Total', axis=1)

##avg of GWP over 5 years
df_final1 = df_final1.assign(NWP_Avg=df_final1[['NWP (£m) 2016YE', 'NWP (£m) .12017YE', 'NWP (£m) .22018YE', 'NWP (£m) .32019YE','NWP (£m) .42020YE']].mean(axis=1))

##std of GWP over 5 years
df_final1 = df_final1.assign(NWP_STD=df_final1[['NWP (£m) 2016YE', 'NWP (£m) .12017YE', 'NWP (£m) .22018YE', 'NWP (£m) .32019YE','NWP (£m) .42020YE']].std(axis=1))

df_final1.columns

#Merging
df_merged=pd.merge(df_final1, df_Underwriting, on='Firms', how='inner')




##GCI--growth/fall.

df_merged['GCI_2016-2017'] = df_merged.apply(lambda row: (row['Gross claims incurred (£m).12017YE']- row['Gross claims incurred (£m)2016YE']) / row['Gross claims incurred (£m)2016YE'] *100 if row['Gross claims incurred (£m)2016YE'] != 0 else None, axis=1)

df_merged['GCI_2017-2018'] = df_merged.apply(lambda row: (row['Gross claims incurred (£m).22018YE']- row['Gross claims incurred (£m).12017YE']) / row['Gross claims incurred (£m).12017YE'] *100 if row['Gross claims incurred (£m).12017YE'] != 0 else None, axis=1)

df_merged['GCI_2018-2019'] = df_merged.apply(lambda row: (row['Gross claims incurred (£m).32019YE']- row['Gross claims incurred (£m).22018YE']) / row['Gross claims incurred (£m).22018YE'] *100 if row['Gross claims incurred (£m).22018YE'] != 0 else None, axis=1)

df_merged['GCI_2019-2020'] = df_merged.apply(lambda row: (row['Gross claims incurred (£m).42020YE']- row['Gross claims incurred (£m).32019YE']) / row['Gross claims incurred (£m).32019YE'] *100 if row['Gross claims incurred (£m).32019YE'] != 0 else None, axis=1)

##GWI--growth/fall.
df_merged['GWP_2016-2017'] = df_merged.apply(lambda row: (row['GWP (£m).12017YE']- row['GWP (£m)2016YE']) / row['GWP (£m)2016YE'] *100 if row['GWP (£m)2016YE'] != 0 else None, axis=1)

df_merged['GWP_2017-2018'] = df_merged.apply(lambda row: (row['GWP (£m).22018YE']- row['GWP (£m).12017YE']) / row['GWP (£m).12017YE'] *100 if row['GWP (£m).12017YE'] != 0 else None, axis=1)

df_merged['GWP_2018-2019'] = df_merged.apply(lambda row: (row['GWP (£m).32019YE']- row['GWP (£m).22018YE']) / row['GWP (£m).22018YE'] *100 if row['GWP (£m).22018YE'] != 0 else None, axis=1)

df_merged['GWP_2019-2020'] = df_merged.apply(lambda row: (row['GWP (£m).42020YE']- row['GWP (£m).32019YE']) / row['GWP (£m).32019YE'] *100 if row['GWP (£m).32019YE'] != 0 else None, axis=1)

##SCR coverage ratio, if <100% then firm is holding enough capital to meet the requirement. The size of the buffer (i.e. surplus over 100%) can be important. 
df_merged = df_merged.assign(SCR_Avg=df_merged[['SCR coverage ratio2016YE', 'SCR coverage ratio.12017YE', 'SCR coverage ratio.22018YE', 'SCR coverage ratio.32019YE','SCR coverage ratio.42020YE']].mean(axis=1))


##Net combined ratio – (incurred losses plus expenses) / earned premiums. This is a ratio that can indicate the profitability of a firm. If this is less than 100% it indicates a profit. 
df_merged = df_merged.assign(NCR_Avg=df_merged[['Net combined ratio2016YE', 'Net combined ratio.12017YE', 'Net combined ratio.22018YE', 'Net combined ratio.32019YE','Net combined ratio.42020YE']].mean(axis=1))

##Filtering out Loss Making and Bad captial holding Firms
##Firms with SCR<1--bad capital holding
less_than_one_SCR = df_merged[df_merged['SCR_Avg'] < 1]
print(less_than_one)


##Firms with NCR>1--Loss makin
more_than_one_NCR = df_merged[df_merged['NCR_Avg'] > 1]
print(more_than_one_SCR)

#Combination of loss making and bad capital holding.
badholding_LossMaking = df_merged[(df_merged['SCR_Avg'] < 1) & (df_merged['NCR_Avg'] > 1)][['Firms', 'NCR_Avg', 'SCR_Avg','GWP_Avg','GWP_Total']]
print(badholding_LossMaking)
            
##Firms with Good capital holding yet making loss.'SCR_Avg'] >= 1) and ['NCR_Avg'] > 1)
More_than_1_SCR_NCR_more_1 = df_merged[(df_merged['SCR_Avg'] >= 1) & (df_merged['NCR_Avg'] > 1)][['Firms', 'NCR_Avg', 'SCR_Avg','GWP_Avg','GWP_Total']]
print(More_than_2_SCR)

sorted_df = More_than_1_SCR_NCR_more_1.sort_values(by=More_than_1_SCR_NCR_more_1.columns.tolist(), ascending=False)
print(sorted_df)

#checks
#suma = df_merged.loc[(df_merged[['GCI_2016-2017', 'GCI_2017-2018', 'GCI_2018-2019','GCI_2019-2020']] == 0).sum(axis=1) >= 3]

#df_final1 = df_final1.drop(df_final1[(df_final1[['GWP (£m)2016YE', 'GWP (£m).12017YE', 'GWP (£m).22018YE', 'GWP (£m).32019YE','GWP (£m).42020YE']] == 0).sum(axis=1) >= 3].index)
df=df_merged.copy()
#nan_cols_mask = (df_merged[['GCI_2016-2017', 'GCI_2017-2018', 'GCI_2018-2019', 'GCI_2019-2020']].isna()).sum(axis=1) >= 2
#result_df = df[nan_cols_mask]
#print(result_df)

df
#df.dropna(thresh=1, subset=['GCI_2016-2017', 'GCI_2017-2018', 'GCI_2018-2019', 'GCI_2019-2020'], inplace=True)
#df.dropna(thresh=2, subset=['GCI_2016-2017', 'GCI_2017-2018', 'GCI_2018-2019', 'GCI_2019-2020'], inplace=True)

df_2=df.copy()

##droping nan values in any of 'GCI_2016-2017', 'GCI_2017-2018', 'GCI_2018-2019', 'GCI_2019-2020'
df_2.dropna(subset=['GCI_2016-2017', 'GCI_2017-2018', 'GCI_2018-2019', 'GCI_2019-2020'], inplace=True)


df_2['GCI_pct_change'] = df_2[['GCI_2016-2017', 'GCI_2017-2018', 'GCI_2018-2019', 'GCI_2019-2020']].sum(axis=1).pct_change(periods=4)
df_2['GWP_pct_change'] = df_2[['GWP_2016-2017', 'GWP_2017-2018', 'GWP_2018-2019', 'GWP_2019-2020']].sum(axis=1).pct_change(periods=4)

df_changing_profiles=df_2[['Firms','GWP_Total','GCI_pct_change','GWP_pct_change']]


df_changing_profiles=df_changing_profiles[['Firms','GWP_Total','GCI_pct_change','GWP_pct_change']].sort_values(by=['GCI_pct_change','GWP_pct_change'], ascending=False)

df_2.hist(figsize=(12, 10))
plt.show()

import matplotlib as plt


df_merged.info()

####################Outlier detection using z-score.###########
##using GWP ---Reject
df_2['Z_Score_GWP_2016']=(df_2['GWP (£m)2016YE']-df_2['GWP_Avg'])/df_2['GWP_STD']
df_2['Z_Score_GWP_2017']=(df_2['GWP (£m).12017YE']-df_2['GWP_Avg'])/df_2['GWP_STD']
df_2['Z_Score_GWP_2018']=(df_2['GWP (£m).22018YE']-df_2['GWP_Avg'])/df_2['GWP_STD']
df_2['Z_Score_GWP_2019']=(df_2['GWP (£m).32019YE']-df_2['GWP_Avg'])/df_2['GWP_STD']
df_2['Z_Score_GWP_2020']=(df_2['GWP (£m).42020YE']-df_2['GWP_Avg'])/df_2['GWP_STD']
df_2['Zcd']=(df_2['Z_Score_GWP_2016']+df_2['Z_Score_GWP_2017']+df_2['Z_Score_GWP_2018']+df_2['Z_Score_GWP_2019']+df_2['Z_Score_GWP_2020'])/5

average_GWP_2016 = df_2["GWP (£m)2016YE"].mean()
average_NWP_2016=  df_2["NWP (£m) 2016YE"].mean()
df_2['average_chang2_2016']=df_2['NWP (£m) 2016YE']/df_2['NWP_Avg']*100
##############################################################3Graphs

##Df for graphs.

df_2.info()
a=df_General.columns
b=df_2.columns
df_graphs= df_2.filter(items=a)

df_graphs_2=df_2[[	"GWP_Total",
	'GWP_Avg',
	'GWP_STD',
	'NWP_Total',
	'NWP_Avg',
	'NWP_STD',
	'Gross claims incurred (£m)2016YE',
	'Gross claims incurred (£m).12017YE',
   'Gross claims incurred (£m).22018YE',
   'Gross claims incurred (£m).32019YE',
   'Gross claims incurred (£m).42020YE',
	'Net combined ratio2016YE',
	'Net combined ratio.12017YE',
	'Net combined ratio.22018YE',
	'Net combined ratio.32019YE',
	'Net combined ratio.42020YE',
	'GCI_2016-2017',
	'GCI_2017-2018',
	'GCI_2018-2019',
	'GCI_2019-2020',
	'GWP_2016-2017',
	'GWP_2017-2018',
	'GWP_2018-2019',
	'GWP_2019-2020',
   'SCR_Avg',
	'NCR_Avg',
'GCI_pct_change',
'GWP_pct_change']]

df_graphs_2.info()
df_graphs_2_astype=df_graphs_2.astype('float')
Total_df_graphs = pd.concat([df_graphs_astype, df_graphs_2_astype], axis=1)

df_graphs.to_csv('df_graphs.csv')
df_graphs_2.to_csv('df_graphs_2.csv')
Total_df_graphs.to_csv('Total_df_graphs.csv')

# Calculate correlation matrix
corr = df_graphs_2_astype.corr()

# Plot heatmap of correlation matrix
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

from pandas_profiling import ProfileReport
profile=ProfileReport(Total_df_graphs,explorative=True)
profile.to_file('output_3.html')

dt.show(Total_df_graphs)
import dtale as dt
import dtale.app as dtale_app
dtale_app.USE_NGROK = False
d=dtale.show(df_graphs_2_astype)
d.open_browser()   

dt.show(Total_df_graphs)
d = dt.show(df_graphs_2_astype)
d.open_browser()  

################Scatter plot GWP and NWP##################
plt.scatter(df_graphs['GWP (£m)2016YE'], df_graphs['NWP (£m) 2016YE'])
plt.xlabel('GWP')
plt.ylabel('NWP')
plt.title('GWP vs NWP')
plt.show()


df_Full_3 = df_Full_3.astype('float')
df_General_astype=df_General.loc[:, df_General.columns != 'Firms']
df_General_astype=df_General_astype.astype('float')
df_General_astype.info()

plt.scatter(df_2['GWP_Total'], df_2['Net combined ratio.12017YE'])
plt.xlabel('GWP')
plt.ylabel('Net combined ratio')
plt.title('GWP vs Net Combined Ratio')
plt.show()

# Plot histograms of all columns
df_Full_3.hist(bins=15, figsize=(15, 10), grid=False)
plt.tight_layout()

# Plot boxplots of all columns
df_graphs_astype=df_graphs.loc[:, df_graphs.columns != 'Firms']
df_graphs_astype=df_graphs_astype.astype('float')

df_graphs_2_astype=df_graphs_2.astype('float')

plt.figure(figsize=(15, 10))
df_graphs_astype.boxplot()
plt.xticks(rotation=45)


##Df for graphs.
a=df.columns
df_graphs=df[[]]

#########Clustering Analysis###################################
#####data prep####
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df_cluster=df_final1.copy()
df_cluster = df_cluster.loc[~(df_cluster[['NWP (£m) 2016YE', 'NWP (£m) .12017YE', 'NWP (£m) .22018YE', 'NWP (£m) .32019YE','NWP (£m) .42020YE']] == 0).all(axis=1)]
df_cluster = df_cluster.loc[~(df_cluster[['GWP (£m)2016YE', 'GWP (£m).12017YE', 'GWP (£m).22018YE', 'GWP (£m).32019YE','GWP (£m).42020YE']] == 0).all(axis=1)]



### drop rows when all 3 column values of NWP and GWP are equal to zero
#GWP
df_cluster = df_cluster.drop(df_cluster[(df_cluster[['GWP (£m)2016YE', 'GWP (£m).12017YE', 'GWP (£m).22018YE', 'GWP (£m).32019YE','GWP (£m).42020YE']] == 0).sum(axis=1) >= 3].index)
#NWP
df_cluster = df_cluster.drop(df_cluster[(df_cluster[['NWP (£m) 2016YE', 'NWP (£m) .12017YE', 'NWP (£m) .22018YE', 'NWP (£m) .32019YE','NWP (£m) .42020YE']] == 0).sum(axis=1) >= 3].index)

#Merging
z=df_Underwriting.columns
df_cluster_merged=pd.merge(df_cluster, df_Underwriting, on='Firms', how='inner')

### drop rows when all 3 column values of GCI and NCR are equal to zero
#GCI
df_cluster_merged = df_cluster_merged.loc[~(df_cluster_merged[['Gross claims incurred (£m)2016YE', 'Gross claims incurred (£m).12017YE', 'Gross claims incurred (£m).22018YE', 'Gross claims incurred (£m).32019YE','Gross claims incurred (£m).42020YE']] == 0).all(axis=1)]

df_cluster_merged = df_cluster_merged.drop(df_cluster_merged[(df_cluster_merged[['Gross claims incurred (£m)2016YE', 'Gross claims incurred (£m).12017YE', 'Gross claims incurred (£m).22018YE', 'Gross claims incurred (£m).32019YE','Gross claims incurred (£m).42020YE']] == 0).sum(axis=1) >= 3].index)

##NCR####
df_cluster_merged = df_cluster_merged.loc[~(df_cluster_merged[['Net combined ratio2016YE', 'Net combined ratio.12017YE', 'Net combined ratio.22018YE', 'Net combined ratio.32019YE','Net combined ratio.42020YE']] == 0).all(axis=1)]

df_cluster_merged = df_cluster_merged.drop(df_cluster_merged[(df_cluster_merged[['Net combined ratio2016YE', 'Net combined ratio.12017YE', 'Net combined ratio.22018YE', 'Net combined ratio.32019YE','Net combined ratio.42020YE']] == 0).sum(axis=1) >= 3].index)

df_cluster_merged

df_cluster_merged.info()

########General test, not included as part of code #######
df_suma=df_cluster_merged.copy()
df_suma = df_suma.assign(GWP_Total=df_suma[['GWP (£m)2016YE', 'GWP (£m).12017YE', 'GWP (£m).22018YE', 'GWP (£m).32019YE','GWP (£m).42020YE']].sum(axis=1))
df_suma = df_suma.assign(NWP_Total=df_suma[['NWP (£m) 2016YE', 'NWP (£m) .12017YE', 'NWP (£m) .22018YE', 'NWP (£m) .32019YE','NWP (£m) .42020YE']].sum(axis=1))

df_suma_GWP_NWP=df_suma[['Firms','GWP_Total','NWP_Total']]
df_cluster_merged.info()

#changing cat to num for the alg to run
from sklearn.preprocessing import LabelEncoder
df_cluster_merged_hold=df_cluster_merged.copy()

df_cluster_merged["Firms"] = df_cluster_merged[["Firms"]].apply(LabelEncoder().fit_transform)
summary_stats=df_cluster_merged_flt.describe()

df_cluster_merged_flt = df_cluster_merged.astype('float')

scaler = StandardScaler()
df_cluster_merged_scaled = scaler.fit_transform(df_cluster_merged)
#df_cluster_merged_scaled=df_cluster_merged_scaled.dropna()

  
## Choosing optimal K##########
df_merged_clusterclustdist = pd.Series(0.0,index = range(1,21))

for k in range(1,21):
    Firms_anyK = KMeans(n_clusters=k).fit(df_cluster_merged_scaled)
    df_merged_clusterclustdist[k] = Firms_anyK.inertia_
    
plt.figure()
plt.plot(df_merged_clusterclustdist)
plt.xlim([0,7])

df_merged_cluster = KMeans(n_clusters=4, random_state=1234).fit(df_cluster_merged_scaled)
df_merged_cluster.cluster_centers_
df_merged_cluster.inertia_
df_merged_cluster.labels_
df_cluster_merged_scaled=pd.DataFrame(df_cluster_merged_scaled)
df_merged_cluster_label = df_cluster_merged_scaled.copy()
df_merged_cluster_label.columns=df_cluster_merged_flt.columns
df_merged_cluster_label=pd.DataFrame(df_merged_cluster_label,columns = df_cluster_merged_flt.columns)
df_cluster_merged_flt["Cluster4"] =df_merged_cluster.labels_
df_merged_cluster_label["Cluster4"] = df_merged_cluster.labels_
df_merged_cluster_centroids = pd.DataFrame(df_merged_cluster.cluster_centers_,columns = df_cluster_merged_flt.columns)

df_merged_cluster_label["Cluster4"].value_counts()

######Attempts on Visualisation###########
sns.pairplot(df_merged_cluster_label['EoF for SCR (£m)2016YE'], hue = "Cluster4")
df_merged_cluster_label.columns
df_merged_cluster_label_1=df_merged_cluster_label.copy()
df_merged_cluster_label_1 = df_merged_cluster_label_1.rename(columns={'Firms': 'Firms_1'})

df_merged_cluster_label_1.insert(0, 'Firms', df_cluster_merged_flt['Firms'])

df_merged_cluster_label.to_csv('iris_with_label.csv')

df_cluster_merged_flt

cluster=df_cluster_merged_flt[['Cluster4','Firms']]

cluster_1=pd.DataFrame(df_cluster_merged_flt[['Cluster4', 'Firms']][df_cluster_merged_flt['Cluster4'] == 1])

cluster_0=pd.DataFrame(df_cluster_merged_flt[['Cluster4', 'Firms']][df_cluster_merged_flt['Cluster4'] == 0])

cluster_3=pd.DataFrame(df_cluster_merged_flt[['Cluster4', 'Firms']][df_cluster_merged_flt['Cluster4'] == 3])

df_merged_cluster_label_profile = df_merged_cluster_label.groupby("Cluster4").mean()

for i in df_merged_cluster_label_profile.columns:
    plt.figure()
    df_merged_cluster_label_profile[i].plot.bar()
    plt.title(i)

# Comparing distribution
for i in df_merged_cluster_label_profile.columns:
    df_merged_cluster_label.boxplot(column = i,by = "Cluster4")




#NWP/GWP- Risk
df_2['NWP/GWP_2016']=df_2.apply(lambda row: row['NWP (£m) 2016YE'] / row['GWP (£m)2016YE'] if row['GWP (£m)2016YE'] != 0 else None, axis=1)
df_2['NWP/GWP_2017']=df_2.apply(lambda row: row['NWP (£m) .12017YE']/ row['GWP (£m).12017YE'] if row['GWP (£m).12017YE'] != 0 else None, axis=1)
df_2['NWP/GWP_2018']=df_2.apply(lambda row: row['NWP (£m) .22018YE'] / row['GWP (£m).22018YE'] if row['GWP (£m).22018YE'] != 0 else None, axis=1)
df_2['NWP/GWP_2019']=df_2.apply(lambda row: row['NWP (£m) .32019YE'] / row['GWP (£m).32019YE'] if row['GWP (£m).32019YE'] != 0 else None, axis=1)
df_2['NWP/GWP_2020']=df_2.apply(lambda row: row['NWP (£m) .42020YE']/ row['GWP (£m).42020YE']if row['GWP (£m).42020YE']!= 0 else None, axis=1)

df_final3.to_excel('df_final3.xlsx')
df_final4=df_final3.copy()

df_final3.columns

import matplotlib.pyplot as plt

# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(df_final3['Net combined ratio'],
        df_final3['Firms'],
        color="red",
        marker="o")
# set x-axis label
ax.set_xlabel("year", fontsize = 14)
# set y-axis label
ax.set_ylabel("lifeExp",
              color="red",
              fontsize=14)

# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.bar(df_final3['Firms'], df_final3['Net combined ratio'],color="blue", alpha=0.5)
ax2.set_ylabel("gdpPercap",color="blue",fontsize=14)
plt.show()
# save the plot as a file
fig.savefig('two_different_y_axis_for_single_python_plot_with_twinx.jpg',
            format='jpeg',
            dpi=100,
            bbox_inches='tight')

ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.bar(df['MISC_RENEWALCOUNT'], df["len"],color="blue", alpha=0.5),#marker="o")
ax2.set_ylabel("Volume",color="blue",fontsize=14)
plt.show()
# save the plot as a file
fig.savefig('two_different_y_axis_for_single_python_plot_with_twinx.jpg'+'_'+str(i),
format='jpeg',
dpi=100,
bbox_inches='tight')

df_final3.drop[('Firms')]
plt.scatter(df_final3['Firms'],df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income($)')

df_2.summary()


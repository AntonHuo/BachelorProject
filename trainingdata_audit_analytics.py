import pandas as pd

#load the first file
csv_file1 = "restatements_audit_analytics.csv"
csv_data1 = pd.read_csv(csv_file1, low_memory = False)
csv_df1 = pd.DataFrame(csv_data1)

#this csv has only one column, I need to split it into 3 columns
csv_df1['year'] = csv_df1['year;ric;effect'].map(lambda x:x.split(';')[0])
csv_df1['ric'] = csv_df1['year;ric;effect'].map(lambda x:x.split(';')[1])
csv_df1['effect'] = csv_df1['year;ric;effect'].map(lambda x:x.split(';')[2])

csv_df1['year']=csv_df1['year'].astype(int)

print(csv_df1)

#load the second file
csv_file2 = "testdateimitlabels.csv"
csv_data2 = pd.read_csv(csv_file2, low_memory = False)
csv_df2 = pd.DataFrame(csv_data2)
print(csv_df2)

#Merge two dataframes
csv_fusion = pd.merge(csv_df1 , csv_df2 , how='left', on=['ric','year'])

csv_audit = csv_fusion.drop(['year;ric;effect', 'all','relevant','relevant5%'], axis=1)

print(csv_audit)


#save the new dataframe as a csv file
csv_audit.to_csv('testdata_audit_analytics.csv',index=False)
from pymongo import MongoClient
import pandas as pd

#connect to the database
host = '134.99.112.190'
client = MongoClient(host, 27017)
db = client.finfraud3
db.authenticate("read_user", "tepco11x?z")

print('gut')

COLLECTION = "original"
db_coll_ori = db[COLLECTION]

#download all data as a dataframe
data = db_coll_ori.find()
data = list(data)

df = pd.DataFrame(data)

print('bitte')

#tranfer the dataframe as a dictionary
data_dict=df.to_dict(orient= 'dict')


#select all keys without quarterly information
keys = []
yearly_data = data_dict['1998'][0]
tech_names = {'FQ-1', 'FQ-3', 'FQ-4', 'FQ0'}
yearly_data_without_FQ = {key: value for key, value in yearly_data.items() if key not in tech_names}
for b in yearly_data_without_FQ:
    keys.append(b)

print(keys)

dic = dict()

#select all data for each key
for x in range(0, df.shape[0]):
    for y in range(1998, 2017):
        for key1 in keys:
            yearly_data = data_dict[str(y)][x]

            if key1 in yearly_data.keys():
                dic.setdefault(key1, []).append(data_dict[str(y)][x][key1])
            else:
                dic.setdefault(key1, []).append('nan')

#transfer the dictionary as dataframe and save it as a csv file.
dateicsv=pd.DataFrame(dic)

dateicsv.to_csv('testdatei.csv',index=False)



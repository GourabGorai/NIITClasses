import pandas as pd
import csv
def convert_views(views):
    if 'K' in views:
        return float(views.replace('K', '')) * 1_000
    elif 'M' in views:
        return float(views.replace('M', '')) * 1_000_000
    else:
        return float(views)
data=pd.read_csv('social media influencers - youtube.csv')
print(data.head())
print(data.tail())
print(data.describe())
df=pd.DataFrame(data)
print(df[df['youtuber name'].isna()])
data.info()
print(data.sort_values(by="avg views",ascending=True))
data2=pd.DataFrame(data)
data2['Subscribers']=data2['Subscribers'].apply(convert_views)
mv=data2[data2['Subscribers']>1000000]
print(mv)
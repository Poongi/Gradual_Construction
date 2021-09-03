import pandas as pd
import numpy as np



df = pd.read_csv("./example/IMDB_Dataset.csv", index_col=False)
df['sentiment'][df['sentiment']=='positive'] = '1'
df['sentiment'][df['sentiment']=='negative'] = '0'


for i in range(3000) :
    with open('./example/IMDB/'+'review_'+str(i)+'.txt', 'w') as f:
        f.write(df.iloc[i]['review'])

for i in range(3000) :
    with open('./example/IMDB/'+'label_'+str(i)+'.txt', 'w') as f:
        f.write(df.iloc[i]['sentiment'])


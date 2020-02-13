<<<<<<< HEAD
import pandas as pd
import numpy as np

df = pd.read_json('./tweets.json', lines=True)
df_user = df[df['text'].str.contains('$btc', regex=False) | 
    df['text'].str.contains('$BTC', regex=False)]
df_user = df_user['user'].to_dict()

=======
import pandas as pd
import numpy as np

df = pd.read_json('./tweets.json', lines=True)
df_user = df[df['text'].str.contains('$btc', regex=False) | 
    df['text'].str.contains('$BTC', regex=False)]
df_user = df_user['user'].to_dict()

>>>>>>> 5624d9f9ba101a024cf76ba4732c28fe55115448
unique_names = set([df_user[i]['screen_name'] for i in df_user.keys()])
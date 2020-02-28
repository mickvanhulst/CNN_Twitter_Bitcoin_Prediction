import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    df1 = pd.read_json('../data/processed/train_tweets.json', lines=True)
    df2 = pd.read_json('../data/processed/test_tweets.json', lines=True)
    df = pd.concat([df1, df2])

    # Visualize tweets per user distribution
    # df['count'] = df.groupby('user')['user'].transform('count')
    # bins = np.linspace(0, df['count'].max(), 25)
    # bins = [round(x) for x in bins]
    # df['binned'] = pd.cut(df['count'], bins)
    # df.groupby([df.binned]).agg('count')['count'].plot(kind='bar')

    # plt.xlabel('Bins of counts of tweets per user')
    # plt.ylabel('n amount of tweets')
    # plt.show()

    # Visualize distribution of datetime grouped by year/month
    df.groupby([df.created_at.dt.year, df.created_at.dt.month]).agg('count')['created_at'].plot(kind="bar")
    plt.xlabel('Created at (month/year combination)')
    plt.ylabel('n amount of tweets')
    # plt.title('Distribution of tweets for each month/year combination')
    plt.show()

    print(len(df1), len(df2), len(df))

    # df = df[df['created_at'] >= '2017-06-01 00:00:00']
    # df.groupby([df.created_at.dt.year, df.created_at.dt.month]).agg('count')['created_at'].plot(kind="bar")
    # plt.xlabel('Created at (month/year combination)')
    # plt.ylabel('n amount of tweets')
    # plt.show()



if __name__ == '__main__':
    main()

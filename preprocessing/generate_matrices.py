import pandas as pd
import numpy as np
import json
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import euclidean_distances
import gensim
import matplotlib.pyplot as plt
from gen_cust_features_matrix import gen_custom_features_tweet, TFIDF

def tfidf_matching(tweets, keywords, binary=True, partial=True):
    '''
    Check if keywords occur in bag of words, then either set binary values or perform a count.
    Also check for partial matchings, so either keyword fear matches fearful word in bag of words or
    word in bag of words fear matches keyword fearful.
    '''
    tokens = [word_tokenize(text) for text in tweets]
    bag_of_words = []
    bag_of_words.extend([item for sublist in tokens for item in sublist])

    keyword_occurrence = np.zeros(len(keywords))
    for idx, keyword in enumerate(keywords):
        kw, tfidf_score = keyword
        if partial:
            cnt = np.sum([1 for w in bag_of_words if (w in kw) | (kw in w)])
            if cnt > 0:
                if binary:
                    keyword_occurrence[idx] = 1
                else:
                    keyword_occurrence[idx] = cnt
        else:
            if kw in bag_of_words:
                if binary:
                    keyword_occurrence[idx] = 1
                else:
                    keyword_occurrence[idx] = bag_of_words.count(kw)
    return keyword_occurrence

def cluster_matching(tweets, model, centroids, binary=True):
    '''
    This approach assigns words to previously generated clusters. We added an option for either binary or count values.
    '''
    tokens = [word_tokenize(text) for text in tweets]
    bag_of_words = []
    bag_of_words.extend([item for sublist in tokens for item in sublist])
    keyword_occurrence = [0 for i in range(0, len(centroids))]

    for word in bag_of_words:
        try:
            word_vector = model[word]
        except:
            continue
#        word_vector = model[word]
        # Calculate distances with centroids and update to one for corresponding cluster.
        dist = euclidean_distances(centroids, [word_vector])
        min_idx = list(dist).index(min(dist))
        if binary:
            keyword_occurrence[min_idx] = 1
        else:
            keyword_occurrence[min_idx] = bag_of_words.count(word)

    return keyword_occurrence


def generate_array(tweets, keywords, model, centroids, tfidf_vectorizer, form=1, binary=True, partial=True):
    if form == 1:
        keyword_occurrence = tfidf_matching(tweets, keywords, binary, partial)
    elif form == 2:
        keyword_occurrence = gen_custom_features_tweet(tweets, tfidf_vectorizer)
    else:
        keyword_occurrence = cluster_matching(tweets, model, centroids, binary)

    return keyword_occurrence

def save_matrix_plot(matrix, c, n, mode, train, binary):
    if train:
        phase = 'train'
    else:
        phase = 'test'
    if binary:
        type = 'binary'
        color_map = type
    else:
        type = 'count'
        color_map ='hot'
    plt.figure(figsize=(8, 15))
    plt.imshow(matrix, cmap=color_map, interpolation='nearest', aspect='auto')
    plt.tight_layout()
    plt.colorbar()
    plt.savefig('../analysis/plots/matrices/{}/{}_{}_{}_{}'.format(mode, type, phase, c, n))
    plt.close()

def save_matrix_plot_avg(matrix, c, mode, train, binary):
    if train:
        phase = 'train'
    else:
        phase = 'test'
    if binary:
        type = 'binary'
    else:
        type = 'count'
    color_map = 'hot'
    plt.figure(figsize=(8, 15))
    plt.imshow(matrix, cmap=color_map, interpolation='nearest', aspect='auto')
    plt.tight_layout()
    plt.colorbar()
    plt.savefig('../analysis/plots/matrices/avg_plots/avg_all_{}_{}_{}_{}'.format(mode, type, phase, c))
    plt.close()

def main():
    '''
    Different forms:
    1 --> generate using TFIDF
    0 --> generate using clusters.
    '''
    # Set a few variables for different tests.
    train = True

    binary = False
    tfidf_amt_of_words = 500
    mode = ['keywords_tfidf_p', 'keywords_tfidf', 'keywords_clusters', 'custom_features'][3]
    partial = True if mode == 'keywords_tfidf_p' else False
    name_folder = mode + '/binary' if binary else mode + '/count'
    name_folder = name_folder + '/train' if train else name_folder + '/test'
    data_file = 'train_tweets' if train else 'test_tweets'

    tweets = pd.read_json('../data/processed/{}.json'.format(data_file), lines=True)
    btc_prices = pd.read_json('../data/processed/btc_prices.json', lines=True)

    # Load word2vec model and get centroids
    model = gensim.models.Word2Vec.load('../data/models/word2vec.model')
    centroids = np.load('../data/processed/centroids_25.npy')

    # Get keyword data
    with open('../data/processed/keywords.json', "r+") as f:
        keywords_dict = json.load(f)

    keywords = keywords_dict[str(tfidf_amt_of_words)]
    del keywords_dict

    # Some additional settings.
    form = 1 if 'keywords_tfidf' in name_folder else 2 if 'custom' in name_folder else 0
    size_matrix = len(keywords) if 'keywords_tfidf' in name_folder else 126 if 'custom' in name_folder else len(centroids)

    # For custom features
    df_temp = pd.read_json('../data/processed/train_tweets.json', lines=True)
    tfidf_vectorizer = TFIDF(df_temp['text_kw'].tolist())
    del df_temp

    # Set vars
    classes = []
    trade_classes = []
    matrices = []
    matrix_date_identifier = []
    meta_data_users = []
    meta_data_btc = []
    interval = 7

    # Statistics
    cnts = []
    avg_kw_per_matrix = 0
    tot_matrices = 0
    tot_dates = 0

    '''
    We look at the development of an individual user w.r.t. keyword and BTC prices. 
    We get the minimum/max date in the dataset and add the time interval to the minimum. After this we loop over
    all dates and then look back interval timesteps per date. For each date we filter users that have
    tweeted on that particular day and fill in the matrix based on the user's tweets.
    
    We save user meta-data and if the price went up/down at t+1.
    '''
    # start at min + interval, end at max tweet.
    min_max_interval = np.arange(tweets['created_at'].min() + np.timedelta64(interval - 1, 'D'),
                                 tweets['created_at'].max() + np.timedelta64(1, 'D'), dtype='datetime64[D]')
    avg_matrices = {}

    for date in min_max_interval:
        tot_dates += 1
        if date not in btc_prices['Date'].dt.date.values[1:]:
            print('Date {} not found in BTC df, ignoring the date'.format(date))
            continue
        print('----------------------------------------')
        print('Total percentage of dates processed {}'.format(round((tot_dates / len(min_max_interval)) * 100, 2)))
        print('Collected {} non-zero matrices thus far, percentage of matrix non-zero'
              ' is {}%'.format(tot_matrices, round(avg_kw_per_matrix, 5)))
        print('----------------------------------------')
        users = tweets['user'][tweets['created_at'].dt.date == date].unique()
        date_interval = np.arange(date - np.timedelta64(interval - 1, 'D'),
                                  date + np.timedelta64(1, 'D'), dtype='datetime64[D]')
        for user in users:
            tweets_user = tweets[(tweets['user'] == user)]
            #todo: move this below
            user_meta_data = tweets_user[['verified', 'followers_count', 'friends_count',
                               'listed_count', 'favourites_count', 'statuses_count', 'unix_months',
                               'twitter_user_id']].fillna(0).drop_duplicates().values.squeeze()
            if user_meta_data.ndim > 1:
                # Change in e.g. followers overtime, so we pick latest data.
                user_meta_data = user_meta_data[-1]

            # This should generate 1 matrix per user.
            matrix = None
            matrix_btc = None
            for day in date_interval:
                dt_btc_md = btc_prices[['perc_diff_close','perc_diff_high', 'perc_diff_low', 'perc_diff_open', 'perc_diff_vol_f',
                                        'perc_diff_vol_t']][btc_prices['Date'].dt.date == day].drop_duplicates().values.squeeze()
                tweets_user_on_date = tweets_user['text_kw'][tweets_user['created_at'].dt.date == day].tolist()
                # if date_interval[-1]:
                #     #todo: add user_metadata here for date.
                if len(tweets_user_on_date) == 0:
                    keyword_occurrence = np.array([0 for i in range(0, size_matrix)])
                else:
                    keyword_occurrence = generate_array(tweets_user_on_date, keywords, model, centroids, tfidf_vectorizer, form=form,
                                                        binary=binary, partial=partial)
                if matrix_btc is not None:
                    matrix_btc = np.dstack((matrix_btc, dt_btc_md))
                else:
                    matrix_btc = dt_btc_md
                if matrix is not None:
                    matrix = np.dstack((matrix, keyword_occurrence))
                else:
                    matrix = keyword_occurrence

            if not np.all(matrix == 0):
                tot_matrices += 1

                cnts.append(np.count_nonzero(matrix) / (size_matrix * interval))
                avg_kw_per_matrix = np.mean(cnts)
                dt_plus_1 = date + np.timedelta64(1, 'D')
                trade_profitable = btc_prices['trade_profitable'][btc_prices['Date'].dt.date == dt_plus_1].values[0]
                trade_class = btc_prices['trade_class'][btc_prices['Date'].dt.date == dt_plus_1].values[0]

                if trade_profitable not in avg_matrices.keys():
                    avg_matrices[trade_profitable] = [matrix, 1]
                else:
                    avg_matrices[trade_profitable] = [avg_matrices[trade_profitable][0] + matrix, avg_matrices[trade_profitable][1] + 1]

                if (tot_matrices % 500) == 0:
                    save_matrix_plot(matrix.squeeze(), trade_profitable, tot_matrices, mode, train, binary)

                meta_data_users.append(user_meta_data.astype(float))
                matrix_date_identifier.append(date.astype(float))
                matrices.append(matrix.squeeze())
                meta_data_btc.append(matrix_btc.squeeze())
                classes.append(trade_profitable)
                trade_classes.append(trade_class)

                if len(matrices) >= 1500:
                    # Save matrices to numpy file every 1500 matrices or if last run.
                    np.save('../data/processed/matrices/{}/md_up_until{}'.format(name_folder, tot_matrices),
                            np.array(meta_data_users).squeeze())
                    np.save('../data/processed/matrices/{}/md_btc_up_until{}'.format(name_folder, tot_matrices),
                            np.array(meta_data_btc).squeeze())
                    np.save('../data/processed/matrices/{}/dt_up_until{}'.format(name_folder, tot_matrices),
                            np.array(matrix_date_identifier).squeeze())
                    np.save('../data/processed/matrices/{}/dp_up_until_{}'.format(name_folder, tot_matrices),
                            np.array(matrices).squeeze())
                    np.save('../data/processed/matrices/{}/c_up_until_{}'.
                            format(name_folder, tot_matrices), np.array(classes).squeeze())
                    np.save('../data/processed/matrices/{}/tc_up_until_{}'.
                            format(name_folder, tot_matrices), np.array(trade_classes).squeeze())
                    meta_data_users = []
                    meta_data_btc = []
                    matrices = []
                    matrix_date_identifier = []
                    classes = []
                    trade_classes = []

    if len(matrices) > 0:
        # Save remaining matrices
        np.save('../data/processed/matrices/{}/md_up_until{}'.format(name_folder, tot_matrices),
                np.array(meta_data_users).squeeze())
        np.save('../data/processed/matrices/{}/dt_up_until{}'.format(name_folder, tot_matrices),
                np.array(matrix_date_identifier).squeeze())
        np.save('../data/processed/matrices/{}/dp_up_until_{}'.format(name_folder, tot_matrices),
                np.array(matrices).squeeze())
        np.save('../data/processed/matrices/{}/c_up_until_{}'.
                format(name_folder, tot_matrices), np.array(classes).squeeze())
        np.save('../data/processed/matrices/{}/md_btc_up_until{}'.format(name_folder, tot_matrices),
                np.array(meta_data_btc).squeeze())
        np.save('../data/processed/matrices/{}/tc_up_until_{}'.
                format(name_folder, tot_matrices), np.array(trade_classes).squeeze())

        for k in avg_matrices.keys():
            matrix_avg_k = avg_matrices[k][0].squeeze() / avg_matrices[k][1]
            save_matrix_plot_avg(matrix_avg_k, k, mode, train, binary)


if __name__ == '__main__':
    main()
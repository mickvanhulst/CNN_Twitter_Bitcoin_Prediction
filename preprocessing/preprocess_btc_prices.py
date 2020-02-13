import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('../data/btc_prices/Coinbase_BTCUSD_d.csv', skiprows=1)

    # Goal is to have all data starting at may 1sth, so we get a day before that
    # as we need to calculate the price difference. This price difference calculation
    # results in 2017-04-30 having NA, which we remove afterwards.
    df['perc_diff_close'] = df['Close'].diff(-1) / df['Close']
    df['perc_diff_high'] = df['High'].diff(-1) / df['High']
    df['perc_diff_low'] = df['Low'].diff(-1) / df['Low']
    df['perc_diff_open'] = df['Open'].diff(-1) / df['Open']
    df['perc_diff_vol_f'] = df['Volume From'].diff(-1) / df['Volume From']
    df['perc_diff_vol_t'] = df['Volume To'].diff(-1) / df['Volume To']

    # Taker fee (so when buying) is .4%, maker (selling) is 0%.
    df["price_diff"] = df['Close'].diff(-1)
    price_diff = df["price_diff"] - (df['Close'].shift(-1) * 0.004)
    df['trade_profitable'] = np.where(price_diff > 0, 1, 0)

    #todo: add perc differences for volume etc. This means more to a network than the actual number.

    # Make it a 3 class problem (-1 for neg, 0 for no trade, 1 for pos).
    #TODO: Added a min profit perc to do anything. Else difference is never zero. Staying absent might be important.
    df['trade_class'] = np.where((np.abs(price_diff) > 0) & (np.abs(price_diff) > (df['Close'] * 0.01)),
                                          np.where(price_diff > 0, 1, -1), 0)
    df = df.dropna()

    df.to_json('../data/processed/btc_prices.json', lines=True, orient='records')

if __name__ == '__main__':
    main()
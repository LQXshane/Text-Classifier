import pandas as pd
# from IPython import embed


def readFile(file):

    df_raw = pd.read_csv(file)

    idx = range(0, len(df_raw.columns),2)

    tweet_contents = df_raw.iloc[:, idx]

    del  df_raw

    tweet_contents = tweet_contents.drop(0)

    tweet_contents = tweet_contents.drop_duplicates()

    return tweet_contents


def myJoiner(df, labels, col, label):
    df_res = pd.DataFrame()

    for i in labels:
        # print (i)
        df_new = pd.DataFrame(df[i])
        df_new.columns = col

        df_res = df_new.append(df_res, ignore_index=True)

    df_res = df_res.dropna()
    df_res.index = range(len(df_res))
    df_res['label'] = label

    return df_res.dropna()

def relabel(file, positive, negative):

    print ("positive labels are: ", positive)
    print ("negative labels are: ", negative)

    tweets = readFile(file)
    data_0 = myJoiner(tweets, negative, ['contents'], '0')
    data_1 = myJoiner(tweets, positive, ['contents'], '1')

    df_res = pd.concat([data_0, data_1])
    df_res.index = range(len(df_res))
    df_res = df_res.sample(frac=1) #shuffle the docs
    df_res.contents = df_res.contents.str.replace('\r', ' ')

    return df_res


# data = relabel('../../Crwdworkers/Trump/Categories_Trump.csv', ['3,3,1',
#        '3,2,2', '2,5,0', '2,4,1', '2,3,2', '1,6,0', '1,5,1', '1,4,2',
#        '1,3,3', '0,7,0', '0,6,1', '0,5,2', '0,4,3'], ['7,0,0', '6,1,0', '5,2,0', '5,1,1', '4,3,0', '4,2,1'])
#
# print (data.head())
# print (len(data))
#
# data.to_csv('../categories/no_border/trump.csv', index=False)
#
# embed()
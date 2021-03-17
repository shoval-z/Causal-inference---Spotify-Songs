from sklearn.linear_model import LogisticRegression, LinearRegression
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.linear_model import Lasso, Ridge
import plotly.graph_objects as go


def artist_names_with_more_than_x_songs(x):
    """
    returns a list of tje artist in the dataset with more than x songs
    :param x: threshold for number of songs
    :return: a list of artist names.
    """
    spotify_song_info = pd.read_csv("song_info.csv")
    artist_count = spotify_song_info[['artist_name', 'song_name']]
    artist_count = artist_count.groupby(by='artist_name', as_index=False).count()
    artist_count = artist_count.rename(columns={'song_name': 'cnt'})
    artist_count = artist_count.sort_values(by='cnt', ascending=False)
    artist_count = artist_count[['cnt', 'artist_name']]
    artist_with_more_than_x = artist_count[artist_count['cnt'] > x]
    return artist_with_more_than_x['artist_name'].unique()


def propensity_score(X, T, df, t1, t2):
    """
    finds the PS score for each data point. If overlap does not hold, we trimmed the DataFrame so it will hold
    :param X: the features that are used to calculate the PS scores.
    :param T: the treatment column that is used to calculate the PS scores.
    :param df: the original df of the whole dataset
    :param t1: the treatment group
    :param t2: the control group
    :return: treamed_df - the remaining df (such that overlap hold),
            unwanted_all - the data that was trimmed so we can analyze it,
            wanted_pair - if the given pair contains enough data from the treatment group and control group
            model - the LR model so we can get the PS score for the IPW
    """
    print(t1, t2)
    wanted_pair = True
    propensity_score_dict = dict()
    unwanted_amount = dict()
    treamed_df = pd.DataFrame()
    model = LogisticRegression(C=1, class_weight='balanced')
    new_x = [p for p, t1, t2 in zip(X, T[t1], T[t2]) if t1 != 0 or t2 != 0]
    new_t = [t1 for t1, t2 in zip(T[t1], T[t2]) if t1 != 0 or t2 != 0]
    model.fit(new_x, new_t)
    propensity_score_dict[(t1, t2)] = model.predict_proba(new_x)[:, -1]
    new = np.c_[new_t, propensity_score_dict[(t1, t2)]]
    new_0 = [p for p, t in zip(new[:, 1], new[:, 0]) if t == 0]
    new_1 = [p for p, t in zip(new[:, 1], new[:, 0]) if t == 1]

    # This part finds the data we want to trim
    categorial = ['artist_name', 'playlist']
    df = df.join(pd.get_dummies(df[categorial]))
    new_df = df[(df[t1] != 0) | (df[t2] != 0)]
    new_df_0 = new_df[new_df[t1] == 0].reset_index()
    new_df_1 = new_df[new_df[t1] == 1].reset_index()
    new_0_np = np.array(new_0)
    outliers_idx_0 = np.where(new_0_np < min(new_1))[0]
    new_1_np = np.array(new_1)
    outliers_idx_1 = np.where(new_1_np > max(new_0))[0]
    unwanted_0 = new_df_0.index.isin(outliers_idx_0)
    unwanted_1 = new_df_1.index.isin(outliers_idx_1)
    wanted_df_0 = new_df_0[~unwanted_0]
    wanted_df_1 = new_df_1[~unwanted_1]
    unwanted_df_0 = new_df_0[unwanted_0]  # the rows from original df with extreme values
    unwanted_df_1 = new_df_1[unwanted_1]  # the rows from original df with extreme values
    unwanted_amount[(t1, t2)] = len(unwanted_df_0) + len(unwanted_df_1)

    unwanted_all = pd.concat((unwanted_df_0, unwanted_df_1))
    treamed_df = pd.concat([treamed_df, wanted_df_0])
    treamed_df = pd.concat([treamed_df, wanted_df_1])

    ratio = len(wanted_df_1) / len(wanted_df_0)
    print(ratio)
    if ratio < 1 / 3 or ratio > 3:
        wanted_pair = False
    treamed_df.drop_duplicates(inplace=True)
    treamed_df.reset_index(drop=True, inplace=True)
    treamed_df.drop('index', axis=1, inplace=True)
    return treamed_df, unwanted_all, wanted_pair, model


def load_data_after_trimming(df, t1):
    """
    scale the data after trimmindg
    :param df: the trimmed df
    :param t1: the treatment group
    :return: X- the scale features.
            T_key - the column that represent the treatment group
            Y- the column that represent the output- popularity
            df = the new df
    """
    df = df.drop(
        ['artist_name', 'playlist'], axis=1)
    scaler = MinMaxScaler()
    T_key_name = ['key_0', 'key_1', 'key_2', 'key_3',
                  'key_4', 'key_5', 'key_6', 'key_7', 'key_8', 'key_9', 'key_10',
                  'key_11']
    X = df.drop(T_key_name, axis=1)
    X = scaler.fit_transform(X.drop(['song_popularity'], axis=1))
    T_key = df[t1]
    Y = df['song_popularity']
    return X, T_key, Y, df


def load_data(CATE=False):
    """
    arrange the the dataset- drop the unwanted rows or columns and scale the data
    :param CATE: whether we want the look only on artist with more the x songs or not (here x=3)
    :return: X- the scale features.
            T_key - the column that represent the treatment group
            Y- the column that represent the output- popularity
            df = the new df
            original_df = the new df without the one-hot encoding for artist and playlist
    """
    spotify_song_data = pd.read_csv("song_data.csv")
    spotify_song_info = pd.read_csv("song_info.csv")
    df = spotify_song_info.join(spotify_song_data, lsuffix='', rsuffix='song_name')
    print(f"len before CATE {len(df['artist_name'].unique())}")
    if CATE:
        wanted_artist = artist_names_with_more_than_x_songs(x=3)
        df = df[df['artist_name'].isin(wanted_artist)]
    categorial = ['artist_name', 'playlist']
    original_df = df.copy()
    df = df.join(pd.get_dummies(df[categorial]))
    df = df.join(pd.get_dummies(df.key, prefix='key'))
    original_df = original_df.join(pd.get_dummies(df.key, prefix='key'))
    df = df.drop(
        ['song_name', 'song_namesong_name', 'album_names', 'artist_name', 'playlist', 'key',
         'danceability'], axis=1)
    original_df = original_df.drop(['song_name', 'song_namesong_name', 'album_names', 'key', 'danceability'], axis=1)
    scaler = MinMaxScaler()
    T_key_name = ['key_0', 'key_1', 'key_2', 'key_3',
                  'key_4', 'key_5', 'key_6', 'key_7', 'key_8', 'key_9', 'key_10',
                  'key_11']
    X = df.drop(T_key_name, axis=1)
    X = scaler.fit_transform(X.drop(['song_popularity'], axis=1))
    T_key = df[T_key_name]
    Y = df['song_popularity']
    return X, T_key, Y, df, original_df


def propensity_score_bootstrap(new_x, model):
    """
    use the lr model to get the PS score for the wanted samples
    :param new_x: the wanted samples
    :param model: th LR model
    :return: the PS scores
    """
    return model.predict_proba(new_x)[:, -1]


def s_learner(new_x, new_t, new_y, model_name='Lasso'):
    """
    find the S-learner estimator
    :param new_x: the samples
    :param new_t: the corresponding treatment
    :param new_y:  the corresponding result
    :param model_name: which model to use
    :return: the S-learner estimator
    """
    if model_name == 'Lasso':
        model = Lasso(alpha=0.2)
    if model_name == 'Ridge':
        model = Ridge(alpha=0.5)
    Xnew = np.c_[new_x, new_t]
    model.fit(Xnew, new_y)
    f_1 = model.predict(np.c_[new_x, np.ones(new_x.shape[0])])
    f_0 = model.predict(np.c_[new_x, np.zeros(new_x.shape[0])])
    return np.mean(f_1 - f_0)


def T_learner(new_x, new_y, new_df, t1, model_name='Lasso'):
    """
    find the T-learner estimator
    :param new_x: the samples
    :param new_y:  the corresponding result
    :param new_df:  the DF that matches the given pair
    :param t1:  the treatment column name
    :param model_name: which model to use
    :return: the T-learner estimator
    """
    if model_name == 'Ridge':
        model_1 = Ridge(alpha=0.5)
        model_0 = Ridge(alpha=0.5)

    if model_name == 'Lasso':
        model_1 = Lasso(alpha=0.5)
        model_0 = Lasso(alpha=0.5)

    X_1 = new_x[new_df[t1] == 1]
    X_0 = new_x[new_df[t1] == 0]
    y_1 = new_y[new_df[t1] == 1]
    y_0 = new_y[new_df[t1] == 0]

    model_1.fit(X_1, y_1)
    f_1 = model_1.predict(new_x)
    model_0.fit(X_0, y_0)
    f_0 = model_0.predict(new_x)
    return np.mean((f_1 - f_0))


def IPW_bootstrap(p_s, new_t, new_y):
    """
    find the IPW estimator
    :param p_s: the propensity scores for each wanted sample
    :param new_t: the treatment column
    :param new_y: the output column
    :return: the IPW estimator
    """
    tmp1 = np.sum(new_t * new_y / p_s) / len(new_t)
    tmp2 = np.sum((1 - new_t) * new_y / (1 - p_s)) / len(new_t)
    return (tmp1 - tmp2)


def bootstrap(new_x, new_t, new_y, new_df, t1, model, name='s_learner'):
    """
    return lists of the bootstrap estimators for the ATE/CATE. we will create the CI from that list
    :param new_x: the full samples
    :param new_t: the treatment column
    :param new_y: the output column
    :param new_df: the new filtered df that in corresponding to the samples
    :param t1: the treatment column name
    :param model: the lr model to find the ps scores
    :param name: the estimator we currently want to calculate
    :return: lists of the estimators
    """
    tmp_1, tmp_2, tmp_3 = list(), list(), list()
    n = [*range(len(new_df))]
    for b in range(50):
        n_b = np.random.choice(n, len(n))
        df_b = new_df.loc[n_b]
        df_b = df_b.loc[:, (df_b != 0).any(axis=0)]
        t_b = new_t.take(n_b)
        y_b = new_y.take(n_b)
        x_b = new_x[n_b]
        if name == 's_learner':
            tmp_2.append(s_learner(x_b, t_b, y_b, model_name='Lasso'))
            tmp_3.append(s_learner(x_b, t_b, y_b, model_name='Ridge'))
        if name == 't_learner':
            tmp_2.append(T_learner(x_b, y_b, df_b, t1, model_name='Lasso'))
            tmp_3.append(T_learner(x_b, y_b, df_b, t1, model_name='Ridge'))
        if name == 'IPW':
            propensity_score_b = propensity_score_bootstrap(x_b, model)
            tmp_1.append(IPW_bootstrap(propensity_score_b, t_b, y_b))
    return tmp_1, tmp_2, tmp_3


def CI_val(ATE_dict, name='IPW'):
    """
    create the CI for the wanted method
    :param ATE_dict: A dictionary that contain for each treatment-control pair, a list of the estimators.
        a.k {(t1,t2):list(estimators)}
    :param name: the current method
    :return: the mean value, CIs and list of pairs
    """
    lasso_mean, ridge_mean, lasso_ci, ridge_ci, ipw_mean, ipw_ci = list(), list(), list(), list(), list(), list()
    list_keys = list()
    if name is not 'IPW':
        for key, ate_dict in ATE_dict.items():
            for key2, item in ate_dict.items():
                mean_val = np.mean(item)
                std_val = 2 * np.std(item) / np.sqrt(50)
                if key == 'lasso':
                    lasso_mean.append(mean_val)
                    lasso_ci.append([mean_val - std_val, mean_val + std_val])
                    list_keys.append(key2)
                if key == 'ridge':
                    ridge_mean.append(mean_val)
                    ridge_ci.append([mean_val - std_val, mean_val + std_val])
        return lasso_mean, ridge_mean, lasso_ci, ridge_ci, list_keys
    else:
        for key, item in ATE_dict.items():
            mean_val = np.mean(item)
            std_val = 2 * np.std(item) / np.sqrt(50)
            ipw_mean.append(mean_val)
            ipw_ci.append([mean_val - std_val, mean_val + std_val])
            list_keys.append(key)
        return ipw_mean, ipw_ci, list_keys


def main():
    """
    run the entire code- find the ATE/CATE CI for each pair
    :return: create '.npy' files that contain the CI for each pair
    """
    CATE = True
    X, T_key, Y, df, original_df = load_data(CATE=CATE)
    print(f" the len after CATE is {len(original_df['artist_name'].unique())}")
    CATE_key_list = [('key_0', 'key_2'), ('key_0', 'key_3'), ('key_0', 'key_5'), ('key_0', 'key_6'),
                     ('key_0', 'key_7'), ('key_0', 'key_9'), ('key_0', 'key_10'), ('key_0', 'key_11'),
                     ('key_1', 'key_6'),
                     ('key_2', 'key_3'), ('key_2', 'key_5'), ('key_2', 'key_6'), ('key_2', 'key_7'), ('key_2', 'key_8'),
                     ('key_2', 'key_9'), ('key_3', 'key_4'), ('key_3', 'key_5'), ('key_3', 'key_6'), ('key_3', 'key_8'),
                     ('key_4', 'key_5'), ('key_4', 'key_7'), ('key_5', 'key_6'), ('key_5', 'key_7'),
                     ('key_5', 'key_9'), ('key_5', 'key_11'), ('key_6', 'key_7'), ('key_6', 'key_9'),
                     ('key_6', 'key_10'),
                     ('key_6', 'key_11'), ('key_7', 'key_8'), ('key_7', 'key_9'), ('key_7', 'key_10'),
                     ('key_7', 'key_11'), ('key_8', 'key_10'), ('key_9', 'key_10'), ('key_10', 'key_11')]
    ATE_dict_s = {'lasso': dict(), 'ridge': dict()}
    ATE_dict_t = {'lasso': dict(), 'ridge': dict()}
    ATE_ipw = dict()
    df_dict = dict()
    T_key_name = ['key_0', 'key_1', 'key_2', 'key_3',
                  'key_4', 'key_5', 'key_6', 'key_7', 'key_8', 'key_9', 'key_10',
                  'key_11', 'artist_name', 'playlist']
    tmp_original = original_df.drop(T_key_name, axis=1)
    df_dict['names'] = [i for i in tmp_original.columns]
    df_dict['all data'] = tmp_original.mean().values
    for i, t1 in enumerate(T_key.columns):
        for j, t2 in enumerate(T_key.columns):
            if t1 == t2:
                continue
            if (t1, t2) in ATE_dict_s['ridge'].keys() or (t2, t1) in ATE_dict_s['ridge'].keys():
                continue
            if CATE:
                if (t1, t2) not in CATE_key_list:
                    continue
            new_df, unwanted_df, wanted_pair, lr_model = propensity_score(X, T_key, original_df, t1, t2)
            if not wanted_pair:
                continue
            new_x, new_t, new_y, new_df = load_data_after_trimming(new_df, t1)
            new_df.reset_index(drop=True, inplace=True)
            new_y.reset_index(drop=True, inplace=True)
            new_t.reset_index(drop=True, inplace=True)
            _, ATE_dict_s['lasso'][(t1, t2)], ATE_dict_s['ridge'][(t1, t2)] = bootstrap(new_x, new_t, new_y, new_df, t1,
                                                                                        model=lr_model,
                                                                                        name='s_learner')
            _, ATE_dict_t['lasso'][(t1, t2)], ATE_dict_t['ridge'][(t1, t2)] = bootstrap(new_x, new_t, new_y, new_df, t1,
                                                                                        model=lr_model,
                                                                                        name='t_learner')
            ATE_ipw[(t1, t2)], _, _ = bootstrap(new_x, new_t, new_y, new_df, t1, model=lr_model, name='IPW')

    s_lasso_mean, s_ridge_mean, s_lasso_ci, s_ridge_ci, s_list_keys = CI_val(ATE_dict_s, name='s_learner')
    t_lasso_mean, t_ridge_mean, t_lasso_ci, t_ridge_ci, t_list_keys = CI_val(ATE_dict_t, name='t_learner')
    ipw_mean, ipw_ci, list_keys_ipw = CI_val(ATE_ipw, name='IPW')

    np.save(f'CI_S_learner_LASSO_key_{CATE}.npy', s_lasso_ci)
    np.save(f'CI_S_learner_RIDGE_key_{CATE}.npy', s_ridge_ci)
    np.save(f'CI_T_learner_LASSO_key_{CATE}.npy', t_lasso_ci)
    np.save(f'CI_T_learner_RIDGE_key_{CATE}.npy', t_ridge_ci)
    np.save(f'CI_IPW_key_{CATE}.npy', ipw_ci)


if __name__ == '__main__':
    main()

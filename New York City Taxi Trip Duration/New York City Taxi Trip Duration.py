import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, learning_curve, validation_curve, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_log_error
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv('E:/files/Kaggle/New York City Taxi Trip Duration/train.csv')
test = pd.read_csv('E:/files/Kaggle/New York City Taxi Trip Duration/test.csv')
# 打车时间段，打车距离，打车路段交通情况，天气
# print(train.head())
# print(train.info())
# print(test.info())
train = train.drop(train[(train['trip_duration'] > 85000)].index)
train = train.drop(train[(train['trip_duration'] < 120)].index)
train['log_trip_duration'] = np.log1p(train['trip_duration'])

# 去除经纬度异常值
train = train.drop(train[(train['dropoff_latitude'] < 40.6)].index)
train = train.drop(train[(train['dropoff_latitude'] > 40.9)].index)
train = train.drop(train[(train['dropoff_longitude'] < -74.05)].index)
train = train.drop(train[(train['dropoff_longitude'] > -73.7)].index)
train = train.drop(train[(train['pickup_latitude'] < 40.6)].index)
train = train.drop(train[(train['pickup_latitude'] > 40.9)].index)
train = train.drop(train[(train['pickup_longitude'] < -74.05)].index)
train = train.drop(train[(train['pickup_longitude'] > -73.7)].index)
city_long_border = (-74.05, -73.7)
city_lat_border = (40.6, 40.9)

# 上车地点聚类，各路段拥堵情况不同，并加上虚拟变量
coord_pickup = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                          test[['pickup_latitude', 'pickup_longitude']].values))
coord_dropoff = np.vstack((train[['dropoff_latitude', 'dropoff_longitude']].values,
                           test[['dropoff_latitude', 'dropoff_longitude']].values))
coords = np.hstack((coord_pickup,coord_dropoff))# 4 dimensional data
sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=20, batch_size=10000).fit(coords[sample_ind])
for df in (train,test):
    df.loc[:, 'pickup_dropoff_loc'] = kmeans.predict(df[['pickup_latitude', 'pickup_longitude',
                                                         'dropoff_latitude','dropoff_longitude']])

kmean_train= pd.get_dummies(train['pickup_dropoff_loc'],prefix='loc', prefix_sep='_')
kmean_test= pd.get_dummies(test['pickup_dropoff_loc'],prefix='loc', prefix_sep='_')
train  = pd.concat([train,kmean_train],axis=1)
test= pd.concat([test,kmean_test],axis=1)


# 圆上两点距离
def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


# 两点间曼哈顿距离
def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b


def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))


train.loc[:, 'distance_haversine'] = haversine_array(train['pickup_latitude'].values, train['pickup_longitude'].values,
                                                     train['dropoff_latitude'].values,
                                                     train['dropoff_longitude'].values)
test.loc[:, 'distance_haversine'] = haversine_array(test['pickup_latitude'].values, test['pickup_longitude'].values,
                                                    test['dropoff_latitude'].values, test['dropoff_longitude'].values)

train.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(train['pickup_latitude'].values,
                                                                    train['pickup_longitude'].values,
                                                                    train['dropoff_latitude'].values,
                                                                    train['dropoff_longitude'].values)
test.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(test['pickup_latitude'].values,
                                                                   test['pickup_longitude'].values,
                                                                   test['dropoff_latitude'].values,
                                                                   test['dropoff_longitude'].values)

train.loc[:, 'direction'] = bearing_array(train['pickup_latitude'].values, train['pickup_longitude'].values,
                                          train['dropoff_latitude'].values, train['dropoff_longitude'].values)
test.loc[:, 'direction'] = bearing_array(test['pickup_latitude'].values, test['pickup_longitude'].values,
                                         test['dropoff_latitude'].values, test['dropoff_longitude'].values)

# 寻找时间的关系，几点，星期几，几月
train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)
train.loc[:, 'pickup_date'] = train['pickup_datetime'].dt.date
test.loc[:, 'pickup_date'] = test['pickup_datetime'].dt.date
train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)

train['Month'] = train['pickup_datetime'].dt.month
test['Month'] = test['pickup_datetime'].dt.month

train['Hour'] = train['pickup_datetime'].dt.hour
test['Hour'] = test['pickup_datetime'].dt.hour

train['dayofweek'] = train['pickup_datetime'].dt.dayofweek
test['dayofweek'] = test['pickup_datetime'].dt.dayofweek

# 极端天气
weather_event = ['20160110', '20160113', '20160117', '20160123',
                 '20160205', '20160208', '20160215', '20160216',
                 '20160224', '20160225', '20160314', '20160315',
                 '20160328', '20160329', '20160403', '20160404',
                 '20160530', '20160628']
weather_event = pd.Series(pd.to_datetime(weather_event, format = '%Y%m%d')).dt.date
train['extreme_weather'] = train.pickup_date.isin(weather_event).map({True: 1, False: 0})
test['extreme_weather'] = test.pickup_date.isin(weather_event).map({True: 1, False: 0})

# 地点与速度聚类，时间段与速度聚类
train.loc[:, 'avg_speed_h'] = 1000 * train['distance_haversine'] / train['trip_duration']
train.loc[:, 'avg_speed_m'] = 1000 * train['distance_dummy_manhattan'] / train['trip_duration']

for gby_col in ['Hour', 'hourofweek',
                'dayofweek', 'pickup_dropoff_loc']:
    gby = train.groupby(gby_col).mean()[['avg_speed_h', 'avg_speed_m']]
    gby.columns = ['%s_gby_%s' % (col, gby_col) for col in gby.columns]
    train = pd.merge(train, gby, how='left', left_on=gby_col, right_index=True)
    test = pd.merge(test, gby, how='left', left_on=gby_col, right_index=True)

fr1 = pd.read_csv(
    'E:/files/Kaggle/New York City Taxi Trip Duration/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv',
    usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
fr2 = pd.read_csv(
    'E:/files/Kaggle/New York City Taxi Trip Duration/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv',
    usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
test_street_info = pd.read_csv(
    'E:/files/Kaggle/New York City Taxi Trip Duration/new-york-city-taxi-with-osrm/fastest_routes_test.csv',
    usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
train_street_info = pd.concat((fr1, fr2))
train = train.merge(train_street_info, how='left', on='id')
test = test.merge(test_street_info, how='left', on='id')


# 不知道什么意思的特征
for gby_cols in [['Hour', 'pickup_dropoff_loc']]:
    coord_speed = train.groupby(gby_cols).mean()[['avg_speed_h']].reset_index()
    coord_count = train.groupby(gby_cols).count()[['id']].reset_index()
    coord_stats = pd.merge(coord_speed, coord_count, on=gby_cols)
    coord_stats = coord_stats[coord_stats['id'] > 100]
    coord_stats.columns = gby_cols + ['avg_speed_h_%s' % '_'.join(gby_cols), 'cnt_%s' %  '_'.join(gby_cols)]
    train = pd.merge(train, coord_stats, how='left', on=gby_cols)
    test = pd.merge(test, coord_stats, how='left', on=gby_cols)

group_freq = '60min'
df_all = pd.concat((train, test))[['id', 'pickup_datetime', 'pickup_dropoff_loc']]
train.loc[:, 'pickup_datetime_group'] = train['pickup_datetime'].dt.round(group_freq)
test.loc[:, 'pickup_datetime_group'] = test['pickup_datetime'].dt.round(group_freq)

dropoff_counts = df_all \
    .set_index('pickup_datetime') \
    .groupby([pd.TimeGrouper(group_freq), 'pickup_dropoff_loc']) \
    .agg({'id': 'count'}) \
    .reset_index().set_index('pickup_datetime') \
    .groupby('pickup_dropoff_loc').rolling('240min').mean() \
    .drop('pickup_dropoff_loc', axis=1) \
    .reset_index().set_index('pickup_datetime').shift(freq='-120min').reset_index() \
    .rename(columns={'pickup_datetime': 'pickup_datetime_group', 'id': 'dropoff_cluster_count'})

train['dropoff_cluster_count'] = train[['pickup_datetime_group', 'pickup_dropoff_loc']].merge(dropoff_counts, on=['pickup_datetime_group', 'pickup_dropoff_loc'], how='left')['dropoff_cluster_count'].fillna(0)
test['dropoff_cluster_count'] = test[['pickup_datetime_group', 'pickup_dropoff_loc']].merge(dropoff_counts, on=['pickup_datetime_group', 'pickup_dropoff_loc'], how='left')['dropoff_cluster_count'].fillna(0)


coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                    train[['dropoff_latitude', 'dropoff_longitude']].values,
                    test[['pickup_latitude', 'pickup_longitude']].values,
                    test[['dropoff_latitude', 'dropoff_longitude']].values))

pca = PCA()
pca.fit(coords)
train['pickup_pca0'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 0]
train['pickup_pca1'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 1]
train['dropoff_pca0'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
train['dropoff_pca1'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
test['pickup_pca0'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 0]
test['pickup_pca1'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 1]
test['dropoff_pca0'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
test['dropoff_pca1'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
train.loc[:, 'pca_manhattan'] = np.abs(train['dropoff_pca1'] - train['pickup_pca1']) + np.abs(train['dropoff_pca0'] - train['pickup_pca0'])
test.loc[:, 'pca_manhattan'] = np.abs(test['dropoff_pca1'] - test['pickup_pca1']) + np.abs(test['dropoff_pca0'] - test['pickup_pca0'])

# _ = input('Press [Enter] to continue.')

# ----------------------------------------------- 训练模型并评价 ---------------------------------------------------------

def regressor_moduel(estimator, X_train, y_train, X_train1, X_cv, y_train1, y_cv):
    print('cross_val_score: ', end='')
    # print(cross_val_score(estimator, X_train, y_train, cv=10).mean())

    estimator.fit(X_train1, y_train1)
    print('cv score: ', end='')
    print(estimator.score(X_cv, y_cv))
    print('train score: ', end='')
    print(estimator.score(X_train1, y_train1))
    y_predict = estimator.predict(X_cv)
    y_cv = np.expm1(y_cv)
    y_predict = np.expm1(y_predict)
    print(np.sqrt(mean_squared_log_error(y_cv, y_predict)))

    
# 自定义xgboost eval_metric，RMSLE
# 在使用xgboost之前，先对y进行log1p转换，然后metric选择rmse就好了。预测出的结果也是log后的，所以真实结果还需要进行exp转换一下。
def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'RMSLE', np.sqrt(mean_squared_log_error(preds, labels))

y_train = train['log_trip_duration']
# selected_features = ['distance_haversine', 'distance_dummy_manhattan', 'direction',
#                      'extreme_weather', 'vendor_id','total_travel_time','total_distance','number_of_steps',
#                      'avg_speed_m_gby_Hour', 'avg_speed_m_gby_hourofweek',
#                      'avg_speed_m_gby_dayofweek', 'avg_speed_m_gby_pickup_dropoff_loc']
do_not_use_for_training = ['id', 'log_trip_duration', 'pickup_datetime', 'dropoff_datetime',
                           'pickup_latitude', 'pickup_longitude','dropoff_latitude', 'dropoff_longitude',
                           'trip_duration', 'store_and_fwd_flag','passenger_count',
                           'pickup_cluster','dropoff_cluster',
                           'pickup_date', 'avg_speed_h', 'avg_speed_m',
                           'pickup_lat_bin', 'pickup_long_bin',
                           'center_lat_bin', 'center_long_bin',
                           'pickup_dt_bin', 'pickup_datetime_group']
selected_features = [f for f in train.columns if f not in do_not_use_for_training]
X_train = train[selected_features]
X_test = test[selected_features]

X_train=X_train.values # dataframe to numpy.array
y_train=y_train.values
X_test=X_test.values

print(len(selected_features))
print(X_train.shape,X_test.shape)
print(X_train.columns.values.tolist())

_ = input('Press [Enter] to continue.')

kf = KFold(n_splits=10)
kf.get_n_splits(X_train)

for train_index, test_index in kf.split(X_train):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_kf_train, X_kf_test = X_train[train_index], X_train[test_index]
    y_kf_train, y_kf_test = y_train[train_index], y_train[test_index]

X_train1, X_cv, y_train1, y_cv = train_test_split(X_train, y_train, test_size=0.2, random_state=33)
    
dtrain = xgb.DMatrix(X_train1, label=y_train1)
dvalid = xgb.DMatrix(X_cv, label=y_cv)
dtest = xgb.DMatrix(X_test)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

# Try different parameters! My favorite is random search :)
xgb_pars = {'min_child_weight': 10, 'eta': 0.05, 'colsample_bytree': 0.3, 'max_depth': 12,
            'subsample': 0.8, 'lambda': 0.2, 'nthread': -1, 'booster': 'gbtree', 'silent': 1,
            'gamma': 0.05, 'objective': 'reg:linear', 'eval_metric': 'rmse'}

xgb_model = xgb.train(xgb_pars, dtrain, 200, watchlist, early_stopping_rounds=250,
                  maximize=False, verbose_eval=15)


# ----------------------------------------------- 输出csv结果 -----------------------------------------------------------

y_predict=xgb_model.predict(dtest)

y_predict = np.expm1(y_predict)

xgb_submission = pd.DataFrame({'id': test['id'], 'trip_duration': y_predict})
print(xgb_submission.shape)
xgb_submission.to_csv('E:/files/Kaggle/New York City Taxi Trip Duration/xgb_submission.csv', index=False)

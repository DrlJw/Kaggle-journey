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
# 上车地点经纬度可视化
# fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
# ax[0].scatter(train['pickup_longitude'].values[:500000], train['pickup_latitude'].values[:500000],
#               color='blue', s=1, label='train',marker='o',alpha=0.2)
# ax[1].scatter(test['pickup_longitude'].values[:500000], test['pickup_latitude'].values[:500000],
#               color='green', s=1, label='test',marker='o',alpha=0.2)
# fig.suptitle('Train and test area complete overlap.')
# ax[0].legend(loc=0)
# ax[0].set_ylabel('latitude')
# ax[0].set_xlabel('longitude')
# ax[1].set_xlabel('longitude')
# ax[1].legend(loc=0)
# plt.ylim(city_lat_border)
# plt.xlim(city_long_border)
# plt.show()

coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                    train[['dropoff_latitude', 'dropoff_longitude']].values))

sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])

# 上车地点聚类，各路段拥堵情况不同
train.loc[:, 'pickup_cluster'] = kmeans.predict(train[['pickup_latitude', 'pickup_longitude']])
train.loc[:, 'dropoff_cluster'] = kmeans.predict(train[['dropoff_latitude', 'dropoff_longitude']])
test.loc[:, 'pickup_cluster'] = kmeans.predict(test[['pickup_latitude', 'pickup_longitude']])
test.loc[:, 'dropoff_cluster'] = kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']])

# train.loc[:, 'center_latitude'] = (train['pickup_latitude'].values + train['dropoff_latitude'].values) / 2
# train.loc[:, 'center_longitude'] = (train['pickup_longitude'].values + train['dropoff_longitude'].values) / 2
# test.loc[:, 'center_latitude'] = (test['pickup_latitude'].values + test['dropoff_latitude'].values) / 2
# test.loc[:, 'center_longitude'] = (test['pickup_longitude'].values + test['dropoff_longitude'].values) / 2

# fig, ax = plt.subplots(ncols=1, nrows=1)
# ax.scatter(train.pickup_longitude.values[:500000], train.pickup_latitude.values[:500000], s=10, lw=0,
#            c=train.pickup_cluster[:500000].values, cmap='autumn', alpha=0.2)
# ax.set_xlim(city_long_border)
# ax.set_ylim(city_lat_border)
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')
# plt.show()

def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


# 圆上两点距离
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

# train.loc[:, 'avg_speed_h'] = 1000 * train['distance_haversine'] / train['trip_duration']
# train.loc[:, 'avg_speed_m'] = 1000 * train['distance_dummy_manhattan'] / train['trip_duration']
# fig, ax = plt.subplots(ncols=3, sharey=True)
# ax[0].plot(train.groupby('Hour').mean()['avg_speed_m'], 'bo-', lw=2, alpha=0.7)
# ax[1].plot(train.groupby('dayofweek').mean()['avg_speed_m'], 'go-', lw=2, alpha=0.7)
# ax[2].plot(train.groupby('Month').mean()['avg_speed_m'], 'ro-', lw=2, alpha=0.7)
# ax[0].set_xlabel('Hour of Day')
# ax[1].set_xlabel('Day of Week')
# ax[2].set_xlabel('Month of Year')
# ax[0].set_ylabel('Average Speed')
# fig.suptitle('Average Traffic Speed by Date-part')
# plt.show()

# print(train['pickup_date'].head())# 2016-03-14
# print(train['pickup_datetime'].head())# 2016-03-14 17:24:55
# print(train.groupby('passenger_count').size()) # 乘客数量的数量分布
# print(test.groupby('passenger_count').size())

# YN = train[["store_and_fwd_flag", "trip_duration"]].groupby(['store_and_fwd_flag'],as_index=False).mean()
# print(YN)   # N958 Y1080

# passenegerNUM = train[["passenger_count", "trip_duration"]].groupby(['passenger_count'],as_index=False).mean()
# print(passenegerNUM)
# print(train[(train['passenger_count']==0)]['trip_duration'])

# plt.hist(train['log_trip_duration'].values, bins=100)
# plt.xlabel('log(trip_duration)')
# plt.ylabel('number of train records')
# plt.show()

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

# _ = input('Press [Enter] to continue.')

# ----------------------------------------------- 训练模型并评价 ---------------------------------------------------------

y_train = train['log_trip_duration']
selected_features = ['distance_haversine', 'direction', 'dayofweek', 'Hour', 'Month', 'pickup_cluster',
                     'total_distance', 'total_travel_time', 'number_of_steps','extreme_weather']
X_train = train[selected_features]
X_test = test[selected_features]


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


X_train=X_train.values # dataframe to numpy.array
y_train=y_train.values
X_test=X_test.values

kf = KFold(n_splits=10)
kf.get_n_splits(X_train)

for train_index, test_index in kf.split(X_train):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_kf_train, X_kf_test = X_train[train_index], X_train[test_index]
    y_kf_train, y_kf_test = y_train[train_index], y_train[test_index]

dtrain = xgb.DMatrix(X_kf_train, label=y_kf_train)
dvalid = xgb.DMatrix(X_kf_test, label=y_kf_test)
dtest = xgb.DMatrix(X_test)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

# Try different parameters! My favorite is random search :)
xgb_pars = {'min_child_weight': 10, 'eta': 0.04, 'colsample_bytree': 0.8, 'max_depth': 15,
            'subsample': 0.75, 'lambda': 2, 'nthread': -1, 'booster' : 'gbtree', 'silent': 1, 'gamma' : 0,
            'objective': 'reg:linear', 'eval_metric': 'rmse'}

xgb_model = xgb.train(xgb_pars, dtrain, 200, watchlist, early_stopping_rounds=250,
                  maximize=False, verbose_eval=15)


# ----------------------------------------------- 输出csv结果 -----------------------------------------------------------

y_predict=xgb_model.predict(dtest)

y_predict = np.expm1(y_predict)

xgb_submission = pd.DataFrame({'id': test['id'], 'trip_duration': y_predict})
print(xgb_submission.shape)
xgb_submission.to_csv('E:/files/Kaggle/New York City Taxi Trip Duration/xgb_submission.csv', index=False)

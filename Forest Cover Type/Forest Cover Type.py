import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve, validation_curve, GridSearchCV
from sklearn.model_selection import cross_val_score,KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn import svm
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv('E:/files/Kaggle/Forest Cover Type/train.csv')
test = pd.read_csv('E:/files/Kaggle/Forest Cover Type/test.csv')


def data_plus(train):
    # 增加噪声样本，导致交叉验证集失效，但对测试集有提升
    df = train.copy()
    df['Id'] = df['Id'] + 15120
    df['Horizontal_Distance_To_Hydrology'] = df['Horizontal_Distance_To_Hydrology'] + np.random.randint(-10, 10)
    df['Elevation'] = df['Elevation'] + np.random.randint(-50, 50)
    df['Horizontal_Distance_To_Roadways'] = df['Horizontal_Distance_To_Roadways'] + np.random.randint(-10, 10)
    df1 = train.copy()
    df1['Id'] = df1['Id'] + 15120
    df1['Horizontal_Distance_To_Hydrology'] = df1['Horizontal_Distance_To_Hydrology'] + np.random.randint(-10, 10)
    df1['Elevation'] = df1['Elevation'] + np.random.randint(-50, 50)
    df1['Horizontal_Distance_To_Roadways'] = df1['Horizontal_Distance_To_Roadways'] + np.random.randint(-10, 10)
    df2 = train.copy()
    df2['Id'] = df2['Id'] + 15120
    df2['Horizontal_Distance_To_Hydrology'] = df2['Horizontal_Distance_To_Hydrology'] + np.random.randint(-10, 10)
    df2['Elevation'] = df2['Elevation'] + np.random.randint(-50, 50)
    df2['Horizontal_Distance_To_Roadways'] = df2['Horizontal_Distance_To_Roadways'] + np.random.randint(-10, 10)
    train = train.append(df, ignore_index=True)
    train = train.append(df1, ignore_index=True)
    train = train.append(df2, ignore_index=True)
    del df
    del df1
    del df2
    return train


def features(df):
    df['Soil_Type12_32'] = df['Soil_Type32'] + df['Soil_Type12']
    df['Soil_Type23_22_32_33'] = df['Soil_Type23'] + df['Soil_Type22'] + df['Soil_Type32'] + df['Soil_Type33']
    df['HF1'] = df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Fire_Points']
    df['HF2'] = abs(df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Fire_Points'])
    df['HR1'] = abs(df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Roadways'])
    df['HR2'] = abs(df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Roadways'])
    df['FR1'] = abs(df['Horizontal_Distance_To_Fire_Points'] + df['Horizontal_Distance_To_Roadways'])
    df['FR2'] = abs(df['Horizontal_Distance_To_Fire_Points'] - df['Horizontal_Distance_To_Roadways'])

    df['Elevation_Vertical'] = df['Elevation']+df['Vertical_Distance_To_Hydrology']
    df['Neg_Elevation_Vertical'] = df['Elevation']-df['Vertical_Distance_To_Hydrology']

    df['slope_hyd'] = (df['Horizontal_Distance_To_Hydrology'] ** 2 + df['Vertical_Distance_To_Hydrology'] ** 2) ** 0.5
    df.slope_hyd = df.slope_hyd.map(lambda x: 0 if np.isinf(x) else x)  # remove infinite value if any

    # Mean distance to Amenities
    df['Mean_Amenities'] = (df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Hydrology + df.Horizontal_Distance_To_Roadways) / 3
    # Mean Distance to Fire and Water
    df['Mean_Fire_Hyd1']=(df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Hydrology) / 2
    df['Mean_Fire_Hyd2']=(df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Roadways) / 2

    df['Hillshade_mean'] = (df['Hillshade_9am'] + df['Hillshade_Noon'] + df['Hillshade_3pm']) / 3
    df['Elevation'] = np.log1p(df['Elevation'])
    df['Shadiness_morn_noon'] = df.Hillshade_9am/(df.Hillshade_Noon+1)
    df['Shadiness_noon_3pm'] = df.Hillshade_Noon/(df.Hillshade_3pm+1)
    df['Shadiness_morn_3'] = df.Hillshade_9am/(df.Hillshade_3pm+1)
    df['Shadiness_morn_avg'] = (df.Hillshade_9am+df.Hillshade_Noon)/2
    df['Shadiness_afternoon'] = (df.Hillshade_Noon+df.Hillshade_3pm)/2
    df["Hillshade-9_Noon_diff"] = df["Hillshade_9am"] - df["Hillshade_Noon"]
    df["Hillshade-noon_3pm_diff"] = df["Hillshade_Noon"] - df["Hillshade_3pm"]
    df["Hillshade-9am_3pm_diff"] = df["Hillshade_9am"] - df["Hillshade_3pm"]

    # # Mountain Trees
    df['Slope*Elevation'] = df['Slope'] * df['Elevation']

    df['Neg_HorizontalHydrology_HorizontalFire'] = (
                df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Fire_Points'])
    df['Neg_HorizontalHydrology_HorizontalRoadways'] = (
                df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Roadways'])
    df['Neg_HorizontalFire_Points_HorizontalRoadways'] = (
                df['Horizontal_Distance_To_Fire_Points'] - df['Horizontal_Distance_To_Roadways'])

    df['MeanNeg_Mean_HorizontalHydrology_HorizontalFire'] = (df['Horizontal_Distance_To_Hydrology'] - df[
        'Horizontal_Distance_To_Fire_Points']) / 2
    df['MeanNeg_HorizontalHydrology_HorizontalRoadways'] = (df['Horizontal_Distance_To_Hydrology'] - df[
        'Horizontal_Distance_To_Roadways']) / 2
    df['MeanNeg_HorizontalFire_Points_HorizontalRoadways'] = (df['Horizontal_Distance_To_Fire_Points'] - df[
        'Horizontal_Distance_To_Roadways']) / 2

    df["Vertical_Distance_To_Hydrology"] = abs(df['Vertical_Distance_To_Hydrology'])

    df['Neg_Elev_Hyd'] = df.Elevation - df.Horizontal_Distance_To_Hydrology * 0.2

    return df


# train=data_plus(train)
train=features(train)
test=features(test)

print(train.shape)


def classifier_moduel(estimator, X_train, y_train, X_train1, X_cv, y_train1, y_cv):
    # print('cross_val_score: ', end='')
    # print(cross_val_score(estimator, X_train, y_train, cv=5).mean())
    estimator.fit(X_train1, y_train1)
    print('cv score: ', end='')
    print(estimator.score(X_cv, y_cv))
    print('train score: ', end='')
    print(estimator.score(X_train1, y_train1))
    estimator_y_predict = estimator.predict(X_cv)
    print(classification_report(y_cv, estimator_y_predict))


feature = [col for col in train.columns if col not in ['Cover_Type', 'Id', 'Soil_Type7', 'Soil_Type8',
                                                       'Soil_Type9', 'Soil_Type15', 'Soil_Type21', 'Soil_Type9',
                                                       'Soil_Type25', 'Soil_Type27', 'Soil_Type28', 'Soil_Type34',
                                                       'Soil_Type36']]
X_train = train[feature]
X_test = test[feature]
y_train = train['Cover_Type']

X_train = X_train.values
y_train = y_train.values
X_test = X_test.values

# X_train1, X_cv, y_train1, y_cv = train_test_split(X_train, y_train, test_size=0.25, random_state=33)

rfc = RandomForestClassifier(n_estimators=150,bootstrap=True)

print(cross_val_score(lgb, X_train, y_train, cv=5).mean())


# classifier_moduel(rfc, X_train, y_train, X_train1, X_cv, y_train1, y_cv)

#
# rfc.fit(X_train, y_train)
# sub = pd.DataFrame({"Id": test['Id'],"Cover_Type": rfc.predict(X_test)})
# sub.to_csv("E:/files/Kaggle/Forest Cover Type/rfc_sub.csv", index=False)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve, validation_curve, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
import seaborn as sns
import warnings

warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

train = pd.read_csv('E:/files/Kaggle/titanic/train.csv')
test = pd.read_csv('E:/files/Kaggle/titanic/test.csv')

# print(X_train.head())
# print(X_train.info())
# print(X_train.describe())

# Explore Age distibution
g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color="Red", shade = True)
g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], ax =g, color="Blue", shade= True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])
plt.show()

train['Embarked'].fillna('S', inplace=True)
test['Embarked'].fillna('S', inplace=True)

train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)

test['Fare'].fillna(test['Fare'].mean(), inplace=True)

train['Family'] = train['SibSp'] + train['Parch'] + 1
test['Family'] = test['SibSp'] + test['Parch'] + 1

train['isAlone'] = 1
train.loc[(train['Family'] == 1), 'isAlone'] = 0
test['isAlone'] = 1
test.loc[(test['Family'] == 1), 'isAlone'] = 0

train["Fare"] = train["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
test["Fare"] = test["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

def get_person(passenger):
    age, sex = passenger
    return 'child' if age < 10 else sex


train['Person'] = train[['Age', 'Sex']].apply(get_person, axis=1)
test['Person'] = test[['Age', 'Sex']].apply(get_person, axis=1)

'''fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(10, 5))
# sns.factorplot('Person',data=titanic_df,kind='count',ax=axis1)
sns.countplot(x='Person', data=X_train, ax=axis1)
# average of survived for each Person(male, female, or child)
person_perc = X_train[["Person", "Survived"]].groupby(['Person'], as_index=False).mean()
sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male', 'female', 'child'])
plt.show()'''

_ = input('Press [Enter] to continue.')

# ----------------------------------------------------------------------------------------------------------------------

y_train = train['Survived']

selected_features = ['Pclass', 'Person', 'Embarked', 'Family', 'Fare', 'isAlone']
# selected_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch','Fare']
X_train = train[selected_features]
X_test = test[selected_features]

# 离散值向量化
dict_vec = DictVectorizer(sparse=False)
X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
X_test = dict_vec.fit_transform(X_test.to_dict(orient='record'))

X_train1, X_cv, y_train1, y_cv = train_test_split(X_train, y_train, test_size=0.25, random_state=33)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def plot_validation_curve(estimator, title, X, y, param_name, param_range, ylim=None):
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=10, scoring="mean_squared_error", n_jobs=4)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with " + title)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    if ylim is not None:
        plt.ylim(*ylim)
    else:
        pass
    plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="darkorange")
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="navy")
    plt.legend(loc="best")
    plt.show()


def grid_search_model(estimator, X, Y, parameters, cv):
    CV_model = GridSearchCV(estimator=estimator, param_grid=parameters, cv=cv)
    CV_model.fit(X, Y)
    print(CV_model.cv_results_)
    print("Best Score:", CV_model.best_score_, " / Best parameters:", CV_model.best_params_)


def classifier_moduel(estimator, X_train, y_train, X_train1, X_cv, y_train1, y_cv):
    print('cross_val_score: ', end='')
    print(cross_val_score(estimator, X_train, y_train, cv=10).mean())

    estimator.fit(X_train1, y_train1)
    print('cv score: ', end='')
    print(estimator.score(X_cv, y_cv))
    print('train score: ', end='')
    print(estimator.score(X_train1, y_train1))
    estimator_y_predict = estimator.predict(X_cv)
    print(classification_report(y_cv, estimator_y_predict))


if __name__ == '__main__':
    xgb = XGBClassifier(max_depth=3, random_state=10, learning_rate=0.5, gamma=5)
    rfc = RandomForestClassifier(max_depth=3, n_estimators=80)
    SVM = svm.SVC()

    # plot_validation_curve(xgb, 'xgb', X_train, y_train, 'learning_rate', [0.01,0.05,0.1,0.5,1.0,2.0,3,4])
    # plot_validation_curve(xgb, 'xgb', X_train, y_train, 'gamma', [0.5,0.75,1.0,2.0,3.0,5.0])
    # plot_learning_curve(xgb, 'xgb', X_train, y_train, (0.7, 1.01), cv=10, n_jobs=4)
    # plt.show()

    classifier_moduel(rfc, X_train, y_train, X_train1, X_cv, y_train1, y_cv)
    classifier_moduel(xgb, X_train, y_train, X_train1, X_cv, y_train1, y_cv)

    ## Search grid for optimal parameters
    '''rf_param_grid = {"max_depth": [None],
                     "max_features": [1, 3, 10],
                     "min_samples_split": [2, 3, 10],
                     "min_samples_leaf": [1, 3, 10],
                     "bootstrap": [False],
                     "n_estimators": [100, 300],
                     "criterion": ["gini"]}

    gsRFC = GridSearchCV(rfc, param_grid=rf_param_grid, cv=10, scoring="accuracy", n_jobs=4, verbose=1)

    gsRFC.fit(X_train, y_train)

    RFC_best = gsRFC.best_estimator_
    # Best score
    print(gsRFC.best_score_)'''

    classifier_moduel(rfc, X_train, y_train, X_train1, X_cv, y_train1, y_cv)

    _ = input('Press [Enter] to continue.')

    # ------------------------------------------------------------------------------------------------------------------

    xgb.fit(X_train, y_train)
    y_predict = xgb.predict(X_test)

    xgb_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y_predict})
    xgb_submission.to_csv('E:/files/Kaggle/titanic/xgb_submission.csv', index=False)

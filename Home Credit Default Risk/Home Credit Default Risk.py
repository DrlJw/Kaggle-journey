import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv('E:/files/Kaggle/Home Credit Default Risk/application_train.csv')  # , nrows=30000
test = pd.read_csv('E:/files/Kaggle/Home Credit Default Risk/application_test.csv')
bureau = pd.read_csv('E:/files/Kaggle/Home Credit Default Risk/bureau.csv')
# bureau_balance = pd.read_csv('E:/files/Kaggle/Home Credit Default Risk/bureau_balance.csv')
previous_application = pd.read_csv('E:/files/Kaggle/Home Credit Default Risk/previous_application.csv')
pos_cash_balance = pd.read_csv('E:/files/Kaggle/Home Credit Default Risk/POS_CASH_balance.csv')
installments_payments = pd.read_csv('E:/files/Kaggle/Home Credit Default Risk/installments_payments.csv')
credit_card_balance = pd.read_csv('E:/files/Kaggle/Home Credit Default Risk/credit_card_balance.csv')

y_train = train['TARGET']


def application_train(train, test):
    # 所有贷款申请的静态数据，一行代表我们数据样本中的一笔贷款
    # the main training and testing data with information about each loan application at Home Credit.
    # Every loan has its own row and is identified by the feature SK_ID_CURR.
    # The training application data comes with the TARGET
    # indicating 0: the loan was repaid or 1: the loan was not repaid.
    train['CODE_GENDER'].replace('XNA', np.nan, inplace=True)  # 性别CODE_GENDER
    train['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)  # 工作天数
    train['NAME_FAMILY_STATUS'].replace('Unknown', np.nan, inplace=True)  # 家庭（婚姻）状况
    train['ORGANIZATION_TYPE'].replace('XNA', np.nan, inplace=True)  # 工作类型

    test['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    test['NAME_FAMILY_STATUS'].replace('Unknown', np.nan, inplace=True)
    test['ORGANIZATION_TYPE'].replace('XNA', np.nan, inplace=True)

    train['annuity_income_percentage'] = train['AMT_ANNUITY'] / train['AMT_INCOME_TOTAL']  # 贷款年金/年收入
    train['car_to_birth_ratio'] = train['OWN_CAR_AGE'] / train['DAYS_BIRTH']  # 车龄/年龄
    train['car_to_employ_ratio'] = train['OWN_CAR_AGE'] / train['DAYS_EMPLOYED']  # 车龄/工作天数
    train['children_ratio'] = train['CNT_CHILDREN'] / train['CNT_FAM_MEMBERS']  # 孩子数量/家庭成员数量
    train['credit_to_annuity_ratio'] = train['AMT_CREDIT'] / train['AMT_ANNUITY']  # 贷款金额/贷款年金
    train['credit_to_goods_ratio'] = train['AMT_CREDIT'] / train['AMT_GOODS_PRICE']  # 贷款金额/消费贷款额度
    train['credit_to_income_ratio'] = train['AMT_CREDIT'] / train['AMT_INCOME_TOTAL']  # 贷款金额/年收入
    train['days_employed_percentage'] = train['DAYS_EMPLOYED'] / train['DAYS_BIRTH']  # 工作天数/年龄
    # External sources
    train['external_sources_weighted'] = train.EXT_SOURCE_1 * 2 + train.EXT_SOURCE_2 * 3 + train.EXT_SOURCE_3 * 4
    for function_name in ['min', 'max', 'sum', 'mean', 'std']:
        train['external_sources_{}'.format(function_name)] = eval('np.{}'.format(function_name))(
            train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)
    train['income_credit_percentage'] = train['AMT_INCOME_TOTAL'] / train['AMT_CREDIT']  # 年收入/贷款金额
    train['income_per_child'] = train['AMT_INCOME_TOTAL'] / (1 + train['CNT_CHILDREN'])  # 年收入/孩子数
    train['income_per_person'] = train['AMT_INCOME_TOTAL'] / train['CNT_FAM_MEMBERS']  # 年收入/家庭成员数量
    train['payment_rate'] = train['AMT_ANNUITY'] / train['AMT_CREDIT']  # 贷款年金/贷款金额
    train['phone_to_birth_ratio'] = train['DAYS_LAST_PHONE_CHANGE'] / train['DAYS_BIRTH']  # 改变电话天数/年龄
    train['phone_to_employ_ratio'] = train['DAYS_LAST_PHONE_CHANGE'] / train['DAYS_EMPLOYED']  # 改变电话天数/工作天数

    test['annuity_income_percentage'] = test['AMT_ANNUITY'] / test['AMT_INCOME_TOTAL']
    test['car_to_birth_ratio'] = test['OWN_CAR_AGE'] / test['DAYS_BIRTH']
    test['car_to_employ_ratio'] = test['OWN_CAR_AGE'] / test['DAYS_EMPLOYED']
    test['children_ratio'] = test['CNT_CHILDREN'] / test['CNT_FAM_MEMBERS']
    test['credit_to_annuity_ratio'] = test['AMT_CREDIT'] / test['AMT_ANNUITY']
    test['credit_to_goods_ratio'] = test['AMT_CREDIT'] / test['AMT_GOODS_PRICE']
    test['credit_to_income_ratio'] = test['AMT_CREDIT'] / test['AMT_INCOME_TOTAL']
    test['days_employed_percentage'] = test['DAYS_EMPLOYED'] / test['DAYS_BIRTH']
    # External sources
    test['external_sources_weighted'] = test.EXT_SOURCE_1 * 2 + test.EXT_SOURCE_2 * 3 + test.EXT_SOURCE_3 * 4
    for function_name in ['min', 'max', 'sum', 'mean', 'nanmedian']:
        test['external_sources_{}'.format(function_name)] = eval('np.{}'.format(function_name))(
            test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)
    test['income_credit_percentage'] = test['AMT_INCOME_TOTAL'] / test['AMT_CREDIT']
    test['income_per_child'] = test['AMT_INCOME_TOTAL'] / (1 + test['CNT_CHILDREN'])
    test['income_per_person'] = test['AMT_INCOME_TOTAL'] / test['CNT_FAM_MEMBERS']
    test['payment_rate'] = test['AMT_ANNUITY'] / test['AMT_CREDIT']
    test['phone_to_birth_ratio'] = test['DAYS_LAST_PHONE_CHANGE'] / test['DAYS_BIRTH']
    test['phone_to_employ_ratio'] = test['DAYS_LAST_PHONE_CHANGE'] / test['DAYS_EMPLOYED']

    AGGREGATION_RECIPIES = [
        (['CODE_GENDER', 'NAME_EDUCATION_TYPE'], [('AMT_ANNUITY', 'max'),
                                                  ('AMT_CREDIT', 'max'),
                                                  ('EXT_SOURCE_1', 'mean'),
                                                  ('EXT_SOURCE_2', 'mean'),
                                                  ('OWN_CAR_AGE', 'max'),
                                                  ('OWN_CAR_AGE', 'sum')]),
        (['CODE_GENDER', 'ORGANIZATION_TYPE'], [('AMT_ANNUITY', 'mean'),
                                                ('AMT_INCOME_TOTAL', 'mean'),
                                                ('DAYS_REGISTRATION', 'mean'),
                                                ('EXT_SOURCE_1', 'mean')]),
        (['CODE_GENDER', 'REG_CITY_NOT_WORK_CITY'], [('AMT_ANNUITY', 'mean'),
                                                     ('CNT_CHILDREN', 'mean'),
                                                     ('DAYS_ID_PUBLISH', 'mean')]),
        (['CODE_GENDER', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], [('EXT_SOURCE_1', 'mean'),
                                                                                               ('EXT_SOURCE_2',
                                                                                                'mean')]),
        (['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], [('AMT_CREDIT', 'mean'),
                                                      ('AMT_REQ_CREDIT_BUREAU_YEAR', 'mean'),
                                                      ('APARTMENTS_AVG', 'mean'),
                                                      ('BASEMENTAREA_AVG', 'mean'),
                                                      ('EXT_SOURCE_1', 'mean'),
                                                      ('EXT_SOURCE_2', 'mean'),
                                                      ('EXT_SOURCE_3', 'mean'),
                                                      ('NONLIVINGAREA_AVG', 'mean'),
                                                      ('OWN_CAR_AGE', 'mean'),
                                                      ('YEARS_BUILD_AVG', 'mean')]),
        (['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], [('ELEVATORS_AVG', 'mean'),
                                                                                ('EXT_SOURCE_1', 'mean')]),
        (['OCCUPATION_TYPE'], [('AMT_ANNUITY', 'mean'),
                               ('CNT_CHILDREN', 'mean'),
                               ('CNT_FAM_MEMBERS', 'mean'),
                               ('DAYS_BIRTH', 'mean'),
                               ('DAYS_EMPLOYED', 'mean'),
                               ('DAYS_ID_PUBLISH', 'mean'),
                               ('DAYS_REGISTRATION', 'mean'),
                               ('EXT_SOURCE_1', 'mean'),
                               ('EXT_SOURCE_2', 'mean'),
                               ('EXT_SOURCE_3', 'mean')]),
    ]

    for groupby_cols, specs in tqdm(AGGREGATION_RECIPIES):
        group_object = train.groupby(groupby_cols)
        for select, agg in tqdm(specs):
            groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
            train = train.merge(group_object[select]
                                .agg(agg)
                                .reset_index()
                                .rename(index=str,
                                        columns={select: groupby_aggregate_name})
                                [groupby_cols + [groupby_aggregate_name]],
                                on=groupby_cols,
                                how='left')

    for groupby_cols, specs in tqdm(AGGREGATION_RECIPIES):
        group_object = test.groupby(groupby_cols)
        for select, agg in tqdm(specs):
            groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
            test = test.merge(group_object[select]
                              .agg(agg)
                              .reset_index()
                              .rename(index=str,
                                      columns={select: groupby_aggregate_name})
                              [groupby_cols + [groupby_aggregate_name]],
                              on=groupby_cols,
                              how='left')

    for groupby_cols, specs in tqdm(AGGREGATION_RECIPIES):
        for select, agg in tqdm(specs):
            if agg in ['mean', 'median', 'max', 'min']:
                groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
                diff_name = '{}_diff'.format(groupby_aggregate_name)
                abs_diff_name = '{}_abs_diff'.format(groupby_aggregate_name)

                train[diff_name] = train[select] - train[groupby_aggregate_name]
                train[abs_diff_name] = np.abs(train[select] - train[groupby_aggregate_name])

    for groupby_cols, specs in tqdm(AGGREGATION_RECIPIES):
        for select, agg in tqdm(specs):
            if agg in ['mean', 'median', 'max', 'min']:
                groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
                diff_name = '{}_diff'.format(groupby_aggregate_name)
                abs_diff_name = '{}_abs_diff'.format(groupby_aggregate_name)

                test[diff_name] = test[select] - test[groupby_aggregate_name]
                test[abs_diff_name] = np.abs(test[select] - test[groupby_aggregate_name])

    return train, test


def bureau_feature(bureau):
    # 所有客户之前由其他金融机构提供给信用局的信用报告（对于我们样本中有贷款的客户）
    # 对于我们样本中的每笔贷款，客户在申请日期之前在信用局拥有的信贷数量与行数一样多
    # data concerning client's previous credits from other financial institutions.
    # Each previous credit has its own row in bureau
    # but one loan in the application data can have multiple previous credits.
    bureau['AMT_CREDIT_SUM'].fillna(0, inplace=True)  # 当前信用额度
    bureau['AMT_CREDIT_SUM_DEBT'].fillna(0, inplace=True)  # 当前债务
    bureau['AMT_CREDIT_SUM_OVERDUE'].fillna(0, inplace=True)  # 当前逾期金额
    bureau['CNT_CREDIT_PROLONG'].fillna(0, inplace=True)  # 信贷延长次数

    bureau['DAYS_CREDIT_ENDDATE'][bureau['DAYS_CREDIT_ENDDATE'] < -40000] = np.nan  # 信贷局信贷剩余期限
    bureau['DAYS_CREDIT_UPDATE'][bureau['DAYS_CREDIT_UPDATE'] < -40000] = np.nan  # 最后一次信息更新天数
    bureau['DAYS_ENDDATE_FACT'][bureau['DAYS_ENDDATE_FACT'] < -40000] = np.nan  # 已结束信贷天数
    bureau['AMT_CREDIT_SUM'][bureau['AMT_CREDIT_SUM'] > 10000000] = np.nan
    bureau['AMT_CREDIT_SUM_DEBT'][bureau['AMT_CREDIT_SUM_DEBT'] > 50000000] = np.nan

    bureau['AMT_CREDIT_DEBT_RATE'] = bureau['AMT_CREDIT_SUM_DEBT'] / (1 + bureau['AMT_CREDIT_SUM'])

    bureau['bureau_credit_active_binary'] = (bureau['CREDIT_ACTIVE'] != 'Closed').astype(int)  # 当前信贷状态
    bureau['bureau_credit_enddate_binary'] = (bureau['DAYS_CREDIT_ENDDATE'] > 0).astype(int)
    features = pd.DataFrame({'SK_ID_CURR': bureau['SK_ID_CURR'].unique()})

    groupby = bureau.groupby(by=['SK_ID_CURR'])

    g = groupby['AMT_CREDIT_DEBT_RATE'].agg('mean').reset_index()  # 当前信贷距当前天数
    g.rename(index=str, columns={'AMT_CREDIT_DEBT_RATE': 'AMT_CREDIT_DEBT_RATE'}, inplace=True)
    features = features.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['DAYS_CREDIT'].agg('count').reset_index()  # 当前信贷距当前天数
    g.rename(index=str, columns={'DAYS_CREDIT': 'bureau_number_of_past_loans'}, inplace=True)
    features = features.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['CREDIT_TYPE'].agg('nunique').reset_index()  # 信贷类型（用途）
    g.rename(index=str, columns={'CREDIT_TYPE': 'bureau_number_of_loan_types'}, inplace=True)
    features = features.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['bureau_credit_active_binary'].agg('mean').reset_index()
    g.rename(index=str, columns={'bureau_credit_active_binary': 'bureau_credit_active_binary'}, inplace=True)
    features = features.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['AMT_CREDIT_SUM_DEBT'].agg('sum').reset_index()
    g.rename(index=str, columns={'AMT_CREDIT_SUM_DEBT': 'bureau_total_customer_debt'}, inplace=True)
    features = features.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['AMT_CREDIT_SUM'].agg('sum').reset_index()
    g.rename(index=str, columns={'AMT_CREDIT_SUM': 'bureau_total_customer_credit'}, inplace=True)
    features = features.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['AMT_CREDIT_SUM_OVERDUE'].agg('sum').reset_index()
    g.rename(index=str, columns={'AMT_CREDIT_SUM_OVERDUE': 'bureau_total_customer_overdue'}, inplace=True)
    features = features.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['CNT_CREDIT_PROLONG'].agg('sum').reset_index()
    g.rename(index=str, columns={'CNT_CREDIT_PROLONG': 'bureau_average_creditdays_prolonged'}, inplace=True)
    features = features.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['bureau_credit_enddate_binary'].agg('mean').reset_index()
    g.rename(index=str, columns={'bureau_credit_enddate_binary': 'bureau_credit_enddate_percentage'}, inplace=True)
    features = features.merge(g, on=['SK_ID_CURR'], how='left')

    features['bureau_average_of_past_loans_per_type'] = \
        features['bureau_number_of_past_loans'] / features['bureau_number_of_loan_types']

    features['bureau_debt_credit_ratio'] = \
        features['bureau_total_customer_debt'] / features['bureau_total_customer_credit']

    features['bureau_overdue_debt_ratio'] = \
        features['bureau_total_customer_overdue'] / features['bureau_total_customer_debt']

    return features


def previous_application_feature(prev_applications):
    # 样本中申请贷款客户的之前所有在Home Credit申请贷款的记录
    # previous applications for loans at Home Credit of clients who have loans in the application data.
    # Each current loan in the application data can have multiple previous loans.
    # Each previous application has one row and is identified by the feature SK_ID_PREV.
    prev_applications['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)  # 先前申请第一次支付时间
    prev_applications['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)  # 先前申请第一次应该到期时间
    prev_applications['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)  # 先前申请第一次实际到期时间
    prev_applications['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)  # 先前申请最后一次实际到期时间
    prev_applications['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)  # 先前申请期望的结束时间

    features = pd.DataFrame({'SK_ID_CURR': prev_applications['SK_ID_CURR'].unique()})

    PREVIOUS_APPLICATION_AGGREGATION_RECIPIES = []
    for agg in ['mean', 'min', 'max', 'sum', 'var']:
        for select in ['AMT_ANNUITY',
                       'AMT_APPLICATION',
                       'AMT_CREDIT',
                       'AMT_DOWN_PAYMENT',
                       'AMT_GOODS_PRICE',
                       'CNT_PAYMENT',
                       'DAYS_DECISION',
                       'HOUR_APPR_PROCESS_START',
                       'RATE_DOWN_PAYMENT'
                       ]:
            PREVIOUS_APPLICATION_AGGREGATION_RECIPIES.append((select, agg))
    PREVIOUS_APPLICATION_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], PREVIOUS_APPLICATION_AGGREGATION_RECIPIES)]

    for groupby_cols, specs in tqdm(PREVIOUS_APPLICATION_AGGREGATION_RECIPIES):
        group_object = prev_applications.groupby(groupby_cols)
        for select, agg in tqdm(specs):
            groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
            features = features.merge(group_object[select]
                                      .agg(agg)
                                      .reset_index()
                                      .rename(index=str,
                                              columns={select: groupby_aggregate_name})
                                      [groupby_cols + [groupby_aggregate_name]],
                                      on=groupby_cols,
                                      how='left')

    prev_app_sorted = prev_applications.sort_values(['SK_ID_CURR', 'DAYS_DECISION'])  # 上次申请时间
    prev_app_sorted_groupby = prev_app_sorted.groupby(by=['SK_ID_CURR'])

    prev_app_sorted['previous_application_prev_was_approved'] = (
            prev_app_sorted['NAME_CONTRACT_STATUS'] == 'Approved').astype('int')  # 该月合同状态
    g = prev_app_sorted_groupby['previous_application_prev_was_approved'].last().reset_index()
    features = features.merge(g, on=['SK_ID_CURR'], how='left')

    prev_app_sorted['previous_application_prev_was_refused'] = (
            prev_app_sorted['NAME_CONTRACT_STATUS'] == 'Refused').astype('int')
    g = prev_app_sorted_groupby['previous_application_prev_was_refused'].last().reset_index()
    features = features.merge(g, on=['SK_ID_CURR'], how='left')

    g = prev_app_sorted_groupby['SK_ID_PREV'].agg('nunique').reset_index()
    g.rename(index=str, columns={'SK_ID_PREV': 'previous_application_number_of_prev_application'}, inplace=True)
    features = features.merge(g, on=['SK_ID_CURR'], how='left')

    g = prev_app_sorted.groupby(by=['SK_ID_CURR'])['previous_application_prev_was_refused'].mean().reset_index()
    g.rename(index=str, columns={
        'previous_application_prev_was_refused': 'previous_application_fraction_of_refused_applications'},
             inplace=True)
    features = features.merge(g, on=['SK_ID_CURR'], how='left')

    prev_app_sorted['prev_applications_prev_was_revolving_loan'] = (
            prev_app_sorted['NAME_CONTRACT_TYPE'] == 'Revolving loans').astype('int')
    g = prev_app_sorted.groupby(by=['SK_ID_CURR'])[
        'prev_applications_prev_was_revolving_loan'].last().reset_index()
    features = features.merge(g, on=['SK_ID_CURR'], how='left')

    for number in [1, 3, 5]:
        prev_applications_tail = prev_app_sorted_groupby.tail(number)

        tail_groupby = prev_applications_tail.groupby(by=['SK_ID_CURR'])

        g = tail_groupby['CNT_PAYMENT'].agg('mean').reset_index()  # 申请时的先前信用期限
        g.rename(index=str,
                 columns={'CNT_PAYMENT': 'previous_application_term_of_last_{}_credits_mean'.format(number)},
                 inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = tail_groupby['DAYS_DECISION'].agg('mean').reset_index()
        g.rename(index=str,
                 columns={'DAYS_DECISION': 'previous_application_days_decision_about_last_{}_credits_mean'.format(
                     number)},
                 inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = tail_groupby['DAYS_FIRST_DRAWING'].agg('mean').reset_index()
        g.rename(index=str,
                 columns={
                     'DAYS_FIRST_DRAWING': 'previous_application_days_first_drawing_last_{}_credits_mean'.format(
                         number)},
                 inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

    return features


def installments_feature(installments):
    # 先前信贷的还款历史
    # payment history for previous loans at Home Credit.
    # There is one row for every made payment and one row for every missed payment.
    features = pd.DataFrame({'SK_ID_CURR': installments['SK_ID_CURR'].unique()})

    installments['CALC_DAYS_LATE_PAYMENT'] = installments['DAYS_ENTRY_PAYMENT'] - installments['DAYS_INSTALMENT']
    installments['CALC_PERC_LESS_PAYMENT'] = installments['AMT_PAYMENT'] / installments['AMT_INSTALMENT']
    installments['CALC_PERC_LESS_PAYMENT'].replace(np.inf, 0, inplace=True)
    installments['CALC_DIFF_INSTALMENT'] = installments['AMT_INSTALMENT'] - installments['AMT_PAYMENT']
    installments['CALC_PERC_DIFF_INSTALMENT'] = np.abs(installments['CALC_DIFF_INSTALMENT']) / installments[
        'AMT_INSTALMENT']
    installments['CALC_PERC_DIFF_INSTALMENT'].replace(np.inf, 0, inplace=True)

    INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES = []
    for agg in ['mean', 'min', 'max', 'sum', 'var']:
        for select in ['AMT_INSTALMENT',
                       'AMT_PAYMENT',
                       'DAYS_ENTRY_PAYMENT',
                       'DAYS_INSTALMENT',
                       'NUM_INSTALMENT_NUMBER',
                       'NUM_INSTALMENT_VERSION',
                       'CALC_DAYS_LATE_PAYMENT',
                       'CALC_PERC_LESS_PAYMENT',
                       'CALC_PERC_DIFF_INSTALMENT'
                       ]:
            INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES.append((select, agg))
    INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES)]

    for groupby_cols, specs in tqdm(INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES):
        group_object = installments.groupby(groupby_cols)
        for select, agg in tqdm(specs):
            groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
            features = features.merge(group_object[select]
                                      .agg(agg)
                                      .reset_index()
                                      .rename(index=str,
                                              columns={select: groupby_aggregate_name})
                                      [groupby_cols + [groupby_aggregate_name]],
                                      on=groupby_cols,
                                      how='left')

    return features


def POS_CASH_feature(pos_cash_balance):
    # 申请人先前申请的POS贷和现金贷的月度余额快照
    # monthly data about previous point of sale or cash loans clients have had with Home Credit.
    # Each row is one month of a previous point of sale or cash loan
    # and a single previous loan can have many rows.
    features = pd.DataFrame({'SK_ID_CURR': pos_cash_balance['SK_ID_CURR'].unique()})

    pos_cash_sorted = pos_cash_balance.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'])  # 相对于申请日期的月份余额
    group_object = pos_cash_sorted.groupby('SK_ID_CURR')['CNT_INSTALMENT_FUTURE'].last().reset_index()
    group_object.rename(index=str,
                        columns={'CNT_INSTALMENT_FUTURE': 'pos_cash_remaining_installments'},
                        inplace=True)
    features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

    pos_cash_balance['is_contract_status_completed'] = pos_cash_balance[
                                                           'NAME_CONTRACT_STATUS'] == 'Completed'  # 该月期间的合同状态
    group_object = pos_cash_balance.groupby(['SK_ID_CURR'])['is_contract_status_completed'].sum().reset_index()
    group_object.rename(index=str,
                        columns={'is_contract_status_completed': 'pos_cash_completed_contracts'},
                        inplace=True)
    features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

    POS_CASH_BALANCE_AGGREGATION_RECIPIES = []
    for agg in ['mean', 'min', 'max', 'sum', 'var']:
        for select in ['MONTHS_BALANCE', 'SK_DPD', 'SK_DPD_DEF']:
            POS_CASH_BALANCE_AGGREGATION_RECIPIES.append((select, agg))
    POS_CASH_BALANCE_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], POS_CASH_BALANCE_AGGREGATION_RECIPIES)]

    for groupby_cols, specs in tqdm(POS_CASH_BALANCE_AGGREGATION_RECIPIES):
        group_object = pos_cash_balance.groupby(groupby_cols)
        for select, agg in tqdm(specs):
            groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
            features = features.merge(group_object[select]
                                      .agg(agg)
                                      .reset_index()
                                      .rename(index=str, columns={select: groupby_aggregate_name})
                                      [groupby_cols + [groupby_aggregate_name]],
                                      on=groupby_cols,
                                      how='left')

    return features


def credit_card_feature(credit_card):
    # 申请人先前信用卡的月度余额快照
    # monthly data about previous credit cards clients have had with Home Credit.
    # Each row is one month of a credit card balance, and a single credit card can have many rows.
    features = pd.DataFrame({'SK_ID_CURR': credit_card['SK_ID_CURR'].unique()})

    INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES = []
    for agg in ['mean', 'min', 'max', 'sum', 'var']:
        for select in ['AMT_BALANCE',
                       'AMT_CREDIT_LIMIT_ACTUAL',
                       'MONTHS_BALANCE',
                       'SK_DPD',
                       'SK_DPD_DEF'
                       ]:
            INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES.append((select, agg))
    INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES)]

    for groupby_cols, specs in tqdm(INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES):
        group_object = credit_card.groupby(groupby_cols)
        for select, agg in tqdm(specs):
            groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
            features = features.merge(group_object[select]
                                      .agg(agg)
                                      .reset_index()
                                      .rename(index=str,
                                              columns={select: groupby_aggregate_name})
                                      [groupby_cols + [groupby_aggregate_name]],
                                      on=groupby_cols,
                                      how='left')

    return features


def merge_data(df, train, test, corr_num=0.01):
    # merge dataset to training set base on Correlations that between target and feature
    X_eng = train[['SK_ID_CURR', 'TARGET']].merge(df, how='left', on='SK_ID_CURR')
    X_eng = X_eng.drop(['SK_ID_CURR'], axis=1)
    X_eng_corr = abs(X_eng.corrwith(X_eng['TARGET']))

    selected_features = X_eng_corr[X_eng_corr > corr_num]
    selected_features = selected_features.index.values.tolist()
    selected_features.remove('TARGET')
    selected_features.append('SK_ID_CURR')

    train = train.merge(df[selected_features], how='left', on='SK_ID_CURR')
    test = test.merge(df[selected_features], how='left', on='SK_ID_CURR')

    return train, test


train, test = application_train(train, test)

bureau = bureau_feature(bureau)
train, test = merge_data(bureau, train, test)
del bureau

previous_application = previous_application_feature(previous_application)
train, test = merge_data(previous_application, train, test)
del previous_application

pos_cash_balance = POS_CASH_feature(pos_cash_balance)
train, test = merge_data(pos_cash_balance, train, test)
del pos_cash_balance

installments_payments = installments_feature(installments_payments)
train, test = merge_data(installments_payments, train, test)
del installments_payments

credit_card_balance = credit_card_feature(credit_card_balance)
train, test = merge_data(credit_card_balance, train, test)
del credit_card_balance

print(train.shape, test.shape)

_ = input('Press [Enter] to continue.')

# one-hot encoding of categorical variables
X_train = pd.get_dummies(train)
X_test = pd.get_dummies(test)
X_train, X_test = X_train.align(X_test, join='inner', axis=1)

X_train = X_train.drop(['SK_ID_CURR'], axis=1)
X_test = X_test.drop(['SK_ID_CURR'], axis=1)

del train
# del test

print(X_train.shape, X_test.shape)

_ = input('Press [Enter] to continue.')

# X_train1, X_cv, y_train1, y_cv = train_test_split(X_train, y_train, test_size=0.25, random_state=33)
#
# dtest = xgb.DMatrix(X_test)
# xgb_pars = {'min_child_weight': 4, 'eta': 0.03, 'colsample_bytree': 0.5, 'max_depth': 6,
#             'subsample': 0.5, 'lambda': 0.05, 'alpha': 0.05, 'nthread': -1, 'booster': 'gbtree', 'silent': 1,
#             'gamma': 0.05, 'objective': 'binary:logistic', 'eval_metric': 'auc'}

kf = KFold(n_splits=5, random_state=1024, shuffle=True)
oof_preds = np.zeros(X_train.shape[0])

for n_fold, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
    dtrain = lgb.Dataset(data=X_train.iloc[train_idx],
                         label=y_train.iloc[train_idx],
                         free_raw_data=False, silent=True)
    dvalid = lgb.Dataset(data=X_train.iloc[valid_idx],
                         label=y_train.iloc[valid_idx],
                         free_raw_data=False, silent=True)

    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'nthread': 4,
        'learning_rate': 0.02,  # 02,
        'num_leaves': 20,
        'colsample_bytree': 0.9497036,
        'subsample': 0.8715623,
        'subsample_freq': 1,
        'max_depth': 8,
        'reg_alpha': 0.041545473,
        'reg_lambda': 0.0735294,
        'min_split_gain': 0.0222415,
        'min_child_weight': 60,  # 39.3259775,
        'seed': 0,
        'verbose': -1,
        'metric': 'auc',
    }

    clf = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dvalid],
        early_stopping_rounds=10,
        verbose_eval=100
    )

    oof_preds[valid_idx] = clf.predict(dvalid.data)

print(roc_auc_score(y_train, oof_preds))

#
# clf.fit(X_train1, y_train1, eval_set=[(X_train1, y_train1), (X_cv, y_cv)],
#         eval_metric='auc', verbose=100, early_stopping_rounds=7)

# fig, ax = plt.subplots(figsize=(10, 35))
# xgb.plot_importance(xgb_model, importance_type='gain', ax=ax)
# plt.show()

_ = input('Press [Enter] to continue.')

# y_predict = xgb_model.predict(dtest)
# # y_predict = clf.predict(X_test)
#
# xgb_submission = pd.DataFrame({'SK_ID_CURR': test['SK_ID_CURR'], 'TARGET': y_predict})
# print(xgb_submission.shape)
# xgb_submission.to_csv('E:/files/Kaggle/Home Credit Default Risk/xgb_submission.csv', index=False)

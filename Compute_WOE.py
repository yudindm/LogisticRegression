# -*- coding: utf-8 -*-
from os import environ
from os.path import join, expanduser
import sqlalchemy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def calc_woe_iv(X_train, y_train, cat_col):
    if (np.isnan(X_train['edu_cd']).any()):
        raise Exception('Вычисление WoE для переменных, содержащих NaN, не реализовано.')
    bad = X_train[cat_col][y_train == 0]
    bads = bad.groupby(bad).count()

    good = X_train[cat_col][y_train == 1]
    goods = good.groupby(good).count()

    result = pd.DataFrame({'bads': bads, 'goods': goods})
    result.loc[np.isnan(result['bads']), 'bads'] = 0
    result.loc[np.isnan(result['goods']), 'goods'] = 0
    
    #If a particular bin contains no event or non-event,
    #you can use the formula below to ignore missing WOE.
    #We are adding 0.5 to the number of events and non-events in a group.
    result['num_adjust'] = 0
    result.loc[
        (result['goods'] == 0) | (result['bads'] == 0),
        'num_adjust'] = 0.5

    result['goods_pct'] = (result['goods'] + result['num_adjust']) / result['goods'].sum()
    result['bads_pct'] = (result['bads'] + result['num_adjust']) / result['bads'].sum()
    result['woe'] = np.log(result['bads_pct'] / result['goods_pct'])
    result['iv'] = (result['bads_pct'] - result['goods_pct']) * result['woe']

    return result

#Подключение к Oracle
secrets_filepath = join(expanduser('~'), 'guardian_secrets')

with open(secrets_filepath, 'r', encoding='utf-8-sig') as secrets_file:
    lines = secrets_file.readlines()
    secrets = dict(
        line.rstrip().split("=", maxsplit=1)
        for line in lines if line.rstrip() != '')

environ["NLS_LANG"] = "AMERICAN_AMERICA.UTF8"

egn = sqlalchemy.create_engine('oracle+cx_oracle://crm_user:{0}@bc15-aix01:1521/?service_name=crm'.format(secrets['crm_crm_user_pass']))
cnn = egn.connect()

#Загрузка набора данных
orig_df = pd.read_sql('select * from crm_user.TMP_GAG_CS_NEW_VV_2', cnn)
#copy_df = pd.read_csv("c:\\users\\crm\\Documents\\LogisticRegression\\gag_cs_new.csv")

#Загрузка описания типов колонок
col_t = pd.read_sql(
    'select lower(column_name) column_name, column_status, column_type '
    'from crm_user.lib_scor_column_types '
    'where sysdate between df and dt', cnn)
#copy_col = pd.read_csv("c:\\users\\crm\\Documents\\LogisticRegression\\gag_cs_new_cols.csv")

#Переменные из набора данных по которым заданы типы в справочнике
cols = set(orig_df.columns).intersection(set(col_t['column_name']))
#Свободные и зависимая переменные
y_col = 'event'
X_cols = list(cols - set([y_col]))

#Разбиваем набор данных на обучающий и тестовый в пропорции 70/30
X_train, X_test, y_train, y_test = train_test_split(
    orig_df[X_cols], orig_df[y_col], test_size=0.3, random_state=0)

#Создание dummy колонок из колонок типа NOMINAL
cat_columns = col_t[
    (col_t['column_type'] == 'NOMINAL') & \
    (col_t['column_name'].isin(X_cols))
    ]['column_name']

iv_calc_result = {}
for cat_col in cat_columns:
    iv_calc_result[cat_col] = calc_woe_iv(X_train, y_train, cat_col)




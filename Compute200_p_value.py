# -*- coding: utf-8 -*-
"""
Редактор Spyder

Это временный скриптовый файл.
"""
from os import environ
from os.path import join, expanduser
import sqlalchemy
import numpy as np
import pandas as pd
import statsmodels.api as sm

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
df = pd.read_sql('select * from crm_user.TMP_GAG_CS_NEW_VV_2', cnn)

#Загрузка описания типов колонок
col_t = pd.read_sql(
    'select lower(column_name) column_name, column_status, column_type '
    'from crm_user.lib_scor_column_types '
    'where sysdate between df and dt', cnn)

#Создание dummy колонок из колонок типа NOMINAL
cat_columns = col_t[col_t['column_type'] == 'NOMINAL']['column_name']
for cat_column in cat_columns:
    dummies = pd.get_dummies(df[cat_column], prefix=cat_column)
    df = df.join(dummies)

df = df.drop(labels=cat_columns.tolist(), axis=1)

#Определение "корреляции" переменных к целевой
pvalue_single = pd.Series(np.NaN, index=df.columns)
pvalue_errors = {}
for col in pvalue_single.index:
    if col in ['event', 'contact_dt']:
        continue
    sm_logit = sm.Logit(df['event'], df[col])
    try:
        sm_result = sm_logit.fit(disp=0)
        pvalue_single[col] = sm_result.pvalues[0]
    except Exception as ex:
        pvalue_errors[col] = ex



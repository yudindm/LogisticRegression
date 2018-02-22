# -*- coding: utf-8 -*-
from os import environ
from os.path import join, expanduser
import sqlalchemy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class woe_result(object):
    'Результаты биннинга'
    def __init(self):
        self.cat_col = None
        self.iv = None
        self.bin_map = None
        self.sum_table = None

def connect_to_oracle():
    secrets_filepath = join(expanduser('~'), 'guardian_secrets')
    with open(secrets_filepath, 'r', encoding='utf-8-sig') as secrets_file:
        lines = secrets_file.readlines()
        secrets = dict(
            line.rstrip().split("=", maxsplit=1)
            for line in lines if line.rstrip() != '')
    
    environ["NLS_LANG"] = "AMERICAN_AMERICA.UTF8"
    
    egn = sqlalchemy.create_engine('oracle+cx_oracle://crm_user:{0}@bc15-aix01:1521/?service_name=crm'.format(secrets['crm_crm_user_pass']))
    return egn.connect()

def calc_woe_iv(X_train, y_train, cat_col):
    if (np.isnan(X_train['edu_cd']).any()):
        raise Exception('Вычисление WoE для переменных, содержащих NaN, не реализовано.')
    bad = X_train[cat_col][y_train == 0]
    bads = bad.groupby(bad).count()

    good = X_train[cat_col][y_train == 1]
    goods = good.groupby(good).count()

    result = woe_result()
    
    result.cat_col = cat_col
    
    sum_tab = pd.DataFrame({'bads': bads, 'goods': goods}, dtype=np.float64)
    sum_tab.loc[np.isnan(sum_tab['bads']), 'bads'] = 0
    sum_tab.loc[np.isnan(sum_tab['goods']), 'goods'] = 0

    total_cnt = (sum_tab['bads'] + sum_tab['goods']).sum()
    misc_bins = sum_tab[((sum_tab["bads"] + sum_tab["goods"]) / total_cnt) < 0.01]
    if len(misc_bins) > 1:
        bin_map = [[cat] for cat in sum_tab.index if cat not in misc_bins.index]
        bin_map = bin_map + [list(cat for cat in misc_bins.index)]
    else:
        bin_map = [[cat] for cat in sum_tab.index]

    result.bin_map = bin_map
#    result.sum_table = sum_tab
    result.sum_table = merge_bins(sum_tab, result.bin_map)
    
    calc_woe(result.sum_table)
    result.iv = result.sum_table['iv'].sum()
    
    while len(result.sum_table) > 3:
        sorted_sum_tab = result.sum_table.copy()
        sorted_sum_tab.sort_values(by='woe')
        closest_bin = find_closest_bin(sorted_sum_tab)
        new_bin_map = combine_bins(result.bin_map, closest_bin.index[0])
        new_sum_tab = merge_bins(sum_tab, new_bin_map)
        calc_woe(new_sum_tab)
        new_iv = new_sum_tab['iv'].sum()
        if (abs(new_iv - result.iv) / result.iv) > 0.05:
            break
        result.sum_table = new_sum_tab
        result.iv = new_iv
        result.bin_map = new_bin_map
    
    return result

def calc_woe(df):
    #If a particular bin contains no event or non-event,
    #you can use the formula below to ignore missing WOE.
    #We are adding 0.5 to the number of events and non-events in a group.
    df['num_adjust'] = 0
    df.loc[
        (df['goods'] == 0) | (df['bads'] == 0),
        'num_adjust'] = 0.5

    df['goods_pct'] = (df['goods'] + df['num_adjust']) / df['goods'].sum()
    df['bads_pct'] = (df['bads'] + df['num_adjust']) / df['bads'].sum()
    df['woe'] = np.log(df['bads_pct'] / df['goods_pct'])
    df['iv'] = (df['bads_pct'] - df['goods_pct']) * df['woe']
    
def find_closest_bin(df_woe):
    return df_woe['woe'].rolling(window=2).apply(lambda x: x[1] - x[0]).nsmallest(1)

def combine_bins(bin_map, bin_cat_to):
    result = []
    for bin_cat in bin_map:
        if str(bin_cat) == bin_cat_to:
            result[len(result) - 1] = result[len(result) - 1] + bin_cat
        else:
            result.append(bin_cat)
    return result

def merge_bins(df_cat, bin_map):
    df_bins = pd.DataFrame(columns=['goods', 'bads'])
    for bin_cat in bin_map:
        bin_counts = df_cat.loc[bin_cat].sum()
        bin_counts.name = str(bin_cat)
        df_bins = df_bins.append(bin_counts)
    return df_bins

#Подключение к Oracle
#cnn = connect_to_oracle()

#Загрузка набора данных
#orig_df = pd.read_sql('select * from crm_user.TMP_GAG_CS_NEW_VV_2', cnn)
orig_df = pd.read_csv("gag_cs_new.csv")

#Загрузка описания типов колонок
#col_t = pd.read_sql(
#    'select lower(column_name) column_name, column_status, column_type '
#    'from crm_user.lib_scor_column_types '
#    'where sysdate between df and dt', cnn)
col_t = pd.read_csv("gag_cs_new_cols.csv")

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

iv_calc_results = {}
for cat_col in cat_columns:
    iv_calc_results[cat_col] = calc_woe_iv(X_train, y_train, cat_col)

woe_series = []
for iv_calc_result in iv_calc_results.items():
    bin_map = iv_calc_result[1].bin_map
    col_name = iv_calc_result[1].cat_col
    sum_tab = iv_calc_result[1].sum_table

    cat_woe_map = {}
    for r in sum_tab.iterrows():
        for bin in bin_map:
            if str(bin) == r[0]:
                for cat in bin:
                    cat_woe_map[cat] = r[1].woe

    woe_ser = X_train[col_name].transform(lambda x: cat_woe_map[x])
    woe_ser.name = woe_ser.name + '_WOE'
    woe_series.append(woe_ser)    

X_train = X_train.join(woe_series)

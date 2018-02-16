from os import environ
from os.path import join, expanduser
import sqlalchemy
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

#https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

secrets_filepath = join(expanduser('~'), 'guardian_secrets')

with open(secrets_filepath, 'r', encoding='utf-8-sig') as secrets_file:
    lines = secrets_file.readlines()
    secrets = dict(
        line.rstrip().split("=", maxsplit=1)
        for line in lines if line.rstrip() != '')

environ["NLS_LANG"] = "AMERICAN_AMERICA.UTF8"

egn = sqlalchemy.create_engine('oracle+cx_oracle://crm_user:{0}@bc15-aix01:1521/?service_name=crm'.format(secrets['crm_crm_user_pass']))
cnn = egn.connect()

df = pd.read_sql('select * from crm_user.TMP_GAG_CS_NEW_VV_2', cnn)
col_t = pd.read_sql(
    'select lower(column_name) column_name, column_status, column_type '
    'from crm_user.lib_scor_column_types '
    'where sysdate between df and dt', cnn)

cat_columns = col_t[col_t['column_type'] == 'NOMINAL']['column_name']
for cat_column in cat_columns:
    dummies = pd.get_dummies(df[cat_column], prefix=cat_column)
    df = df.join(dummies)

df = df.drop(labels=cat_columns.tolist(), axis=1)

y = ['event']
X = col_t[~col_t['column_name'].isin(y+cat_columns.tolist())]['column_name'].tolist()

logreg = LogisticRegression()
rfe = RFE(logreg, 15)
rfe = rfe.fit(df[X], df['event'])

Xf = np.array(X)[rfe.support_].tolist()

for f in Xf:
    Xf1 = [f2 for f2 in Xf if f2 != f]
    logit_model = sm.Logit(df['event'], df[Xf1])
    try:
        result=logit_model.fit()
        print('Success without {0}'.format(f))
    except:
        print('Error without {0}'.format(f))


Xf1 = [f2 for f2 in Xf if f2 != 'mnth_from_first_restr']
logit_model = sm.Logit(df['event'], df[Xf1])


X_train, X_test, y_train, y_test = train_test_split(df[Xf1], df['event'], test_size=0.3, random_state=0)

#train
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

#predicting
y_pred = logreg.predict(X_test)

#score result on test set
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))



kfold = KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)

print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


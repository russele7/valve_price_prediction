#!/usr/bin/env python
# coding: utf-8

# # 1. Функции

# In[63]:


# Функция для расчета метрик
def metrics(y_true, y_predict):
  mae_score = MAE(y_true, y_predict)
  mse_score = MSE(y_true, y_predict)
  print('MAE = {}'.format(round(mae_score,2)))
  print('MSE = {}'.format(round(mse_score,2)))
  return mae_score, mse_score


# In[64]:


# Визуализация соотношения y_true / y_predict на тесте

def true_predict_ratio(y_true, y_predict, xy_max):
    if xy_max == None:
        xy_max = max(y_true.max(), y_predict.max())
    plt.figure(figsize = (10,10))
    sns.scatterplot(x = y_true.values, y = y_predict)
    plt.title('Соотношение y_true / y_predict')
    plt.xlabel('y_true')
    plt.ylabel('y_predict')
    plt.xlim([-10, xy_max + 10])
    plt.ylim([-10, xy_max + 10])
    plt.savefig('pict.pdf', facecolor='w', edgecolor='b',
        orientation='portrait', bbox_inches='tight')


# In[65]:


# Визуализация важности признаков

def importance_visualisation(model_fitted, cols):
  try:
    importances = pd.Series(data = abs(model_fitted.feature_importances_), index = list(cols))
  except:
    importances = pd.Series(data = abs(model_fitted.best_estimator_.feature_importances_), index = list(cols))

  importances = importances.sort_values(ascending = False)
  plt.figure(figsize = (10,4))
  sns.barplot(x = list(importances.index) , y = importances)
  plt.xticks(rotation = 90)
  plt.title('Значимость признаков')
  plt.show()


# In[66]:


# Визуализация предсказаний для новых позиций
def true_predict_visual_uniq(y_true, y_predict, train_set):
  
  hues = list(train_set.columns)
  fig, ax = plt.subplots(len(hues), 1,  figsize=(6, 6 * len(hues)))
  ax_lim = np.array(list(y_true) + list(y_predict)).max()
  i = 0
  for i in range(len(hues)):
    if hues[i] == 'body':
      sns.scatterplot(x = y_true, y = y_predict, 
                    hue = train_set[hues[i]].replace(dict((v,k) for k, v in dict_body_num.items())).values, ax = ax[i])
    elif hues[i] == 'size':
      sns.scatterplot(x = y_true, y = y_predict, 
                    hue = train_set[hues[i]].replace(dict(zip((train_set['size'].unique()) , 
                                                          [str(value) for value in X['size'].unique()]))).values, ax = ax[i])
      
    else:
      sns.scatterplot(x = y_true, y = y_predict, hue = train_set[hues[i]].values, ax = ax[i])
    ax[i].set_title(hues[i])
    ax[i].set_xlabel('y_true_test')
    ax[i].set_ylabel('y_predict_test')
    ax[i].set_xlim([-50, ax_lim + 50])
    ax[i].set_ylim([-50, ax_lim + 50])  


# # 1. Загрузка и предварительный анализ данных

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid') #, {'grid.color': '.1', 'text.color': '.1', 'text.size' : '40'})
sns.set(font_scale=1)
import plotly.express as px

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as MAE , mean_squared_error as MSE, make_scorer
from sklearn.svm import SVR
from sklearn.metrics import make_scorer
from sklearn.externals import joblib

import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import timeit


# Загружаем данные

# In[6]:


total_spec = pd.read_excel('total_spec_work.xlsx', index_col = 0)
total_spec


# # 2. EDA анализ - подготовка

# Некоторые материалы являются аналогами (отличаются технологией получения - литье или поковка). Можем их объединить

# In[7]:


dict_body = {
    'A216 WCB' : 'WCB',
    'A350 LF2' : 'LF2_LCB_LCC', 'A352 LCC' : 'LF2_LCB_LCC', 'A352 LCB' : 'LF2_LCB_LCC',


    'A182 F11' : 'F11_WC6', 'A217 WC6' : 'F11_WC6', 
    'A182 F9' : 'F9_C12', 'A217 C12' : 'F9_C12',
    'A182 F316' : 'F316_CF8M', 'A351 CF8M' : 'F316_CF8M',
    'A182 F321' : 'F321',
    'A182 F347' : 'F347_CF8C', 'A351 CF8C' : 'F347_CF8C',
    'ASTM A351 CF8' : 'F304_CF8', 'ASTM A182 F304/304L' : 'F304_CF8'
}


# In[8]:


total_spec['body'] = total_spec['body'].replace(dict_body)


# # 3. Подготовка данных для обучения моделей

# ## 3.1 Первым делом стоит перевести body в численный тип данных. Т.к. есть понимание какой материал дороже а какой дешевле. Для наглядности создадим новый числовой столбец и сохраним старый категориальный а потом посмотрим на важность признаков в моделях

# # Источник, по которому подобраны коэффициенты для материалов корпуса  
# https://ru.scribd.com/doc/237305322/Material-Reference-Chart

# In[9]:


dict_body_num = {
    'A216 WCB' : 3, 'WCB' : 3,
    'A350 LF2' : 4, 'A352 LCC' : 4, 'A352 LCB' : 4, 'LF2_LCB_LCC' : 4,
    'A182 F11' : 5, 'A217 WC6' : 5,  'F11_WC6' : 5,
    'A182 F9' : 6, 'A217 C12' : 6, 'F9_C12' : 6,
    'F304_CF8' : 8,
    'A182 F316' : 9, 'A351 CF8M' : 9, 'F316_CF8M' : 9,
    'A182 F321' : 10, 'F321' : 10,
    'A182 F347' : 11, 'A351 CF8C' : 11, 'F347_CF8C' : 11,
    
    'A351 CK3MCuN': 14,
    'UNS S32205': 14,

    'B381 F2' : 140
}


# In[10]:


total_spec['body_num'] = total_spec['body'].replace(dict_body_num)

total_spec = total_spec.loc[:, ['valve_type', 'size', 'class_rating', 'body', 'connection', 'actuator',
       'trim', 'seat', 'face_to_face', 'weight', 'specification', 'body_num',
       'usd_unit_price']]


# ## 3.2 Отделение целевого признака. Выделение количественных и категориальных признаков

# In[11]:


X = total_spec.drop(columns = ['usd_unit_price'])
y = total_spec['usd_unit_price']


# In[12]:


num_features = list(X.columns[[col != 'object' for col in X.dtypes]])
cat_features = list(X.columns[[col == 'object' for col in X.dtypes]])


# Дальше будет следующая последовательность:  
# 1. Обработка категориальных признаков, потому что если сначала сделать разделение на обучающую и тестовую выборки то есть риск что не все варианты категориальных признаков попадут в каждую из выборок.  
# 2. Разделение на обучающую и тестовую выборки.  
# 3. Нормализация числовых признаках - тут уже нужно отдельно обрабатывать обучающую и тестовую выборки

# ## 3.4  ordinal преобразованные данные:

# In[13]:


encoder = OrdinalEncoder()

X_ord = X.copy()
encoder.fit(X_ord[cat_features])

X_ord[cat_features] = X_ord[cat_features].fillna("nan_value")
X_ord[cat_features] = pd.DataFrame(encoder.transform(X_ord[cat_features]), columns = cat_features)
X_ord[cat_features] = X_ord[cat_features].astype(int)


# In[14]:


X_ord.body.unique()


# In[15]:


print('Введите количество моделей')
n_models = int(input())


# In[16]:


rand_best_params_list = []
for i in range(n_models):
    ## 3.5 Разделение на train и eval
    total_spec_train, total_spec_eval,     X_train, X_eval,      X_ord_train, X_ord_eval,      y_train, y_eval = train_test_split(total_spec , X, X_ord, y, test_size = 0.1, shuffle=True) 
    ## 3.7 Стандартизация
    scaler = StandardScaler()

    X_st_train = X_train.copy()
    X_st = X.copy()
    X_st_eval = X_eval.copy()
    X_st_train[num_features] = scaler.fit_transform(X_st_train[num_features])
    X_st[num_features] = scaler.transform(X_st[num_features])
    X_st_eval[num_features] = scaler.transform(X_st_eval[num_features])

    X_ord_st_train = X_ord_train.copy()
    X_ord_st = X_ord.copy()
    X_ord_st_eval = X_ord_eval.copy()
    X_ord_st_train[num_features] = scaler.fit_transform(X_ord_st_train[num_features])
    X_ord_st[num_features] = scaler.transform(X_ord_st[num_features])
    X_ord_st_eval[num_features] = scaler.transform(X_ord_st_eval[num_features])
    
    ## 5.1 lgbm + gridsearch + default score
    
#     pg_grid = {'boosting_type' : ['gbdt'],
#             'num_leaves': [8,10,12],
#             'max_depth':  [7,8,9] ,
#             'n_estimators' : [200, 400, 800],
#             'min_data_in_leaf': [2, 3]}
#             # 'number_iterations' : [10,50,100],
#             # 'function_fraction' : [0.8, 1],
#             # 'bagging_fraction' : [0.8, 1]}
            
    pg_random_search = {'boosting_type' : ['gbdt'],
            'num_leaves': list(range(5,51,5)),
            'max_depth':  list(range(1,15,1)),
             'n_estimators' : [200, 500, 800, 1000],
                  'min_data_in_leaf': list(range(1,10,1))}

    # Задаем модель
#     model_lgbm1 = GridSearchCV(estimator = LGBMRegressor( objective= 'mse', verbose = 10),
#                            param_grid= pg_LGBM1, cv=3, n_jobs=-1 , verbose=10, scoring = 'neg_mean_squared_error')
    
    
    model = RandomizedSearchCV(estimator= LGBMRegressor( objective= 'regression', verbose = 50), scoring = 'neg_mean_squared_error', 
                               param_distributions= pg_random_search, 
                               n_iter = 100, n_jobs=-1, verbose = 50, cv = 3)
    
     # Обучаем модель
    model.fit(X_ord_st, y, categorical_feature = cat_features)
    rand_best_params_list.append(list(model.best_params_.values()) + [(model.best_score_)])
    print('Результаты по модели №',i)
    print(model.best_params_)
    
    # сохраняем модель    
    joblib.dump(model.best_estimator_, 'model_fitted_{}.pkl'.format(i))   


# In[ ]:


print('А теперь введите количество моделей, которые будем брать для СРЕДНЕГО предсказания')
n_models_pred = int(input())


# In[24]:


list_models = []
for i in range(n_models_pred):
    list_models.append(joblib.load('model_fitted_{}.pkl'.format(i)))        


# In[25]:


def model_mean(list_of_models, X):
    pred = np.zeros(len(X))
    for model in list_of_models:
        pred += model.predict(X)
    pred = pred / len(list_of_models)
    return pred


# # Теперь загружаем и предобрабатываем тестовый датасет

# In[26]:


test_set = pd.read_excel('total_spec_test.xlsx', index_col = 0)

X_test = test_set[['valve_type', 'size', 'class_rating', 'body', 'connection', 'actuator',
       'trim', 'seat', 'face_to_face', 'weight', 'specification']]


# In[27]:


X_test['body'] = X_test['body'].replace(dict_body)
X_test['body_num'] = X_test['body'].replace(dict_body_num)


# In[28]:


## 3.7 Ordinalencoder
X_test_ord = X_test.copy()
X_test_ord[cat_features] = X_test_ord[cat_features].fillna("nan_value")
X_test_ord[cat_features] = pd.DataFrame(encoder.transform(X_test_ord[cat_features]), columns = cat_features)
X_test_ord[cat_features] = X_test_ord[cat_features].astype(int)
## 3.7 Стандартизация
X_test_ord_st = X_test_ord.copy()
X_test_ord_st[num_features] = scaler.transform(X_test_ord_st[num_features])


# In[29]:


test_set['predict_price'] = model_mean(list_models, X_test_ord_st)
test_set['delta_usd'] = test_set['usd_unit_price'] - test_set['predict_price']

def mape(y_true, y_predict):
  return 100. * (y_true - y_predict) / y_true

test_set['mape'] = mape(test_set['usd_unit_price'], test_set['predict_price'] )


# In[30]:


test_set.to_excel('results.xlsx')


# In[61]:


print('метрики на тестовой выборке:')
metrics(test_set['usd_unit_price'],  model_mean(list_models, X_test_ord_st))
true_predict_ratio(test_set['usd_unit_price'],  model.predict(X_test_ord_st), None)


# In[ ]:


print('Расчет закончен. Предсказанные цены и график можно посмотреть в файлах results.xlsx и pict.pdf')


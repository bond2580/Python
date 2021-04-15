#!/usr/bin/env python
# coding: utf-8

# In[242]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# In[243]:


#データ内の文字データをダミー変数に,数字データを標準化 
def pre(data, targ):
    data_y = data[targ]
    data_x = data.drop(targ, axis=1)
    
    drop=[]
    
    for name in data_x.columns:
        if type(data_x[name][0]) == str:
            drop.append(name)
        else:
            data_x[name] == (data_x[name] - data_x[name].mean()) / data_x[name].std()

    
    dum = pd.get_dummies(data_x[drop]) #ダミーを作成
    data_x = data_x.drop(drop, axis=1)
    data_x = pd.concat([data_x, dum], axis=1)
    
    return data_x, data_y #説明変数と目的変数を返す


# In[244]:


#ファイルの読み込み
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
len(test)


# In[245]:


#訓練データの作成
train = train.drop(["PassengerId", "Cabin", "Ticket", "Name"], axis=1)
train.isna().sum()
train = train.fillna(train.median())
train = train.dropna()


# In[246]:


#テストデータの作成
Target = pd.read_csv("gender_submission.csv")
Target = pd.merge(test, Target, on="PassengerId")
Id = Target["PassengerId"]
len(Target)


# In[247]:


Target = Target.drop(["PassengerId", "Cabin", "Ticket", "Name"], axis=1)
Target = Target.fillna(train.median())
Target = Target.dropna()


# In[248]:


#訓練データ, テストデータを目的変数, 説明変数に分割
train_X, train_y = pre(train, "Survived")
test_X, test_y = pre(Target, "Survived")


# In[249]:


#ランダムサーチ用の分類器とパラメータ
params = {LogisticRegression(): {
            "C": [10**i for i in range(-2, 2)],
            "penalty": ["l1", "l2", "elasticnet"],
            "max_iter": [i for i in range(1, 100, 10)],
            "random_state": [42]
          },
          SVC(): {
           "C": [10**i for i in range(-2, 2)],
            "kernel": ["linear", "rbf", "poly", "sigmoid"],
            "random_state": [42]
          },
          DecisionTreeClassifier(): {
            "max_depth":[i for i in range(1, 10)],
            "random_state": [i for i in range(10, 20)]  
          },
           RandomForestClassifier(): {
            "n_estimators": [i for i in range(10, 20)],
            "max_depth": [i for i in range(1, 10)],
            "random_state": [i for i in range(100)]
          }}

max_score = 0
best_model = None
best_param = None


# In[257]:


#精度の最も高くなるモデルとパラメータをランダムサーチ

max_score = 0

for model, param in params.items():
    clf = RandomizedSearchCV(model, param, cv=5)
    clf.fit(train_X, train_y)
    
    score = clf.score(test_X, test_y)
    print(model)
    print("Score: %f" %score)
    
    if max_score < score:
          pred = clf.predict(test_X)
          max_score = score
          best_model = model.__class__.__name__ 
          best_param = clf.best_params_

        


# In[271]:


#提出用データの作成
df = pd.DataFrame([Id, test_y]).T


# In[272]:


#提出用ファイルをcsvファイルとして保存
df.to_csv("submission.csv", index=False)


# In[ ]:





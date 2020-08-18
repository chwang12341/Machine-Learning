## knn_model_joblib_save.py
## 導入Python 數據處理套件 numpy and pandas
import numpy as np
import pandas as pd
## 導入sklearn(Scikit Learn)
## 導入sklearn的數據集
from sklearn import datasets
## 導入分割train data 與 test data的套件
from sklearn.model_selection import train_test_split
## 導入KNN 模型 - RadiusNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
## 導入Iris 數據集
iris_raw_dataset = datasets.load_iris()
## 將我們要用來預測的自變數與應變數轉成dataframe
## data 是我們要觀察的變數
X_df = pd.DataFrame(iris_raw_dataset.data)
## target 是我們希望能預測的變數
Y_df = pd.DataFrame(iris_raw_dataset.target)
## 將數據分割為Train Data 與 Test Data
## test_size: 代表你要切多少比例給test data，剩下的就會成為train data
## random_state: 隨機數種子，設定為一樣的隨機樹種子，被隨機分割出來的data編號才會一樣喔
X_train, X_test, y_train, y_test = train_test_split(X_df, Y_df, test_size = 0.2, random_state = 0)
##建立 KNN Model
knn_model = RadiusNeighborsClassifier(radius=2)
## 訓練model, fit進model，創建屬於這個數據集的KNN模型, fit參數接受 train data 是matrix，test data 是array
## 用訓練集來創進KNN model, ravel()將多維轉換成一維matrix
knn_model.fit(X_train, y_train.values.ravel())

## 導入joblib套件，joblib為sklearn的外部套件
from sklearn.externals import joblib
## 保存模型
joblib.dump(knn_model,"model/knn_model.pkl")
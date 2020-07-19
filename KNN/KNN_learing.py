## 導入Python 數據處理套件 numpy and pandas
import numpy as np
import pandas as pd
## 導入sklearn(Scikit Learn)
## 導入sklearn的數據集
from sklearn import datasets
## 導入分割train data 與 test data的套件
from sklearn.model_selection import train_test_split
## 導入KNN 模型
from sklearn.neighbors import KNeighborsClassifier
## 導入Iris 數據集
iris_raw_dataset = datasets.load_iris()
## 查看有哪些資料分類，列出Key值
print(iris_raw_dataset.keys()) #dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
## 印出 feature 數值
print(iris_raw_dataset['data'])
## 目標種類: 分成三類Setosa、Versicolor、Virginaica，分別用0、1、2代表
print(iris_raw_dataset['target'])
## 目標種類的名稱
print(iris_raw_dataset['target_names'])
## 印出描述資料集的DATA
print(iris_raw_dataset['DESCR'])
## 印出屬性name
print(iris_raw_dataset['feature_names']) #['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
## 印出目標種類
print(np.unique(iris_raw_dataset.target)) #iris_raw_dataset
## 將我們要用來預測的自變數與應變數轉成dataframe
## data 是我們要觀察的變數
X_df = pd.DataFrame(iris_raw_dataset.data)
## target 是我們希望能預測的變數
Y_df = pd.DataFrame(iris_raw_dataset.target)
## 將數據分割為Train Data 與 Test Data
## test_size: 代表你要切多少比例給test data，剩下的就會成為train data
## random_state: 隨機數種子，設定為一樣的隨機樹種子，被隨機分割出來的data編號才會一樣喔
X_train, X_test, y_train, y_test = train_test_split(X_df, Y_df, test_size = 0.2, random_state = 0)
## 印出有多少資料
print(len(Y_df )) #150
## 查看 train 與 test 分別的資料筆數
print(len(y_train)) #120
print(len(y_test)) #30
##建立 KNN Model
knn_model = KNeighborsClassifier()
## 訓練model, fit進model，創建屬於這個數據集的KNN模型, fit參數接受 train data 是matrix，test data 是array
## 用訓練集來創進KNN model, ravel()將多維轉換成一維matrix
knn_model.fit(X_train, y_train.values.ravel())
## 利用 test data裡的 X 來預測 y
print(knn_model.predict(X_test))
# [2 1 0 2 0 2 0 1 1 1 2 1 1 1 2 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0]
## 查看實際y
print(y_test.values.ravel())
# [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0]
## 這是 test data X 預測y是什麼的機率 
print(knn_model.predict_proba(X_test))
## 模型預測的準確率(Accuracy)
print(knn_model.score(X_test, y_test))
## 優化
## KNN Classifier 參數設
knn_model = KNeighborsClassifier(n_neighbors = 10, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
knn_model.fit(X_train, y_train.values.ravel())
print(knn_model.score(X_test, y_test)) #1.0
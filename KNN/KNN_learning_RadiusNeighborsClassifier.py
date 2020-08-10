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


## Step 4: radius_neighbors 和 radius_neighbors_graph 實作
## a. radius_neighbors: 找到指定半徑下，一個或多個的近鄰，它會返回數據集中每個點的索引和距離值
## b. radius_neighbors_graph: 計算x中的點與在指定半徑內的近鄰的加權圖
## c. 步驟: 先指定一個或多個資料點，然後設定半徑，查看radius_neighbors與radius_neighbors_graph
## 找到指定半徑下，一個或多個的近鄰，它會返回數據集中每個點的索引和距離值
## 指訂一個或多個點資料，我這邊隨便設定一個資料，必須先從list轉array，然後再.reshape(1, -1)，才能使用
X = [5.8, 2.8, 3.8, 6]
X = np.array(X).reshape(1, -1)
RN = knn_model.radius_neighbors(X, radius=10)
print(RN)
print(np.asarray(RN[0][0]))
# print(np.asarray(RN[1][2]))
## 計算x中的點與在指定半徑內的近鄰的加權圖
## radius neighbors graph
RNG = knn_model.radius_neighbors_graph(X, radius=10)
print(RNG)
print(RNG.toarray())


# ## 利用 test data裡的 X 來預測 y
# print(knn_model.predict(X_test))
# ## 查看實際y
# print(y_test.values.ravel())
# ## 這是 test data X 預測y是什麼的機率 
# print(knn_model.predict_proba(X_test))
# ## 模型預測的準確率(Accuracy)
# print("radius = 2 , score :",knn_model.score(X_test, y_test)) ## 0.8333333333333334
# print("Accuracy: ",knn_model.score(X_test, y_test)*100,"%")


## 優化
## KNN Classifier 參數設
knn_model = RadiusNeighborsClassifier(radius=1, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
knn_model.fit(X_train, y_train.values.ravel())
print("radius = 1, score :",knn_model.score(X_test, y_test)) # 1.0
print("Accuracy: ",knn_model.score(X_test, y_test)*100,"%")
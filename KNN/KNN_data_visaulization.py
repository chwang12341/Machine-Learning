## KNN Data Visualization
## import package
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
## 導入畫圖套件
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
## 導入Iris 數據集
iris_raw_dataset = datasets.load_iris()
## 使用數據data 的前兩列 來預測
X = pd.DataFrame(iris_raw_dataset.data[:, :2])
y = pd.DataFrame(iris_raw_dataset.target)
# 將數據分割為Train Data 與 Test Data
# test_size: 代表你要切多少比例給test data，剩下的就會成為train data
# random_state: 隨機數種子，設定為一樣的隨機樹種子，被隨機分割出來的data編號才會一樣喔
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.2, random_state = 0)
## Train data Data Visualization - - - - - - - - - - - - - - - - - - 
## 每一個距離大小 step size
ss = 0.06
## Coler Map
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']) 
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
##建立 KNN Model
knn_model = KNeighborsClassifier(n_neighbors = 10, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
## df transform to np.array(for the next step)
X_array = X_train.values
y_array = y_train.values
## Train model
knn_model.fit(X_array , y_array )
## 創建測試數據矩陣
## 抽取X data裡的 第一列當x軸，第二列當y軸
x_min, x_max = X_array[:, 0].min() - 1, X_array[:, 0].max() + 1 
y_min, y_max = X_array[:, 1].min() - 1, X_array[:, 1].max() + 1 
xx, yy = np.meshgrid(np.arange(x_min, x_max, ss), np.arange(y_min, y_max, ss))
## (預測的數據狀況)
Z_predict_data = knn_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z_predict_data = Z_predict_data.reshape(xx.shape)
plt.figure(figsize=(12, 8))
## 預測的範圍圖
plt.pcolormesh(xx, yy, Z_predict_data, cmap=cmap_light)
## 實際的數據
plt.scatter(X_array[:, 0], X_array[:, 1], c=y_array.ravel(), cmap=cmap_bold)
## 設定圖片名稱跟邊界
plt.title("Train Data data visualization")
plt.xlim(xx.min(), xx.max()) 
plt.ylim(yy.min(), yy.max())
plt.show()
## Test data Data Visualization - - - - - - - - - - - - - - - - - - 
## 每一個距離大小 step size
ss = 0.06
## Coler Map
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']) 
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
##建立 KNN Model
knn_model = KNeighborsClassifier(n_neighbors = 10, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
## df transform to np.array(for the next step)
X_array = X_test.values
y_array = y_test.values
## Train model
knn_model.fit(X_array , y_array )
## Predict Accuracy
print(knn_model.score(X_test, y_test)) #0.7666666666666667
## 創建測試數據矩陣
## 抽取X data裡的 第一列當x軸，第二列當y軸
x_min, x_max = X_array[:, 0].min() - 1, X_array[:, 0].max() + 1 
y_min, y_max = X_array[:, 1].min() - 1, X_array[:, 1].max() + 1 
xx, yy = np.meshgrid(np.arange(x_min, x_max, ss), np.arange(y_min, y_max, ss))
## (預測的數據狀況)
Z_predict_data = knn_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z_predict_data = Z_predict_data.reshape(xx.shape)
plt.figure(figsize=(12, 8))
## 預測的範圍圖
plt.pcolormesh(xx, yy, Z_predict_data, cmap=cmap_light)
## 實際的數據
plt.scatter(X_array[:, 0], X_array[:, 1], c=y_array.ravel(), cmap=cmap_bold)
## 設定圖片名稱跟邊界
plt.title("Test Data data visualization")
plt.xlim(xx.min(), xx.max()) 
plt.ylim(yy.min(), yy.max())
plt.show()
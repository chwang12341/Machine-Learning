## 導入交叉驗證(Cross Validation)的套件
from sklearn.model_selection import cross_val_score
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
## 導入Iris Data 當我們這次實作的數據庫
iris_flower = datasets.load_iris()
X = iris_flower.data
y = iris_flower.target
##建立 KNN Model
## KNN Classifier 參數設定
knn_model = KNeighborsClassifier(n_neighbors = 10, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
## 交叉驗證(Cross Validation)法實作
accuracy = cross_val_score(knn_model, X, y, cv=10, scoring="accuracy")
print(accuracy)
print(accuracy.mean()*100,'%')
## 找最佳K值
## 設定欲找尋的k值範圍
k_value_range = range(3,34)
## 裝測試結果的平均分數
k_value_scores = []
for k in k_value_range:
    knn_model = KNeighborsClassifier(n_neighbors = k, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
    accuracy = cross_val_score(knn_model, X, y, cv=10, scoring="accuracy")
    print("K值: "+ str(k) +" Accuracy: "+ str(accuracy.mean()))
    k_value_scores.append(accuracy.mean())
 
## Data Visualization
import matplotlib.pyplot as plt
 
plt.plot(k_value_range,k_value_scores, marker = 'o')
plt.title("找尋最佳KNN裡的K值", fontproperties="SimSun")
plt.xlabel('K 值', fontproperties="SimSun")
plt.ylabel('Accuracy')
plt.show()
### Machine Learning-交叉驗證(Cross Validation)-找到KNN中適合的K值-Scikit Learn一步一步實作教學



[Github完整程式碼](https://github.com/chwang12341/Machine-Learning/tree/master/Cross-Validation)

yoyo~~ 相信大家在實作KNN的過程中，可能會跟我一樣遇到k值如何選定的問題，這邊就來介紹一下我在網路上找到的方法-交叉驗證(cross validation)，它可以幫助我們分析k值取多少會得到多高的準確度(Accuracy)，很厲害的

#### 1. 交叉驗證(Cross validation)是什麼?


簡單來說就是將整個數據集(Dataset)切成許多集(組)，一部份的組作為訓練集，另一部分做為測試集，達到訓練與評價模型的效果

#### 2. 為什麼需要交叉驗證?

a. 如果我們只用一組的訓練集來訓練模型，會導致我們的結果可能會有偏差，也就是換了訓練集資料後，訓練出來的模型預測能力可能不同，但都是源自同一個資料集來創建模型，更有效的方法是，將不同的資料組輪流成為訓練集，然後平均得出來的結果，就能更客觀的了解模型的預測能力
b. 我們的數據集不夠大量，有限的資源下要獲得更多的資訊 
c. 可以找到合適的模型或參數，像是找到KNN中符合資料的K值，讓機器學習模型表現最好

### 3. 交叉驗證的方法?

> a.留出法 (holdout cross validation)
> b. k折交叉驗證法 (k-fold Cross Validation)
> c.留一法 (Leave one out cross validation) 
> d. Bootstrap Sampling

#### 4. 留出法 (holdout cross validation)

![img](https://cdn-images-1.medium.com/max/800/1*84xdjxSoxBsJZ-p0oLecsA.png)



a. 說明: 最簡單的交叉驗證方法，直接將原始數據隨機切成三等份，訓練集、驗證集和測試集
b. 缺點: 
i.如果只做一次劃分，切割三份的比例，與切割後的分布是否與原數據相同等因素很難拿捏
ii. 不同的劃分方式，會導致不同的最佳模型
iii. 用來訓練模型的數據(訓練集)更少了

#### 5. k折交叉驗證法 (k-fold Cross Validation)



![img](https://cdn-images-1.medium.com/max/800/1*HlngPGEn9OoqbhHDlcpRNw.png)

a. 說明: 改進了留出法對數據劃分可能存在的缺點，首先將數據集切割成k組，然後輪流在k組中挑選一組作為測試集，其它都為訓練集，然後執行測試，進行了k次後，將每次的測試結果平均起來，就為在執行k折交叉驗證法 (k-fold Cross Validation)下模型的性能指標



b. k取值: 通常都取10，資料量大時，建議設小，反之設大，這樣在資料數量小的時候能多分一點數據給訓練集來訓練模型

#### 6. 留一法 (Leave one out cross validation)  

![img](https://cdn-images-1.medium.com/max/800/1*N8nIrLrmz014X7oNKcIwaA.png)

a. 說明: 很像k折交叉驗證法，但把k值設為樣本數量T，然後每次測試集只有一個樣本，其餘樣本的是訓練集，進行T次的訓練與預測後，得到評價模型結果
b. 使用時機: 數據量真的很少的時候
c. 缺點: 數據量大的話，需要大量的運算

#### 7. 自助抽樣法 (Bootstrap Sampling) 

![img](https://cdn-images-1.medium.com/max/800/1*xukaTv4vTeco9CdTAlus2g.png)


a. 說明: 在有N個數據資料中，進行N次的隨機抽出組成一個新的數據集，由於時隨機所以可能有資料被重複抽出，有的完全沒有被選中，這時組成的新數據集即為訓練集，有大概36.8%的數據不會被選中，沒有被抽中的數據為驗證集


b. 使用時機: 除非數據量真的非常少，不然通常不會被使用


c. 優點: 會有36.8%的數據不會被選成訓練集，對於資料量小的的數據集，不用切成更小的測試集，也就不會影響模型的預測能力


d. 缺點: 訓練集的分布與原始數據一定不同，需要引入估計偏差的方法



#### 8. 如何利用交叉驗證(Cross Validation)來找尋最佳KNN中的K值?



#### 實作



在Machine Learning-KNN演算法- Python實作-Scikit Learn一步一步實作教學 這篇文章中我們一起學習了如何實作KNN，這邊我就不對KNN的套件多做介紹，直接使用喔

**a. 導入交叉驗證(Cross Validation)的套件**

```
from sklearn.model_selection import cross_val_score
```

**b. 導入KNN的套件 與 數據處理所需的套件**

```
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
```

**c. 導入Iris Data 當我們這次實作的數據庫**

```
iris_flower = datasets.load_iris()
X = iris_flower.data
y = iris_flower.target
```

**d. 建立KNN模型**

```
##建立 KNN Model
## KNN Classifier 參數設定
knn_model = KNeighborsClassifier(n_neighbors = 10, weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’, metric_params=None, n_jobs=1)
```

**e. 交叉驗證(Cross Validation)法實作**

i. cv: 將資料分成幾組，如上面我們提到的k折交叉驗證法 (k-fold Cross Validation)來實作交叉驗證(Cross Validation)的功能
ii. scoring: 得分方式 ，accuracy: 為一種方法，用以顯示準確度得值(越高越準確)

**小筆記: scoring除了有’accuracy’，也有計算平均方差的'mean_squared_error', 這個方法通常是用在評斷回歸模型的好壞**

```
accuracy = cross_val_score(knn_model, X, y, cv=10, scoring=”accuracy”)
print(accuracy)
print(accuracy.mean()*100,’%’)
```

 iii. 執行結果

![img](https://cdn-images-1.medium.com/max/800/1*ERIZIN90cw3vA89cY-behg.png)



vi. 可以從accuracy串列中，看到有些組是100%，有些組則是大概93%，也就是說如果我們剛好取的是第一組，就會誤以為模型準確度(Accuracy)一定是100%，但實際上只是剛好取到這樣的資料組合





#### **f. 最佳K值**


 
i. 找到最佳K值，也就是KNN模型參數裡面的n_neighbors值
ii. 我這邊想設定我查找的k值範圍落在3~33，也就是說我想知道當K值等於3~33的時候，模型表現的狀況，來決定我的最佳K值

```
## 設定欲找尋的k值範圍
k_value_range = range(3,34)
## 裝測試結果的平均分數
k_value_scores = []
for k in k_value_range:
 knn_model = KNeighborsClassifier(n_neighbors = k, weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’, metric_params=None, n_jobs=1)
 accuracy = cross_val_score(knn_model, X, y, cv=10, scoring=”accuracy”)
 print(“K值: “, k)
 print(“Accuracy: “, accuracy.mean())
 k_value_scores.append(accuracy.mean())
print(k_value_scores)
## 找一個最佳的k值，由於k的初始值我們設在3，所以要加三
print("最佳K值: " ,k_value_scores.index(max(k_value_scores))+3)
```


**g. 資料視覺化**

i. fontproperties=”SimSun”: 宋體，解決中文顯示問題

```
## Data Visualization
import matplotlib.pyplot as plt
 
plt.plot(k_value_range,k_value_scores, marker = ‘o’)
plt.title(“找尋最佳KNN裡的K值”, fontproperties=”SimSun”)
plt.xlabel(‘K 值’, fontproperties=”SimSun”)
plt.ylabel(‘Accuracy’)
plt.show()
```

**h. 完整程式碼**

```
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
knn_model = KNeighborsClassifier(n_neighbors = 10, weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’, metric_params=None, n_jobs=1)
## 交叉驗證(Cross Validation)法實作
accuracy = cross_val_score(knn_model, X, y, cv=10, scoring=”accuracy”)
print(accuracy)
print(accuracy.mean()*100,’%’)
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
print(k_value_scores)
## 找一個最佳的k值，由於k的初始值我們設在3，所以要加三
print("最佳K值: " ,k_value_scores.index(max(k_value_scores))+3)
## Data Visualization
import matplotlib.pyplot as plt
 
plt.plot(k_value_range,k_value_scores, marker = ‘o’)
plt.title(“找尋最佳KNN裡的K值”, fontproperties=”SimSun”)
plt.xlabel(‘K 值’, fontproperties=”SimSun”)
plt.ylabel(‘Accuracy’)
plt.show()
```

**i. 執行結果**



```
[1.         0.93333333 1.         1.         1.         0.86666667
 0.93333333 0.93333333 1.         1.        ]
96.66666666666669 %
K值: 3 Accuracy: 0.9666666666666666
K值: 4 Accuracy: 0.9666666666666666
K值: 5 Accuracy: 0.9666666666666668
K值: 6 Accuracy: 0.9666666666666668
K值: 7 Accuracy: 0.9666666666666668
K值: 8 Accuracy: 0.9666666666666668
K值: 9 Accuracy: 0.9733333333333334
K值: 10 Accuracy: 0.9666666666666668
K值: 11 Accuracy: 0.9666666666666668
K值: 12 Accuracy: 0.9733333333333334
K值: 13 Accuracy: 0.9800000000000001
K值: 14 Accuracy: 0.9733333333333334
K值: 15 Accuracy: 0.9733333333333334
K值: 16 Accuracy: 0.9733333333333334
K值: 17 Accuracy: 0.9733333333333334
K值: 18 Accuracy: 0.9800000000000001
K值: 19 Accuracy: 0.9733333333333334
K值: 20 Accuracy: 0.9800000000000001
K值: 21 Accuracy: 0.9666666666666666
K值: 22 Accuracy: 0.9666666666666666
K值: 23 Accuracy: 0.9733333333333334
K值: 24 Accuracy: 0.96
K值: 25 Accuracy: 0.9666666666666666
K值: 26 Accuracy: 0.96
K值: 27 Accuracy: 0.9666666666666666
K值: 28 Accuracy: 0.9533333333333334
K值: 29 Accuracy: 0.9533333333333334
K值: 30 Accuracy: 0.9533333333333334
K值: 31 Accuracy: 0.9466666666666667
K值: 32 Accuracy: 0.9466666666666667
K值: 33 Accuracy: 0.9466666666666667
```

![img](https://cdn-images-1.medium.com/max/800/1*dZp8D4LyigyzErgAfGXctA.png)

**可以從印出的數據與圖中找到在我們設定的範圍中(3~33)最佳的K值為13、18跟20，他們的Accuracy為0.9800000000000001，數據圖中可以清楚看到有三個點有最高的準確度(Accuracy)，所以就取它們之一當K值囉**



> 這樣我們就可以驗證我們的模型囉!! 也可以找到最佳的K值(參數)，來幫助我們建立最佳的預測模型!! 希望有幫助到大家，可以開始驗證起來





## Reference

[**K-Fold Cross Validation and GridSearchCV in Scikit-Learn**
*Python is one of the most popular open-source languages for data analysis (along with R), and for good reason. With…*randomforests.wordpress.com](https://randomforests.wordpress.com/2014/02/02/basics-of-k-fold-cross-validation-and-gridsearchcv-in-scikit-learn/)

[**[Day29\]機器學習：交叉驗證！ - iT 邦幫忙::一起幫忙解決難題，拯救 IT 人的一天**
*倒數第二天！ 今天要來看機器學習中很重要的 交叉驗證(Cross validation) ： 一般來說我們會將數據分為兩個部分，一部分用來訓練，一部分用來測試，交叉驗證是一種統計學上將樣本切割成多個小子集的做測試與訓練。…*ithelp.ithome.com.tw](https://ithelp.ithome.com.tw/articles/10197461)


https://kknews.cc/code/994o5g5.html
# Machine Learning - 標準化（StandardScaler）- 快速完成數據標準化 - Sklearn 套件教學 



哈囉哈囉，不知道大家在閱讀這篇前，有先閱讀過我的上一篇針對標準化說明的文章嗎XD，那篇詳細說明了標準化是什麼、有什麼用、標準化種類介紹及公式計算，Python程式碼實作等等，而這篇就不會重複上一篇的觀念喔，這一篇會帶大家使用Sklearn（強大的的機器學習套件）來進行各種標準化的實作喔，那讓我們開始吧！！





## 標準化的步驟 



#### Step1: 導入數據集與所需的套件 



+ 這邊我使用的是鼎鼎大名的鳶尾花數據集（Iris Dataset），相信大家進行機器學習練習的路上，時常會以它來當作數據集，所以對它很不陌生 

+ 簡單介紹一下鳶尾花數據集，數據集中有 "sepal_length", "sepal_width", "petal_length", "petal_width", "species", "species_id"這些特徵欄位，也就是有花萼長度與寬度、花瓣的長度與寬度，和花的種類與標籤編號這些特徵資料，非常適合進行分類問題的分析，像是Machine Learning中的KNN近鄰演算法，這我也會在之後的文章與大家介紹喔

```Python
## 導入sklearn 標準化套件
from sklearn import preprocessing

## 導入繪圖套件
import plotly.express as px

## 導入數據處理套件
import pandas as pd

## 使用Plotly Express的內建數據集
iris_data = px.data.iris()

## 顯示數據
iris_data
```

**執行結果**

|      | sepal_length | sepal_width | petal_length | petal_width |   species | species_id |
| :--- | -----------: | ----------: | -----------: | ----------: | --------: | ---------: |
| 0    |          5.1 |         3.5 |          1.4 |         0.2 |    setosa |          1 |
| 1    |          4.9 |         3.0 |          1.4 |         0.2 |    setosa |          1 |
| 2    |          4.7 |         3.2 |          1.3 |         0.2 |    setosa |          1 |
| 3    |          4.6 |         3.1 |          1.5 |         0.2 |    setosa |          1 |
| 4    |          5.0 |         3.6 |          1.4 |         0.2 |    setosa |          1 |
| ...  |          ... |         ... |          ... |         ... |       ... |        ... |
| 145  |          6.7 |         3.0 |          5.2 |         2.3 | virginica |          3 |
| 146  |          6.3 |         2.5 |          5.0 |         1.9 | virginica |          3 |
| 147  |          6.5 |         3.0 |          5.2 |         2.0 | virginica |          3 |
| 148  |          6.2 |         3.4 |          5.4 |         2.3 | virginica |          3 |
| 149  |          5.9 |         3.0 |          5.1 |         1.8 | virginica |          3 |

150 rows × 6 columns



![image1](images\image1.PNG)





#### Step2: 原始資料進行繪圖 



+ 這邊我拿'sepal_length' 與 'petal_length' 特徵來當X軸與Y軸，顏色用'species'來根據種類繪製不同顏色，大家也可以自己試試拿不同的特徵欄位來當X軸與Y軸喔  

```Python
## 這邊我拿'sepal_length' 與 'petal_length' 特徵來當X軸與Y軸，顏色用'species'來根據種類繪製不同顏色 
## 原始資料進行繪圖
fig = px.scatter(iris_data, x = 'sepal_length', y = 'petal_length', color = 'species')

## 顯示圖像
fig.show()
```



**執行結果**



![image2](images\image2.PNG)







#### Step3: 標準化前的數據準備 



+ 由於進行標準化的過程中，我們輸入進去的數據不可以是字串（string）形式，所以我們需要把species這個特徵欄位拿掉，然而species_id這個特徵欄位雖然是數值，但是它不需要進行標準化，所以也拿掉 

+ 拿掉species這個特徵欄位的同時，記得將其保存於一個變數之中，它會是我們後面繪圖，需要用來以顏色區別鳶尾花種類的依據



```Python
## 將species特徵欄位數據獨立出來，目的是讓正規化後的圖，可以用嚴肅想是物種不同
iris_species = iris_data.loc[:,'species']

## 將沒辦法進行標準化的特徵欄位先拿出來
iris_data = iris_data.drop(['species','species_id'], axis = 1)

## 顯示數據
iris_data
```

**執行結果**



![image3](images\image3.PNG)



#### Step4: 標準化（Standard Scaler） 



###### 方法一： Z-Score 將數據呈現正態分佈與中心化 



+ 說明：數據將會在進行標準化（Z-Score）後，會呈現正態分佈喔，而且標準化後的數據集平均值為0標準差為1  （詳細的說明與算法會在前一篇我針對標準化文章的介紹喔，有興趣的大家可以參考參考） 

+ 使用套件： 使用sklearn中的preprocessing.StandardScaler()來執行

+ 函數格式： sklearn.preprocessing.StandardScaler(copy = True, with_mean = True, with_std = True) 



+ 參數說明

  ```
  copy: 在原始資料中進行縮放，預設為
  
  True with_mean: 在標準化縮放前，將數據的分佈進行中心化處理，預設為True 
  
  with_std: 將數據資料縮放成單位標準差，預設為True 
  ```

  

+ 程式碼範例： 我先將數據進行標準化- Z-Score後，並將"species"種類加回這個數據集中，並進行視覺化

```Python
## 使用標準化Z-Score套件
z_score_scaler = preprocessing.StandardScaler()

## 對數據進行標準化
iris_z_score = z_score_scaler.fit_transform(iris_data)

## 轉換成DataFrame
iris_z_score = pd.DataFrame(iris_z_score)

## 將species數據結合起來
iris_z_score['species'] = iris_species


print(iris_z_score)

## 繪圖，由於進行標準化後的欄位名稱會改變成數字，所以記得將X軸與Y軸的值，給予對應的數值
fig = px.scatter(iris_z_score, x = 0, y = 2, color = 'species')

## 顯示圖像
fig.show()
```

**執行結果**

![image4](images\image4.PNG)



**完整程式碼**

```Python
## 導入sklearn 標準化套件
from sklearn import preprocessing

## 導入繪圖套件
import plotly.express as px

## 導入數據處理套件
import pandas as pd

## 使用Plotly Express的內建數據集
iris_data = px.data.iris()

## 顯示數據
iris_data

## 這邊我拿'sepal_length' 與 'petal_length' 特徵來當X軸與Y軸，顏色用'species'來根據種類繪製不同顏色 
## 原始資料進行繪圖
fig = px.scatter(iris_data, x = 'sepal_length', y = 'petal_length', color = 'species')

## 顯示圖像
fig.show()

## 將species特徵欄位數據獨立出來，目的是讓正規化後的圖，可以用嚴肅想是物種不同
iris_species = iris_data.loc[:,'species']

## 將沒辦法進行標準化的特徵欄位先拿出來
iris_data = iris_data.drop(['species','species_id'], axis = 1)

## 顯示數據
iris_data

## 使用標準化Z-Score套件
z_score_scaler = preprocessing.StandardScaler()

## 對數據進行標準化
iris_z_score = z_score_scaler.fit_transform(iris_data)

## 轉換成DataFrame
iris_z_score = pd.DataFrame(iris_z_score)

## 將species數據結合起來
iris_z_score['species'] = iris_species


print(iris_z_score)

## 繪圖，由於進行標準化後的欄位名稱會改變成數字，所以記得將X軸與Y軸的值，給予對應的數值
fig = px.scatter(iris_z_score, x = 0, y = 2, color = 'species')

## 顯示圖像
fig.show()
```



+ 小叮嚀： 由於進行標準化後欄位名稱會改變成數字，所以記得將X軸與Y軸的值賦予對應的數值喔，大家可以自行使用這四個特徵來做X軸與Y軸的搭配喔 

+ 我們來看一下計算Z-Score所需的平均與標準差的值，在這四個特徵中分別為多少 

+ 從上面的結果可以看出，雖然會有一點點的誤差，但基本上進行標準化- Z-Score後的數據，平均值為0，標準差為1喔





#### 方法二： Min-Max - 線性歸一化 



+ 說明： 數據將會在進行Min-Max線性歸一化後，歸一化後的數據值會介於0～1之間 

+ 使用套件： 使用sklearn中的preprocessing.MinMaxScaler()來執行 

+ 函數格式： sklearn.preprocessing.MinMaxScaler(feature_range = (0,1), copy = True) 

+ 參數說明

  ```
  copy: 在原始資料中進行縮放，預設為True 
  
  feature_range: 設定輸出結果的數據資料值範圍，預設為（0,1），代表介於0～1之間 
  ```

+ 程式碼範例：

```Python
## 使用標準化Min-Max套件
min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0,1))

## 對數據進行標準化
iris_min_max = min_max_scaler.fit_transform(iris_data)

## 轉換為Data Frame格式
iris_min_max = pd.DataFrame(iris_min_max)

## 接回species種類特徵的資料
iris_min_max['species'] = iris_species

print(iris_min_max)

## 繪圖，由於進行標準化後的欄位名稱會改變成數字，所以記得將X軸與Y軸的值，給予對應的數值
fig = px.scatter(iris_min_max, x = 0, y = 2, color = 'species')

## 顯示圖像
fig.show()
```

**執行結果**

![image5](images\image5.PNG)



完整程式碼

```Python
## 導入sklearn 標準化套件
from sklearn import preprocessing

## 導入繪圖套件
import plotly.express as px

## 導入數據處理套件
import pandas as pd

## 使用Plotly Express的內建數據集
iris_data = px.data.iris()

## 顯示數據
iris_data

## 這邊我拿'sepal_length' 與 'petal_length' 特徵來當X軸與Y軸，顏色用'species'來根據種類繪製不同顏色 
## 原始資料進行繪圖
fig = px.scatter(iris_data, x = 'sepal_length', y = 'petal_length', color = 'species')

## 顯示圖像
fig.show()

## 將species特徵欄位數據獨立出來，目的是讓正規化後的圖，可以用嚴肅想是物種不同
iris_species = iris_data.loc[:,'species']

## 將沒辦法進行標準化的特徵欄位先拿出來
iris_data = iris_data.drop(['species','species_id'], axis = 1)

## 顯示數據
iris_data

## 使用標準化Min-Max套件
min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0,1))

## 對數據進行標準化
iris_min_max = min_max_scaler.fit_transform(iris_data)

## 轉換為Data Frame格式
iris_min_max = pd.DataFrame(iris_min_max)

## 接回species種類特徵的資料
iris_min_max['species'] = iris_species

print(iris_min_max)

## 繪圖，由於進行標準化後的欄位名稱會改變成數字，所以記得將X軸與Y軸的值，給予對應的數值
fig = px.scatter(iris_min_max, x = 0, y = 2, color = 'species')

## 顯示圖像
fig.show()
```





#### 方法三： MaxAbs 標準化 



+ 說明： 進行MaxAbs標準化後，因為計算公式的關係，當X為最大值時，標準化後等於1，最小值則會轉化為-1，所以數據會縮放到介於-1～1之前喔，這樣的方式使我們的數據分佈的型態不會有變化喔 

+ 由於這次用的數據集中不會有負數的值，所以不能呈現出介於-1～1之間的數據集數值，我在前一篇的教學中，有手動加入負值 並自行寫函式來實現MaxAbs，有興趣的大家可以參考看看XD 

+ 使用函數： 使用preprocessing.MaxAbsScaler()來執行 

+ 函數格式：sklearn.preprocessing.MaxAbsScaler(copy = True) 

+ 參數說明 

  copy: 在原始資料中進行縮放，預設為True 

+ 程式碼範例

```Python
## 使用標準化MaxAbs函數
maxabs_scaler = preprocessing.MaxAbsScaler()

## 對數據進行標準化轉換
iris_maxabs = maxabs_scaler.fit_transform(iris_data)

## 轉換成Data Frame格式
iris_maxabs = pd.DataFrame(iris_maxabs)

## 接回species特徵種類的資料
iris_maxabs['species'] = iris_species

print(iris_maxabs)

## 繪圖，由於進行標準化後的欄位名稱會改變成數字，所以記得將X軸與Y軸的值，給予對應的數值
fig = px.scatter(iris_maxabs, x = 0, y = 2, color = 'species')

## 顯示圖像
fig.show()
```

**執行結果**



![image6](images\image6.PNG)



**完整程式碼**

```Python
## 導入sklearn 標準化套件
from sklearn import preprocessing

## 導入繪圖套件
import plotly.express as px

## 導入數據處理套件
import pandas as pd

## 使用Plotly Express的內建數據集
iris_data = px.data.iris()

## 顯示數據
iris_data

## 這邊我拿'sepal_length' 與 'petal_length' 特徵來當X軸與Y軸，顏色用'species'來根據種類繪製不同顏色 
## 原始資料進行繪圖
fig = px.scatter(iris_data, x = 'sepal_length', y = 'petal_length', color = 'species')

## 顯示圖像
fig.show()

## 將species特徵欄位數據獨立出來，目的是讓正規化後的圖，可以用嚴肅想是物種不同
iris_species = iris_data.loc[:,'species']

## 將沒辦法進行標準化的特徵欄位先拿出來
iris_data = iris_data.drop(['species','species_id'], axis = 1)

## 顯示數據
iris_data

## 使用標準化MaxAbs函數
maxabs_scaler = preprocessing.MaxAbsScaler()

## 對數據進行標準化轉換
iris_maxabs = maxabs_scaler.fit_transform(iris_data)

## 轉換成Data Frame格式
iris_maxabs = pd.DataFrame(iris_maxabs)

## 接回species特徵種類的資料
iris_maxabs['species'] = iris_species

print(iris_maxabs)

## 繪圖，由於進行標準化後的欄位名稱會改變成數字，所以記得將X軸與Y軸的值，給予對應的數值
fig = px.scatter(iris_maxabs, x = 0, y = 2, color = 'species')

## 顯示圖像
fig.show()
```





#### 方法四： RobustScaler 



+ 說明： 當數據中出現異常值（離群點），如果使用Z-Score會導致數據失去這樣的特性此時就是RobustScaler的時機點，它對於數據中心化與縮放的統計資料是基於百分位數的，所以不受少量離群點的影響，它也擁有較強的參數調節能力，能針對數據中心化與縮放進行更強的調節 

+ 使用方法： 使用preprocessing.RobustScaler來執行 

+ 函數格式： sklearb.preprocessing.RobustScaler(with_centering = True, with_scaling = True, quantile_range = (25.0,75.0), copy = True) 

+ 參數說明 

  ```
  copy: 在原始資料中進行縮放，預設為True 
  
  with_centering: 在資料縮放前，先將數據資料中心化，預設為True 
  
  with_scaling: 將數據資料縮放到四分位數的範圍內，預設為True 
  
  quantile_range: 預設為（25.0,75.0），也就是IQR(Interquartile Range)的計算方法，它代表第1四分位數（前25%的分位數）與第3四分位數（前75%的分位數） 之間的範圍距離 
  ```

． 程式碼範例

```Python
## 使用標準化RobustScaler函數
rs_scaler = preprocessing.RobustScaler()

## 對數據進行標準化
iris_rs = rs_scaler.fit_transform(iris_data)

## 轉換為Data Frame格式
iris_rs = pd.DataFrame(iris_rs)

## 將species特徵種類的資料接上
iris_rs['species'] = iris_species

print(iris_rs)

## 繪圖，由於進行標準化後的欄位名稱會改變成數字，所以記得將X軸與Y軸的值，給予對應的數值
fig = px.scatter(iris_rs, x = 0, y = 2, color = 'species')

## 顯示圖像
fig.show()
```

**執行結果**



![image7](images\image7.PNG)



**完整程式碼**

```Python
## 導入sklearn 標準化套件
from sklearn import preprocessing

## 導入繪圖套件
import plotly.express as px

## 導入數據處理套件
import pandas as pd

## 使用Plotly Express的內建數據集
iris_data = px.data.iris()

## 顯示數據
iris_data

## 這邊我拿'sepal_length' 與 'petal_length' 特徵來當X軸與Y軸，顏色用'species'來根據種類繪製不同顏色 
## 原始資料進行繪圖
fig = px.scatter(iris_data, x = 'sepal_length', y = 'petal_length', color = 'species')

## 顯示圖像
fig.show()

## 將species特徵欄位數據獨立出來，目的是讓正規化後的圖，可以用嚴肅想是物種不同
iris_species = iris_data.loc[:,'species']

## 將沒辦法進行標準化的特徵欄位先拿出來
iris_data = iris_data.drop(['species','species_id'], axis = 1)

## 顯示數據
iris_data

## 使用標準化RobustScaler函數
rs_scaler = preprocessing.RobustScaler()

## 對數據進行標準化
iris_rs = rs_scaler.fit_transform(iris_data)

## 轉換為Data Frame格式
iris_rs = pd.DataFrame(iris_rs)

## 將species特徵種類的資料接上
iris_rs['species'] = iris_species

print(iris_rs)

## 繪圖，由於進行標準化後的欄位名稱會改變成數字，所以記得將X軸與Y軸的值，給予對應的數值
fig = px.scatter(iris_rs, x = 0, y = 2, color = 'species')

## 顯示圖像
fig.show()
```



+ **由於數據集中沒有什麼異常值，所以RobustScaler與MaxAbs繪製出來的圖形幾乎一樣**





**這一篇我們學會了如何使用sklearn裡面的套件來幫助我們完成數據標準化（Standard Scaler），大家可以自行試試拿拿看手邊的數據進行標準化，如果有在學習機器學習演算法的大家，也可以試試有沒有進行標準化（Standard Scaler），數據預測準確率是否有明顯差距，感謝大家的閱讀，希望有幫助到您～～**





## Reference



[https://www.itread01.com/content/1547172790.html](https://www.itread01.com/content/1547172790.html?fbclid=IwAR3wujndpUdF5o0ZdgdIGK4Yr2EJytUKQo25KNmiXbjpkup4qjCFpMO8QW8)

[http://estat.ncku.edu.tw/topic/desc](http://estat.ncku.edu.tw/topic/desc?fbclid=IwAR3N6BzprAAbmhFN89GmLdX8t5h5R4jgRWuFjPZVMcaUk06XPTq907L3ZqM) stat/base/variance.html

[https://kknews.cc/zh-tw/code8q32em4.html](https://l.facebook.com/l.php?u=https%3A%2F%2Fkknews.cc%2Fzh-tw%2Fcode8q32em4.html%3Ffbclid%3DIwAR3pQcPur7CJoDKxWFJB3rY-nZO92THWt_7D7jaPWJm4IKK6j4fZC4Z3mfA&h=AT3y8mp6sNYJUBjQcvXv5AlSvP2ZmwSUJoje6XnTfNuvD2bjmJbRfD3If3KAK_trdG-pq25EfJVrMBJZkVP4tIH3XADbj1O6by5DG9iw1QsByGXThM7md4rh4ZqcEMVl98ImUg)

[https://ithelp.ithome.com.tw/articles/10197357](https://ithelp.ithome.com.tw/articles/10197357?fbclid=IwAR2ekeNJ6uiMhi7d_BZvDakYWJIwuICdFfwwZ2X2LxoRyrl-XILjPWBfHgU)


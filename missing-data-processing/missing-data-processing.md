# Machine Learning - 給自己的機器學習筆記 - 被數據集中空空的欄位嚇到了嗎 - 數據集中的缺失值如何處理?



嗨嗨~~相信大家也有遇到過，別人傳給你一份裡面一堆空值的數據集，這時候會先放空看著數據集，然後心裡想這是什麼，數據這麼不齊全，要怎麼分析，由於工作中常用到，所以這篇想來紀錄一下如何處理缺失值?有哪些方法可以使用?



## 1.使用 Sklearn的套件



### 函數介紹
```
class sklearn.impute.SimpleImputer(*, missing_values = nan, strategy = 'mean', fill_value=None, verbose = None, copy=True, add_indicator = False
```



### 參數介紹

1. missing_values: 設定數據集中什麼代表缺失值，一般情況下DataFrame 中的空值(pd.NA)也會轉换成np.nan，所以通常都是填人np.nan(空值)

2. strategy: 設定策略，也就是如何去填補這些缺失值，有四個選項，為mean、median 、most_frequent和constant

+ mean: 由該列的均值填充，只能用在數值型數據
+ median:由該列的中位數填充，只能用在數值型數
+ most_frequent: 由該列的眾數填充，可以用在數值型和字串型數據，如果有很多這樣的值，那最小的會被傳回
+ constant: 自行定義要填充的值，會使用fill_value來設定要填入缺失值的值

3. fill_value: strategy = "constant"的時候，設定 fill_value的值來替換掉缺失值，預設情況下，用在數值型數據就會用0填充，而字符串或對象數據類型則會使用"'missing_value"來填充

4. verbose: 傳入整數，預設為0，用來控制mputer的詳細程度


5. copy: 傳入True/False，預設為True，當為True時，會對沒有填充的數據集(x)創建副本，如果為False，在任何可能的地方進行插補，但在如下的情況下，就算copy = False最後也將創建一個新的副本

+ 當x不是浮點數組
+ 當x被編碼為CSR矩陣
+ 當add indicator = True

6. add indicator:

+ 傳入True/False，預設為False，當設置為True時，會於數據後面加上n列由0跟1構成與原數據大小相等的數據，0表示非空值，而1表示該位置為空值，有點像是判斷是否有空值的索引方法

+ 如果設置為True，則MissingIndicator的轉換會會將堆疊到imputer轉换的輸出中，這可以使預測的估計器(predictive estimator)儘管在進行了imputation，但依然能夠解法缺失問題，如果某個特徵在進行和擬和/訓練的時候，沒有缺失值，那即使在變換/測試的時候有缺失值，該特徵都不會顯示在缺失指標上



### 屬性介紹

1. statistics_: 每個特徵列的填充值
2. indicator_: 用於為缺失值增加二進制指標，如果add_indicator是False，那這邊就會為None



### 實作範例



#### 參數應用

+ 所有strategy比較

```Python
## 導入SimpleImputer套件
from sklearn.impute import SimpleImputer
## 導入NumPy套件
import numpy as np

## 構建數據集
x = [[1, 2, np.nan], [28, np.nan, 36], [np.nan, 58, 66]]
print('Datasets: ', x)

## 設定SimpleImputer
## 缺失值以均值填充
imp_mean = SimpleImputer(missing_values = np.nan, strategy = 'mean')

## 缺失值以中位數填充
imp_median = SimpleImputer(missing_values = np.nan, strategy ='median')

## 缺失值以眾数填充
imp_most_frequent = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')

## 缺失值以自定義的值填充
imp_constant = SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value = 100)

## 構建訓練集
train_set = [[2, 6, 7], [9, 7, 3], [np. nan, 68, 60]]
print('Training Set: ', train_set)

## 訓練
imp_mean.fit(train_set)
imp_median.fit(train_set)
imp_most_frequent.fit(train_set)
imp_constant.fit(train_set)


## 填補缺失值的結果
print('Mean:')
print(imp_mean.transform(x))

print('Most Frequent:')
print(imp_most_frequent.transform(x))

print('Median:')
print(imp_median.transform(x))

print('Constant:')
print(imp_constant.transform(x))
```
**執行結果**

```
Datasets:  [[1, 2, nan], [28, nan, 36], [nan, 58, 66]]
Training Set:  [[2, 6, 7], [9, 7, 3], [nan, 68, 60]]
Mean:
[[ 1.          2.         23.33333333]
 [28.         27.         36.        ]
 [ 5.5        58.         66.        ]]
Most Frequent:
[[ 1.  2.  3.]
 [28.  6. 36.]
 [ 2. 58. 66.]]
Median:
[[ 1.   2.   7. ]
 [28.   7.  36. ]
 [ 5.5 58.  66. ]]
Constant:
[[  1.   2. 100.]
 [ 28. 100.  36.]
 [100.  58.  66.]]
```






+ 當設置add_indicator = True

```Python
## 導入SimpleImputer套件
from sklearn.impute import SimpleImputer
## 導入NumPy套件
import numpy as np

## 構建數據集
x = [[1, 2, np.nan], [28, np.nan, 361], [np.nan, 58, 66]]
print('Datasets: ', x)

# 設定SimpleImputer
## 缺失值以均值填充
imp_mean = SimpleImputer(missing_values = np.nan, strategy = 'mean', add_indicator = True, copy = True)

## 構建訓練集
train_set = [[2, 6, 7], [9, 7, 31], [np.nan, 68, 60]]
print('Training Set: ', train_set)

## 訓練
imp_mean.fit(train_set)

## 填補缺失值的結果
print('Mean:')
print(imp_mean.transform(x))
```
**執行結果**

```
Datasets:  [[1, 2, nan], [28, nan, 361], [nan, 58, 66]]
Training Set:  [[2, 6, 7], [9, 7, 31], [nan, 68, 60]]
Mean:
[[  1.           2.          32.66666667   0.        ]
 [ 28.          27.         361.           0.        ]
 [  5.5         58.          66.           1.        ]]
```





+ 從结果可以看出後面多了一列由0跟1组成的列，由於我們的訓練集最後一列有空值(np.nan)，所以為1，其它則為0



#### 屬性應用

+ statistics & indicator
```Pytgon
print('Statistics: ', imp_mean.statistics_) print('Indicator: ', imp_mean.indicator_)
```
**執行結果**

+ Scikit-Learn處理缺失值的方法: https://scikit-learn.org/stable/modules/classes.html?highlight=impute#module-sklearn.impute
+ Simplelmputer官網參考網址:  https://scikit-learn.org/stable/modules/generated/sklearn.impute.Simplelmputer.html#sklearn.impute.Simplelmputer





## 2. 使用Pandas的方法- DataFrame 處理



### 構建數據集


```Python
## 導入套件
import numpy as np
import pandas as pd

## 構建數據集
dataset = pd. DataFrame([[1,2,3], [8,7,9], [np.nan, 28, 66], [2, 3, 8], [np.nan, 6, 8], [1, 10, 28]], columns = ['A', 'B', 'C'])

dataset
```
**執行結果**

![1](images\1.PNG)




### 自行定義填充缺失值的值- 以固定值填充


+ 填入數值
```Python
## 將缺失值填充為整數100
dataset['A'] = dataset['A'].fillna(100)

dataset
```
**執行結果**

![2](images\2.PNG)





+ 填入字符串
```Python
## 將缺失值填充為字符串Missing value
dataset['A'] = dataset['A'].fillna('Missing Value')

dataset
```
**執行結果**

![3](images\3.PNG)





### 用均值填充

```Python
## 用A列的均值填充
dataset['A'] = dataset['A'].fillna(dataset['A'].mean())

dataset
```
**執行結果**

![4](images\4.PNG)





### 用眾數填充

```Python
## 用A列的眾數填充
dataset['A'] = dataset['A'].fillna(dataset ['A'].mode()[0])

dataset
```
**執行結果**

![5](images\5.PNG)





### 用前後的數據填充

+ 用前一筆的數據填充



**方法一**

```Python
## 用前一筆數據填充
dataset['A'] = dataset['A'].fillna(method = 'pad')

dataset
```
**執行結果**

![6](images\6.PNG)



**方法二**

```Python
## 用前一筆數據填充
dataset['A'] = dataset['A'].fillna(method = 'ffill')

dataset
```
**執行結果**

![7](images\7.PNG)




+ 用後一筆數據填充
```Python
## 用後一筆數據填充
dataset['A'] = dataset['A'].fillna(method = 'bfill')

dataset
```
**執行結果**

![8](images\8.PNG)





### 用插值法填充

```Python
## interpolate: 使用插值方法來填充，預設為method = 'linear'線性方法
dataset['A'] = dataset['A'].interpolate()

dataset
```
**執行結果**



![9](images\9.PNG)





**接下來會有一篇專門講interpolate()的用法喔**
























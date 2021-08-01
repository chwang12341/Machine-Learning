# Machine Learning - 給自己的機器學習筆記 - Kaggle競賽必備!! - LightGBM (Light Gradient Boosting Machine) - 實作演練 - 筆記(二)







## 5. 參數介紹



### XGBoost、LightGBM、CatBoost重要參數比較



![img](https://miro.medium.com/max/2000/1*A0b_ahXOrrijazzJengwYw.png)

圖片來源: https://towardsdatascience.com/catboost-vs-light.gbm-vs-xgboost-5f93620723db







### LightGBM常用超參數說明



**處理過擬和的參數**

| 參數| 說明|
|---|---|
learning_rate | 設定學習率,這會影響每棵樹的最終結果，GBM採用每棵樹的輸出得到初始的預測值 ，接著進行疊代，learning rate控制了預測變化大小的程度，常用的值有0.1,0.001, 0.003 |
| max_depth| 設定樹的最大深度|
| num_leaves| 設定完整樹的葉子數量，預設為31|
| min_data_in_leaf | 設定一個樹葉上能擁有的最小數據數量，預設為20|



**控制速度**


| 參數| 說明|
|---|---|
| feature_fraction | 需有前提: 提升方法設定為隨機森林的時候才可以使用，設定選擇多少比例 的參數來構建樹，像是0.8，就表示每個疊代，LightGBM會隨機選擇80%的參數來構建樹 |
|bagging_fraction| 設定每個疊代用的數據比例，一般被用來加速訓練，並且可以防止過擬和
| num_iterations| 設定boosting的疊代次數，預設為100



**輸入輸出參數**

| 參數| 說明|
|---|---|
|categorical_feature|設定類別的索引，像是categorical_feature = 0, 2, 6，表示第0、第2、第6列都為類別變量 |
|max_bin| 設定特徵被分箱的最大數量|
|ignore_column| 與categorical_feature一樣，只是是設定完全忽略的類別索引|
| save_binary| 為True的時候，可以把我們的數據集存成二進位文件，下次讀取的時候，可以省掉許多的時間，也可以節省許多的内存空間 |



**度量方法**

1. metric: 設定模型的損失函數，以下為幾種常用的迴歸和分類的損失函數

+ mae: 絕對值平均誤差
+ mse: 均方誤差
+ + binary_logloss: 二元分類對數損失
+ multi_logloss: 多分類對數損失



**重要參數**

1. Task: 設定需要在數據上執行的任務、訓練或是預測

2. application: 設定模型的應用，非常重要!!有迴歸或是分類，預設為迴歸模型


+ regression: 迴歸
+ binary: 二元分類
+ multiclass: 多類分類


3. boosting: 設定執行的算法類型，預設為gbdt

+ gbdt: 傳统的梯度提升樹
+ rf: 隨機森林
+ dart: 使用dropout的多重迴歸樹相加總
+ goss: 基於梯度的單側取樣


4. num_boost_round: 設定提升的疊代數量，通常為100以上

5. device: 預設為cpu，可以傳人gpu





**其他控制參數**

1. early_stopping_round: 設定當評估方法在幾輪中沒有提升就提前停止，避免過多的疊代，而這
個參數就是設定幾輪的數量

2. lambda: 設定正則化的參數，值介於0-1

3. min_gain_to_split: 設定進行樹分裂時，最小的增長數量，用來控制樹中有用的分裂數量

4. max_cat_threshold: 預設為32，限制分類特徵考慮的分割點數


完整參數可以參考: https://lightgbm.readthedocs.io/en/latest/Parameters.html





## 6. 實作



### 安裝LightGBM
```
pip install lightgbm
```



### 導入套件

```Python
## 導入lightgbm
import lightgbm as lgb
## 導入Scikit-Learn的評量套件
from sklearn import metrics
from sklearn.metrics import mean_squared_error
## 導入Scikit-Learn的內建數據集
from sklearn. datasets import load_iris
## 導入Scikit-Learn用來拆分訓練集和測試集的套件
from sklearn.model_selection import train_test_split
```



### 載入數據集

```Python
## 載入數據集
iris_dataset = load_iris()
data = iris_dataset.data
target = iris_dataset.target

## 拆分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split (data, target, test_size = 0.3)
```





### 將數據集轉成gb特徵的數據集格式

```Python
## 創建成符合lgb特徵的數據集格式
## 將數據保存成LightGBM二進位文件，加載速度更快，占用更少內存空間
## 訓練集
lgb_train = lgb.Dataset(X_train, y_train)

## 測試集
lgb_test = lgb.Dataset(X_test, y_test, reference = lgb_train)
```





### 設定Igb參數

```Python
## 撰寫訓練用的參數
params = {
    'task': 'train',
    ## 算法類型
    'boosting': 'gbdt',
    'num_trees': 100,
    'num_leaves': 20,
    'max_depth': 6,
    'learning_rate': 0.04,
    ## 構建樹時的特徵選擇比例
    'feature_fraction': 0.5,
    'feature_fraction_seed': 8,
    "bagging_fraction":0.5,
    ## k 表示每k次迭代就進行bagging
    'bagging_freq':5,
    ## 如果數據集樣本分布不均衡，可以幫助明顯提高準確率
    'is_unbalance': True,
    'verbose':0,
    ## 目標函數
    'objective': 'regression',
    ## 度量指標
    'metric': {'rmse', 'auc'},
    # 度量輸出的頻率
    'metric_freq': 1,
}
```





### 訓練模型

```Python

## 訓練模型
test_results = {}
lgbm = lgb.train(params, lgb_train, valid_sets = lgb_test, num_boost_round = 20, early stopping_rounds = 5, evals_result = test_results)

## 保存模型
lgbm.save_model('save_model.txt')
```





### 預測測試集

```Python
## 預測測試集
## 在訓練期間有啟動early_stopping_rounds， 就可以透過best_iteration來從最佳送代中獲得預測結果
y_pred = lgbm.predict(X_test, num_iteration = lgbm.best_iteration)
print(y_pred)
```

**執行結果**

```
[0.87122796 1.2157111  0.87122796 0.87122796 1.2157111  1.2157111
 1.2157111  1.05520253 0.87122796 0.87122796 0.87122796 0.87122796
 1.05520253 0.87122796 0.87122796 1.05520253 1.05520253 0.92051469
 1.2157111  0.87122796 1.2157111  1.05520253 0.87122796 1.05520253
 0.87122796 0.87122796 0.91811222 1.2157111  1.2157111  0.87122796
 1.2157111  1.2157111  1.2157111  0.96739896 0.92051469 1.2157111
 1.2157111  1.2157111  1.05520253 1.2157111  1.05520253 0.92051469
 1.05520253 0.87122796 1.2157111 ]
```





### 評估模型好壞

```Python
## 評估模型的好壞
## RMSE
rmse = mean_squared_error (y_test, y_pred) ** 0.5
print('RMSE of the model: ', rmse)
```

**執行結果**

```
RMSE of the model:  0.8199142880537924
```





### 視覺化



**安裝套件**

```
pip install graphviz
```



**視覺化**

```Python
## 視覺化
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (28, 30))
ax = fig.subplots()
lgb.plot_tree(lgbm, tree_index = 1, ax = ax) plt.show()
```



**視覺化-訓練結果**

```Python
## 訓練結果視覺化
ax = lgb.plot_metric(test_results, metric = 'auc')
plt.show()
```

**執行結果**

![image4](images\image4.png)



**視覺化 - 特徵重要性排序**

```Python
## 視覺化 • 特徵重要性排序
ax = lgb.plot_importance(Igbm, max_num_features = 10)
plt.show()
```

**執行結果**

![image5](images\image5.png)





**大功告成！！恭喜恭喜**





## Reference

https://en.wikipedia.org/wiki/LightGBM
論文： 1999 REITZ LECTURE GREEDY FUNCTION APPROXIMATION: A GRADIENT BOOSTING MACHINE1 By Jerome H. Friedman
https://zhuanlan.zhihu.com/p/52583923
https://kknews.cc/zh-tw/tech/y3a3x8j.html
https://github.com/denistanjingyu/LTA-Mobility-Sensing-Project
https://codertw.com/程式語言/510420/
https://lightgbm.readthedocs.io/en/latest/Parameters.html https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db


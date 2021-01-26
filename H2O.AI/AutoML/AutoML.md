# Machine Learning 機器學習 - H2O.AI - AutoML (Automatic Machine Learning) 自動機器學習 - 不知道手邊數據該使用什麼模型來分析嗎? - 讓H2OAutoML來幫助您 - 超強大的 H2OAutoML 教學筆記



## 1. 動機

嗨~~ 今天來跟大家介紹我最近學習的一個超強大的方法 - AutoML (Automatic Machine Learning)，我們過去要建立機器學習模型時，總是要好好思考著這個數據到底需要用什麼演算法來分析好，而終於選則好演算法，也訓練好後，卻不知道它是不是最佳的解，但難道要我們一個一個演算法去嘗試看看嗎，看誰的效能最好嗎?





## 2. AutoML (Automatic Machine Learning) 自動機器學習





+ 簡單來說它幫助我們一次性的比較各類型演算法模型的表現，並根據評估指標進行排名，讓我們可以清楚地瞭解應該用什麼機器學習的演算法模型
+ 讓非專業的人士也能夠輕鬆進行機器學習實驗







## 3. H2OAutoML() 參數介紹

+ H2OAutoML() 函式



### 必須有的停止參數



|參數|說明|
|---|---|
|max_runtime_secs|設置執行時間，AutoML 執行到最後產出報告 Stacked Ensemble 模型的時間，預設為0，表示沒有時間限制，但如果max_runtme_secs和max_models都沒有設定的話，它會動態調整到1小時|
|max_models|設定 AutoML 執行模型的最大數量，不包含 Stacked Ensembble 模型，預設為 NULL / None|

**補充:** Stacked Ensembles 是將很多不同的演算法組合，造就最佳的模型表現，所以它在 H2O 產出的排行榜都是第一





### 可選的參數

|參數|說明|
|---|---|
|nfolds|設定 k-fold cross-validation (k折交叉驗證)的 k 值，但設定值要大於2，預設為5，如果為0，則將禁用交叉驗證法，與禁用 Stacked Ensembles (因此降低了總體最佳模型的表現)|
|balance_classes|設定是否對類別少的數據進行過度採樣(oversample)，來平衡類別的分佈，預設情況下關閉此選項的功能，並且可以增加數據框(data frame)的大小，這個選項只適用於分類(classification)問題，如果數據集過度採樣(oversample)超過了使用 max_after_balance_size 參數計算出來的最大值，那多數的類別就會被減少採樣(undersample)來滿足限制|
|class_sampling_factors|設定按每個類別(依照辭典順序)的過度 / 減少採樣(over / under-sampling)比率，預設的情況下，在訓練過程中會自動計算能讓類別平衡的比率，但前提是須設置 balance_classes = True|
|max_after_balance_size|設置平衡好的類別數量後，訓練數據的最大相對大小 (前提: 必須啟動 balance_classes)，預設為5.0，傳入值可以小於1.0|
|max_runtime_secs_per_model|設置 AutoML 運行中訓練每個模型的最長時間，預設為0(表示禁用，也就是沒有限制)，要注意一點，設定此參數可能會影響 AutoML 的重現性(reproducibility)|
|stopping_metric|設定用於早期停止的指標，預設為Auto，可以選擇的選項有:|
+ Auto: 預設情況下，logloss用於分類(classification)問題，deviance用於迴歸(regression)問題
+ deviance(平均殘餘偏差)
+ logloss
+ MSE
+ RMSE
+ MAE
+ RMSLE
+ AUC: 位於 ROC 曲線底下的面積
+ AUCPR: 位於 Precision-Recall曲線底下的面積
+ lift_top_group
+ misclassification
+ mean_per_class_error

|參數|說明|
|---|---|
|stopping_tolerance|設置用於基於度量(metric-based)停止標準的相對公差，用來停止 AutoML 執行時間時的網格搜尋和訓練個別模型，如果數據集至少有一百萬行，則設置為0.001 ; 否則，此值取決於數據集的大小和非 NA 率(non-NA-rate)，在這種情況下，此值將計算為 1 / sqrt(nrows * non-NA-rat)|
|stopping_rounds|設定回合數，根據移動平均值來檢查，當停止的度量指標(e.g. AUC)沒有在指定的回合數裡有所提升，就會停止模型訓練，在 AutoML 中，這個控制可以提早將隨機網格搜尋(random grid searches)和各別模型停止，預設值為3，必須要是非負整數(non-negative integer)，如果要完全禁用，就設置為0|
|sort_metric|設定在 AutoML運行結束時，用於對最終排行榜進行排序的度量指標，可以有的選項有:|

+ AUTO: 預設情況下，AUC會用二進制分類(binary classification)，mean_per_class_error 用在多項式迴歸，而 deviance 用在迴歸(regression)
+ deviance: 平均殘餘偏差 (mean residual deviance)
+ logloss
+ MSE
+ RMSE
+ MAE
+ RMSLE
+ AUC: 位於 ROC 曲線底下的面積
+ AUCPR: Precision-Recall 曲線底下的面積
+ mean_per_class_error


|參數|說明|
|---|---|
|seed|設置可重複(重現)性的種子，必須為整數，預設值為 NULL / None，但 AutoML 僅能在某些條件下保證可重現性，因為性能關係，預設的情況下 H2O 的深度學習模型不可重現，因此如果我們需要可重現性，則 exclude_algos 必須包含 "Deep Learning"，此外必須使用 max_models，因為 max_runtime_secs 受到資源限制，這說明著，如果兩次運行之間的計算資源有所不同，則 AutoML 可以在一次運行和另一次運行之間訓練更多的模型|
|project_name|設置項目名稱，標籤指定 AutoML 項目的字符串，預設為 NULL / None，表示會根據訓練框架的 ID 自動生成項目名稱，透過多次調用 AutoML 函數中，指定相同的項目名稱，就可以訓練更多的模型並將其添加到現有的 AutoML 項目 (條件: 在隨後的運行中都使用相同的訓練框架(training frame))|
|exclude_algos|設置要跳過的算法，使用 list / vector 來裝載演算法名稱，舉個例子: exclude_algos = ["GLM", "DeepLearning", "DRF" ]，預設為 None / Null，表示當這個時候如果也未設置 include_algos 參數的選項，則將使用所有適合的 H2O 算法，它和 include_algos 互斥|
|include_algos|設置計算上要包含的算法，使用 list / vector 來裝載演算法名稱，舉個例子: include_algos = ["GLM", "DeepLearning", "DRF" ]，預設為 None / Null，表示當這個時候如果也未設置 exclude_algos 參數的選項，則將使用所有適合的 H2O 算法，它和 exclude_algos 互斥，可用的算法有:|

+ DRF: 包含了隨機森林模型(Random Forest) 和極端隨機森林(XRT)模型
+ GLM
+ XGBoost (XGBoost GBM)
+ GBM (H2O GBM)
+ DeepLearning (Fully-connected multi-layer artificial neural network)
+ StackedEnsemble



|參數|說明|
|---|---|
|modeling_plan|設置 AutoML 引擎將使用的建模步驟列表，但因為可能取決於其他因素，它們不會都被執行 |
|preprocessing|設定要執行預處理(preprocessing)的步驟列表，但目前僅支援 ["target_encoding"]|
|exploitation_ratio|設定開發階段和探勘階段的預算比率，設定值只能介於 0~1之間，預設為0，表示開發階段處於禁用階段，但目前還處於實驗性的，所以如果真的要設定值，建議嘗試0.1的預算比率比較好|
|monotone_constraints|設置代表單調約束(monotone constraints)的映射值，使用+1代表強制增加約束，-1代表減少約束|
|keep_cross_validation_predictions|設定是否保留交叉驗證(Cross-Validation)的預測，預設為False，如果我們想要對相同的 AutoML 對象進行重複運行，則要設置為True，因為在 AutoML 中需要交叉驗證(CV)預測才能建構其他的 Stack Ensemble 模型|
|keep_cross_validation_models|設定是否保留交叉驗證的模型，保留交叉驗證的模型，可能會在 H2O 集群中消耗大量的內存空間，預設為False|
|keep_cross_validatiom_fold_assignment|啟動這個選項功能來保留交叉驗證的折疊分配，預設為False|
|vernosity|可選參數，在訓練期間印出後端資訊的詳細程度，必須是"debug"、"info"、"warn"之一，預設為 NULL / None，表示禁用客戶端的日誌記錄|
|export_checkpoints_dir|設置產生的模型將自動導出到的目錄位置|







## 4. H2OAutoML.train() 參數介紹





+ H2OAutoML.train()


### 必須有的參數

|參數|說明|
|---|---|
|y|響應列的列名，也就是要成為目標值（應變數）的列名|
|training_frame|指定訓練集|


### 可選的參數

|參數|說明|
|---|---|
|x|指定預測變量(自變量)特徵的列名，以串列的形式將欲成為預測變量的列名裝進，如果除了響應變量(應變量)列名以外的列名都需要用到，就不需特別設置|
|validation_frame|設定測試集數據，當nfolds = 0的時候才適用次選項，不然 nfolds > 1 的時候會被忽略，設定它用來在早期停止個別模型和停止網格搜尋|
|leaderboard_frame|設定一個數據框(data frame)，來對最終排行榜上的模型評分，如果沒有設定的話，則排行榜會使用交叉驗證指標，或如果 nfolds = 0 關掉交叉驗證的情況下，則會從訓練框(training_frame)中自動稱成 leaderboard_frame|
|blending_frame|設定用於計算預測(predictions)的框架(frame)，該框架被當成是 Stacked Ensemble Models Metalearner 的訓練框架(training_frame)，如果有設定的話，AutoML會使用 Blending(又稱 Holdout Stacking)來訓練 Stacked Ensembles，而不是預設的交叉驗證(Cross-Validation)|
|fold_column|設定一列用於分配每次觀察的交叉驗證(Cross-Validatiom)折疊索引，它是用在 AutoML 執行個別模型時，來覆寫預設值(default)、隨機(randomized)、5折交叉驗證(5-fold cross-validation)等方案的方式|
|weight_column|設定帶有觀察權重的列，如果指定某個觀察值的權重為0，就等於從數據中排除它；如果指定為2表示將該行重複兩次，不能設置為負值|





## 5. 實作






### STEP 1: 導入 H2O 套件與啟動(初始化) H2O

```Python
## 導入H2O套件
import h2o

## 導入H2O中的AutoML套件
from h2o.automl import H2OAutoML


## 初始化H2O
h2o.init()
```

**執行結果**

![image8](images\image8.PNG)



### STEP 2: 導入數據集，並把數據集中的自變量名稱裝入串列


+ winequality-white.csv - 這是一個關於用各種特徵來預測紅酒品質的數據集喔，我會放在 Githuhb 中，大家可以自行下載

網路上也可以找到下載連結 - https://archive.ics.uci.edu/ml/datasets/wine+quality

```Python
## 導入數據集
wine_data = h2o.import_file("data/winequality-white.csv")

## 定義我們的預測變量，也就是自變量x
predictors = wine_data.columns

## 將響應變量(目標值)，也就是應變量y拿掉
predictors.remove('quality')

## 顯示數據集
wine_data
```

**執行結果**

![image9](images\image9.PNG)



### STEP 3: 將數據集切割成訓練集與測試集

+ 將數據集拆成訓練集與測試集，並設置比例為0.7

```Python
## 將數據集拆成訓練集與測試集，並設置比例為0.7，隨機種子設123456
dataset_split = wine_data.split_frame(ratios = [0.7], seed = 123456)

## 顯示分割結果
print(dataset_split)

## 設定對應的數據集給訓練集與測試集，70%給訓練集，30%給測試集
wine_train = dataset_split[0]
wine_test = dataset_split[1]

## 顯示訓練集與測試集大小
print("Training Set: ", wine_train.shape)
print("Test Set: ", wine_test.shape)

```
**執行結果**



![image10](images\image10.PNG)



### STEP 4: 導入 AutoML 套件與啟動它，來訓練各種模型

+ 這邊我設定最大的模型數量(max_models)為20，最大的執行時間(max_runtimes_secs)為200，因為這邊是迴歸問題，所以我設定 sort_metric = "deviance"
+ 因為這樣的設定可能導致還沒跑完model就達最大的執行時間(max_runtimes_secs)，所以我將設定改成最大的模型數量(max_models)為10就好

```Python
## 導入AutoML套件
from h2o.automl import H2OAutoML

## 設定H2OAutoML
aml = H2OAutoML(max_models = 20, max_runtime_secs = 200, seed = 1, sort_metric = "deviance")

## 啟動H2OAutoML來訓練模型
aml.train(x = predictors, y = 'quality', training_frame = wine_train, validation_frame = wine_test)

```

**執行結果**



![image11](images\image11.PNG)





### STEP 5: 印出模型排行榜

+ 小筆記: 最佳的模型會被存為 aml.leader 喔

```Python
## 印出模型排行榜
lb = aml.leaderboard
print(lb)

## 印出所有行數
# lb.head(rows = lb.nrows)

## 顯示最佳模型資訊
print(aml.leader)

## 顯示最佳模型的詳細資訊
print(aml.leader.metalearner)
```
**執行結果**



![image12](images\image12.PNG)

結果: 排行第一名的模型為 model_id 是 StackedEnsemble_AllModles_AutoML的模型





### STEP 6: 視覺化 - 最佳模型底下的各種演算法模型的標準化係數比較

```Python
## 取得最佳模型的model_id，也就是模型資料中的 Model Key
metalearner = h2o.get_model(aml.leader.metalearner()['name'])

## 視覺化: 最佳模型底下的各種演算法模型的標準化係數比較
metalearner.std_coef_plot()
```

執行結果

![image13](images\image13.PNG)



### 疑惑解答: 為什麼明明是抓取第一名的模型來視覺化標準化係數，裡面卻出現這麼多演算法模型的名稱，一個模型不是應該只有一種演算法?


重要筆記: 由於 Stacked Ensemble是組合了很多種的演算法來構建模型，所以它的分數也當然會比較好，下一步去印它裡面的標準化係數的大小比較圖時，才會出現那麼多的演算法名稱，一般的情況下，我們會選擇 Stacked Ensemble 以外的第一名演算法模型當成算法



### STEP 7:  拿表現最佳的模型來預測測試集，和評估最佳模型性能表現

```Python
## 拿最佳模型預測測試集資料
preds = aml.leader.predict(wine_test)
print(preds)

## 評估最佳模型的性能表現
score = aml.leader.model_performance(wine_test)
print(score)

## 關閉H2O
h2o.shutdown()

```

執行結果



![image14](images\image14.PNG)







想要更詳細的 AutoML 用法，可以直接參考官網 - https://dos.h2o.ai/h2o/latest-stable/h2o-docs/automl.html?highlight=autml，但提醒大家一件事，因為我發現 H2O 會保留舊版的線上文檔，所以還是要確定一下版本喔，今天我們一起學習了好多東西，超級強大的 AutoML 會帶給我們非常多的幫助，感謝大家的閱讀喔







## Reference

https://dos.h2o.ai/h2o/latest-stable/h2o-docs/automl.html?highlight=autml

https://kknews.cc/zh-tw/tech/qgvjzzr.html



















































































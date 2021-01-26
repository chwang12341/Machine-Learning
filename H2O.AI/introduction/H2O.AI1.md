# Machine Learning - 近年來我們不可忽略的AI公司 - Gartner評價為最具願景(VISIONARIES)的公司之一 - H2O.AI 介紹與官網文件整理 - 幫助大家快速查詢需要的功能和如何安裝與運作 - Python 環境建置與FLOW環境建置教學  - 啟動H2O.AI的init參數詳解 - 下篇





嗨嗨，對於強大的 H2O，相信大家看了上篇後，跟我一樣都躍躍欲試了，那就讓我們一起安裝與啟用 H2O吧





## 1. Python 環境中如何安裝





### Step 1: 先到下載頁面 - https://www.h2o.ai/download/，如圖點選紅色圓圈圈起來的地方，或可以自己選擇要下載的版本或方式





![downloads](images\downloads.png)







### Step 2: 點選 INSTALL IN PYTHON

大家可以根據自己習慣用的程式語言來找對應的官方教學喔





![download_python](images\download_python.png)







### Step 3: 打開 Anaconda Prompt，執行指令





1. 下載所需的套件
```
pip install request
pip install tabulate
pip install "colorama>=0.3.8"
pip install future
```

2. 下載 H2O
```
# 移除掉現有 Python版本 的 H2O 模組
pip uninstall h2o

# 安裝最新版本的 H2O Python 模組
pip install http://h2o-release.s3.amazonaws.com/h2o/rel-zermelo/2/Python/h2o-3.32.0.2-py2.py3-none-any.whl

```


小叮嚀：這邊大家要小心，因為未來可能會有更新的 H2O 版本出現，所以最後一個用來下載 H2O Python 版本的指令會需要更新喔，但大家也不用擔心，只要點進 Step 2 我所提到的頁面，就能知道官網最新版本的下載指令喔





## 2. 如何在 Python 中啟動 H2O.AI

```Python
## 導入H2O套件
import h2o


## 初始化和啟動H2O
h2o.init()

```

執行結果



![start_h2o](images\start_h2o.PNG)








執行結果會顯示各種系統資訊喔，如果大家點擊啟動後開啟的server - localhost:54321 連結，就會連到 H2O.AI 內建的 Web UI 介面 - FLOW


這樣也就能確定有安裝好 H2O.AI喔





## 3. h2o.init() 參數介紹



+ 小叮嚀: init()為初始化 H2O 的方法，裡面擁有眾多可以調整的參數，但大部分的使用情況下，使用者只需要寫 h2o.ini()即可，不需要額外的設定參數


|參數|說明|
|---|---|
|url|要連結的伺服器 URL (可以用它來替代ip + port + https)|
|ip|執行 H2O 的伺服器 IP 位置|
|port|H2O 服務器正在監聽哪個端口號|
|name|集群名稱(Cluster Name)，如果沒有設定，就會連接到現有的集群，並不會檢查名稱，如果有設定，就會只有當集群名稱匹配才會連結，如果沒有找到，並決定啟動本地端的實列，就會將其當成是集群名稱，如果設置為None，就會產生一個隨機的實列|
|https|如果設置為 True，就會透過 https:// 連接，而不是透過 http:// 連結|
|insecure|當使用 https 來連結時，將它設置為 True，將會禁用 SSL 證書驗證|
|username|當使用基本身份驗證時，所設定的用戶名稱|
|password|當使用基本身份驗證時，所設定的密碼|
|cookies|要添加到每個請求(Request)的 Cookie 列表|
|proxy|設定代理服務器的地址|
|start_h2o|如果設置為 False，則在與現存的服務器連結失敗的狀況下，不要嘗試啟動 H2O|
|n_threads|設定當啟動新的 H2O 時，要使用的內核數量|
|ice_root|設置新 H2O 服務器的臨時文件目錄位置|
|log_dir|設定啟動新實列後，要保存 H2O 日誌的目錄位置，如果連接到現有的節點就會被忽略|
|log_level|設定當啟動新實列時，H2O 記錄程序級別，預設為 INFO，選項有TRACE、DEBUG、IINFO、WARN、ERRR、FATA，當連接到現有節點時就會被忽略|
|enable_assertions|在 Java 中為新的 H2O 伺服器，啟動斷言(assertions)|
|max_mem_size|設定用於新 H2O 服務器的最大內存空間，整數輸入，單位將會被預設為 gigabytes，可以透過傳入字串來設定不同單位，像是160M，代表160 megabytes|
||筆記: 如果沒有設定 max_mem_size，則 H2O 分配的內存空間將會由 Java 虛擬機(JVM)的預設內存空間來配置，大小會根據 Java 的版本，但通常為實體機器內存空間的25%|
|min_mem_size|設定用於新 H2O 服務器的最小內存空間，整數輸入，單位將會被預設為 gigabytes，可以透過傳入字串來設定不同單位，像是160M，代表160 megabytes|
|strict_version_check|如果設定為 True，則當客戶端與服務端的版本不同時，就會報錯|
|ignore_config|設定是否要執行處理a.h2oconfig文件，預設為False|
|extra_classpath|列出當從 Python 啟動 H2O 時， Java classpath 應包含的庫(libraries)的路徑|
|kwargs|所有其他(不推薦使用的屬性)|
|jvm_custom_args|實列化JVM H2O 中，使用者自行定義的參數，忽略掉已經運行且有客戶連結的 H2O|
|bind_to_localhost|為一個標誌，用來設定是否應將 H2O 實列的訪問限制在本地端的機器(預設的情況)，或是可以從網路上的其他機器來進行訪問 H2O|





## 4. 補充: 客制化 - 初始化時設定一些系統資訊

今天我們希望配置一塊指定的內存大小，與指定的線程數給 H2O 使用，我們就需要修改 init() 裡面的參數


+ 程式碼範例: 我們想提供 H2O 實列 6GB 的內存空間，並且它只能使用四個內核
```Python
import h2o

h2o.init(nthreads = 4, max_mem_size = 6)
```




## 5. 如何安裝 H2O.AI 的編譯器 FLOW?





### Step 1: 先到下載頁面 - https://www.h2o.ai/downloads/，如圖點選紅色圈圈起來的地方



![downloads](images\downloads.png)





### Step 2: 點選 Download



![flow_download](images\flow_download.png)



### Step 3: 

打開命令提示字元(cmd)，切換(cd)到下載下來的壓縮檔目錄，並執行解壓縮

解壓縮指令
```
unzip h2o-3.32.0.1.zip
```

當然也可以手動解壓縮喔



**小叮嚀: 大家要根據自己下載下來的壓縮檔名稱修改上面的指令喔**





### Step 4: 

切換(cd)到解壓縮完的資料夾目錄裡 ex. cd h2o-3.32.0.1，接下來執行下面的指令

```
java -jar h2o.jar
```

![flow_command](images\flow_command.PNG)



### Step 5:

打開瀏覽器，進到 http://localhost:54321





![flow_interface](images\flow_interface.PNG)





進到這個頁面就算安裝完成囉




### 補充: FLOW 的內建教學怎麼使用



#### Step 6: 點選右邊的 Browse installed packs



![flow_example](images\flow_example.png)







#### Step 7: 點選 examples -> 點選大家需要的 AI 範例

這邊我點選 K-Means 來試試，就會出現非常詳細地如何使用 K-Means 教學，非常好用!!





![K-means](images\K-Means.PNG)



FLOW 的範例中詳細地記載了 AI 算法的使用步驟，真的太方便了!!執行方式也和 Jupyter Notebook看起來非常相似





確定安裝好後，也可以在 Python 環境中啟動 H2O.AI 後，我們就能開啟在 H2O.AI 的旅程，我也非常期待接下來要如何運用，那就希望大家能期待接下來的教學文了!!





## Reference

https://www.h2o.ai/

https://docs.h2o.ai/h2o/latest-stable/h2o-docs/index.html























































































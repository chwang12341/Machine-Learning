# Machine Learning - 近年來我們不可忽略的AI公司 - Gartner評價為最具願景(VISIONARIES)的公司之一 - H2O.AI 介紹與官網文件整理 - 幫助大家快速查詢需要的功能和如何安裝與運作 - Python 環境建置與FLOW環境建置教學  - 啟動H2O.AI的init參數詳解 - 上篇





YOYO~~ 今天想來跟大家介紹一個非常新且非常非常厲害的AI開發框架，我個人非常喜歡它的宗旨 - **Our mission is to make AI for everyone**，它想將AI帶給所有的人，現在只要一聽到AI開發，大家可能直覺就會想到Scikit-Learn、TensorFlow、Keras、PyTorch等等，非常眾多的開發平台，但近年來有一間公司不斷崛起，擴展速度跟飛的一樣快，也就是我們今天文章的主角 - H2O.A，大家是不是跟我一樣期待







## 1. H2O.AI 公司介紹 - Oxdata

+ Oxdata位於加利福尼亞州的山景城，並於2014年推出了一個獨立開源機器學習平台 - H2O.AI





### Gartner 評價為最具願景(VISIONARIES)的公司之一

+ 2020年 Gartner 針對數據科學和機器學習的平台分析報告(如下圖)，從過去2019、2018的報告中，可以看出 H2O.AI 在座標軸上不斷向右靠，一路被評為相當具願景的公司



![gartner](images\gartner.png)




+ H2O.AI 公司產品介紹



![products](images\products.PNG)



**產品**:

1. **H2O**: 具有線性可擴展性的完全開源的分佈式內存機器學習平台
2. **H2O Driverless AI**: 
    + **說明:** H2O無人駕駛AI透過使用自動化技術能在幾分鐘或幾小時內，而不是幾個月內完成關鍵機器學習任務，從而使資料科學家能夠更快，更高效地處理目標項目
    + **過程:** 透過提供的自動功能工程、模型驗證、模型調整、模型選擇與部屬、機器學習可解釋性、帶來自己的方法(配方)、時間序列和自動管道生成以進行模型評分
    + **目的:** H2O無人駕駛AI為公司提供了可擴展的客制化資料科學平台，來達到各行各業中每間企業的各種需求滿足

 

![driveless-ai](images\driveless-ai.png)



3. **Sparkling Water**:
    + **說明:** Sparkling Water 允許使用者可以在 H2O 中快速可擴展的機器學習算法和 Spark 的功能結合
    + **解釋:** Spark 是以MapReduce 模型繼承而來，它更有效地支援各種類型的計算，Apache Spark 是一個為速度和通用目標設計的集群計算平台(想要更瞭解Spark，可以參考這篇https://bigdatafinance.tw/index.php/tech/coding/253-spark-spark，我覺得非常厲害且完整)
    + **目的:** 結合兩個開源環境為想要使用 Spark SQL 進行查詢的客戶，提供了無縫的體驗，將結果輸入到 H2O 中以構建模型並進行預測分析，對於任何給定的問題，工具之間更好的互操作性，可提供更好的體驗給客戶



![spark](images\spark.png)




4. **H2O Wave**
+ **說明:** H2O Wave 是一個用於構建美觀且交互式AI應用程式(Application)的應用程式開發框架
    + **目的:** 借助 H2O Wave，數據科學家、機器學習工程師和軟體開發工程師，與領域的專家們合作，可以在數小時(而不是數週)內構建新的AI應用程式，以滿足公司商務上的需求



![wave](images\wave.PNG)





**圖片來源 : H2O.AI 官網**





## 2. H2O.AI 是什麼?



+ **介紹:** H2O 是具有線性可擴展性的完全開源的分佈式內存機器學習平台
+ **支援算法:** H2O 支援最廣泛使用的統計和機器學習算法，包含 gradient boosted machines、廣義線性模型(generalized linear models)、深度學習(deep learning)等
+ **行業領先:** H2O 還具有行業領先的 AutoML 功能，該功能可自動運行所有算法及它的超參數，以生成並列出最佳模型的排行榜
+ **全球使用量:** H2O平台已在全球有超過 18,000 個組織在使用，並且在 R & Python 社區中都非常的受歡迎
+ **支援語言:** H2O.AI 平台核心程式是由 Java 編寫而成的，但它的 Rest API 允許用戶使用外部程式或腳本訪問 H2O 的所有功能，平台的接口支援 Python、R、Scala、Java、JSON和CoffeeScript / JavaScript，還有一個它內建的 Web 開發介面，名為 FLOW
+ **架構描述:** 可以由多種資源導入，並支援多種程式語言，由 Java 撰寫，具快速、可擴展性、分散式的計算引擎，最後導出獨一無二的模型產品



![H2O-3-arch](images\H2O-3-arch.jpg)





## 3. H2O 提供了哪些服務(功能)



圖的左邊是它具備的所有章節大項，右邊是我整理的大項中的細項喔



![image1](images\image1.png)



![image2](images\image2.png)



![image3](images\image3.png)







因為文件會隨著時間可能會增加很多功能，所以大家可以直接參考官網的文檔目錄喔!!

線上文檔: https://docs.h2o.ai/h2o/latest-stable/h2o-docs/index.html

有任何問題也可以到官網的討論區提問喔: https://stackoverflow.com/questions/tagged/h2o







## 4. Algorithm - H2O 有哪些演算法



![Algorithm](images\Algorithm.png)





對於這間被評為最具願景，但名氣卻低調到讓很多人可能沒聽過的公司，我真的覺得太扯了XD，H2O.AI 真的非常強大，有了對這間公司的瞭解後，大家是不是也很興奮地要給它好好研究一番XD，想再次強調我真的很喜歡的宗旨 - **Our mission is to make AI for everyone**，也勉勵自己，希望有一天也能夠做出那麼厲害的產品





## Reference

https://www.h2o.ai/

https://docs.h2o.ai/h2o/latest-stable/h2o-docs/index.html



























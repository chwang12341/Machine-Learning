# Machine Learning - 給自己的機器學習筆記 - Kaggle競賽必備!! - LightGBM (Light Gradient Boosting Machine) - 介紹與原理 - 筆記(一)





## 1. LightGBM是什麼?



+ 全名Light Gradient Boosting Machine

+ 由**微軟**公司於2017年四月釋出的
+ 為一款基於決策樹(Decision Tree)學習算法的梯度提升框架
+ 具有快速、分布式和高性能的特性


在論文中，作者說明他們是如何開發出LightGBM的方法的，他們在傳統的演算法 - Gradient Boosting Decision Tree(GBDT)上添加他們提出的Gradient-based One-Side Sampling (GOSS) 和 Exclusive Feature Bunding (EFB)方法，就成為了LightGBM

**補充:**

+ Gradient-based One-Side Sampling (GOSS): 為一種有選擇性的取樣，方法只取樣梯度(gradient) 較大的數據點來計算information gain,並省略掉其他的數據點，由於gradient較大的數據對計算 information gain有較多的影響力，所以就省略掉梯度較小的數據點，而這也被作者證明不太會對準確度有影響

+ Exclusive Feature Bundling (EFB): 绑定互斥的特徵(Mutually Exclusive Feature)來降低維度，作者表示NP-Hard Problem為直接找出最佳化绑定的方式，但是Greedy算法可以獲得準確度接近的近似解

Mutually Exclusive Feature: 兩個很少同時採用非零值的特徵(two features that rarely take nonzero values simultaneously)







## 2. Boosting演算法的發展歷史 - XGBoost、LightGBM、CatBoost



**為什麼Boosting演算法那麼熱門?**



他們對於訓練資料有限、訓練時間少、專業知識少的引數調優系統中，還是能發揮極大的作用

+ XGBoost最早是在2014年由陳天奇提出的一個研究專案

+ Microsoft在2017年1月釋出了LightGBM(稳定版本)

+ 同年2017年四月，俄羅斯的Yandex科技公司(一家在俄羅斯領先的科技公司）開源了CatBoost

![img](https://miro.medium.com/max/1400/1*i0CA9ho0WArOj-0UdpuKGQ.png)

圖片來源: https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db







## 3. LightGBMXGBoost?





![Screenshot-from-2019-03-27-23-09-47-1](https://user-images.githubusercontent.com/45563371/89561372-fb172900-d84a-11ea-8127-5e1c06c69044.png)



**XGBoost**



+ 一層一層的往下分裂(如圖)

+ XGBoost樹以水平方向生長,它所生長的是樹的層次

+ XGBoost採取預分類演算法(presorted algorithm)和基於直方圖的演算法來運算出最佳分割



**LightGBM**



+ 由葉子(leaf)的方向分裂(如圖) - 使用對增益最大的節點進行更進一步的分解方式，這樣可以省下大量分解所耗的資源

+ LightGBM樹以垂直方向生長，它所生長的是樹的葉子

+ LightGBM挑選具有最大誤差的樹葉往下生長，如果生長一樣的樹葉量，生長葉子的方法可以比用層的方法減少更多的loss

+ 基於梯度的單側採樣(GOSS)技術來過濾資料例項，來搜尋分割值
+ GOSS的方法: 簡單來說就是保留擁有最大梯度的資料，並在其他具有小梯度的例項上採用隨機取樣

舉例: 假設我手邊有1萬筆資料，其中的2千筆有比較高的梯度，所以被我都保留，剩下的8千筆，隨機選擇個20%，那最後在發現的分割值基礎上，我們選擇了全部數據1萬筆裡面的3千600筆數據

+ 執行GOSS的時候為了保持相同的資料分布，在計算資訊增益的時候，GOSS提供了小梯度的數據一個常數乘數，所以在減少資料例項的數量和保持學習決策樹的準確度上取得了非常好的平衡



我覺得這篇:  https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/510420/  寫得非常非常厲害,它幫我們詳細比較了XGBoost、LightGBM 和CatBoost，也清楚寫出方法的過程原理，推薦大家閱讀





## 4. 為什麼使用LightGBM這麼熱門?



由於它具有以下的優勢，讓它迅速的被廣泛使用

+ 名稱中有個Light，表示它有很快的訓練速度

+ 占用很少的內存空間
+ 支援並行的學習方式
+ 它可以用來處理大量的數據
+ 還可以支援GPU學習
+ 擁有更高的準確率







## 5. 任何情況都適用LightGBM嗎?



當然不是囉，LightGBM對於過擬和(overfitting)很敏感，所以不適合用於小型數據集，對於小型數據集非常容易overfitting







## Reference



https://en.wikipedia.org/wiki/LightGBM
論文： 1999 REITZ LECTURE GREEDY FUNCTION APPROXIMATION: A GRADIENT BOOSTING MACHINE1 By Jerome H. Friedman
https://zhuanlan.zhihu.com/p/52583923
https://kknews.cc/zh-tw/tech/y3a3x8j.html
https://github.com/denistanjingyu/LTA-Mobility-Sensing-Project
https://codertw.com/程式語言/510420/
https://lightgbm.readthedocs.io/en/latest/Parameters.html https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db

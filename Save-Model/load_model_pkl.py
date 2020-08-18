## load_model_pkl.py
## 導入joblib套件，joblib為sklearn的外部套件
from sklearn.externals import joblib
## 載入模型
knn_model = joblib.load("model/knn_model.pkl")
## 載入iris data
iris_raw_dataset = datasets.load_iris()
## 裝入特徵資料與標籤資料
X, y = iris_raw_dataset.data, iris_raw_dataset.target
## 測試模型
## 拿前十筆特徵資料預測它們的標籤類別
## 列出前二十筆的特徵資料
print(X[:20])
## 預測它們的標籤
print(knn_model.predict(X[0:20]))
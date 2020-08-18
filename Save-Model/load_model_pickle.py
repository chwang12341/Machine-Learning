## load_model_pickle.py
## 載入模型
with open('model/knn_model.pickle', 'rb') as f:
 knn_model = pickle.load(f)
## 載入iris data
iris_raw_dataset = datasets.load_iris()
## 裝入特徵資料與標籤資料
X, y = iris_raw_dataset.data, iris_raw_dataset.target
## 測試模型
## 拿前十筆特徵資料預測它們的標籤類別
## 列出前十筆的特徵資料
print(X[:10])
## 預測它們的標籤
print(knn_model.predict(X[0:10]))
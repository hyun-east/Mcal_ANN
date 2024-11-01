import numpy as np
import pandas as pd
import pickle

with open("/mnist_trained_model.pkl", "rb") as f:
    network = pickle.load(f)

# 데이터 불러오기 및 전처리
Test = pd.read_csv(".gitignore/mnist_test.csv")
X_test = Test.iloc[100, 1:] / 255.0
Y_test = np.eye(10)[Test.iloc[100, 0]]

# 모델 예측 수행
predictions = network.predict(X_test)

print(Y_test, predictions)
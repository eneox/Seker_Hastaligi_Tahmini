from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("diabetes.csv")

seker_hastalari = data[data["Outcome"] == 1]
saglikli_insanlar = data[data["Outcome"] == 0]


plt.scatter(saglikli_insanlar.Age, saglikli_insanlar.Glucose,
            color="green", label="sağlıklı", alpha=0.4)

plt.scatter(seker_hastalari.Age, seker_hastalari.Glucose,
            color="blue", alpha=0.4)


y = data.Outcome.values
x_ = data.drop(["Outcome"], axis=1)

x = (x_ - np.min(x_))/(np.max(x_)-np.min(x_))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1)

sayac = 1
for k in range(1, 11):
    knn_yeni = KNeighborsClassifier(n_neighbors=k)
    knn_yeni.fit(x_train, y_train)
    print(sayac, "  ", "Doğruluk oranı: %", knn_yeni.score(x_test, y_test)*100)
    sayac += 1

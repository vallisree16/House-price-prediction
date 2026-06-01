import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Dataset
data = {
    "Area": [500, 700, 900, 1100, 1300, 1500, 1700],
    "Price": [2500000, 3500000, 4500000, 5500000, 6500000, 7500000, 8500000]
}

df = pd.DataFrame(data)

print(df)

# Graph
plt.scatter(df["Area"], df["Price"])
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("House Price Prediction")
plt.show()

# Train model
X = df[["Area"]]
y = df["Price"]

model = LinearRegression()
model.fit(X, y)

print("Model Trained Successfully!")

# Prediction
area = int(input("Enter House Area: "))

prediction = model.predict([[area]])

print("Predicted Price =", int(prediction[0]))

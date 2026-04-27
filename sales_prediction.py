import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
data = pd.DataFrame({
    'TV': [230, 44, 17, 151, 180],
    'Radio': [37, 39, 45, 41, 10],
    'Newspaper': [69, 45, 69, 58, 58],
    'Sales': [22, 10, 9, 18, 12]
})
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)
print("Predicted Sales:", predictions)
print("Error:", error)

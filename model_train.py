import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset (example: house_data.csv)
data = pd.read_csv("house_data.csv")

# Independent and dependent features
X = data[['area', 'bedrooms', 'bathrooms', 'location_score']]
y = data['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open('house_price_model.pkl', 'wb'))

print("✅ Model trained and saved successfully!")

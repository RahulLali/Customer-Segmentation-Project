import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/customer_segmentation.csv')

df['Income'] = df['Income'].fillna(df['Income'].mean())

df = df.drop_duplicates()

current_year = 2026

df['Age'] = current_year - df['Year_Birth']

df['Total_Spending'] = (
    df['MntWines'] +
    df['MntFruits'] +
    df['MntMeatProducts'] +
    df['MntFishProducts'] +
    df['MntSweetProducts'] +
    df['MntGoldProds']
)

df['Children'] = df['Kidhome'] + df['Teenhome']

selected_features = [
    'Income',
    'Age',
    'Total_Spending',
    'Children',
    'Recency'
]

X = df[selected_features]

scaler = joblib.load('models/scaler.pkl')

X_scaled = scaler.transform(X)

model = joblib.load('models/kmeans_model.pkl')

df['Cluster'] = model.predict(X_scaled)

plt.figure(figsize=(10, 6))

plt.scatter(
    df['Income'],
    df['Total_Spending'],
    c=df['Cluster']
)

plt.title('Customer Segmentation Clusters')

plt.xlabel('Income')

plt.ylabel('Total Spending')

plt.savefig('static/images/cluster.png')

print("Cluster graph saved successfully")

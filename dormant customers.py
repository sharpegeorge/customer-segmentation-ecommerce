# Import necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('data.csv', encoding='ISO-8859-1')

# Drop rows with missing values in core columns
data = data.dropna(subset=['Quantity', 'UnitPrice', 'CustomerID'])

# Remove rows where Quantity is negative
data = data[data['Quantity'] >= 0]

# Convert InvoiceDate to datetime format
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], errors='coerce')

# Check for invalid or future InvoiceDate values (which can result in unrealistic weeks)
invalid_dates = data[data['InvoiceDate'].isna()]
print(f"Found {len(invalid_dates)} rows with invalid dates.")

# Remove rows with invalid dates
data = data.dropna(subset=['InvoiceDate'])

# Ensure there are no future dates (should not be possible in most cases)
fixed_date = pd.to_datetime('2011-12-09')
data = data[data['InvoiceDate'] <= fixed_date]

# Create a TotalSpend column
data['TotalSpend'] = data['Quantity'] * data['UnitPrice']

# Get the most recent purchase date for each customer
most_recent_purchase = data.groupby('CustomerID')['InvoiceDate'].max().reset_index()

# Calculate weeks since last purchase relative to the fixed date (09/12/2011)
most_recent_purchase['WeeksSinceLastPurchase'] = (fixed_date - most_recent_purchase['InvoiceDate']).dt.days / 7

# Aggregate again to include new features
customer_agg = data.groupby('CustomerID').agg(
    TotalSpendPerCustomer=('TotalSpend', 'sum'),
    PurchaseFrequency=('InvoiceNo', 'nunique'),
    AverageSpendPerTransaction=('TotalSpend', 'mean'),
    AverageQuantityPerTransaction=('Quantity', 'mean')
).reset_index()

# Merge the weeks since last purchase with the aggregated data
customer_agg = pd.merge(customer_agg, most_recent_purchase[['CustomerID', 'WeeksSinceLastPurchase']], on='CustomerID')

# Scale features for clustering
features = ['WeeksSinceLastPurchase', 'TotalSpendPerCustomer']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(customer_agg[features])

# Elbow Method to find the optimal number of clusters
inertia = []
k_values = range(1, 11)  # Test for 1 to 10 clusters
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot Elbow Method to determine optimal number of clusters
plt.figure(figsize=(8, 6))
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

# Apply K-Means Clustering with the optimal number of clusters
optimal_clusters = 3

kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
customer_agg['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualize with a scatter plot
plt.figure(figsize=(8, 6))

# Create scatter plot with clusters
for cluster in range(optimal_clusters):
    cluster_data = customer_agg[customer_agg['Cluster'] == cluster]
    plt.scatter(cluster_data['WeeksSinceLastPurchase'], cluster_data['TotalSpendPerCustomer'], s=50, alpha=0.6)

plt.xlabel('Weeks Since Last Purchase')
plt.ylabel('Total Spending')
plt.title('Customer Clusters Based on Time Since Last Purchase and Total Spending')
plt.legend()
plt.show()

# Show the cluster means
print(customer_agg.groupby('Cluster')[['WeeksSinceLastPurchase', 'TotalSpendPerCustomer']].mean())

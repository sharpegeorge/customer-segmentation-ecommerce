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

# Extract datetime components
data['Month'] = data['InvoiceDate'].dt.month
data['Year'] = data['InvoiceDate'].dt.year

# Create a TotalSpend column
data['TotalSpend'] = data['Quantity'] * data['UnitPrice']

# Add a feature for shopping in December (Christmas shopping indicator)
data['IsDecember'] = (data['Month'] == 12).astype(int)

# Aggregate again to include the new feature
customer_agg = data.groupby('CustomerID').agg(
    TotalSpendPerCustomer=('TotalSpend', 'sum'),
    PurchaseFrequency=('InvoiceNo', 'nunique'),
    AverageSpendPerTransaction=('TotalSpend', 'mean'),
    AverageQuantityPerTransaction=('Quantity', 'mean'),
    DecemberPurchases=('IsDecember', 'sum')
).reset_index()

# Create a new column to separate December and Non-December spending
monthly_spend = data.groupby(['CustomerID', 'Month'])['TotalSpend'].sum().reset_index()

# Filter December spend and non-December spend
december_spend = monthly_spend[monthly_spend['Month'] == 12]
non_december_spend = monthly_spend[monthly_spend['Month'] != 12]

# Merge December and Non-December spending for each customer
spend_comparison = december_spend[['CustomerID', 'TotalSpend']].rename(columns={'TotalSpend': 'DecemberSpend'})
spend_comparison = pd.merge(spend_comparison, non_december_spend.groupby('CustomerID')['TotalSpend'].sum().reset_index(), on='CustomerID', how='left')
spend_comparison = spend_comparison.rename(columns={'TotalSpend': 'NonDecemberSpend'})

# Calculate December-to-Total Spend Ratio
spend_comparison['DecemberToTotalSpendRatio'] = spend_comparison['DecemberSpend'] / (spend_comparison['DecemberSpend'] + spend_comparison['NonDecemberSpend'])

# Add other aggregated features
spend_comparison = pd.merge(spend_comparison, customer_agg[['CustomerID', 'TotalSpendPerCustomer', 'PurchaseFrequency']], on='CustomerID', how='left')

# Handle NaN values - Fill NaN values with 0 for spending columns
spend_comparison = spend_comparison.fillna(0)

# Scale features
features = ['DecemberToTotalSpendRatio', 'TotalSpendPerCustomer', 'PurchaseFrequency']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(spend_comparison[features])

# Apply K-Means Clustering
from sklearn.cluster import KMeans

# Determine the optimal number of clusters using the Elbow Method
inertia = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot the Elbow curve
plt.figure(figsize=(8, 6))
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method to Determine Optimal Clusters')
plt.show()

# After observing the elbow plot, we select the optimal number of clusters (e.g., 3)
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
spend_comparison['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualize with a scatter plot
plt.figure(figsize=(8, 6))

# Create scatter plot with clusters
for cluster in range(optimal_clusters):
    cluster_data = spend_comparison[spend_comparison['Cluster'] == cluster]
    plt.scatter(cluster_data['DecemberToTotalSpendRatio'], cluster_data['TotalSpendPerCustomer'], s=50, alpha=0.6)

plt.xlabel('December to Total Spending Ratio')
plt.ylabel('Total Spending')
plt.title('Customer Clusters based on December Spending Behavior and Total Spending')
plt.legend()
plt.show()

# Show the cluster means
print(spend_comparison.groupby('Cluster')[['DecemberToTotalSpendRatio', 'TotalSpendPerCustomer', 'PurchaseFrequency']].mean())

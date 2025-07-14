# Import libraries
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
data = pd.read_csv('data.csv', encoding='ISO-8859-1')
data = data.dropna()  # Remove rows with missing values
data = data[data['Quantity'] >= 0]  # Remove rows with negative quantities

# Initialize the label encoder
label_encoder = LabelEncoder()

# Apply Label Encoding to categorical features
data['StockCode'] = label_encoder.fit_transform(data['StockCode'].astype(str))
data['Country'] = label_encoder.fit_transform(data['Country'].astype(str))
data['InvoiceNo'] = label_encoder.fit_transform(data['InvoiceNo'].astype(str))

# Convert InvoiceDate to datetime format
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], errors='coerce')

# Extract datetime components
data['Month'] = data['InvoiceDate'].dt.month
data['Day'] = data['InvoiceDate'].dt.day
data['Year'] = data['InvoiceDate'].dt.year
data['DayOfWeek'] = data['InvoiceDate'].dt.weekday

# Drop the InvoiceDate column as it's no longer needed
data = data.drop(['InvoiceDate'], axis=1)

# Drop unnecessary columns
data = data.drop(['Description'], axis=1)

# Create a new column for Total Spend (Quantity * UnitPrice)
data['TotalSpend'] = (data['Quantity'] * data['UnitPrice']).astype(int)

# Aggregate data to create customer-level features
customer_agg = data.groupby('CustomerID').agg(
    TotalSpendPerCustomer=('TotalSpend', 'sum'),
    PurchaseFrequency=('InvoiceNo', 'nunique'),
    AverageSpendPerTransaction=('TotalSpend', 'mean'),
    AverageQuantityPerTransaction=('Quantity', 'mean')
).reset_index()

# Merge the aggregated data back with the original data
data = data.merge(customer_agg, on='CustomerID', how='left')

# Drop duplicate rows after merging
data = data.drop_duplicates(subset=['CustomerID'])

# Standardize features
scaler = StandardScaler()
features_to_scale = ['TotalSpendPerCustomer', 'PurchaseFrequency', 'AverageSpendPerTransaction', 'AverageQuantityPerTransaction']
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# Elbow Method to determine optimal k
inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data[features_to_scale])  # Use only the scaled features for clustering
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.plot(range(1, 10), inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Apply K-Means with optimal k
optimal_k = 3 
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(data[features_to_scale])  # Use only the scaled features for clustering

# Add cluster labels to the dataset
data['Cluster'] = clusters

# Visualize with a scatter plot (multiply TotalSpendPerCustomer by 9375)
plt.figure(figsize=(8, 6))

# Create scatter plot with clusters in different colors (TotalSpendPerCustomer multiplied by 9375)
for cluster in range(optimal_k):
    cluster_data = data[data['Cluster'] == cluster]
    plt.scatter(cluster_data['PurchaseFrequency'], cluster_data['TotalSpendPerCustomer'] * 9375, 
                s=50, alpha=0.6)

plt.xlabel('Purchase Frequency')
plt.ylabel('Total Spending (Multiplied by 9375)')
plt.title('Customer Clusters Based on Overall Spending Habits')
plt.legend()
plt.show()

# Show the cluster means
print(data.groupby('Cluster')[['PurchaseFrequency', 'TotalSpendPerCustomer', 'AverageSpendPerTransaction', 'AverageQuantityPerTransaction']].mean())

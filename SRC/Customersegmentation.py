import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

df = pd.read_excel(r"C:\Users\hp\Desktop\RESEARCH PROJECT\Beauty_Skincare_Customer_Dataset_Research_1000.xlsx")

print(df.head())

print(df.info())

print(df.isnull().sum())

# Fill numerical columns with mean
df["Annual_Income_LPA"] = df["Annual_Income_LPA"].fillna(df["Annual_Income_LPA"].mean())

df["Spending_Score"] = df["Spending_Score"].fillna(df["Spending_Score"].mean())

df["Average_Purchase_Amount_INR"] = df["Average_Purchase_Amount_INR"].fillna(df["Average_Purchase_Amount_INR"].mean())

# Fill categorical columns with mode
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])

df["Shopping_Frequency"] = df["Shopping_Frequency"].fillna(df["Shopping_Frequency"].mode()[0])

df["Favorite_Product"] = df["Favorite_Product"].fillna(df["Favorite_Product"].mode()[0])

df["Preferred_Shopping_Mode"] = df["Preferred_Shopping_Mode"].fillna(df["Preferred_Shopping_Mode"].mode()[0])

print(df.isnull().sum())

print("Number of Duplicate Records:", df.duplicated().sum())

df = df.drop_duplicates()

print("Number of Duplicate Records:", df.duplicated().sum())

print(df.shape)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# Encode categorical columns
df["Gender"] = label_encoder.fit_transform(df["Gender"])
df["Shopping_Frequency"] = label_encoder.fit_transform(df["Shopping_Frequency"])
df["Favorite_Product"] = label_encoder.fit_transform(df["Favorite_Product"])
df["Preferred_Shopping_Mode"] = label_encoder.fit_transform(df["Preferred_Shopping_Mode"])

print(df.head())

print(df.dtypes)

pd.set_option('display.max_columns', None)
print(df.head())

# -----------------------------
# Feature Scaling
# -----------------------------

# Select only the features used for clustering
features = df[[
    "Age",
    "Gender",
    "Annual_Income_LPA",
    "Spending_Score",
    "Shopping_Frequency",
    "Favorite_Product",
    "Preferred_Shopping_Mode",
    "Average_Purchase_Amount_INR"
]]

# Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Convert scaled data into a DataFrame
scaled_df = pd.DataFrame(scaled_features, columns=features.columns)

print("\nScaled Dataset")
print(scaled_df.head())

# --------------------------
# K-MEANS CLUSTERING
# --------------------------

# --------------------------
# ELBOW METHOD
# --------------------------

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6,3))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# --------------------------
# K-MEANS MODEL
# --------------------------

kmeans = KMeans(n_clusters=3, random_state=42)

kmeans.fit(scaled_features)

# --------------------------
# ASSIGN CLUSTER LABELS
# --------------------------

df["KMeans_Cluster"] = kmeans.labels_

print(df.head())

print(df["KMeans_Cluster"].value_counts())

# --------------------------
# K-MEANS CLUSTER VISUALIZATION
# --------------------------

plt.figure(figsize=(6,3))

plt.scatter(
    df["Annual_Income_LPA"],
    df["Spending_Score"],
    c=df["KMeans_Cluster"],
    cmap="viridis",
    s=40
)

# Plot Centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

plt.scatter(
    centroids[:,2],      # Annual Income
    centroids[:,3],      # Spending Score
    s=300,
    c='red',
    marker='X',
    label='Centroids'
)

plt.title("K-Means Customer Segmentation")
plt.xlabel("Annual Income (LPA)")
plt.ylabel("Spending Score")
plt.legend()
plt.grid(True)

plt.show()

# --------------------------
# AGGLOMERATIVE CLUSTERING
# --------------------------

agg = AgglomerativeClustering(n_clusters=3)
df["Agg_Cluster"] = agg.fit_predict(scaled_features)
print(df.head())
print(df["Agg_Cluster"].value_counts())

# --------------------------
# AGGLOMERATIVE VISUALIZATION
# --------------------------

plt.figure(figsize=(6,3))

plt.scatter(
    df["Annual_Income_LPA"],
    df["Spending_Score"],
    c=df["Agg_Cluster"],
    cmap="viridis",
    s=40
)

plt.title("Agglomerative Customer Segmentation")
plt.xlabel("Annual Income (LPA)")
plt.ylabel("Spending Score")

plt.colorbar(label="Cluster")
plt.grid(True)

plt.show()

# --------------------------
# DBSCAN CLUSTERING
# --------------------------
dbscan = DBSCAN(eps=1.8, min_samples=5)

df["DBSCAN_Cluster"] = dbscan.fit_predict(scaled_features)

print(df.head())

print(df["DBSCAN_Cluster"].value_counts())

# --------------------------
# DBSCAN VISUALIZATION
# --------------------------

plt.figure(figsize=(6,3))

plt.scatter(
    df["Annual_Income_LPA"],
    df["Spending_Score"],
    c=df["DBSCAN_Cluster"],
    cmap="viridis",
    s=40
)

plt.title("DBSCAN Customer Segmentation")
plt.xlabel("Annual Income (LPA)")
plt.ylabel("Spending Score")

plt.colorbar(label="Cluster")
plt.grid(True)

plt.show()

# --------------------------
# SILHOUETTE SCORE
# --------------------------

from sklearn.metrics import silhouette_score

# K-Means Silhouette Score
kmeans_silhouette = silhouette_score(
    scaled_features,
    df["KMeans_Cluster"]
)

# Agglomerative Silhouette Score
agg_silhouette = silhouette_score(
    scaled_features,
    df["Agg_Cluster"]
)

# DBSCAN - exclude noise points (-1)
dbscan_mask = df["DBSCAN_Cluster"] != -1

dbscan_silhouette = silhouette_score(
    scaled_features[dbscan_mask],
    df.loc[dbscan_mask, "DBSCAN_Cluster"]
)

print("\nSilhouette Scores")
print("K-Means:", kmeans_silhouette)
print("Agglomerative:", agg_silhouette)
print("DBSCAN:", dbscan_silhouette)

# --------------------------
# DAVIES-BOULDIN INDEX
# --------------------------

from sklearn.metrics import davies_bouldin_score

# K-Means
kmeans_db = davies_bouldin_score(
    scaled_features,
    df["KMeans_Cluster"]
)

# Agglomerative
agg_db = davies_bouldin_score(
    scaled_features,
    df["Agg_Cluster"]
)

# DBSCAN - exclude noise points (-1)
dbscan_mask = df["DBSCAN_Cluster"] != -1

dbscan_db = davies_bouldin_score(
    scaled_features[dbscan_mask],
    df.loc[dbscan_mask, "DBSCAN_Cluster"]
)

print("\nDavies-Bouldin Index")
print("K-Means:", kmeans_db)
print("Agglomerative:", agg_db)
print("DBSCAN:", dbscan_db)

# --------------------------
# CALINSKI-HARABASZ INDEX
# --------------------------

from sklearn.metrics import calinski_harabasz_score

# K-Means
kmeans_ch = calinski_harabasz_score(
    scaled_features,
    df["KMeans_Cluster"]
)

# Agglomerative
agg_ch = calinski_harabasz_score(
    scaled_features,
    df["Agg_Cluster"]
)

# DBSCAN - exclude noise points (-1)
dbscan_ch = calinski_harabasz_score(
    scaled_features[dbscan_mask],
    df.loc[dbscan_mask, "DBSCAN_Cluster"]
)

print("\nCalinski-Harabasz Index")
print("K-Means:", kmeans_ch)
print("Agglomerative:", agg_ch)
print("DBSCAN:", dbscan_ch)

# --------------------------
# K-MEANS CLUSTER PROFILING
# --------------------------

cluster_profile = df.groupby("KMeans_Cluster")[
    [
        "Age",
        "Annual_Income_LPA",
        "Spending_Score",
        "Shopping_Frequency",
        "Average_Purchase_Amount_INR"
    ]
].mean()

print("\nK-Means Cluster Profile")
print(cluster_profile)

# --------------------------
# FINAL ALGORITHM COMPARISON
# --------------------------

comparison = pd.DataFrame({
    "Algorithm": ["K-Means", "Agglomerative", "DBSCAN"],
    "Silhouette Score": [
        kmeans_silhouette,
        agg_silhouette,
        dbscan_silhouette
    ],
    "Davies-Bouldin Index": [
        kmeans_db,
        agg_db,
        dbscan_db
    ],
    "Calinski-Harabasz Index": [
        kmeans_ch,
        agg_ch,
        dbscan_ch
    ]
})

print("\nFinal Algorithm Comparison")
print(comparison)

print(df["KMeans_Cluster"].value_counts())
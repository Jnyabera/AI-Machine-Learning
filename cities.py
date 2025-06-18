# SDG 11: Sustainable Cities - Public Transport Route Optimization with Clustering
# Author: James Nyabera Ouma
# Description: AI-driven solution using unsupervised learning (KMeans clustering) to optimize public transport routes.

# --- IMPORT LIBRARIES ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import folium
from folium.plugins import MarkerCluster

# --- LOAD DATA ---
# Example data: Simulated GPS coordinates for bus stops (in a real case, import from CSV or API)
data = {
    'stop_id': range(1, 21),
    'latitude': [-1.28, -1.29, -1.30, -1.27, -1.31, -1.25, -1.26, -1.28, -1.30, -1.29,
                 -1.33, -1.32, -1.34, -1.31, -1.35, -1.36, -1.30, -1.29, -1.28, -1.27],
    'longitude': [36.82, 36.83, 36.84, 36.81, 36.82, 36.79, 36.78, 36.80, 36.83, 36.85,
                  36.86, 36.87, 36.85, 36.88, 36.89, 36.90, 36.86, 36.84, 36.82, 36.81]
}
bus_stops = pd.DataFrame(data)

# --- KMEANS CLUSTERING ---
coords = bus_stops[['latitude', 'longitude']]

# Determine optimal K using Elbow method
inertia = []
k_range = range(2, 8)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(coords)
    inertia.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.show()

# Train final model with chosen K
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
bus_stops['cluster'] = kmeans.fit_predict(coords)

# --- VISUALIZE ON MAP ---
map_center = [-1.30, 36.83]
m = folium.Map(location=map_center, zoom_start=13)

colors = ['red', 'blue', 'green', 'purple']
for i in range(optimal_k):
    cluster_group = bus_stops[bus_stops['cluster'] == i]
    cluster_map = MarkerCluster().add_to(m)
    for _, row in cluster_group.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Stop {row['stop_id']} (Cluster {row['cluster']})",
            icon=folium.Icon(color=colors[i])
        ).add_to(cluster_map)

# Save map
m.save("clustered_bus_stops_map.html")

# --- OPTIONAL: SUGGEST ROUTES IN EACH CLUSTER (placeholder logic) ---
# Sort stops in each cluster by latitude as a proxy for route ordering
for i in range(optimal_k):
    cluster_group = bus_stops[bus_stops['cluster'] == i].sort_values('latitude')
    print(f"\nSuggested route order for Cluster {i}:")
    print(cluster_group[['stop_id', 'latitude', 'longitude']])

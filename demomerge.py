import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

# Load the dataset
data = pd.read_csv('Merged.csv', encoding="ISO-8859-1")
print(data)

# Extract and preprocess the 'NIC Name' column (industry names)
industry_names = data['NIC Name'].fillna("").str.lower()  
industry_names
print(industry_names.describe())

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(industry_names)

# Applying K-Means clustering
num_clusters = 5  
kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
kmeans.fit(tfidf_matrix)

# Assign clusters to each industry name and add to the DataFrame
data['Cluster'] = kmeans.labels_

# Display results
# Check the cluster distribution
print("Cluster Distribution:")
print(data['Cluster'].value_counts())

#  Review sample industry names in each cluster
for i in range(num_clusters):
    print(f"\nCluster {i} Samples:")
    print(data[data['Cluster'] == i]['NIC Name'].head(10))

# Save the results to a new CSV
output_path = "C:\\Users\\DELL\\Desktop\\Datasets\\demo-merge\\Clustered_Industries.csv"
data.to_csv(output_path, index=False)
print(f"Clustered data saved to {output_path}")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Streamlit app title
st.title("Cluster Visualization of Business Categories")

# Upload CSV file with clustering results
uploaded_file = st.file_uploader("demo-merge/Clustered_Industries.csv", type="csv")
if uploaded_file:
    # Load the CSV file into a DataFrame
    data = pd.read_csv(uploaded_file)
    
    # Ensure 'Cluster' column exists
    if 'Cluster' not in data.columns:
        st.error("Uploaded file does not contain a 'Cluster' column.")
    else:
        # Display cluster distribution
        st.write("Cluster Distribution:")
        cluster_counts = data['Cluster'].value_counts()
        
        # Visualization 1: Bar Chart for Cluster Counts 
        fig, ax = plt.subplots()
        cluster_counts.plot(kind="bar", color="skyblue", ax=ax)
        ax.set_title("Cluster Distribution")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Number of Items")
        st.pyplot(fig)
        
        # Check if TF-IDF or similar feature columns are in the data
        feature_columns = [col for col in data.columns if col not in ['Cluster']]
        if len(feature_columns) < 2:
            st.error("Not enough feature columns for PCA scatter plot.")
        else:
            # Visualization 2: Scatter Plot of Clusters using PCA for dimensionality reduction
            # Extract feature columns for PCA
            feature_data = data[feature_columns]
            
            # Apply PCA to reduce to 2 components for visualization
            pca = PCA(n_components=2)
            pca_components = pca.fit_transform(feature_data)
            
            # Scatter plot
            fig, ax = plt.subplots()
            scatter = ax.scatter(pca_components[:, 0], pca_components[:, 1], c=data['Cluster'], cmap="viridis", alpha=0.6)
            ax.set_title("Business Categories Clustering (PCA)")
            ax.set_xlabel("PCA Component 1")
            ax.set_ylabel("PCA Component 2")
            plt.colorbar(scatter, ax=ax, label="Cluster")
            st.pyplot(fig)

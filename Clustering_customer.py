import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

model = pickle.load(open('Clustering_customer.sav', 'rb'))

df=pd.read_excel("output_clusterss.xlsx")
features = ['Age', 'Years Employed', 'Income']
X = df[features]

st.title('Bank Credit Card Customer Segmentation')

numClusters = st.slider("Select Number of Clusters", min_value=1, max_value=10, value=3)

model.n_clusters = numClusters
clusters = model.fit_predict(X)

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(X['Years Employed'], X['Income'], c=clusters, cmap='viridis')
ax.set_xlabel('Years Employed')
ax.set_ylabel('Income')

legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
st.pyplot(fig)
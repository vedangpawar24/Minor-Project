#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Data Preprocessing:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
data=pd.read_csv("D:\Corizo Project\spotify dataset.csv")


# In[2]:


#Data Analysis and Visualizations:
import matplotlib.pyplot as plt
import seaborn as sns

plt.hist(data['duration_ms'], bins=20)
plt.xlabel('Duration (ms)')
plt.ylabel('Count')
plt.show()


# In[3]:


#Correlation Matrix:
numeric_data = data.select_dtypes(include='number')
corr_matrix = numeric_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')


# In[4]:


#Clustering by Playlist Genres and Names:
from sklearn.cluster import KMeans

X = data[['danceability','energy','key']]
kmeans = KMeans(n_clusters=3)
data['genre_cluster'] = kmeans.fit_predict(X)


# In[5]:


X = data[['danceability','energy','key']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[12]:


n_clusters = 4  
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['Cluster'], cmap='viridis')
plt.title('Clustered Data')
plt.show()


# In[13]:


from sklearn.cluster import KMeans
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Plot silhouette scores
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different Cluster Numbers')
plt.show()


# In[14]:


k = 4  # Choose the optimal number of clusters

kmeans = KMeans(n_clusters=k, random_state=42)
data['cluster'] = kmeans.fit_predict(X_scaled)


# In[15]:


k=0
plt.figure(figsize = (18,14))
for i in X.columns:
    plt.subplot(4,4, k + 1)
    sns.distplot(X[i])
    plt.xlabel(i, fontsize=11)
    k +=1


# In[18]:


#Building the Model:
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, data['playlist_genre'], test_size=0.2)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)


# In[17]:


#Final Results:
accuracy = model.score(X_test, y_test)
print(f'Model Accuracy: {accuracy}')

user_input = pd.DataFrame({'danceability': [0.748], 'energy': [0.916], 'key': [6]})
recommended_genre = model.predict(user_input)
recommended_songs = data[data['playlist_genre'] == recommended_genre[0]]


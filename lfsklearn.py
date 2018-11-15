'''
Example of how you might pull some useful code into a library of functions
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import train_test_split

# Define a function that runs clustering on a df and returns the clusters for each row in df
def Cluster(df, numClusters):
    # Create a k-means clusterer.  
    kmeans = KMeans(init='random', n_clusters=numClusters, n_init=10, max_iter=10)

    # Train it
    kmeans.fit(df)

    # Make some predictions about which cluster each sample belongs to
    clusters = kmeans.predict(df)
    
    return clusters


# Generalise to a function
def ClusterScatter(df, xFeature, yFeature, clusterFeature):
    #Plot the clusters obtained using k means
    plt.figure()
    scatter = plt.scatter(df[xFeature],df[yFeature],c=df[clusterFeature],s=50)
    plt.title('K-Means Clustering')
    plt.xlabel(xFeature)
    plt.ylabel(yFeature)


def PCAPlot(df, numClusters):
    # Visualize the results on PCA-reduced data
    reduced_data = PCA(n_components=2).fit_transform(df[df.columns[:64]])
    kmeans = KMeans(init='k-means++', n_clusters=numClusters, n_init=10)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

def DecisionTreeClassifier(X,y):
    # Create a model - can use gini or entropy for the criterion
    model = tree.DecisionTreeClassifier(criterion = "gini", max_depth = 12, min_samples_split = 300, min_samples_leaf = 150)

    # Split into training and test sets (4 sets in total) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

    # Use training data to fit model (i.e. train the model) 
    model.fit(X_train, y_train)

    # Use training inputs to predict training outputs
    y_hat_train = model.predict(X_train)

    # Use test inputs to predict test outputs
    y_hat_test = model.predict(X_test)
    
    return y_test, y_hat_test


def CorrPlot(df):
    # Draw a chart showing correlation for original data
    names = list(df)
    fig, ax = plt.subplots(figsize=(5,5))
    mat = ax.matshow(df.corr())
    ax.set_xticks(np.arange(0,5,1))
    ax.set_yticks(np.arange(0,5,1))
    ax.set_xticklabels(names, rotation = 45)
    ax.set_yticklabels(names)
    fig.colorbar(mat)
    plt.show()

def PCAApply(df):
    # Standardise the data to have mean 0 and stdev 1
    scaler = StandardScaler()
    data_std = scaler.fit_transform(df)   # standardise
    data_std = pd.DataFrame(data_std)       # turn into a data frame
    data_std.columns = list(df)           # add the original column names back on
    
    # Perform principal component analysis on the standardised data
    pca = PCA(n_components=5)
    pca.fit(data_std)

    # Transform the data back into the shape we want
    PC_df = pd.DataFrame(pca.transform(data_std), index=df.index, columns=['PC1','PC2','PC3','PC4','PC5'])
    return pca, PC_df

def ExplainedVarCumSum(pca):
    # Get the explained variance for each PC as a %
    exp_var_ratio = pca.explained_variance_ratio_
    pca_explained_variance_cumsum = exp_var_ratio.cumsum()

    # Draw a chart of the cumulative sum of explained variance as each PC is added
    index = np.arange(len(pca_explained_variance_cumsum))+1
    plt.bar(index, pca_explained_variance_cumsum)
    plt.xlabel("PC")
    plt.ylabel("% variance explained")
    plt.title("PCA: Explained Variance Cumulative Sum")

def ExplainedVar(pca):
    # Get the explained variance for each PC as a %
    exp_var_ratio = pca.explained_variance_ratio_

    # Draw a chart of the explained variance for each PC
    index = np.arange(len(exp_var_ratio))+1
    plt.bar(index, exp_var_ratio)
    plt.xlabel("PC")
    plt.ylabel("% variance explained")
    plt.title("PCA: Explained Variance Ratio")
# THIS CODE WILL READ MULTIPLE FILES
# import glob

# path = r'E:\Phase 2\Project as homework' # use your path
# all_files = glob.glob(path + "/*.csv")

# li = []

# for filename in all_files:
#     df = pd.read_csv(filename, index_col=None, header=0)
#     li.append(df)

# frame = pd.concat(li, axis=0, ignore_index=True)
import pandas as pd
import numpy as np
import os

#READING FILES
stock = [file for file in os.listdir('E:\Phase 2\Project as homework')]

for file in stock:
    print(file)

All_stock = pd.DataFrame()

#BELOW LINE OF CODE WILL READ EACH FILE, REPLACE NULL WITH MEAN VALUE FOR EACH COLUMN
for file in stock:
    DF = pd.read_csv('E:\Phase 2\Project as homework/'+file)
    for i in DF.columns[DF.isnull().any(axis=0)]:     #---Applying Only on variables with NaN values
        DF[i].fillna(DF[i].mean(),inplace=True)
    All_stock = pd.concat([All_stock, DF])

# All_Month.head()

# All_stock.to_csv("E:\Phase 2\Project as homework\All_stock.csv",index=False)

All_stock.info()
All_stock['Symbol'].value_counts()
All_stock['Series'].value_counts()

a = All_stock.tail(1000)

All_stock.isnull().sum()

All_stock_1 = All_stock.drop(['Date','Series','Company Name','Industry','ISIN Code'], axis = 1)

a = All_stock_1.tail(1000)

All_stock_1.isnull().sum()

All_stock_na = All_stock_1[All_stock_1.isna().any(axis=1)]

# All_stock_null = All_stock_1[All_stock_1.isnull().any(axis=1)]

All_stock_1.dropna(inplace=True)

print(len(All_stock))
print(len(All_stock_1))

############################################################
cor = All_stock_1.corr()

columns = cor[cor["Close"]>0.7]["Close"]
columns_1 = cor[cor["Close"]<(-0.65)]["Close"]

C = pd.concat([columns,columns_1],axis=1)
columns_list = C.index
columns_list

All_stock_corr = pd.DataFrame(All_stock_1,columns = columns_list)
#CREATING TARGET VARIABLE############################################################

All_stock_1['Target'] = All_stock_1['Close'].shift(1)
All_stock_1['Target'] = All_stock_1['Target'].fillna(0)

a = All_stock_1.head(1000)

# All_stock_1 = pd.DataFrame(All_stock_1)

All_stock_cluster = All_stock_corr.values
#######################################################################################
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
LE.fit_transform(All_stock_cluster[:,0]) 

All_stock_cluster[:,0] = LE.fit_transform(All_stock_cluster[:,0]) 


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()

All_stock_cluster = sc_x.fit_transform(All_stock_cluster)
#######################################################################################
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss = []
for i in range(1, 16):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(All_stock_cluster)
    wcss.append(kmeans.inertia_)

with plt.style.context(('fivethirtyeight')):
    plt.figure(figsize=(10,6))
    plt.plot(range(1, 16), wcss)
    plt.title('The Elbow Method with k-means++\n',fontsize=25)
    plt.xlabel('Number of clusters')
    plt.xticks(fontsize=20)
    plt.ylabel('WCSS (within-cluster sums of squares)')
    plt.vlines(x=5,ymin=0,ymax=250000,linestyles='--')
    plt.text(x=5.5,y=110000,s='5 clusters seem optimal choice \nfrom the elbow position',
             fontsize=25,fontdict={'family':'Times New Roman'})
    plt.show()

#######################################################################################
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(All_stock_cluster)

# Visualising the clusters
plt.scatter(All_stock_cluster[y_kmeans == 0, 1], All_stock_cluster[y_kmeans == 0, 2], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(All_stock_cluster[y_kmeans == 1, 1], All_stock_cluster[y_kmeans == 1, 2], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(All_stock_cluster[y_kmeans == 2, 1], All_stock_cluster[y_kmeans == 2, 2], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(All_stock_cluster[y_kmeans == 3, 1], All_stock_cluster[y_kmeans == 3, 2], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(All_stock_cluster[y_kmeans == 4, 1], All_stock_cluster[y_kmeans == 4, 2], s = 100, c = 'magenta', label = 'Cluster 5')
#plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s = 100, c = 'black', label = 'Cluster 6')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'black', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
#######################################################################################
x = All_stock_1.drop(['Target'],axis=1).values
y = All_stock_1.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#######################################################################################
x_train[:,0] = LE.fit_transform(x_train[:,0]) 
x_test[:,0] = LE.fit_transform(x_test[:,0]) 

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)



def correlation_matrix(x_train):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure(figsize=(16,12))
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(All_stock_1.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Stock data set features correlation\n',fontsize=15)
    labels=All_stock_1.columns
    ax1.set_xticklabels(labels,fontsize=9)
    ax1.set_yticklabels(labels,fontsize=9)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[0.1*i for i in range(-11,11)])
    plt.show()

correlation_matrix(x_train)

from scipy.stats import pearsonr

for col in cor.columns:
    coef, pval = pearsonr(All_stock_1[col], All_stock_1['Target'])
    print("Correlation b/w target and %s - coef: %.2f, pval: %f" %(col, coef, pval))
#######################################################################################
dfx = pd.DataFrame(data=x_train,columns=All_stock_1.columns[0:13])

from sklearn.decomposition import PCA
pca = PCA(n_components=None)

dfx_pca = pca.fit(dfx)


plt.figure(figsize=(10,6))
plt.scatter(x=[i+1 for i in range(len(dfx_pca.explained_variance_ratio_))],
            y=dfx_pca.explained_variance_ratio_,
           s=200, alpha=0.75,c='orange',edgecolor='k')
plt.grid(True)
plt.title("Explained variance ratio of the \nfitted principal component vector\n",fontsize=25)
plt.xlabel("Principal components",fontsize=15)
plt.xticks([i+1 for i in range(len(dfx_pca.explained_variance_ratio_))],fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Explained variance ratio",fontsize=15)
plt.show()


dfx_trans = pca.transform(dfx)
dfx_trans = pd.DataFrame(data=dfx_trans)

#######################################################################################
from sklearn.cluster import KMeans
# Number of clusters
kmeans = KMeans(n_clusters=5)
# Fitting the input data
kmeans = kmeans.fit(dfx_trans)
# Getting the cluster labels
labels = kmeans.predict(dfx_trans)
# Centroid values
centroids = kmeans.cluster_centers_

print(centroids) # From sci-kit learn

dfx_trans['cluster_no']= kmeans.predict(dfx_trans)

# #############################################################################################################
hc_df = x_train.copy()

from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist, squareform

# #############################################################################################################

# list of distance metrics
distance_metrics = ["euclidean", "chebyshev", "mahalanobis", "cityblock"]

# list of linkage methods
linkage_methods = ["single", "complete", "average", "weighted"]

high_cophenet_corr = 0
high_dm_lm = [0, 0]

for dm in distance_metrics:
    for lm in linkage_methods:
        Z = linkage(hc_df, metric=dm, method=lm)
        c, coph_dists = cophenet(Z, pdist(hc_df))
        print(
            "Cophenetic correlation for {} distance and {} linkage is {}.".format(
                dm.capitalize(), lm, c
            )
        )
        if high_cophenet_corr < c:
            high_cophenet_corr = c
            high_dm_lm[0] = dm
            high_dm_lm[1] = lm

# ########################################################################################3

# # list of linkage methods
# linkage_methods = ["single", "complete", "average", "centroid", "ward", "weighted"]

# high_cophenet_corr = 0
# high_dm_lm = [0, 0]

# for lm in linkage_methods:
#     Z = linkage(hc_df, metric="euclidean", method=lm)
#     c, coph_dists = cophenet(Z, pdist(hc_df))
#     print("Cophenetic correlation for {} linkage is {}.".format(lm, c))
#     if high_cophenet_corr < c:
#         high_cophenet_corr = c
#         high_dm_lm[0] = "euclidean"
#         high_dm_lm[1] = lm
        
#  ####################################################################################
 
#  # list of linkage methods
# linkage_methods = ["single", "complete", "average", "centroid", "ward", "weighted"]

# # lists to save results of cophenetic correlation calculation
# compare_cols = ["Linkage", "Cophenetic Coefficient"]
# compare = []

# # to create a subplot image
# fig, axs = plt.subplots(len(linkage_methods), 1, figsize=(15, 30))

# # We will enumerate through the list of linkage methods above
# # For each linkage method, we will plot the dendrogram and calculate the cophenetic correlation
# for i, method in enumerate(linkage_methods):
#     Z = linkage(hc_df, metric="euclidean", method=method)

#     dendrogram(Z, ax=axs[i])
#     axs[i].set_title(f"Dendrogram ({method.capitalize()} Linkage)")

#     coph_corr, coph_dist = cophenet(Z, pdist(hc_df))
#     axs[i].annotate(
#         f"Cophenetic\nCorrelation\n{coph_corr:0.2f}",
#         (0.80, 0.80),
#         xycoords="axes fraction",
#     )

#     compare.append([method, coph_corr])
#######################################################################################

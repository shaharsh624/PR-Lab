import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Binarizer, Normalizer, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits

# Load Iris dataset
df1 = pd.read_csv('../practical4/Iris.csv', index_col=0)
X = df1.drop('Species', axis=1)
y = df1['Species']

# Standardize features
X_std = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
X_mms = pd.DataFrame(MinMaxScaler(feature_range=(0,1)).fit_transform(X), columns=X.columns)
X_rbs = pd.DataFrame(RobustScaler().fit_transform(X), columns=X.columns)
X_norm = pd.DataFrame(Normalizer().fit_transform(X), columns=X.columns)
X_bnz = pd.DataFrame(Binarizer(threshold=0).fit_transform(X), columns=X.columns)
X_imp = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)
df1.ffill(inplace=True)

y_lbenc = pd.Series(LabelEncoder().fit_transform(y))
X_ohenc = pd.DataFrame(OneHotEncoder(sparse_output=False).fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pca = PCA()
X_pca = pd.DataFrame(pca.fit_transform(X), columns=[f'PC{i+1}' for i in range(X.shape[1])])
pca_var = pca.explained_variance_ratio_

plt.plot(range(1, len(pca_var) + 1), pca_var * 100)
plt.title("Explained Variance Ratio for PCA")
plt.xlabel("Components")
plt.ylabel("Variance (%)")
plt.grid()
plt.show()

pca = PCA(n_components=2)
X_pca_2d = pd.DataFrame(pca.fit_transform(X), columns=["PC1", "PC2"])

df1_pca = pd.concat([X_pca_2d, y], axis=1)
colors = ["red", "green", "blue"]
species_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
for sp, cl in zip(species_names, colors):
    sp_class = df1_pca["Species"] == sp
    plt.scatter(df1_pca.loc[sp_class,"PC1"], df1_pca.loc[sp_class,"PC2"], c=cl, label=sp)
plt.legend()
plt.grid()
plt.title('Observations in 2D PCA Space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Linear Discriminant Analysis (LDA) classification
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X_train, y_train)
y_lda = lda.predict(X_test)
print("LDA Classification Accuracy:", accuracy_score(y_test, y_lda) * 100)

X_lda = lda.transform(X)

c_map = y.map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=c_map, cmap='viridis', edgecolor='k', s=50, alpha=0.7)
plt.title('LDA - Iris Dataset')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.colorbar()
plt.grid()
plt.show()

# Load Wine dataset
df2 = pd.read_csv('../practical4/wine_data.csv')

X_df2 = df2.drop('class_label', axis=1)
y_df2 = df2['class_label']
X_df2_std = pd.DataFrame(StandardScaler().fit_transform(X_df2), columns=X_df2.columns)

plt.figure(figsize=(7, 7))
X_df2_std.boxplot()
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(10, 7))
sns.heatmap(X_df2_std.cov(), annot=True, fmt='.2f')
plt.show()

pca_wine = PCA(n_components=2)
X_pca_wine = pd.DataFrame(pca_wine.fit_transform(X_df2_std), columns=['PC1', 'PC2'])

df_pca_wine = pd.concat([X_pca_wine, y_df2], axis=1)
colors = ['red', 'blue', 'green']
for label, c in zip(df_pca_wine['class_label'].unique(), colors):
    label_class = df_pca_wine['class_label'] == label
    plt.scatter(df_pca_wine.loc[label_class, 'PC1'], df_pca_wine.loc[label_class, 'PC2'], c=c, label=label, alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Wine Dataset (2 Components)')
plt.grid()
plt.legend()
plt.show()

digits = load_digits()
X_digits = digits.data
y_digits = digits.target

X_digits_sc = StandardScaler().fit_transform(X_digits)

pca_digits = PCA()
pca_digits_tr = pca_digits.fit_transform(X_digits_sc)

plt.plot(np.cumsum(pca_digits.explained_variance_ratio_) * 100)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio by PCA Components')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
scatter = plt.scatter(pca_digits_tr[:, 0], pca_digits_tr[:, 1], c=y_digits, cmap='Spectral', edgecolor='k', s=60)
plt.colorbar(scatter)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('2D PCA of Digits Dataset')
plt.show()

noise_factor = 0.5
X_noisy = X_digits_sc + noise_factor * np.random.normal(size=X_digits_sc.shape)
X_pca_noisy = pca_digits.fit_transform(X_noisy)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca_noisy[:, 0], X_pca_noisy[:, 1], c=y_digits, cmap='Spectral', edgecolor='k', s=60)
plt.colorbar(scatter)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('2D PCA of Noisy Digits Dataset')
plt.show()

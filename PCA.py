import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import PCA

# Load data
file_path = 'dataset/merged.csv'
data = pd.read_csv(file_path)
data['genres'] = data['genres'].apply(lambda x: x.split('|'))

# Transform the genres into one-hot encoding
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(data['genres'])
genres_encoded_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)

# Concatenation of numerical features with the encoded genres
features = pd.concat([data[['rating']], genres_encoded_df], axis=1)

# Normalize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Application for PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(features_scaled)
principal_df = pd.DataFrame(data = principal_components,
                            columns = ['Principal Component 1', 'Principal Component 2'])

print(principal_df.head())

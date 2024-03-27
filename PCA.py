import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import PCA

# Étape 1: Chargement des données
# Remplacer 'data.csv' par le chemin de votre fichier de données
file_path = 'dataset/merged.csv'
data = pd.read_csv(file_path)
data['genres'] = data['genres'].apply(lambda x: x.split('|'))

# Étape 2: Prétraitement des données
# Transformation des genres en one-hot encoding
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(data['genres'])
genres_encoded_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)

# Concaténation des caractéristiques numériques avec les genres encodés
features = pd.concat([data[['rating']], genres_encoded_df], axis=1)

# Étape 3: Normalisation des données
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Étape 4: Application de la PCA
pca = PCA(n_components=2)  # Vous pouvez ajuster le nombre de composantes ici
principal_components = pca.fit_transform(features_scaled)
principal_df = pd.DataFrame(data = principal_components,
                            columns = ['Principal Component 1', 'Principal Component 2'])

# Affichage des résultats
print(principal_df.head())

# Librerias
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, RobustScaler
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import joblib

# Carga de datos
df= pd.read_csv("C:\\Users\\camil\\OneDrive\\Documentos\\ML_P\\src\\data\\raw\\popular_python_projects.csv")

# Limpieza
df["last_commit"] = pd.to_datetime(df["last_commit"])
df["date"] = pd.to_datetime(df["date"])

df.drop(columns=["repo_url", "item", "language"], inplace= True)

df.dropna(axis=0, inplace=True)

# Feature engeneering
lc_y_m_d = [x[0] for x in df["last_commit"].astype(str).str.split(" ")]
df["last_commit"] = lc_y_m_d
df["last_commit"] = pd.to_datetime(df["last_commit"])
df["days_since_lc"] = (df["date"] - df["last_commit"]) / timedelta(days=1)
df.drop(columns="last_commit", inplace=True)

popular_users = list(df["username"].value_counts()[df["username"].value_counts() > 1].index)
n_repos = [df[df["username"] == x]["repo_name"].nunique() for x in popular_users]
df = pd.merge(df, pd.DataFrame({'username':popular_users, 'repos_by_user':n_repos}), 'outer', on='username')
df["repos_by_user"].fillna(1, inplace=True)

# NLP
sia = SentimentIntensityAnalyzer()

res = {}

for i, x in df.iterrows():
    des = x["description"]
    repo_name = x["repo_name"]
    res[repo_name] = sia.polarity_scores(des)

vaders = pd.DataFrame(res).T

vaders = vaders.reset_index().rename(columns={"index":"repo_name"})
df = df.merge(vaders, how="left")

df.drop(columns=['neg', 'neu', 'pos'], inplace=True)

# Label Encoder
le = LabelEncoder()

for x in df[["repo_name", "description", "username"]].columns:
    df[x] = le.fit_transform(df[x])

# Prerando la columna Date para poder escalarla
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year

df.drop(columns="date", inplace=True)

# Escalado
RobEsc = RobustScaler()

esc_features = RobEsc.fit_transform(df)

df[['rank', 'repo_name', 'stars', 'forks', 'username', 'issues',
       'description', 'days_since_lc', 'repos_by_user', 'compound', 'day',
       'month', 'year']] = esc_features

# Cluster
df_cluster = df.drop(columns="forks") 

kmeans = MiniBatchKMeans(5, random_state=42).fit(df_cluster)

clusterized = kmeans.predict(df_cluster)
df_cluster["forks"] = df["forks"]
df_cluster["cluster_result"] = clusterized

df_cluster_0 = df_cluster[df_cluster["cluster_result"] == 0]
df_cluster_1 = df_cluster[df_cluster["cluster_result"] == 1]
df_cluster_2 = df_cluster[df_cluster["cluster_result"] == 2]
df_cluster_3 = df_cluster[df_cluster["cluster_result"] == 3]
df_cluster_4 = df_cluster[df_cluster["cluster_result"] == 4]

# Separación en train y test
train_4, test_4 = train_test_split(df_cluster_4, test_size=0.2, random_state=42)

# Separación en train y val
X = train_4.drop(columns="forks")
y = train_4["forks"]

X_train, X_val, y_train, y_val= train_test_split(X, y, test_size=0.2, random_state=42)

# GridSearch
param_grid = {
    'max_depth': [0, 1, 2, 3, 4, 5],
    'min_samples_leaf': np.arange(1,10),
    'max_features': np.arange(2,12),
    'splitter':["best", "random"], 
    'min_samples_leaf': np.arange(2,5)
}

decicsion_tree_r = DecisionTreeRegressor(random_state=42)

grid_search = GridSearchCV(decicsion_tree_r,
                           param_grid,
                           cv=5,
                           scoring='neg_root_mean_squared_error',
                           n_jobs=-1
                          )

# Entrenamiento
grid_search.fit(X_train, y_train)

dtr = grid_search.best_estimator_

# Guardando el modelo

modelo = joblib.dump(dtr, "mymodel_4.pkl")
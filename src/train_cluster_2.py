# Librerias
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, RobustScaler
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import joblib
import re
import pickle

# Carga de datos
df = pd.read_csv("C:\\Users\\camil\\Documents\\GitHub\\MachineLearningProject_TB\\src\\data\\raw\\popular_python_projects.csv")

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

signos = re.compile(r"(\.)|(\;)|(\:)|(\!)|(\?)|(\¿)|(\@)|(\,)|(\))|(\()|(\))|(\[)|(\])|(\d+)")

def signs_description(des):
    return signos.sub('', des.lower())

df['description'] = df['description'].apply(signs_description)

def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"  
                               u"\U0001F680-\U0001F6FF"  
                               u"\U0001F1E0-\U0001F1FF"  
                               u"\U00002500-\U00002BEF"  
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642" 
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text.lower())


df['description'] = df['description'].apply(remove_emojis)

stopwords_words = stopwords.words('english')

def remove_stopwords(df):
    return " ".join([word for word in df.split() if word not in stopwords_words])

df['description'] = df['description'].apply(remove_stopwords)

def word_lemmatizer(x):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in x.split()])

df['description'] = df['description'].apply(word_lemmatizer)

vectorizer_c = CountVectorizer()
vectorizer_c.fit(df['description'])

X_baseline_c = vectorizer_c.transform(df['description'])

kmeans = MiniBatchKMeans(8, random_state=42).fit(X_baseline_c)

df['cluster'] = kmeans.labels_

# Label Encoder
le = LabelEncoder()

df = df.drop(columns=["description"])

df["repo_name"] = le.fit_transform(df["repo_name"])
df["username"] = le.fit_transform(df["username"])

# Prerando la columna Date para poder escalarla
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year

df.drop(columns="date", inplace=True)

# Escalado
RobEsc = RobustScaler()

esc_features = RobEsc.fit_transform(df[['repo_name', 'rank', 'stars', 'forks', 'username', 'issues',
       'days_since_lc', 'repos_by_user', 'day', 'month', 'year']])

df[['repo_name', 'rank', 'stars', 'forks', 'username', 'issues',
       'days_since_lc', 'repos_by_user', 'day', 'month', 'year']] = esc_features

# Cluster
df_cluster = df.drop(columns="forks") 

kmeans = MiniBatchKMeans(3, random_state=42).fit(df_cluster)

clusterized = kmeans.predict(df_cluster)
df_cluster["forks"] = df["forks"]
df_cluster["cluster_result"] = clusterized

df_cluster_0 = df_cluster[df_cluster["cluster_result"] == 0]
df_cluster_1 = df_cluster[df_cluster["cluster_result"] == 1]
df_cluster_2 = df_cluster[df_cluster["cluster_result"] == 2]


# Separación en train y test
train_1, test_1 = train_test_split(df_cluster_2, test_size=0.2, random_state=42)

# Separación en train y val
X = train_1.drop(columns="forks")
y = train_1["forks"]

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

with open('mymodel_2.pkl', 'wb') as f:
    pickle.dump(dtr, f)

# Examen TEXT MINING


import pandas as pd
import numpy as np
import seaborn as sns
import csv as csv
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from bs4 import BeautifulSoup as bs
import requests as request
import pandas as pd

df = pd.read_csv("e_commerce_data.csv")

print(df.head(10))

print(df.info())

#(c) À l'aide du package seaborn. Afficher, sous forme de graphique en barres, la distribution de la variable 'marks'.

sns.countplot(x='marks', data=df)
plt.title('Distribution des notes')
plt.xlabel('Notes')
plt.ylabel("Nombre d occurences")
plt.show()

# d) Identifier les 10 compagnies les plus présentes dans le dataset.


a = df["companies"].value_counts().head(10)

print(a)

#(e) Afficher la répartition des commentaires positifs et négatifs sur ces 10 compagnies.

top_companies = df['companies'].value_counts().head(10).index

filtered_df = df[df['companies'].isin(top_companies)]

distribution = filtered_df.groupby(['companies', 'marks']).size().unstack(fill_value=0)

print(distribution)


# (f) Reprendre les questions (d) et (e) avec la variable 'language'.

b = df["language"].value_counts().head(10)

top_languages = df['language'].value_counts().head(10).index

filtered_df1 = df[df['language'].isin(top_languages)]

distribution1 = filtered_df1.groupby(['language', 'marks']).size().unstack(fill_value=0)

print(distribution1)

# (g) Créer un DataFrame nommé df_en en conservant uniquement les commentaires en anglais.


df_en = df[df["language"] == "en"][["comment","language"]]

print(df_en.head())

# (2) Visualisation de données

#a) Afficher, sur un graphe, la distribution de la longueur des commentaires. Vous limiterez l'axe des abscisses entre 0 et 3000.

# Insérez votre code ici

df["comment_length"]= df["comment"].str.len()

print(df.head(30))

sns.histplot(data=df, x='comment_length', bins=30)
plt.show()


#(b) Ajouter une nouvelle colonne nommée 'comment_category' au DataFrame df_en qui catégorise les commentaires en fonction de leur longueur : "court", "moyen", "long". Définissez les seuils vous-même.

df["comment_lenght"] = df["comment"].str.len()

def categorize_comment(lenght):
    if lenght < 50:
     return 'court'
    elif lenght < 150:
       return 'moyen'
    else:
       return 'long'
    
df["comment_category"] = df["comment_lenght"].apply(categorize_comment)

print(df["comment_category"])

# (c) Afficher, à l'aide d'un countplot, la répartition des notes en fonction de la longueur des commentaires

sns.countplot(data=df, x='comment_category', hue='marks')

plt.xlabel("Categories de longueur de commentaire")
plt.ylabel("Nombre de commentaires")
plt.title("Repartition des notes selon la longueur des commentaires")
plt.legend(title="Note")
plt.show()

# (d) Créer, à partir des colonnes 'experience_date' et 'comment_date', les variables comm_day, comm_month et comm_year qui correspondent aux jour, mois et année de ces évènements.

df['experience_date'] = pd.to_datetime(df['experience_date'])
df['comment_date'] = pd.to_datetime(df['comment_date'])

# Extraction des composantes jour, mois et année pour experience_date
df['exp_day'] = df['experience_date'].dt.day
df['exp_month'] = df['experience_date'].dt.month
df['exp_year'] = df['experience_date'].dt.year

# Extraction des composantes jour, mois et année pour comment_date
df['comm_day'] = df['comment_date'].dt.day
df['comm_month'] = df['comment_date'].dt.month
df['comm_year'] = df['comment_date'].dt.year

print(df[['experience_date', 'exp_day', 'exp_month', 'exp_year', 'comment_date', 'comm_day', 'comm_month', 'comm_year']].head())



#(e) Afficher l'évolution de la répartition des notes en fonction des jours, mois et années des commentaires.


def plot_marks_distribution_over_time(df, time_unit, title):
    grouped = df.groupby([time_unit, 'marks']).size().unstack(fill_value=0)
    grouped.plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.title(f"Répartition des notes par {title}")
    plt.xlabel(title)
    plt.ylabel("Nombre de commentaires")
    plt.legend(title="Mark", labels=["Négatif (-1)", "Positif (1)"])
    plt.tight_layout()
    plt.show()

# Évolution par jour
plot_marks_distribution_over_time(df, 'comm_day', "jour du mois")

# Évolution par mois
plot_marks_distribution_over_time(df, 'comm_month', "mois")

# Évolution par année
plot_marks_distribution_over_time(df, 'comm_year', "année")



#3. Wordclouds
#L'objectif est de construire deux wordclouds, se focalisant, respectivement, sur les commentaires positifs et les commentaires négatifs.

 #   (a) Compiler tous les commentaires positifs de df_en dans une variable text_pos de type str. Faire de même pour les commentaires négatifs dans une variable text_neg.

print(df_en.head())

df_en["marks"] = df["marks"]

positives = df_en[df_en["marks"] >= 1]
negatives = df_en[df_en["marks"] <= -1]

# Compiler les commentaires en une seule chaîne de caractères
text_pos = " ".join(positives['comment'].astype(str).tolist())
text_neg = " ".join(negatives['comment'].astype(str).tolist())

# Vérification rapide (facultative)
print("Nombre de commentaires positifs :", len(positives))
print("Nombre de commentaires négatifs :", len(negatives))
print("\nExtrait de text_pos:\n", text_pos[:300])
print("\nExtrait de text_neg:\n", text_neg[:300])

#    (b) Importer la classe stopwords du package nltk.corpus.

from nltk.corpus import stopwords

import nltk
nltk.download('stopwords')



#    (c) Initialiser une variable stop_words contenant des mots vides anglais.
stop_words = set(stopwords.words('english'))

#    (d) Afficher stop_words.

print(stop_words)


from wordcloud import WordCloud
import matplotlib.pyplot as plt

#(e) Afficher le wordcloud des commentaires positifs.

wordcloud_pos = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words).generate(text_pos)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud des commentaires positifs")
plt.show()

 #   (f) Afficher le wordcloud des commentaires négatifs.

wordcloud_neg = WordCloud(width=800, height=400, background_color='black', stopwords=stop_words, colormap='Reds').generate(text_neg)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud des commentaires négatifs")
plt.show()

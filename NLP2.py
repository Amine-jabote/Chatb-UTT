# Description: Ce fichier contient le code principal de notre chatbot. Il permet de traiter les demandes de l'utilisateur et de générer 
# des réponses en fonction de l'intention de l'utilisateur.
# Auteur: Mohamed Amine Jabote/ Youssef Sidqui/ Freddy Pouna Wantou


import nltk
import spacy
from openai import OpenAI
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import re

from intention_detection import predire_intention_utilisateur
from technology_detection import detecter_technologie_demandee
from extraire_UE import extraire_info_matiere
from subject_technology_search import trouver_matieres_par_technologie
from tech_response_generation import reponse_technologie
from nettoyage_input import nettoyer_entree_utilisateur
from intention import traiter_intention_utilisateur
from extraire_metier_demande import extraire_metier_demande
from trouver_matieres_par_metier import trouver_matieres_par_metier
from trouver_matieres_par_metier import generer_reponse_matieres_par_metier

nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load("fr_core_news_sm")


model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_json('knowledge.json')

def search(query, df):
    query_embedding = model.encode([query])[0]
    df['distance'] = df['embeddings'].apply(lambda line: np.dot(line, query_embedding))
    df.sort_values(by='distance', ascending=False, inplace=True)
    results = df.head(5)
    return results

def extraire_code_matiere(mots):
    return next((mot for mot in mots if re.match(r'[a-zA-Z]{2}\d{2}', mot)), None)

def extraire_information_ue(mots):
    code_matiere = extraire_code_matiere(mots)
    if code_matiere:  # Vérifiez que code_matiere n'est pas None
        code_matiere = code_matiere.upper()
        df2 = pd.read_json('knowledge.json')
        return df2.loc[code_matiere]['description']
    else:
        return "Le code de la matière n'a pas été trouvé dans votre saisie."

def generate_answer(intention, question, context):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Tu es un chatbot qui guide les étudiants dans le choix de leurs matières et les aide à la construction de leur projet professionnel. Lors de tes réponses utilise le noms des matières qui sont des codes composées de deux lettres et deux chiffres. c'est le premier mot qui t'es transmis"},
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"utilse ces informations surles matières de l'utt pour répondre à la question de l'utilisateur: {context},stp limite toi strictement aux informations transmis dans ce contexte lorsque tu réponds."}
        ]
    )
    return response.choices[0].message.content

def extraire_information_metier(phrase):
    metier_demande = extraire_metier_demande(phrase)
    matieres_trouvees = trouver_matieres_par_metier(metier_demande)
    info_matiere = []
    df2 = pd.read_json('knowledge.json')
    for key, value in matieres_trouvees.items():
        for matiere in value:
            code_matiere = matiere['Code'].upper()
            info_matiere.append(df2.loc[code_matiere]['description'])   
    return info_matiere

def definition_context(intention, utilisateur_saisie, mots):
    if intention == "informations_matière" or intention == "Branches_ue":
        return extraire_information_ue(mots)
    elif intention == "metiers":
        return extraire_information_metier(utilisateur_saisie)
    else:
        result = search(utilisateur_saisie, df)
        context = result['description'].tolist()
        return context

while True:
    utilisateur_saisie = input("Bonjour ! Je suis chatb'UTT et je suis là pour vous aider à choisir vos UE.\n Posez moi n'importe quelle question ? \n")
    
    if utilisateur_saisie.lower() == 'exit':
        break

    mots = nettoyer_entree_utilisateur(utilisateur_saisie)
    intention = predire_intention_utilisateur(utilisateur_saisie)
    context = definition_context(intention, utilisateur_saisie, mots)
    reponse = generate_answer(intention,utilisateur_saisie,context)
    print(reponse)

    

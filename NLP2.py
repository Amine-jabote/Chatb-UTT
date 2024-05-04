# Description: Ce fichier contient le code principal de notre chatbot. Il permet de traiter les demandes de l'utilisateur et de générer 
# des réponses en fonction de l'intention de l'utilisateur.
# Auteur: Mohamed Amine Jabote/ Youssef Sidqui/ Freddy Pouna Wantou


import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Importer les fonctions des autres fichiers
from intention_detection import predire_intention_utilisateur
from technology_detection import detecter_technologie_demandee
from extraire_UE import extraire_info_matiere
from subject_technology_search import trouver_matieres_par_technologie
from tech_response_generation import reponse_technologie
from nettoyage_input import nettoyer_entree_utilisateur
from intention import traiter_intention_utilisateur


nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load("fr_core_news_sm")




# Intégrer la prédiction d'intention dans la fonction traiter_entree_utilisateur
def traiter_entree_utilisateur(phrase):
    
    # Nettoyer l'entrée de l'utilisateur
    mots = nettoyer_entree_utilisateur(phrase)
    
    # Prédire l'intention de l'utilisateur
    intention = predire_intention_utilisateur(phrase)
    
    # Analyser l'intention de l'utilisateur
    
    return traiter_intention_utilisateur(intention, mots, phrase)
    

while True:
    
    utilisateur_saisie = input("Bonjour ! Je suis chatb'UTT et je suis là pour vous aider à choisir vos UE. Pour le moment je ne suis pas si fort mais je vais m'améliorer. pour m'aider veuillez formuler votre demande sous les formes suivantes : \n - Donne moi des informations sur la matière code_UE \n - Propose moi des matières où on utilise <technologie/ex: python,omniscope,Unity...> \n - Propose moi des matières pour le métier <métier ex: Data Scientist> \n - Je veux apprendre <coder en python,....> \n -Quelles sont les modalités d'evaluation de ... \n- exit pour quitter\nQuelle est votre demande ?")
    
    if utilisateur_saisie.lower() == 'exit':
        break
    
    reponse = traiter_entree_utilisateur(utilisateur_saisie)
    print(reponse)

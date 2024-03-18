import re 
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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



# def extraire_metier_demande(phrase):
    # Utiliser une expression régulière pour trouver un motif de métier après "devenir"
    #match = re.search(r'je veux devenir (\w+(?: \w+)*)', phrase)
    #if match:
        #return match.group(1)
    #else:
        #return None

#def trouver_matieres_par_metier(metier_demande):
#    matieres_proposees = []
#    for code_matiere, info_matiere in matieres_database.items():
#        if metier_demande.lower() in [m.lower() for m in info_matiere["Métiers"]]:
#            matieres_proposees.append(code_matiere)
#    return matieres_proposees


# Intégrer la prédiction d'intention dans la fonction traiter_entree_utilisateur
def traiter_entree_utilisateur(phrase):
    
    # Nettoyer l'entrée de l'utilisateur
    mots = nettoyer_entree_utilisateur(phrase)
    
    # Prédire l'intention de l'utilisateur
    intention = predire_intention_utilisateur(phrase)
    
    # Analyser l'intention de l'utilisateur
    
    return traiter_intention_utilisateur(intention, mots, phrase)
    
    if intention == "informations_matière":
        # Chercher un motif correspondant à un code de matière (deux lettres suivies de chiffres)
        code_matiere = next((mot for mot in mots if re.match(r'[a-zA-Z]{2}\d{2}', mot)), None)
        if code_matiere:
            # Récupérer les informations de la matière
            matiere_info = extraire_info_matiere(code_matiere)
            if matiere_info:
                # Construire et formater la réponse
                response = f"{matiere_info['Titre']}({code_matiere})\n"
                response += f"Ce cours est une {matiere_info['Type']} qui concerne les {', '.join(matiere_info['branche'])}({', '.join(matiere_info['Filiere'])}), avec un nombre de crédits qui est {matiere_info['Credits']} crédits, il est disponible en {', '.join(matiere_info['disponibilité'])} et est enseigné en {', '.join(matiere_info['Langue'])}\n"
                response += f"Contenu du cours : \n Les étuiants auront l'occasion d'apprendre \n {matiere_info['Programme']}\n"
                response += "En ce qui concerne la répartition horaire:\n"
                for key, value in matiere_info['Repartition'].items():
                    response += f"- {key}: {value}\n"
                response += f"L'objectif du cours est {matiere_info['Objectif']}\n"
                response += f"Les technologies utilisées :\n Les technologies principales abordées dans ce cours sont {', '.join(matiere_info['Technologies'])}\n"
                response += f"\nPour les modalités d'évaluation {', '.join(matiere_info['Modalite'])} \n"
                response += "\nDébouchés professionnels:\n"
                for metier in matiere_info['Métiers']:
                    response += f"- {metier}\n"
                response += "Pour plus d'informations, veuillez consulter le site de l'UTT ou demander à la scolarité.\n"
                return response
            else:
                return f"Les informations pour la matière {code_matiere} n'ont pas été trouvées."
        else:
            return "Désolé, je ne comprends pas votre demande."
    elif intention == "matières_utilisant_technologie":
        entite_technologie = detecter_technologie_demandee(phrase)
        matieres_trouvees = trouver_matieres_par_technologie(entite_technologie)
        return generer_reponse_matieres_par_technologie(matieres_trouvees,entite_technologie)
    # Ajoute d'autres conditions pour les différentes intentions...
    elif intention == "metiers":
        # Appeler la fonction pour proposer des matières liées à un métier
        metier_demande = extraire_metier_demande(phrase)
        matieres_trouvees = trouver_matieres_par_metier(metier_demande)
        print(matieres_trouvees)


while True:
    
    utilisateur_saisie = input("Bonjour ! Je suis chatb'UTT et je suis là pour vous aider à choisir vos UE. Pour le moment je ne suis pas si fort mais je vais m'améliorer. pour m'aider veuillez formuler votre demande sous les formes suivantes : \n - Donne moi des informations sur la matière code_UE \n - Propose moi des matières où on utilise <technologie/ex: python,omniscope,Unity...> \n - Propose moi des matières pour le métier <métier ex: Data Scientist> \n - Je veux apprendre <coder en python,....> \n -Quelles sont les modalités d'evaluation de ... \n- exit pour quitter\nQuelle est votre demande ?")
    
    if utilisateur_saisie.lower() == 'exit':
        break
    
    reponse = traiter_entree_utilisateur(utilisateur_saisie)
    print(reponse)
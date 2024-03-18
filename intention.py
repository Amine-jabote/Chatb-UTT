from UE_information import traiter_intention_informations_matiere
from Tech_information import traiter_intention_technologie
from Metier_information import traiter_intention_metiers


def traiter_intention_utilisateur(intention, mots,phrase):
    if intention == "informations_matière":
        return traiter_intention_informations_matiere(mots)
    elif intention == "matières_utilisant_technologie":
        return traiter_intention_technologie(phrase)
    elif intention == "metiers":
        return traiter_intention_metiers(mots)
    # Ajoutez d'autres conditions pour les différentes intentions...
    else:
        return "Désolé, je ne comprends pas votre demande."

from Tech_information import traiter_intention_technologie
from Metier_information import traiter_intention_metiers
from extraire_metiers_matiere import traiter_intention_metiers_matiere
from UE_information import traiter_intention_informations_matiere,traiter_intention_nom_ue,traiter_intention_Type_ue,traiter_intention_Nombre_credits,traiter_intention_langue_ue,traiter_intention_Branche_ue, traiter_intention_Programme_ue, traiter_intention_Ouvert_TC, traiter_intention_Objectif_ue, traiter_intention_Modalite_ue





def traiter_intention_utilisateur(intention, mots,phrase):
    if intention == "informations_matière":
        return traiter_intention_informations_matiere(mots)
    elif intention == "nom_ue":
        return traiter_intention_nom_ue(mots)
    elif intention == "Type_ue":
        return traiter_intention_Type_ue(mots)
    elif intention == "Nombre_crédits":
        return traiter_intention_Nombre_credits(mots)
    elif intention == "langue_ue":
        return traiter_intention_langue_ue(mots)
    elif intention == "Branches_ue":
        return traiter_intention_Branche_ue(mots)
    elif intention == "Programme_ue":
        return traiter_intention_Programme_ue(mots)
    elif intention == "Objectif_ue":
        return traiter_intention_Objectif_ue(mots)
    elif intention == "Modalite_ue":
        return traiter_intention_Modalite_ue(mots)
    elif intention == "Ouvert_TC":
        return traiter_intention_Ouvert_TC(mots)
    elif intention == "matières_utilisant_technologie":
        return traiter_intention_technologie(phrase)
    elif intention == "metiers":
        return traiter_intention_metiers(phrase)
    elif intention == "metiers_matiere" :
        return traiter_intention_metiers_matiere(mots)
    # Ajoutez d'autres conditions pour les différentes intentions...
    else:
        return "Désolé, je ne comprends pas votre demande."

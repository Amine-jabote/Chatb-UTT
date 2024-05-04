import re
from extraire_UE import extraire_info_matiere


def construire_reponse_informations_matiere(matiere_info,code_matiere):
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

def traiter_intention_informations_matiere(mots):
    code_matiere = next((mot for mot in mots if re.match(r'[a-zA-Z]{2}\d{2}', mot)), None)
    if code_matiere:
        matiere_info = extraire_info_matiere(code_matiere)
        if matiere_info:
            return construire_reponse_informations_matiere(matiere_info,code_matiere)
        else:
            return f"Les informations pour la matière {code_matiere} n'ont pas été trouvées."
    else:
        return "Désolé, je ne comprends pas votre demande concernant les informations sur une matière."

def traiter_intention_nom_ue(mots):
    code_matiere = next((mot for mot in mots if re.match(r'[a-zA-Z]{2}\d{2}', mot)), None)
    if code_matiere:
        ue_info = extraire_info_matiere(code_matiere)
        if ue_info:
            return f"Le nom ou le titre associé à l'UE {code_matiere} est :  {ue_info['Titre']}"
        else:
            return f"Aucune information trouvée pour l'UE {code_matiere}, veuillez bien vérifier le code de l'UE"
    else:
        return "Je ne parviens pas à comprendre quelle UE vous recherchez."
    
def traiter_intention_Type_ue(mots):
    code_matiere = next((mot for mot in mots if re.match(r'[a-zA-Z]{2}\d{2}', mot)), None)
    if code_matiere:
        ue_info = extraire_info_matiere(code_matiere)
        if ue_info:
            return f"Le type de  l'UE {code_matiere} est :  {ue_info['Type']}"
        else:
            return f"Aucune information trouvée pour l'UE {code_matiere}, veuillez bien vérifier le code de l'UE"
    else:
        return "Je ne parviens pas à comprendre quelle UE vous recherchez."

def traiter_intention_Nombre_credits(mots) :
    code_matiere = next((mot for mot in mots if re.match(r'[a-zA-Z]{2}\d{2}', mot)), None)
    if code_matiere:
        ue_info = extraire_info_matiere(code_matiere)
        if ue_info:
            return f"Le nombre de crédits associé à l'UE {code_matiere} est :  {ue_info['Credits']}"
        else:
            return f"Aucune information trouvée pour l'UE {code_matiere}, veuillez bien vérifier le code de l'UE"
    else:
        return "Je ne parviens pas à comprendre quelle UE vous recherchez."

def traiter_intention_langue_ue(mots) :
    code_matiere = next((mot for mot in mots if re.match(r'[a-zA-Z]{2}\d{2}', mot)), None)
    if code_matiere:
        ue_info = extraire_info_matiere(code_matiere)
        if ue_info:
            return f"La langue d'enseignement de l'UE {code_matiere} est :  {ue_info['Langue']}"
        else:
            return f"Aucune information trouvée pour l'UE {code_matiere}, veuillez bien vérifier le code de l'UE"
    else:
        return "Je ne parviens pas à comprendre quelle UE vous recherchez."
    
def traiter_intention_Branche_ue(mots) :
    code_matiere = next((mot for mot in mots if re.match(r'[a-zA-Z]{2}\d{2}', mot)), None)
    if code_matiere:
        ue_info = extraire_info_matiere(code_matiere)
        if ue_info:
            if len(ue_info['branche']) > 1:
                return f"Les branches de l'UE {code_matiere} sont :  {ue_info['branche']}"
            else :
                return f"La branche de l'UE {code_matiere} est :  {ue_info['branche']}"
        else:
            return f"Aucune information trouvée pour l'UE {code_matiere}, veuillez bien vérifier le code de l'UE"
    else:
        return "Je ne parviens pas à comprendre quelle UE vous recherchez."

def traiter_intention_Programme_ue(mots) :
    code_matiere = next((mot for mot in mots if re.match(r'[a-zA-Z]{2}\d{2}', mot)), None)
    if code_matiere:
        ue_info = extraire_info_matiere(code_matiere)
        if ue_info:
            return f"Le programme de l'UE {code_matiere} est :  {ue_info['Programme']}"
        else:
            return f"Aucune information trouvée pour l'UE {code_matiere}, veuillez bien vérifier le code de l'UE"
    else:
        return "Je ne parviens pas à comprendre quelle UE vous recherchez."
    
def traiter_intention_Objectif_ue(mots) :
    code_matiere = next((mot for mot in mots if re.match(r'[a-zA-Z]{2}\d{2}', mot)), None)
    if code_matiere:
        ue_info = extraire_info_matiere(code_matiere)
        if ue_info:
            return f"L'objectof de l'UE {code_matiere} est :  {ue_info['Objectif']}"
        else:
            return f"Aucune information trouvée pour l'UE {code_matiere}, veuillez bien vérifier le code de l'UE"
    else:
        return "Je ne parviens pas à comprendre quelle UE vous recherchez."

def traiter_intention_Modalite_ue(mots) :
    code_matiere = next((mot for mot in mots if re.match(r'[a-zA-Z]{2}\d{2}', mot)), None)
    if code_matiere:
        ue_info = extraire_info_matiere(code_matiere)
        if ue_info:
            if ue_info['Modalite'] :
                return f"Les modalité d'évaluation de l'UE {code_matiere} sont :  {ue_info['Modalite']}"
            else :
                return f"Malheureusement cette information n'est pas disponible en ce moment veuillez demander plus tard."
        else:
            return f"Aucune information trouvée pour l'UE {code_matiere}, veuillez bien vérifier le code de l'UE"
    else:
        return "Je ne parviens pas à comprendre quelle UE vous recherchez."
    
def traiter_intention_Ouvert_TC(mots) :
    code_matiere = next((mot for mot in mots if re.match(r'[a-zA-Z]{2}\d{2}', mot)), None)
    if code_matiere:
        ue_info = extraire_info_matiere(code_matiere)
        if ue_info:
            if ue_info['OuvertTC'] == True :
                return f"Oui, l'UE {code_matiere} est ouverte au TC (tronc commun)"
            else : 
                return f"Non, l'UE {code_matiere} n'est  pas ouverte au TC (Tronc commun)"
        else:
            return f"Aucune information trouvée pour l'UE {code_matiere}, veuillez bien vérifier le code de l'UE"
    else:
        return "Je ne parviens pas à comprendre quelle UE vous recherchez."

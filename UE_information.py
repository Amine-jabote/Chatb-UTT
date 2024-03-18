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

import json


with open('D:/Devops/Chatbutt/db.json', 'r', encoding='utf-8' ) as file:
    matieres_database = json.load(file)
    
# Fonction pour extraire les informations d'une matière
def extraire_info_matiere(code_matiere):
    code_matiere = code_matiere.upper()
    print(code_matiere)
    if code_matiere in matieres_database:
        return matieres_database[code_matiere]
    else:
        return None
    
def extraire_metiers_matiere(code_matiere):
    code_matiere = code_matiere.upper()
    if code_matiere in matieres_database:
        matiere_info = matieres_database[code_matiere]
        metiers_associés = matiere_info.get("Métiers", [])
        return metiers_associés
    else:
        return []

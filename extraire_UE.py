import json


with open('D:/Devops/Chatbutt/db.json', 'r', encoding='utf-8' ) as file:
    matieres_database = json.load(file)
    
# Fonction pour extraire les informations d'une matière
def extraire_info_matiere(code_matiere):
    print(code_matiere)
    if code_matiere in matieres_database:
        return matieres_database[code_matiere]
    else:
        return None
import json


with open('D:/Devops/Chatbutt/db.json', 'r', encoding='utf-8' ) as file:
    matieres_database = json.load(file)
    

def trouver_matieres_par_technologie(technologies_demandees):
    matieres_trouvees = []
    for code_matiere, info_matiere in matieres_database.items():
        if "Technologies" in info_matiere:
            for technologie in technologies_demandees:
                if technologie.lower() in [tech.lower() for tech in info_matiere["Technologies"]]:
                    matieres_trouvees.append((code_matiere, info_matiere["Titre"]))
                    break
    return matieres_trouvees

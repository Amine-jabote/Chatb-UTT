import json


def trouver_matieres_par_metier(metiers_demande):
    # Charger la base de données depuis le fichier JSON
    with open('D:/Devops/Chatbutt/db.json', 'r', encoding='utf-8') as file:
        matieres_database = json.load(file)
    
    # Dictionnaire pour stocker les matières correspondantes à chaque métier
    matieres_correspondantes_par_metier = {}
    # Parcourir chaque métier de la liste des métiers demandés
    for metier_demande in metiers_demande:
        # Initialiser la liste des matières correspondantes pour ce métier
        matieres_correspondantes = []
        
        # Parcourir la base de données des matières
        for code_matiere, matiere_info in matieres_database.items():
            # Vérifier si le métier demandé fait partie des métiers associés à cette matière
            if metier_demande in matiere_info["Métiers"] or metier_demande.capitalize() in matiere_info["Métiers"]:
                # Ajouter la matière (code et titre) à la liste des matières correspondantes pour ce métier
                matieres_correspondantes.append({"Code": code_matiere, "Titre": matiere_info["Titre"]})
        
        # Stocker les matières correspondantes pour ce métier dans le dictionnaire
        matieres_correspondantes_par_metier[metier_demande] = matieres_correspondantes
    # Retourner le dictionnaire contenant les matières correspondantes pour chaque métier
    return matieres_correspondantes_par_metier



def generer_reponse_matieres_par_metier(matieres__trouvees, metier__demande):
    # Générer la réponse à partir des matières trouvées pour le métier demandé
    if len(matieres__trouvees) > 0:
        return  f"Voici les matières associées au métier de {metier__demande} :\n {trouver_matieres_par_metier(metier__demande)}"
    else:
        return f"Aucune matière n'est associée au métier de {metier__demande}"
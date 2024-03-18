

def reponse_technologie(matieres_trouvees, entite_technologie):
    if not entite_technologie:
        return "Désolé, je n'ai pas compris quelle technologie vous cherchez à apprendre."
    
    if not matieres_trouvees:
        return f"Aucune matière utilisant la technologie '{entite_technologie}' n'a été trouvée."
    
    reponse = f"Les UE où on apprend ou on utilise {entite_technologie} sont les suivantes :\n"
    for code_matiere, titre_matiere in matieres_trouvees:
        reponse += f"- {code_matiere}: {titre_matiere}\n"
    return reponse
from technology_detection import detecter_technologie_demandee
from subject_technology_search import trouver_matieres_par_technologie
from tech_response_generation import reponse_technologie

def traiter_intention_technologie(phrase):
    entite_technologie = detecter_technologie_demandee(phrase)
    matieres_trouvees = trouver_matieres_par_technologie(entite_technologie)
    return reponse_technologie(matieres_trouvees, entite_technologie)
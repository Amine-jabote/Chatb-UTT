import re
from extraire_UE import extraire_metiers_matiere



def traiter_intention_metiers_matiere(mots):
    code_matiere = next((mot for mot in mots if re.match(r'[a-zA-Z]{2}\d{2}', mot)), None)
    metiers_associés = extraire_metiers_matiere(code_matiere)
    if metiers_associés:
        return f"Les métiers associés à la matière {code_matiere} sont :\n" + "\n".join(metiers_associés)
    else:
        return f"Aucun métier associé à la matière {code_matiere} n'a été trouvé."
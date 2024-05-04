import re

def detecter_technologie_demandee(phrase):
    # Liste de noms de technologies courants
    with open('tech.txt', 'r') as f:
        technologies = f.read().splitlines()
    
    # Construire le motif d'expression régulière en utilisant la barre verticale pour séparer les options
    pattern = r'\b(?:' + '|'.join(map(re.escape, technologies)) + r')\b'
    
    # Rechercher les correspondances dans la phrase
    entites_technologie = re.findall(pattern, phrase, flags=re.IGNORECASE)
    
    return entites_technologie

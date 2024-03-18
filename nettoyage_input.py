import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def nettoyer_entree_utilisateur(phrase):
    phrase = phrase.lower()  # Convertir en minuscules pour la normalisation
    phrase = re.sub(r'[^\w\s]', '', phrase)  # Supprimer la ponctuation
    mots = word_tokenize(phrase)  # Tokenization des mots
    mots = [mot for mot in mots if mot not in stopwords.words('french')]  # Supprimer les mots vides
    return mots

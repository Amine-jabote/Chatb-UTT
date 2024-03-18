# intention_detection.py
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])



def predire_intention_utilisateur(phrase):
   
    # Code pour prédire l'intention de l'utilisateur en utilisant le modèle entraîné
    train_data = [
    # Exemples pour l'intention "informations_matière"
    ("donne moi des informations sur la matière", "informations_matière"),
    ("je veux en savoir plus sur la matière Code_UE", "informations_matière"),
    ("quels sont les objectifs de la matière Code_UE", "informations_matière"),
    ("a quoi sert la matière Code_UE", "informations_matière"),
    ("affiche moi les informations de la matière Code_UE", "informations_matière"),
    ("afficher les informations de la matière Code_UE", "informations_matière"),
    ("informations matière Code_UE", "informations_matière"),
    ("informations sur la matière Code_UE", "informations_matière"),
    ("peux tu me donner des informations sur la matière Code_UE", "informations_matière"),
    ("j'aimerais obtenir des renseignements sur la matière Code_UE", "informations_matière"),
    ("pourrais-tu me donner des informations sur la matière Code_UE", "informations_matière"),
    ("est ce que tu peux me donner des informations sur la matière Code_UE", "informations_matière"),
    ("je veux des informations sur la matière Code_UE", "informations_matière"),
    ("pourrais tu afficher les détails de la matière Code_UE", "informations_matière"),
    ("je souhaite obtenir des informations sur la matière Code_UE", "informations_matière"),
    ("peux tu me renseigner sur la matière Code_UE", "informations_matière"),
    ("pourrais-tu me fournir des détails sur la matière Code_UE", "informations_matière"),
    ("j'aimerais en apprendre davantage sur la matière Code_UE", "informations_matière"),
    ("quelles sont les caractéristiques de la matière Code_UE", "informations_matière"),
    ("pourrais tu m'éclairer sur la matière Code_UE", "informations_matière"),
    ("quels sont les aspects essentiels de la matière Code_UE", "informations_matière"),
    ("peux-tu me donner des informations sur la matière Code_UE", "informations_matière"),
    ("je désire connaitre les détails de la matière Code_UE", "informations_matière"),
    ("pourrais tu me fournir des informations sur la matière Code_UE", "informations_matière"),
    ("je souhaiterais obtenir des éclaircissements sur la matière Code_UE", "informations_matière"),
    ("pourrais tu m'expliquer en quoi consiste la matière Code_UE", "informations_matière"),
    ("peux tu me dire ce que je devrais savoir sur la matière Code_UE", "informations_matière"),
    ("j'aimerais etre informé sur la matière Code_UE", "informations_matière"),
    ("pourrais tu me donner un aperçu de la matière Code_UE", "informations_matière"),
    ("j'aimerais obtenir des informations sur la matière Code_UE", "informations_matière"),


    # Exemples pour l'intention "matières_utilisant_technologie"
    ("propose moi des matières où on utilise ", "matières_utilisant_technologie"),
    ("affiche moi les matières utilisant ", "matières_utilisant_technologie"),
    ("quelles matières utilisent ", "matières_utilisant_technologie"),
    ("je veux des matières utilisant ", "matières_utilisant_technologie"),
    ("matières utilisant ", "matières_utilisant_technologie"),
    ("ue utilisant ", "matières_utilisant_technologie"),
    ("matières utilisant la technologie ", "matières_utilisant_technologie"),
    ("ue utilisant la technologie ", "matières_utilisant_technologie"),
    ("matières où on peut apprendre", "matières_utilisant_technologie"),
    ("ue où on peut apprendre", "matières_utilisant_technologie"),
    ("je veux apprendre à coder en ", "matières_utilisant_technologie"),
    ("je veux apprendre à utiliser ", "matières_utilisant_technologie"),
    ("propose des cours qui incluent ", "matières_utilisant_technologie"),
    ("affiche moi les disciplines qui utilisent ", "matières_utilisant_technologie"),
    ("quelles sont les matières qui utilisent ", "matières_utilisant_technologie"),
    ("quelles sont les ue qui utilisent ", "matières_utilisant_technologie"),
    ("je souhaite découvrir des domaines qui emploient ", "matières_utilisant_technologie"),
    ("montre moi des cours qui impliquent l'utilisation de ", "matières_utilisant_technologie"),
    ("montre moi des ue qui impliquent l'utilisation de ", "matières_utilisant_technologie"),
    ("je cherche des matières où l'on utilise la technologie ", "matières_utilisant_technologie"),
    ("je suis intéressé par des domaines qui intègrent ", "matières_utilisant_technologie"),
    ("peux tu me recommander des cours où l'on peut apprendre à utiliser ", "matières_utilisant_technologie"),
    ("je veux apprendre à coder en ", "matières_utilisant_technologie"),
    ("je suis à la recherche de cours où je peux apprendre à utiliser ", "matières_utilisant_technologie"),
    ("quelles sont les disciplines qui incorporent ", "matières_utilisant_technologie"),
    ("quelles sont les ue qui incorporent ", "matières_utilisant_technologie"),
    ("je veux des suggestions de cours qui exploitent ", "matières_utilisant_technologie"),
    ("affiche moi les matières où la technologie est utilisée ", "matières_utilisant_technologie"),
    ("je souhaite explorer des domaines d'études qui utilisent ", "matières_utilisant_technologie"),
    ("peux tu me donner des exemples de matières qui intègrent ", "matières_utilisant_technologie"),
    ("je recherche des cours qui mettent en pratique l'utilisation de ", "matières_utilisant_technologie"),
    ("propose moi des unités d'enseignements où l'on utilise des outils technologies ", "matières_utilisant_technologie"),
    ("quelles sont les matières qui requièrenet l'utilisation de ", "matières_utilisant_technologie"),
    ("je veux découvrir des domaines d'apprentissage qui intègrent ", "matières_utilisant_technologie"),
    ("affiche moi des options de cours où je peux apprendre à utiliser ", "matières_utilisant_technologie"),
    ("je suis à la recherche de matières qui utilisent ", "matières_utilisant_technologie"),
    ("je suis à la recherche de ue qui utilisent ", "matières_utilisant_technologie"),


    # Exemples pour l'intention "metiers"
    ("je veux savoir quels métiers je peux faire avec la matière", "metiers"),
    ("quels sont les métiers liés à la matière", "metiers"),
    ("quels sont les métiers liés à l'ue", "metiers"),
    ("quels métiers puis-je faire avec la matière", "metiers"),
    ("quels métiers puis-je faire avec l'ue", "metiers"),
    ("métiers liés à la matière", "metiers"),
    ("métiers liés à l'ue", "metiers"),
    ("métiers avec la matière", "metiers"),
    ("je veux devenir ", "metiers"),
    ("je veux travailler comme ", "metiers"),
    ("quels sont les débouchés professionnels en lien avec la matière" , "metiers"),
    ("je souhaite explorer les carrières possible avec la matière", "metiers"),
    ("peux tu me donner des informations sur les emplois associés à cette matière ?", "metiers"),
    ("je m'intéresse aux métiers qui impliquent la maitrise de cette matière ", "metiers"),
    ("quelles sont les options de carrière possible avec la matière", "metiers"),
    ("je recherche des informations sur les métiers qui nécessitent la maitrise de cette matière", "metiers"),
    ("quelles sont les possibilités d'emploi en lien avec la matière", "metiers"),
    ("je veux connaitre les différentes opportunités professionnelles en lien avec la matière", "metiers"),
    ("quels sont les métiers accessibles après avoir étudié cette matière", "metiers"),
    ("je suis intéressé par les perspectives professionnelles dans ce domaine", "metiers"),
    ("quels métiers sont disponibles pour ceux qui maitrisent cette matière", "metiers"),
    ("je suis interessé par le sopportunités professionnelles associées à cette matière", "metiers"),
    ("je souhaite découvrir les métiers qui requièrent des compétences dans cette matière", "metiers"),
    ("quels sont les métiers qui nécessitent une connaissances de cette matière", "metiers"),
    ("peux tu me renseigner sur les carrières possibles liées à cette matière ?", "metiers"),
    ("je cherche des informationss sur les métiers qui utilisent les concepts de cette matière", "metiers"),
    ("quels métiers puis-je envisager après avoir étudié cette matière", "metiers"),
    ("je veux explorer les différentes voies professionnelles offertes par cette matière", "metiers"),
    ("quels sont les métierspour lesquels cette matière est utile", "metiers"),
    ("je souhaite en savoir plus sur les métiers accessibles après des études de cette matière", "metiers"),
    ("quels sont les débouchés professionnels après avoir étudié cette matière", "metiers"),
    ("je suis curieux des métiers qui nécessitent une expertise dans cette matière ", "metiers"),
    ("peux tu me fournir des informations sur les métiers qui nécessitent une expertise dans cette matière", "metiers"),
    ("je souhaite explorer les opportunités d'emploi asssociées à cette matière", "metiers"),
    ]
    
    X_train, y_train = zip(*train_data)
    pipeline.fit(X_train, y_train)
    
    return pipeline.predict([phrase])[0]

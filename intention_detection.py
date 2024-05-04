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
    
    # Exemples pour l'intention "nom_ue"
    ("quel est le nom de l'ue ", "nom_ue"),
    ("Peux-tu me dire le nom de l'UE correspondant au code", "nom_ue"),
    ("Comment s'appelle l'UE avec le code ", "nom_ue"),
    ("Quel est le titre de l'UE identifiée par ", "nom_ue"),
    ("J'aimerais connaître le nom de l'UE ", "nom_ue"),
    ("Pourrais-tu me dire le nom de l'UE qui a pour code ", "nom_ue"),
    ("Quel est le nom de l'unité d'enseignement ", "nom_ue"),
    ("Je cherche le nom de l'UE associée à , pourrais-tu m'aider ? ", "nom_ue"),
    ("Sais-tu quel est le nom de l'UE ", "nom_ue"),
    ("Je me demande comment s'appelle l'UE dont le code est ", "nom_ue"),
    ("Quel est le nom de l'unité d'enseignement ", "nom_ue"),
    ("Quel est le nom de l'unité d'enseignement ", "nom_ue"),
    ("Quel est le nom de l'unité d'enseignement ", "nom_ue"),
    
    #Exemples pour l'intention "Type_ue"
    ("Quel est le type de l'unité d'enseignement ", "Type_ue"),
    ("Quel est le type de l'UE nommée  ", "Type_ue"),
    ("Peux-tu me dire le type de l'UE correspondant au code ", "Type_ue"),
    ("Quel est le type de l'UE identifiée par ", "Type_ue"),
    ("Quel est le type de l'UE ", "Type_ue"),
    ("J'aimerais connaître le type de l'UE ", "Type_ue"),
    ("Pourrais-tu me dire le type de l'UE qui a pour code  ", "Type_ue"),
    ("Je cherche le type de l'UE associée à ,pourrais-tu m'aider ? ", "Type_ue"),
    ("Sais-tu quel est le type de l'UE ", "Type_ue"),
    ("Je me demande quel est le type de l'UE dont le code est ", "Type_ue"),
    (" est ce qu'elle est une TM ou une CS ou une HT ou une CT ", "Type_ue"),
    
    #Exemples pour l'intention "Nombre_crédits"
    (" Combien de crédits vaut l'UE nommée ", "Nombre_crédits"),
    (" Peux-tu me dire le nombre de crédits de l'UE correspondant au code ", "Nombre_crédits"),
    (" Quel est le nombre de crédits de l'UE identifiée par  ", "Nombre_crédits"),
    (" Quel est le nombre de crédits de l'UE  ", "Nombre_crédits"),
    (" J'aimerais connaître le nombre de crédits de l'UE  ", "Nombre_crédits"),
    (" Je cherche le nombre de crédits de l'UE associée à ", "Nombre_crédits"),
    (" Sais-tu quel est le nombre de crédits de l'UE ", "Nombre_crédits"),
    (" Je me demande combien de crédits vaut l'UE dont le code est  ", "Nombre_crédits"),
    
    #Exemples pour l'intention "langue_ue"
    ("Quelle est la langue de l'UE nommée ", "langue_ue"),
    ("Peux-tu me dire la langue de l'UE correspondant au code ", "langue_ue"),
    ("Quelle est la langue de l'UE identifiée par ", "langue_ue"),
    ("Quelle est la langue de l'UE ", "langue_ue"),
    ("J'aimerais connaître la langue de l'UE ", "langue_ue"),
    ("Je cherche la langue de l'UE associée à ", "langue_ue"),
    ("Sais-tu quelle est la langue de l'UE ", "langue_ue"),
    ("Je me demande quelle est la langue de l'UE dont le code est ", "langue_ue"),
    
    #Exemples pour l'intention "Branche_ue"
    ("Quelle est la branche de l'UE nommée ", "Branches_ue"),
    ("Peux-tu me dire la branche de l'UE correspondant au code ", "Branches_ue"),
    ("Quelle est la branche de l'UE identifiée par ", "Branches_ue"),
    ("Quelle est la branche de l'UE ", "Branches_ue"),
    ("J'aimerais connaître la branche de l'UE ", "Branches_ue"),
    ("Je cherche la branche de l'UE associée à ", "Branches_ue"),
    ("Sais-tu quelle est la branche de l'UE ", "Branches_ue"),
    ("Je me demande quelle est la branche de l'UE dont le code est ", "Branches_ue"),

    # Exemples pour l'intention "Programme_ue"
    ("Quel est le programme de l'UE nommée ", "Programme_ue"),
    ("Peux-tu me dire le programme de l'UE correspondant au code ", "Programme_ue"),
    ("Quel est le programme de l'UE identifiée par ", "Programme_ue"),
    ("Quel est le programme de l'UE ", "Programme_ue"),
    ("J'aimerais connaître le programme de l'UE ", "Programme_ue"),
    ("Je cherche le programme de l'UE associée à ", "Programme_ue"),
    ("Sais-tu quel est le programme de l'UE ", "Programme_ue"),
    ("Je me demande quel est le programme de l'UE dont le code est ", "Programme_ue"),
    
    #Exemples pour l'intention "Objectif_ue"
    ("Quel est l'objectif de l'UE nommée ", "Objectif_ue"),
    ("Peux-tu me dire l'objectif de l'UE correspondant au code ", "Objectif_ue"),
    ("Quel est l'objectif de l'UE identifiée par ", "Objectif_ue"),
    ("Quel est l'objectif de l'UE ", "Objectif_ue"),
    ("J'aimerais connaître l'objectif de l'UE ", "Objectif_ue"),
    ("Je cherche l'objectif de l'UE associée à ", "Objectif_ue"),
    ("Sais-tu quel est l'objectif de l'UE ", "Objectif_ue"),
    ("Je me demande quel est l'objectif de l'UE dont le code est ", "Objectif_ue"),
    
    #Exemples pour l'intention "Modalite_ue"
    ("Quelles sont les modalités de l'UE nommée ", "Modalite_ue"),
    ("Peux-tu me dire les modalités de l'UE correspondant au code ", "Modalite_ue"),
    ("Quelles sont les modalités de l'UE identifiée par ", "Modalite_ue"),
    ("Quelles sont les modalités de l'UE ", "Modalite_ue"),
    ("J'aimerais connaître les modalités de l'UE ", "Modalite_ue"),
    ("Je cherche les modalités de l'UE associée à ", "Modalite_ue"),
    ("Sais-tu quelles sont les modalités de l'UE ", "Modalite_ue"),
    ("Je me demande quelles sont les modalités de l'UE dont le code est ", "Modalite_ue"),
    
    #Exemples pour l'intention "Ouvert_TC"
    ("Est-ce que l'UE nommée est ouverte au TC ?", "Ouvert_TC"),
    ("Peux-tu me dire si l'UE correspondant au code est ouverte au TC ?", "Ouvert_TC"),
    ("Est-ce que l'UE identifiée par est ouverte au TC ?", "Ouvert_TC"),
    ("Est-ce que l'UE est ouverte au TC ?", "Ouvert_TC"),
    ("Est-ce que l'UE est ouverte au TC ?", "Ouvert_TC"),
    ("Je cherche à savoir si l'UE associée à est ouverte au TC", "Ouvert_TC"),
    ("Sais-tu si l'UE est ouverte au TC ?", "Ouvert_TC"),
    ("Je me demande si l'UE dont le code est est ouverte au TC", "Ouvert_TC"),
    
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
    ("Je veux savoir quels métiers je peux faire avec la matière", "metiers_matiere"),
    ("Quels sont les métiers liés à la matière", "metiers_matiere"),
    ("Quels métiers puis-je faire avec la matière", "metiers_matiere"),
    ("métiers liés à la matière", "metiers_matiere"),
    ("métiers avec la matière", "metiers_matiere"),
    ("Je veux devenir ", "metiers"),
    ("je veux travailler comme ", "metiers"),
    ("je veux travailler en tant que ", "metiers"),
    ("je veux faire comme job ", "metiers"),
    ("je veux faire comme Métier ", "metiers"),
    ("je veux faire comme boulot ", "metiers"),
    ("Mon projet professionnel est", "metiers"),
    ("Quels sont les débouchés professionnels en lien avec la matière" , "metiers_matiere"),
    ("Je souhaite explorer les carrières possible avec la matière", "metiers_matiere"),
    ("Peux tu me donner des informations sur les emplois associés à cette matière ?", "metiers_matiere"),
    ("Je m'intéresse aux métiers qui impliquent la maitrise de cette matière ", "metiers_matiere"),
    ("Quelles sont les options de carrière possible avec la matière", "metiers_matiere"),
    ("Je recherche des informations sur les métiers qui nécessitent la maitrise de cette matière", "metiers_matiere"),
    ("Quelles sont les possibilités d'emploi en lien avec la matière", "metiers_matiere"),
    ("Je veux connaitre les différentes opportunités professionnelles en lien avec la matière", "metiers_matiere"),
    ("Quels sont les métiers accessibles après avoir étudié cette matière", "metiers_matiere"),
    ("Je suis intéressé par les perspectives professionnelles dans ce domaine", "metiers"),
    ("Quels métiers sont disponibles pour ceux qui maitrisent cette matière", "metiers_matiere"),
    ("Je suis interessé par les opportunités professionnelles associées à cette matière", "metiers_matiere"),
    ("Je souhaite découvrir les métiers qui requièrent des compétences dans cette matière", "metiers_matiere"),
    ("Quels sont les métiers qui nécessitent une connaissances de cette matière", "metiers_matiere"),
    ("Peux tu me renseigner sur les carrières possibles liées à cette matière ?", "metiers_matiere"),
    ("Je cherche des informationss sur les métiers qui utilisent les concepts de cette matière", "metiers_matiere"),
    ("Quels métiers puis-je envisager après avoir étudié cette matière", "metiers_matiere"),
    ("Je veux explorer les différentes voies professionnelles offertes par cette matière", "metiers_matiere"),
    ("Quels sont les métiers pour lesquels cette matière est utile", "metiers_matiere"),
    ("Je souhaite en savoir plus sur les métiers accessibles après des études de cette matière", "metiers_matiere"),
    ("Quels sont les débouchés professionnels après avoir étudié cette matière", "metiers_matiere"),
    ("Je suis curieux des métiers qui nécessitent une expertise dans cette matière ", "metiers_matiere"),
    ("Peux tu me fournir des informations sur les métiers qui nécessitent une expertise dans cette matière", "metiers_matiere"),
    ("Je souhaite explorer les opportunités d'emploi asssociées à cette matière", "metiers_matiere"),
    ]
    
    X_train, y_train = zip(*train_data)
    pipeline.fit(X_train, y_train)
    
    return pipeline.predict([phrase])[0]


import re
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

def extraire_metier_demande(phrase):
    print(phrase)
    # Liste des métiers
    metiers = ["ingénieur logiciel", "développeur logiciel", "analyste en assurance qualité logicielle", "chef de projet informatique", "architecte logiciel", "consultant en ingénierie logicielle", "ingénieur en sécurité informatique", "ingénieur en génie logiciel embarqué", "développeur web full-stack", "expert en méthodes agiles", "analyste fonctionnel", "architecte de systèmes d'information", "consultant en système d'information", "ingénieur en système d'information", "analyste en gestion des exigences", "concepteur-développeur en informatique", "analyste en ingénierie des systèmes", "expert en modélisation de systèmes d'information"

            "ingénieur en gestion des connaissances", "analyste en ingénierie des connaissances", "architecte de systèmes de gestion des connaissances", "consultant en ingénierie des connaissances", "développeur d'ontologies", "expert en représentation des connaissances", "analyste en gestion des données", "spécialiste en systèmes multi-agents", "analyste d'organisation", "consultant en gestion des processus", "analyste en systèmes d'information", "architecte d'entreprise", "consultant en gestion du changement", "expert en modélisation d'entreprise", "spécialiste en analyse des flux de travail", "consultant en stratégie organisationnelle", "analyste des opérations", "conseiller en restructuration organisationnelle", "administrateur système", "ingénieur réseau", "administrateur de bases de données", "architecte cloud", "ingénieur en virtualisation", "architecte en sécurité informatique", "consultant en infrastructure informatique", "analyste en gestion des données", "spécialiste en stockage de données", "ingénieur devops", "analyste de données", "architecte de données", "scientifique des données", "ingénieur en données", "consultant en stratégie de données",
            
           "gestionnaire de projet data", "spécialiste en visualisation de données", "analyste de business intelligence", "expert en exploration de données", "développeur big data", "chef de projet informatique", "chef de projet technique", "chef de projet digital", "manager de projet informatique", "directeur de projet informatique", "consultant en gestion de projet", "responsable de bureau de projet", "coordinateur de projet informatique", "chargé de mission en gestion de projet", "analyste de projet", "concepteur d'interfaces utilisateur", "designer d'expérience utilisateur (ux designer)", "ergonome informatique", "développeur d'applications interactives", "architecte de l'interaction", "chercheur en interaction homme-machine", "responsable de la conception multimodale", "consultant en conception d'interfaces", "spécialiste en accessibilité numérique", "analyste en expérience utilisateur (ux analyst)", "analyste fonctionnel", "architecte de systèmes d'information", "consultant en système d'information", "ingénieur en système d'information", "analyste en gestion des exigences", "concepteur-développeur en informatique", "chef de projet informatique", "analyste en ingénierie des systèmes", "expert en modélisation de systèmes d'information"

            
            "concepteur d'ihm", "développeur d'interfaces utilisateur", "ergonome logiciel", "ingénieur en ergonomie des interfaces", "designer d'expérience utilisateur", "architecte cloud", "ingénieur devops", "administrateur système cloud", "spécialiste en virtualisation", "ingénieur en sécurité cloud", "architecte réseau cloud", "administrateur de bases de données cloud", "ingénieur cloud computing", "consultant en architecture cloud", "responsable de l'optimisation énergétique des architectures cloud", "data analyst", "data scientist", "ingénieur big data", "architecte big data", "développeur big data", "ingénieur en intelligence artificielle", "spécialiste en analyse de données", "consultant en big data", "responsable de la gestion des données", "analyste en business intelligence", "consultant en persuasion numérique", "concepteur de systèmes persuasifs", "analyste en ergonomie et persuasion", "ingénieur en conception d'interfaces persuasives", "expert en expérience utilisateur et persuasion", "consultant en développement urbain intelligent", "analyste en technologie des villes intelligentes", "ingénieur en systèmes de transport intelligents", "expert en gouvernance des villes intelligentes", "consultant en travail collaboratif", "analyste en travail coopératif", "ingénieur en travail collaboratif", "expert en plateformes collaboratives", "responsable de projet collaboratif", "analyste en machine learning"

            "consultant en protection des données", "analyste en protection des données", "ingénieur en protection des données", "expert en conformité rgpd", "responsable de la protection des données", "consultant en travail collaboratif", "analyste en travail coopératif", "ingénieur en travail collaboratif", "expert en plateformes collaboratives", "responsable de projet collaboratif", "consultant en systèmes documentaires", "analyste en systèmes documentaires", "ingénieur en systèmes documentaires", "expert en systèmes documentaires", "responsable de projet documentaire", "consultant en décisionnel", "analyste en décisionnel", "ingénieur en décisionnel", "expert en décisionnel", "responsable de projet décisionnel", "consultant en organisation", "analyste en organisation", "ingénieur en organisation", "expert en organisation", "responsable de projet organisationnel", "consultant en processus", "analyste en processus", "ingénieur en processus", "expert en processus", "responsable de projet processus", "consultant en sécurité mobile", "analyste en sécurité mobile", "ingénieur en sécurité mobile", "expert en sécurité mobile", "responsable de projet sécurité mobile", "consultant en fouille de données", "analyste en fouille de données", "ingénieur en fouille de données", "expert en fouille de données", "responsable de projet fouille de données"

            
           "consultant en gestion de projet", "analyste en gestion de projet", "ingénieur en gestion de projet", "expert en gestion de projet", "responsable de projet", "consultant en gestion des si", "analyste en gestion des si", "ingénieur en gestion des si", "expert en gestion des si", "responsable de projet si", "consultant en data analytics", "analyste en data analytics", "ingénieur en data analytics", "expert en data analytics", "responsable de projet data analytics", "consultant en data science", "analyste en data science", "ingénieur en data science", "expert en data science", "responsable de projet data science", "consultant en visualisation", "analyste en visualisation", "ingénieur en visualisation", "expert en visualisation", "responsable de projet visualisation", "consultant en plateformes", "analyste en plateformes", "ingénieur en plateformes", "expert en plateformes", "responsable de projet plateformes"

            
            "consultant en technologies du si", "analyste en technologies du si", "ingénieur en technologies du si", "expert en technologies du si", "responsable de projet technologies du si", "consultant en architectures orientées services", "analyste en architectures orientées services", "ingénieur en architectures orientées services", "expert en architectures orientées services", "responsable de projet architectures orientées services", "physicien", "ingénieur en physique", "chercheur en physique", "expert en physique", "expert en spectroscopie", "ingénieur en caractérisation des matériaux", "chercheur en caractérisation des matériaux", "expert en caractérisation des matériaux", "ingénieur en métallurgie", "chercheur en métallurgie", "expert en métallurgie", "responsable de projet métallurgie"

            
            
           "ingénieur en nanomatériaux", "chercheur en nanomatériaux", "expert en nanomatériaux", "responsable de projet nanomatériaux", "ingénieur en design", "chercheur en design", "expert en design", "responsable de projet design", "ecological designer", "sustainability consultant", "environmental analyst", "green infrastructure planner", "ecosystem management specialist", "climate resilience planner", "sustainable development strategist", "biodiversity conservationist", "natural resource manager", "environmental policy advisor", "ingénieur en mécanique des matériaux", "chercheur en mécanique des matériaux", "expert en mécanique des matériaux", "responsable de projet mécanique des matériaux", "ingénieur en analyse numérique", "chercheur en analyse numérique", "expert en analyse numérique", "responsable de projet analyse numérique", "ingénieur en chimie des matériaux", "chercheur en chimie des matériaux", "expert en chimie des matériaux", "responsable de projet chimie des matériaux", "ingénieur en physique des polymères", "chercheur en physique des polymères", "expert en physique des polymères", "responsable de projet physique des polymères", "ingénieur en semi-conducteurs", "chercheur en semi-conducteurs", "expert en semi-conducteurs", "responsable de projet semi-conducteurs"

            
            
            "ingénieur en caractérisation microscopique", "chercheur en caractérisation microscopique", "expert en caractérisation microscopique", "responsable de projet caractérisation microscopique", "ingénieur en mise en forme des matériaux", "chercheur en mise en forme des matériaux", "expert en mise en forme des matériaux", "responsable de projet mise en forme des matériaux", "ingénieur en communications optiques", "chercheur en communications optiques", "expert en communications optiques", "responsable de projet communications optiques", "ingénieur en management du cycle de vie des matériaux", "chercheur en management du cycle de vie des matériaux", "expert en management du cycle de vie des matériaux", "responsable de projet management du cycle de vie des matériaux"

            
            
           "consultant en stratégie dans le secteur des matériaux", "chercheur en commerce des matériaux", "expert en commerce des matériaux", "responsable de projet commerce des matériaux", "ingénieur en matériaux pour l'énergie", "chercheur en matériaux pour l'énergie", "expert en matériaux pour l'énergie", "responsable de projet matériaux pour l'énergie", "ingénieur en calcul de structures", "ingénieur en mécanique des matériaux", "ingénieur en simulation numérique", "ingénieur en validation des modèles", "technicien en essais mécaniques", "ingénieur en conception mécanique", "ingénieur en fiabilité des structures", "ingénieur en matériaux et structures", "ingénieur en recherche et développement", "ingénieur en mécatronique", "ingénieur en procédés de mise en forme", "ingénieur en normes et réglementation", "chercheur en normes et réglementation", "expert en normes et réglementation", "responsable de projet normes et réglementation", "responsable des achats et des approvisionnements", "analyste des coûts", "ingénieur en optimisation des processus", "consultant en gestion des coûts", "responsable des achats stratégiques", "gestionnaire de contrats d'approvisionnement"

            
            
            "responsable des achats et des approvisionnements", "analyste des coûts", "ingénieur en optimisation des processus", "consultant en gestion des coûts", "responsable des achats stratégiques", "gestionnaire de contrats d'approvisionnement", "responsable environnement", "responsable développement durable", "ingénieur en énergie renouvelable", "chargé de recherche en matériaux pour l'énergie", "ingénieur en développement de solutions énergétiques durables", "responsable en gestion de l'énergie et des matériaux", "expert en matériaux pour la transition énergétique", "ingénieur en efficacité énergétique des matériaux", "technicien en conception de systèmes énergétiques durables", "analyste en stratégies énergétiques et matériaux", "ingénieur en conception mécanique", "ingénieur en simulation numérique", "ingénieur en validation des modèles", "technicien en essais mécaniques", "ingénieur en fiabilité des structures", "ingénieur en matériaux et structures", "ingénieur en recherche et développement", "ingénieur en mécatronique", "ingénieur en automatique", "ingénieur en contrôle-commande", "ingénieur en systèmes embarqués", "ingénieur en robotique", "ingénieur en modélisation et simulation de systèmes dynamiques", "ingénieur en développement de correcteurs automatiques", "ingénieur en conception de systèmes de régulation", "ingénieur en traitement du signal", "ingénieur en instrumentation et mesure", "ingénieur en systèmes temps réel", "ingénieur en simulation numérique et modélisation", "ingénieur en informatique industrielle", "chef de projet", "responsable de projet", "chef de projet informatique", "chef de projet industriel", "chef de projet r&d", "chef de projet technique", "chef de projet web", "chef de projet marketing", "chef de projet événementiel", "chef de projet communication", "chef de projet digital", "chef de projet e-commerce", "ingénieur en réseaux d'entreprise", "ingénieur en sécurité des réseaux", "ingénieur en télécommunications", "ingénieur en réseaux et systèmes", "ingénieur en informatique embarquée"

            
            
            "ingénieur en électronique", "ingénieur en instrumentation et mesure", "ingénieur en capteurs", "ingénieur en systèmes embarqués", "ingénieur en traitement du signal", "ingénieur en automatique", "ingénieur en contrôle-commande", "ingénieur en robotique", "ingénieur en modélisation et simulation de systèmes dynamiques", "ingénieur en développement de correcteurs automatiques", "ingénieur en conception de systèmes de régulation", "ingénieur en systèmes temps réel", "ingénieur en simulation numérique et modélisation", "ingénieur en informatique industrielle", "analyste fonctionnel", "architecte logiciel", "concepteur-développeur en informatique", "ingénieur en système d'information", "développeur web", "designer d'interfaces utilisateur", "développeur d'applications web", "chef de projet informatique"

            
            "ingénieur en informatique industrielle", "ingénieur en informatique embarquée", "ingénieur en télécommunications", "ingénieur en traitement du signal", "développeur en compression de données", "ingénieur en sécurité des communications", "chercheur en intelligence artificielle et apprentissage automatique", "consultant en analyse de données", "data analyste", "analyste de données", "data engineer", "ingénieur données", "développeur de bases de données", "data architect", "architecte de données", "ingénieur en conception de systèmes de communication", "ingénieur en photonique", "analyste de performance des réseaux", "ingénieur en transmission de données", "ingénieur réseau", "administrateur réseau", "architecte réseau", "ingénieur en réseaux et systèmes", "spécialiste en sécurité réseau", "ingénieur cybersecurity", "ingénieur développement logiciels", "ingénieur devops", "ingénieur développeur full-stack"

            
            
            "ingénieur iot", "architecte iot", "ingénieur en réseaux et systèmes", "ingénieur en systèmes embarqués", "ingénieur en traitement de données iot", "ingénieur cloud", "data scientist", "ingénieur des données", "data engineer", "ingénieur data", "analyste des données", "ingénieur en communications unifiées", "architecte de solutions de communications unifiées", "ingénieur en sécurité des communications", "ingénieur cdn", "administrateur de système", "ingénieur en réseaux mobiles", "ingénieur en sécurité des réseaux sans fil", "ingénieur en virtualisation", "ingénieur en datacenter", "architecte de réseaux", "ingénieur en qualité de service (qos)", "cryptographe", "ingénieur cybersécurité", "consultant en sécurité des données", "architecte de sécurité réseau", "ingénieur en réseaux iot", "architecte de réseaux sans fil", "ingénieur en géolocalisation", "ingénieur en navigation par satellite"

            
            
          "ingénieur en machine learning", "ingénieur en deep learning", "ingénieur en intelligence artificielle embarquée", "ingénieur en réseaux et systèmes", "ingénieur en ux", "concepteur d'interfaces utilisateur mobiles", "développeur d'applications mobiles android", "ingénieur en objets connectés multimédia", "ingénieur en traitement d'images", "ingénieur en intelligence artificielle pour la transmission de données", "ingénieur en intelligence artificielle", "ingénieur en apprentissage automatique", "consultant en intelligence artificielle", "spécialiste en data science", "architecte d'algorithmes prédictifs et génératifs", "chercheur en sciences", "analyste de données de recherche", "assistant de recherche", "ingénieur en objets connectés", "consultant en technologies iot", "architecte de systèmes embarqués", "analyste en durabilité", "consultant en développement durable", "chercheur en systèmes socio-écologiques", "analyste numérique", "ingénieur en optimisation", "ingénieur en simulation numérique", "ingénieur en sûreté de fonctionnement", "ingénieur en sécurité des systèmes", "consultant en analyse de risques", "ingénieur en statistiques", "consultant en analyse statistique", "ingénieur en fiabilité et qualité", "ingénieur en modélisation de systèmes discrets", "ingénieur en logistique", "analyste en simulation de processus", "chef de projet qualité", "consultant en amélioration continue", "auditeur qualité", "ingénieur logistique", "chef de projet logistique", "analyste des opérations", "consultant en gestion de la chaîne d'approvisionnement"

            
            
            "ingénieur logistique", "chef de projet logistique", "analyste en gestion des stocks", "consultant en logistique et gestion des stocks", "consultant en gestion de la chaîne d'approvisionnement", "consultant en gestion industrielle", "chef de projet erp", "administrateur de bases de données", "analyste en systèmes d'information", "ingénieur en automatique", "ingénieur en contrôle de processus", "ingénieur en simulation industrielle", "analyste de systèmes de production", "ingénieur en systèmes complexes", "analyste en modélisation de systèmes complexes", "consultant en gestion des systèmes complexes", "chercheur en sciences des systèmes", "ingénieur en simulations", "analyste en modélisation de systèmes", "consultant en optimisation des processus industriels", "responsable logistique", "responsable transport", "responsable import-export", "responsable achats", "responsable approvisionnement", "responsable supply chain", "responsable logistique internationale", "responsable logistique industrielle", "gestionnaire d'entrepôt", "superviseur de la chaîne logistique interne", "planificateur de transport", "analyste de réseau de transport", "spécialiste de la logistique de transport", "gestionnaire de chaîne logistique", "consultant en logistique et supply chain", "consultant en soutien logistique", "ingénieur en mobilité urbaine", "planificateur de transport en commun", "développeur de solutions de mobilité intelligente"

            
            
            "chef de projet en amélioration continue", "responsable qualité", "analyste de processus", "consultant en lean six sigma", "expert en gestion de la performance opérationnelle", "ingénieur industriel", "ingénieur en planification et ordonnancement", "chef de projet en gestion de production", "planificateur de la chaîne logistique", "analyste en optimisation des processus industriels", "consultant en optimisation de la production", "analyste en stratégie tarifaire", "consultant en gestion des revenus", "analyste en économie d'entreprise", "responsable tarification et revenus", "ingénieur en production", "analyste de la chaîne d'approvisionnement", "responsable de la planification de la production", "consultant en gestion de la production", "consultant sap", "analyste fonctionnel sap", "administrateur système sap", "gestionnaire de projet informatique", "chef de projet industrie 4.0", "analyste de données industrielles", "responsable de la logistique et de la chaîne d'approvisionnement 4.0", "architecte de solutions mes/erp", "ingénieur en fiabilité des systèmes", "ingénieur en sécurité des systèmes", "ingénieur en qualité", "analyste en sûreté de fonctionnement", "responsable de la gestion des risques", "consultant en management de la qualité", "ingénieur en maintenance prédictive", "expert en gestion des risques industriels", "consultant en maintenance industrielle", "expert en fiabilité des systèmes", "ingénieur en sécurité industrielle", "analyste des risques professionnels", "consultant en conformité réglementaire", "responsable qhse (qualité, hygiène, sécurité, environnement)", "analyste des risques industriels", "ingénieur en sécurité des installations", "spécialiste en gestion de la sécurité environnementale", "consultant en gestion des risques et crises"

            ]
    
     # Convertir tous les métiers en minuscules
    metiers = [metier.lower() for metier in metiers]
    
    pattern = r'\b(?:' + '|'.join(map(re.escape, metiers)) + r')\b'
    
    # Rechercher les correspondances dans la phrase
    predicted_metiers = re.findall(pattern, phrase.lower(), flags=re.IGNORECASE)
    
    return predicted_metiers

'''
metiers_set = set(metiers)

# Conversion de l'ensemble en une liste
metiers_unique = list(metiers_set)

# Phrases génériques
phrases = ["Je veux devenir {}", "je veux travailler comme {} ", "je veux travailler en tant que {} ","je veux faire comme job {} ", "je veux faire comme Métier {}", "je veux faire comme boulot {}"]

# Génération de données synthétiques

# 1. Collecte des données d'entraînement
training_data = {}
for _ in range(10000):
    metiers_unique = random.choice(metiers)
    phrase = random.choice(phrases).format(metiers_unique)
    training_data[phrase] = [metiers_unique]
    
 # Remplacez par vos données annotées

# 2. Prétraitement des données
class MetierDataset(Dataset):
    def __init__(self, data):
        self.samples = []
        self.vocab = {'UNK': 0}  # Initialisez votre vocabulaire avec UNK
        self.label_encoder = {'UNK': 0}  # Initialisez votre encodeur d'étiquettes avec UNK
        for text, labels in data.items():
            self.samples.append((text, labels))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text, labels = self.samples[idx]

        # Tokenisation
        nlp = spacy.load('fr_core_news_sm')
        doc = nlp(text)
        tokens = [token.text for token in doc]

        # Conversion en indices
        indices = [self.vocab.get(token, self.vocab['UNK']) for token in tokens]

        # Padding des représentations tensorielles pour les amener à la même longueur
        max_length = 20  # Choisissez une longueur maximale appropriée
        if len(indices) < max_length:
            indices += [self.vocab['UNK']] * (max_length - len(indices))
        elif len(indices) > max_length:
            indices = indices[:max_length]

        # Encodage des jetons
        tensor_representation = torch.tensor(indices).float()

        # Conversion des étiquettes
        if len(labels) == 1:  # Si chaque phrase a une seule étiquette
            label = self.label_encoder.get(labels[0], self.label_encoder['UNK'])
        else:  # Si chaque phrase peut avoir plusieurs étiquettes
            label = torch.zeros(len(self.label_encoder))
            for label_name in labels:
                label[self.label_encoder.get(label_name, self.label_encoder['UNK'])] = 1

        return tensor_representation, label





# 3. Construction du modèle
# 3. Construction du modèle
# 3. Construction du modèle
class MetierDetector(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MetierDetector, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.LogSoftmax(dim=-1)  # Modifier dim en 2 pour calculer le softmax sur la deuxième dimension
        )

    def forward(self, x):
        # Convertir les données d'entrée en type Float
        x = x.float()

        _, (h_n, _) = self.rnn(x)
        print("Taille du tenseur avant log_softmax:", h_n.size())  # Ajout de cette ligne pour imprimer la taille du tenseur
        output = self.fc(h_n.squeeze(0))
        return output


# 4. Entraînement du modèle
train_dataset = MetierDataset(training_data)
num_epochs = 10
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Définir la fonction de perte (criterion)
criterion = nn.CrossEntropyLoss()

# Définir l'optimiseur (optimizer)
model = MetierDetector(input_size=len(train_dataset.vocab), hidden_size=64, output_size=len(train_dataset.label_encoder))
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    # Mettre le modèle en mode d'entraînement
    model.train()

    # Créer une barre de progression pour l'entraînement
    train_loader = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

    # Itérer sur les lots de données
    for batch_inputs, batch_labels in train_loader:
        # Réinitialiser le gradient
        optimizer.zero_grad()

        # Propager les données dans le modèle
        outputs = model(batch_inputs)

        # Convertir les sorties en Float
        outputs = outputs.float()

        # Calculer la perte
        loss = criterion(outputs, batch_labels)

        # Rétropropagation du gradient
        loss.backward()

        # Mise à jour des paramètres
        optimizer.step()
        
        batch_labels = batch_labels.long()

        # Mettre à jour la barre de progression avec la perte actuelle
        train_loader.set_postfix(loss=loss.item())  # Affichez la perte actuelle dans la barre de progression

# Après l'entraînement
print("Entraînement terminé !")






# 6. Utilisation du modèle
def extraire_metier_demande(phrase):
    # À faire : Prétraitement de la phrase, conversion en représentation numérique, etc.
    # Exemple fictif :
    tensor_representation = torch.randn(1, 10, 300)  # Exemple de représentation tensorielle de la phrase
    num_classes = 3  # Nombre de classes (métiers) à prédire
    # Utilisation du modèle pour prédire les métiers
    model = MetierDetector(input_size=300, hidden_size=64, output_size=num_classes)
    model.load_state_dict(torch.load('path_to_trained_model.pth'))
    model.eval()
    with torch.no_grad():
        prediction = model(tensor_representation)
    
    # Convertir les prédictions en métiers
    predicted_metiers = []  # À faire selon vos besoins
    
    return predicted_metiers
'''
def traiter_intention_metiers(phrase):
    metier_demande = extraire_metier_demande(phrase)
    matieres_trouvees = trouver_matieres_par_metier(metier_demande)
    return generer_reponse_matieres_par_metier(matieres_trouvees, metier_demande)

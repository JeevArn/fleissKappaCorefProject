a1={
    0: [['vie', 1, 1], ['combat', 4, 1], ['vie', 8, 3], ['combat', 11, 3], ['voyage', 18, 3], ['vie', 303, 3], ['elle', 309, 3], ['vie', 350, 3], ['multitude', 353, 3]],
    1: [['vie', 8, 2], ['combat', 11, 2]], 
    2: [['perception', 65, 1], ['elle', 78, 1]], 
    3: [['monde', 107, 1], ['monde', 127, 1]], 
    4: [['conscience', 117, 2], ['ses', 142, 2]], 
    5: [['idée', 123, 3], ['vision', 150, 1], ['vision', 199, 1]], 
    6: [['gens', 251, 1], ['leurs', 263, 1], ['se', 268, 1], ['leur', 272, 1], ['leurs', 275, 1], ['leurs', 278, 1]], 
    7: [['vie', 303, 2], ['elle', 309, 2], ['vie', 350, 2], ['multitude', 353, 2]], 
    8: [['moments', 332, 2], ['Certains', 394, 2], ['autres', 416, 3], ['leurs', 426, 1]], 
    9: [['conclusion', 434, 2], ['idée', 437, 2]], 
    10: [['idée', 437, 3], ['vie', 440, 1], ['combat', 443, 1], ['métaphore', 447, 1], ['vie', 494, 1]], 
    11: [['guerre', 503, 1], ['s', 507, 1]]
    }

a2={
    0: [['vie', 1, 1], ['combat', 4, 1], ['vie', 8, 3], ['combat', 11, 3], ['voyage', 18, 3], ['vie', 303, 3], ['elle', 309, 3], ['vie', 350, 3], ['multitude', 353, 3]],
    1: [['vie', 8, 2], ['combat', 11, 2]], 
    2: [['perception', 65, 1], ['elle', 78, 1]], 
    3: [['monde', 107, 1], ['monde', 127, 1]], 
    4: [['conscience', 117, 2], ['ses', 142, 2]], 
    5: [['idée', 123, 3], ['vision', 150, 1], ['vision', 199, 1]], 
    6: [['gens', 251, 1], ['leurs', 263, 1], ['se', 268, 1], ['leur', 272, 1], ['leurs', 275, 1], ['leurs', 278, 1]], 
    7: [['vie', 303, 2], ['elle', 309, 2], ['vie', 350, 2], ['multitude', 353, 2]], 
    8: [['moments', 332, 2], ['Certains', 394, 2], ['autres', 416, 3], ['leurs', 426, 1]], 
    9: [['conclusion', 434, 2], ['idée', 437, 2]], 
    10: [['idée', 437, 3], ['vie', 440, 1], ['combat', 443, 1], ['métaphore', 447, 1], ['vie', 494, 1]], 
    11: [['guerre', 503, 1], ['s', 507, 1]]
    }

a3={
    0: [['vie', 1, 1], ['combat', 4, 1], ['vie', 8, 3], ['combat', 11, 3], ['voyage', 18, 3], ['vie', 303, 3], ['elle', 309, 3], ['vie', 350, 3], ['multitude', 353, 3]],
    1: [['vie', 8, 2], ['combat', 11, 2]], 
    2: [['perception', 65, 1], ['elle', 78, 1]], 
    3: [['monde', 107, 1], ['monde', 127, 1]], 
    4: [['conscience', 117, 2], ['ses', 142, 2]], 
    5: [['idée', 123, 3], ['vision', 150, 1], ['vision', 199, 1]], 
    6: [['gens', 251, 1], ['leurs', 263, 1], ['se', 268, 1], ['leur', 272, 1], ['leurs', 275, 1], ['leurs', 278, 1]], 
    7: [['vie', 303, 2], ['elle', 309, 2], ['vie', 350, 2], ['multitude', 353, 2]], 
    8: [['moments', 332, 2], ['Certains', 394, 2], ['autres', 416, 3], ['leurs', 426, 1]], 
    9: [['conclusion', 434, 2], ['idée', 437, 2]], 
    10: [['idée', 437, 3], ['vie', 440, 1], ['combat', 443, 1], ['métaphore', 447, 1], ['vie', 494, 1]], 
    11: [['guerre', 503, 1], ['s', 507, 1]]
    }

# somme du kappa de fleiss pour chaque chaine de coref (pour pouvoir ensuite faire la moyenne)
sum_kappa = 0

# pour chaque chaine de coref de chaque annotator
for ch_a1, ch_a2, ch_a3 in zip(a1.values(), a2.values(), a3.values()):
    #print(ch_a1, "\n", ch_a2, "\n", ch_a3)
    """
    [['vie', 1, 1], ['combat', 4, 1], ['vie', 8, 3], ['combat', 11, 3], ['voyage', 18, 3], ['vie', 303, 3], ['elle', 309, 3], ['vie', 350, 3], ['multitude', 353, 3]] 
    [['vie', 1, 1], ['combat', 4, 1], ['vie', 8, 3], ['combat', 11, 3], ['voyage', 18, 3], ['vie', 303, 3], ['elle', 309, 3], ['vie', 350, 3], ['multitude', 353, 3]] 
    [['vie', 1, 1], ['combat', 4, 1], ['vie', 8, 3], ['combat', 11, 3], ['voyage', 18, 3], ['vie', 303, 3], ['elle', 309, 3], ['vie', 350, 3], ['multitude', 353, 3]]
    """
    len_longest_list = max(len(ch_a1), len(ch_a2), len(ch_a3))
    #print("len_longest_list:",len_longest_list)
    
    correct=0 # somme de la colonne 'correct'
    errone=0 # somme de la colonne 'errone'
    manquant=0 # somme de la colonne 'manquant'

    n_carre=0 # somme des carrés de chaque nombre dans les colonnes 'correct', 'errone' et 'manquant'

    # range de la chaines de coref avec le nombre d'items le plus élevé dans le cas où il serait différent pour les 3 annotateurs
    for i in range(len_longest_list):
        # print(ch_a1[i], ch_a2[i], ch_a3[i])
        # ['vie', 1, 1] ['vie', 1, 1] ['vie', 1, 1]
        
        # pour chaque item dans la chaine de coref courante chez les 3 annotateurs
        for annotation in [ch_a1[i][2], ch_a2[i][2], ch_a3[i][2]]:
            # print([ch_a1[i][2], ch_a2[i][2], ch_a3[i][2]])
            # [1, 1, 1]
            if annotation==1:
                correct+=1
            elif annotation==2:
                errone+=1
            else:
                manquant+=1
            
            n_carre += correct**2 + errone**2 + manquant**2
            
    #print(correct, errone, manquant)

    diviseur=3*len_longest_list

    P_e = (correct/diviseur)**2 + (errone/diviseur)**2 + (manquant/diviseur)**2
    #print("P_e:", P_e)

    P_o = (1/(len_longest_list*3*(3-1))) * (n_carre - (len_longest_list*3))
    #print("P_o:", P_o)

    kappa = (P_e - P_o) / (1 - P_o)
    #print("kappa:", kappa)

    sum_kappa += kappa


moy_kappa = sum_kappa / max(len(a1), len(a2), len(a3))
print("moy_kappa:", moy_kappa)

"""
il faut à présent trouver un moyen de calculer le kappa même quand il y a des chaînes de coréférences et/ou des maillons de chaînes supplémentaires
idées :
 - éventuellement utiliser NaN lorque des maillons ne sont pas présents dans d'autres chaînes de coréférences pour rééquilibrer
 - si le nombre de chaines de coref est différent entre les 3 annotateurs :
    - skipper une chaîne si le premier maillon est différent entre les 3 annotateurs pour trouver la vrai correspondance
    - et ensuite calculer le kappa sur la bonne chaîne de coref
- mais comment comptabilier le fait qu'il y ait une chaine en plus chez l'un des annotateurs dans le kappa ?
    - 
"""

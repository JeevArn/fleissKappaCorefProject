from sklearn.metrics import precision_score, recall_score, f1_score

doc1= {
0: [['discrimination', 1, 1], ['elle', 5, 3], ['discrimination', 31, 1], ['politique', 35, 1], ['qui', 36, 3], ['politique', 94, 3], ['politique', 125, 3], ['discrimination', 151, 3], ['discrimination', 156, 3], ['discrimination', 221, 3], ['politique', 161, 3], ['Elle', 200, 3], ['Elle', 243, 3], ['discrimination', 266, 3], ['elle', 270, 3], ['elle', 390, 3], ['Elle', 410, 3], ['discrimination', 470, 3], ['discrimination', 514, 3], ['discrimination', 551, 3], ['outil', 457, 3], ['discrimination', 604, 3], ['discrimination', 662, 3], ['elle', 583, 2], ['elle', 691, 2], ['discrimination', 758, 3], ['discrimination', 785, 3], ['politique', 798, 3], ['sa', 806, 3], ['elle', 810, 3], ['discrimination', 926, 3], ['elle', 947, 3], ['elle', 955, 3], ['son', 970, 3], ['ses', 980, 3]],
1: [['inégalités', 8, 1], ['Inégalités', 19, 1], ['s', 20, 3], ['inégalités', 52, 1], ['inégalités', 88, 3], ['inégalités', 108, 3], ['inégalités', 119, 3], ['inégalités', 236, 3], ['inégalités', 417, 3], ['inégalités', 827, 3], ['inégalités', 953, 3]],
2: [['monde', 15, 3], ['où', 17, 3]],
3: [['approche', 55, 1], ['son', 64, 1], ['ses', 68, 1]],
4: [['points', 74, 2], ['certains', 101, 2]],
5: [['effet', 80, 2], ['perspective', 84, 2]],
6: [['inégalités', 88, 2], ['inégalités', 108, 2], ['inégalités', 119, 2]],
7: [['politique', 125, 2], ['t', 140, 2]],
8: [['place', 138, 2], ['octroi', 139, 2]],
9: [['discrimination', 151, 2], ['discrimination', 156, 2], ['discrimination', 221, 2]],
10: [['politique', 161, 2], ['Elle', 200, 2], ['Elle', 243, 2], ['elle', 270, 2]],
11: [['opportunités', 205, 2], ['en', 213, 2]],
12: [['créer', 203, 3], ['en', 208, 3], ['en', 213, 3]],
13: [['égalité', 284, 2], ['politique', 303, 2]],
14: [['éducation', 292, 1], ['emploi', 296, 1], ['formation', 300, 1]],
15: [['diversité', 356, 1], ['création', 362, 1]],
16: [['groupes', 370, 1], ['leur', 375, 1]],
17: [['Afrique', 386, 2], ['elle', 390, 2], ['Elle', 410, 2]],
18: [['bourses', 430, 1], ['établissements', 434, 1]],
19: [['politique', 459, 3], ['égalité', 461, 3], ['diversité', 461, 3]],
20: [['fait', 511, 2], ['discrimination', 514, 2], ['discrimination', 551, 2], ['discrimination', 604, 2], ['discrimination', 662, 2]],
21: [['individus', 524, 1], ['ceux', 526, 1], ['bénéficiaires', 530, 3], ['Certains', 544, 1]],
22: [['conviction', 548, 2], ['elle', 583, 2]],
23: [['mérite', 610, 1], ['critère', 615, 1], ['qui', 619, 3]],
24: [['politiques', 627, 2], ['ceux', 644, 2]],
25: [['affaiblit', 664, 3], ['en', 669, 3]],
26: [['appartenance', 676, 2], ['elle', 691, 2], ['ses', 720, 2], ['son', 723, 2], ['son', 729, 2]],
27: [['quelqu\'un', 711, 3], ['ses', 720, 3], ['son', 723, 3], ['son', 729, 3]],
28: [['raisons', 715, 3], ['qui', 716, 3]],
29: [['talent', 724, 2], ['appartenance', 730, 2]],
30: [['France', 749, 1], ['un', 752, 1], ['pays', 755, 3], ['où', 756, 3]],
31: [['Constitution', 769, 1], ['elle', 810, 2], ['Ceci', 829, 1]],
32: [['politique', 798, 2], ['difficulté', 803, 2], ['sa', 806, 2]],
33: [['mise', 843, 3], ['réponse', 856, 3]],
34: [['limite', 869, 1], ['sa', 874, 1]],
35: [['justice', 937, 2], ['elle', 947, 2], ['elle', 955, 2], ['son', 970, 2], ['ses', 980, 2]]}

doc2 = {0: [['cannabis', 1, 1], ['plante', 4, 1], ['Il', 11, 3], ['ses', 24, 3], ['cannabis', 51, 1], ['son', 58, 1], ['cannabis', 66, 1], ['cannabis', 108, 3]],
1: [['Il', 11, 2], ['ses', 24, 2], ['cannabis', 437, 3], ['cannabis', 470, 3], ['cannabis', 429, 3]],
2: [['THC', 70, 1], ['tétrahydrocannabinol', 72, 1], ['substance', 76, 1], ['dernier', 88, 3], ['ses', 95, 3], ['THC', 155, 3], ['il', 169, 3], ['cannabis', 175, 3], ['Il', 196, 3], ['il', 209, 3], ['THC', 220, 3], ['il', 232, 3], ['Il', 240, 3], ['drogue', 272, 3], ['cannabis', 277, 3], ['THC', 296, 3], ['substance', 299, 3], ['sa', 309, 3], ['cannabis', 333, 3]],
3: [['usager', 85, 2], ['dernier', 88, 2], ['ses', 95, 2], ['lui', 112, 2], ['Il', 131, 2], ['il', 169, 2]],
4: [['CBD', 101, 1], ['cannabidiol', 103, 1], ['lui', 112, 3], ['substance', 114, 1], ['qui', 116, 3], ['qui', 124, 3], ['Il', 131, 3], ['drogue', 138, 3], ['CBD', 149, 3]],
5: [['cannabis', 108, 2], ['cannabis', 175, 2], ['en', 184, 2], ['Il', 196, 2]],
6: [['drogue', 138, 2], ['vente', 144, 2]],
7: [['France', 164, 1], ['France', 181, 1], ['France', 318, 3]],
8: [['jeunes', 202, 2], ['risques', 206, 1], ['qu\'', 208, 3]],
9: [['santé', 214, 1], ['santé', 245, 1]],
10: [['effet', 217, 2], ['THC', 220, 2], ['il', 232, 2], ['il', 265, 2], ['THC', 296, 2], ['sa', 309, 2]],
11: [['marché', 344, 2], ['marché', 362, 2]],
11: [['mesure', 405, 3], ['où', 406, 3]],
12: [['source', 410, 3], ['taxation', 418, 1], ['vente', 421, 3], ['en', 424, 1]],
13: [['définitive', 432, 2], ['légalisation', 435, 1], ['sujet', 440, 1]],
14: [['cannabis', 437, 2], ['cannabis', 470, 2]],
15: [['élections', 457, 1], ['programmes', 463, 1]],
16: [['vue', 477, 2], ['problématique', 480, 2]],
17: [['effet', 517, 2], ['ouverture', 520, 2]],
18: [['revanche', 534, 2], ['fait', 537, 2]],
19: [['risques', 567, 1], ['risques', 585, 1]]}

# doc3 Jeevya
doc3_bis = {
    0: [['femmes', 15, 1], ['femmes', 82, 1], ['femmes', 98, 1], ['femmes', 168, 1], ['femmes', 204, 3], ['femmes', 254, 3], ['femmes', 274, 3], ['elles', 292, 3], ['leurs', 294, 3], ['femmes', 301, 3], ['femmes', 336, 3], ['femmes', 399, 3], ['les', 417, 3], ['leurs', 433, 3], ['femmes', 462, 3], ['femmes', 509, 3], ['leurs', 517, 3], ['femmes', 537, 3], ['femmes', 573, 3], ['femmes', 602, 3]], 
    1: [['sport', 21, 1], ['divertissement', 24, 1], ['Il', 34, 2], ['sa', 40, 2], ['son', 43, 2], ['t', 87, 2]],
    'add1': [['discipline', 37, 3], ['sa', 40, 3], ['son', 43, 3]], 
    2: [['sport', 76, 2], ['il', 88, 2]], 
    3: [['hommes', 12, 3],['hommes', 79, 3], ['hommes', 95, 1], ['hommes', 160, 1], ['hommes', 199, 1], ['hommes', 307, 3], ['hommes', 345, 3], ['hommes', 422, 3], ['hommes', 459, 3], ['hommes', 540, 3]], 
    4: [['sportifs', 124, 1], ['leurs', 126, 1]],
    5: [['sponsors', 131, 1], ['ils', 141, 1]], 
    6: [['joueuse', 172, 1], ['ses', 183, 1]], 
    7: [['basketball', 243, 2], ['il', 250, 2]], 
    8: [['femmes', 276, 2], ['elles', 292, 2], ['leurs', 294, 2], ['femmes', 301, 2]], 
    9: [['hommes', 307, 2], ['ceux', 312, 2]], 
    10: [['hommes', 345, 2], ['hommes', 422, 2]], 
    11: [['femmes', 399, 2], ['les', 417, 2], ['leurs', 433, 2], ['femmes', 462, 2]], 
    12: [['sport', 405, 2], ['sport', 525, 2]], 
    13: [['hommes', 459, 2], ['hommes', 540, 2]], 
    14: [['inégalités', 478, 1], ['inégalités', 532, 1], ['inégalités', 590, 1]], 
    15: [['femmes', 509, 2], ['leurs', 517, 2]], 
    16: [['question', 554, 2], ['question', 580, 2]], 
    17: [['femmes', 573, 2], ['femmes', 602, 2]]
    }

# doc3 Marie

doc3 = {
0: [['inégalités', 3, 3], ['inégalité', 67, 3], ['inégalités', 257, 3], ['inégalités', 474, 3], ['inégalités', 478, 3], ['inégalités', 532, 3], ['inégalités', 590, 3]],
1: [['hommes', 12, 3], ['hommes', 79, 3], ['hommes', 95, 3], ['hommes', 160, 3], ['hommes', 199, 3], ['hommes', 307, 3], ['hommes', 345, 3], ['hommes', 422, 3], ['hommes', 459, 3], ['hommes', 540, 3], ['homme', 565, 3], ['hommes', 599, 3]],
2: [['femmes', 15, 1], ['femmes', 82, 1], ['femmes', 98, 1], ['femmes', 168, 1], ['femmes', 276, 3], ['elles', 292, 3], ['leurs', 294, 3], ['femmes', 301, 3], ['femmes', 336, 3], ['femmes', 399, 3], ['les', 417, 3], ['leurs', 433, 3], ['femmes', 462, 3], ['femmes', 509, 3], ['leurs', 517, 3], ['femmes', 537, 3], ['femmes', 573, 3], ['femmes', 602, 3]],
3: [['sport', 21, 1], ['divertissement', 24, 1], ['Il', 34, 2], ['sa', 40, 2], ['son', 43, 2], ['sport', 53, 3], ['sport', 76, 3], ['t', 87, 2], ['sport', 102, 3], ['sport', 115, 3], ['sports', 239, 3], ['sport', 405, 3], ['sport', 525, 3], ['sport', 543, 3], ['sport', 596, 3]],
4: [['disciplines', 37, 3], ['chacune', 38, 3], ['sa', 40, 3], ['son', 43, 3]],
5: [['sport', 76, 2], ['il', 88, 2]],
6: [['hommes', 95, 2], ['hommes', 160, 2], ['hommes', 199, 2]],
7: [['sportifs', 124, 1], ['leurs', 126, 1]],
8: [['sponsors', 131, 1], ['ils', 141, 1]],
9: [['football', 152, 3], ['où', 157, 3]],
10: [['joueuse', 172, 1], ['ses', 183, 1]],
11: [['basketball', 243, 1], ['il', 250, 2], ['où', 249, 3]],
12: [['sociétés', '267', '3'], ['qui', '269', '3']],
13: [['sports', 247, 3], ['où', 249, 3]],
14: [['femmes', 276, 2], ['elles', 292, 2], ['leurs', 294, 2], ['femmes', 301, 2]],
15: [['hommes', 307, 2], ['ceux', 312, 2]],
16: [['hommes', 345, 2], ['hommes', 422, 2]],
17: [['femmes', 399, 2], ['les', 417, 2], ['leurs', 433, 2], ['femmes', 462, 2]],
18: [['sport', 405, 2], ['sport', 525, 2]],
19: [['tennis', 450, 3], ['où', 451, 3]],
20: [['niveau', 449, 3], ['se', 459, 3]],
21: [['rivalise', 460, 3], ['cela', 465, 3]],
22: [['hommes', 459, 2], ['hommes', 540, 2]],
23: [['inégalités', 478, 2], ['inégalités', 532, 2], ['inégalités', 590, 2]],
24: [['femmes', 509, 2], ['leurs', 517, 2]],
25: [['question', 554, 1], ['qui', 557, 3], ['question', 580, 2]],
26: [['femmes', 573, 2], ['femmes', 602, 2]]}


doc4 = {
0: [['inégalités', 1, 1], ['y', 22, 1]], 
1: [['hommes', 4, 2], ['certains', 49, 2]], 
2: [['disparité', 28, 1], ['sa', 35, 1], ['son', 39, 1]], 
3: [['sportifs', 69, 2], ['autres', 72, 2]],
'add1': [['inégalités', 76, 3], ['en', 80, 3]], 
4: [['inégalités', 98, 1], ['inégalités', 124, 1], ['inégalités', 136, 1]], 
5: [['contextes', 130, 2], ['Certains', 132, 2]], 
6: [['femmes', 6, 3], ['femmes', 103, 3], ['femmes', 141, 3], ['femmes', 198, 3], ['femmes', 227, 1], ['femmes', 269, 1], ['femmes', 383, 3], ['femmes', 469, 3], ['femmes', 509, 3], ['femmes', 646, 3], ['femmes', 690, 3]], 
7: [['règles', 251, 2], ['elles', 276, 2]], 
8: [['foyer', 284, 1], ['y', 289, 1]], 
9: [['gens', 287, 2], ['habitudes', 293, 2]], 
10: [['inégalités', 307, 1], ['inégalités', 337, 1]], 
11: [['discrimination', 315, 2], ['résultat', 320, 2]], 
12: [['côté', 352, 2], ['argument', 355, 2]], 
13: [['partisans', 443, 2], ['certains', 491, 2]], 
14: [['principales', 583, 2],['concernées', 584, 3], ['leurs', 596, 1]],
15: [['monde', 605, 1], ['ses', 623, 1]], 
16: [['dynamique', 632, 1], ['là', 637, 1]], 
17: [['inégalités', 659, 1], ['inégalités', 736, 1]], 
18: [['sexisme', 695, 2], ['eSport', 698, 2], ['compétences', 715, 3], ['l', 719, 1]],
'add2': [['passion', 718, 3], ['l', 719, 3]]
}

doc5={0: [['vie', 1, 1], ['combat', 4, 1], ['vie', 8, 3], ['combat', 11, 3], ['voyage', 18, 3], ['vie', 93, 3], ['combat', 97, 3], ['vie', 303, 3], ['elle', 309, 3], ['vie', 350, 3], ['multitude', 353, 3], ['où', 360, 3]],
1: [['vie', 8, 2], ['combat', 11, 2]],
2: [['nature', 24, 3], ['cette', 34, 3]],
3: [['difficultés', 44, 3], ['qui', 51, 3]],
4: [['obstacles', 47, 3], ['qui', 51, 3]],
5: [['succès', 50, 3], ['qui', 51, 3]],
6: [['perception', 65, 1], ['champ', 71, 3], ['elle', 78, 1]],
7: [['monde', 107, 2], ['monde', 127, 2]],
8: [['monde', 127, 3], ['ses', 142, 3]],
9: [['conscience', 117, 2], ['ses', 142, 2]],
10: [['idée', 123, 3], ['cette', 149, 3]],
11: [['vision', 150, 1], ['vision', 199, 1], ['combat', 205, 3]],
12: [['Sisyphe', 169, 3], ['qui', 171, 3], ['lutte', 188, 3]],
13: [['réalité', 220, 3], ['où', 221, 3]],
14: [['gens', 251, 1], ['leurs', 263, 1], ['se', 268, 1], ['leur', 272, 1], ['leurs', 275, 1], ['leurs', 278, 1]],
15: [['vie', 303, 2], ['elle', 309, 2], ['vie', 350, 2], ['multitude', 353, 2]],
16: [['moments', 332, 1], ['qui', 341, 3], ['Certains', 394, 2], ['leurs', 426, 2], ['moments', 476, 3], ['qui', 485, 3]],
17: [['façon', 374, 3], ['dont', 375, 3], ['considération', 383, 3]],
18: [['obstacle', 397, 3], ['occasion', 400, 3], ['épreuve', 405, 3], ['qui', 406, 3]],
19: [['Certains', 394, 3], ['leurs', 426, 3]],
20: [['combat', 419, 3], ['charge', 422, 3], ['qui', 424, 3]],
21: [['conclusion', 434, 2], ['idée', 437, 2]],
22: [['idée', 437, 3], ['métaphore', 447, 3], ['qui', 449, 3]],
23: [['vie', 440, 1], ['combat', 443, 1], ['métaphore', 447, 2], ['vie', 494, 1], ['voyage', 500, 3], ['où', 501, 3]],
24: [['guerre', 503, 1], ['paix', 506, 3], ['s', 507, 1]]}

doc6={0: [['temps', 4, 2], ['Hommes', 7, 2]],
1: [['communautés', 22, 1], ['autres', 34, 3], ['leur', 36, 1], ['se', 39, 1]],
2: [['religion', 43, 1], ['religion', 46, 1], ['religion', 84, 3], ['religion', 106, 3], ['religion', 228, 3], ['celle', 230, 3], ['elle', 235, 3], ['religion', 407, 3], ['religion', 472, 3]],
3: [['reconnaissance', 56, 2], ['sa', 68, 2]],
4: [['Hommes', 7, 3], ['être', 57, 3], ['sa', 66, 3], ['Homme', 91, 3], ['Homme', 242, 3], ['Homme', 260, 3], ['hommes', 270, 3], ['Homme', 284, 3], ['lui', 291, 3], ['ses', 298, 3], ['il', 303, 3], ['s', 309, 3], ['Homme', 317, 3], ['il', 329, 3], ['Homme', 352, 3], ['il', 359, 3], ['son', 513, 3], ['le', 516, 3], ['ses', 525, 3], ['Homme', 415, 3], ['Homme', 441, 3], ['Homme', 481, 3], ['Homme', 504, 3], ['Homme', 532, 3], ['son', 537, 3]],
5: [['principe', 61, 3], ['qui', 64, 3]],
6: [['attitude', 69, 1], ['en', 74, 1], ['qui', 73, 3]],
7: [['communautés', 127, 3], ['s', 134, 3]],
8: [['citoyens', 143, 1], ['leurs', 145, 1]],
9: [['représentants', 146, 3], ['qui', 147, 3]],
10: [['règles', 153, 3], ['qui', 156, 3], ['règles', 447, 3], ['cela', 456, 3], ['règles', 509, 3]],
11: [['lois', 155, 1], ['qui', 156, 3], ['lois', 140, 3], ['lois', 165, 3], ['qui', 166, 3], ['lois', 182, 3], ['lois', 266, 3]],
12: [['vote', 204, 2], ['droit', 207, 2]],
13: [['droits', 190, 3], ['vote', 197, 3], ['respect', 202, 3], ['santé', 212, 3]],
14: [['vie', 212, 2], ['droit', 216, 2], ['vie', 232, 2]],
15: [['société', 226, 2], ['celle', 230, 2], ['elle', 235, 2]],
16: [['croyants', 345, 3], ['communautés', 381, 3]],
17: [['Dieu', 248, 1], ['être', 251, 3], ['qui', 253, 3], ['Homme', 260, 2], ['Dieu', 262, 3], ['Dieu', 357, 3], ['Dieu', 388, 3], ['sa', 93, 1],
['Homme', 292, 3], ['lui', 298, 3], ['ses', 305, 3], ['il', 310, 3], ['Homme', 325, 3], ['conscient', 330, 3], ['il', 337, 3], ['Dieu', 366, 3], ['lui', 367, 3], ['Dieu', 388, 3], ['sa', 393, 3], ['Homme', 451, 3], ['Homme', 491, 3], ['Homme', 515, 3], ['son', 525, 3], ['le', 528, 3], ['ses', 537, 3], ['Homme', 545, 3], ['son', 550, 3], ['Dieu', 266, 3], ['Dieu', 363, 3], ['Dieu', 395, 3], ['sa', 400, 3]],
18: [['libre', 287, 3], ['qui', 290, 3]],
19: [['deux', 413, 3], ['communautés', 416, 3], ['communautés', 127, 3], ['communautés', 381, 3], ['elles', 431, 3]],
20: [['Homme', 361, 2], ['il', 368, 2]],
21: [['coeur', 514, 3], ['qui', 515, 3]],
22:[['aimer', 536, 3], ['cela', 540, 3]]}


doc7={0: [['intelligence', 8, 3], ['Intelligence', 11, 1], ['IA', 13, 3], ['Intelligence', 31, 1], ['bond', 35, 3], ['intelligence', 51, 1], ['elle', 64, 1], ['Intelligence', 105, 1], ['IA', 134, 3], ['Intelligence', 150, 1], ['progrès', 163, 3], ['bond', 167, 3], ['IA', 181, 3], ['société', 226, 2], ['Intelligence', 232, 1]],
1: [['domaines', 26, 2], ['Certains', 28, 2]],
2: [['perte', 113, 1], ['confidentialité', 119, 1], ['dépendance', 129, 1], ['IA', 134, 2]],
3: [['efficacité', 205, 2], ['secteur', 210, 2]],
4: [['avancées', 218, 2], ['inquiétudes', 222, 2]],
5: [['perte', 240, 1], ['confidentialité', 246, 1], ['dépendance', 253, 1]]}


doc8 = {
    0: [['Francesco', 0, 2], ['Benjamin', 3, 2], ['Albert', 8, 2]], 
    1: [['hommes', 14, 1], ['leurs', 31, 1], ['ils', 34, 1]], 
    2: [['mot', 40, 1], ['mot', 48, 1], ['celui', 60, 1], ['il', 62, 1]], 
    3: [['effet', 84, 2], ['productivité', 88, 2], ['Elle', 120, 3]], 
    4: [['production', 105, 1], ['sa', 117, 1], ['Elle', 120, 2]], 
    5: [['économie', 127, 1], ['sa', 135, 3], ['Ici', 146, 1]], 
    6: [['services', 144, 2], ['productivité', 173, 3], ['efficacité', 175, 3], ['derniers', 181, 1]], 
    7: [['individu', 154, 2], ['ensemble', 160, 2]], 
    8: [['gré', 192, 2], ['se', 200, 2], ['celui', 223, 2]], 
    9: [['efficacité', 195, 2], ['travail', 199, 1], ['se', 200, 3]], 
    10: [['enjeu', 211, 2], ['camps', 214, 3], ['celui', 218, 1], ['celui', 223, 3]], 
    11: [['pensée', 236, 2], ['situation', 240, 2]], 
    12: [['débat', 252, 2], ['son', 264, 2]], 
    13: [['question', 255, 1], ['y', 268, 1]], 
    14: [['réponse', 306, 2], ['son', 311, 2]], 
    15: [['travail', 360, 1], ['y', 371, 1], ['ses', 421, 3]], 
    16: [['relation', 375, 1], ['relation', 391, 1]], 
    17: [['effet', 413, 2], ['ses', 421, 2]], 
    18: [['buts', 422, 2], ['y', 428, 2]], 
    19: [['reste', 452, 1], ['le', 456, 1], ['reste', 508, 1], ['reste', 617, 3], ['reste', 642, 3]], 
    20: [['Gary', 470, 1], ['Sa', 479, 1], ['auteur', 564, 3]], 
    21: [['chose', 492, 1], ['la', 503, 1], ['chose', 537, 3], ['la', 637, 3]], 
    22: [['compte', 523, 2], ['ses', 528, 2]], 
    23: [['chose', 537, 2], ['son', 544, 2]], 
    24: [['adeptes', 567, 3], ['eux', 575, 1], ['Ils', 598, 1]], 
    25: [['auteur', 564, 2], ['ses', 592, 2]], 
    26: [['se', 608, 2], ['compte', 613, 2], ['reste', 617, 2], ['reste', 642, 2]], 
    27: [['sorte', 634, 2], ['la', 637, 2]], 
    28: [['finalité', 666, 2], ['sa', 671, 2]], 
    29: [['moyen', 684, 2], ['sa', 687, 2]], 
    30: [['travail', 700, 1], ['travail', 736, 1], ['moyen', 739, 1]], 
    31: [['ennui', 709, 1], ['vice', 712, 2], ['ennui', 731, 1], ['vice', 765, 2]],
    'add1': [['vice', 712, 3], ['vice', 765, 3], ['deuxième', 761, 3], ['tentation', 769, 3]],
    32: [['journées', 743, 1], ['les', 746, 1], ['elles', 756, 1]], 
    33: [['deuxième', 761, 2], ['tentation', 769, 2]],  
    34: [['temps', 801, 1], ['se', 802, 1]], 
    35: [['vide', 850, 2], ['en', 862, 2]], 
    36: [['oui', 878, 2], ['point', 883, 2]], 
    37: [['ennui', 887, 1], ['se', 889, 1]], 
    38: [['échappatoire', 902, 2], ['le', 923, 2]], 
    39: [['travail', 910, 1], ['antidote', 916, 1]], 
    40: [['ennui', 919, 1], ['le', 923, 3], ['ennui', 950, 1], ['ennui', 957, 1], ['sa', 959, 3], ['ennui', 978, 1]], 
    41: [['session', 944, 2], ['travail', 947, 2]], 
    42: [['fois', 953, 2], ['sa', 959, 2]], 
    43: [['solution', 969, 2], ['travail', 973, 2], ['vice', 1016, 3], ['variable', 1019, 3]], 
    44: [['vice', 987, 1], ['variable', 990, 1]], 
    45: [['travail', 993, 1], ['travail', 1028, 1], ['travail', 1041, 1]], 
    46: [['santé', 1012, 2], ['vice', 1016, 2], ['variable', 1019, 2]], 
    47: [['partie', 1044, 2], ['Ici', 1054, 2]], 
    48: [['Baudelaire', 1069, 1], ['Auteur', 1071, 3], ['auteur', 1077, 1], ['son', 1080, 3], ['succès', 1081, 2], ['lui', 1093, 1], ['Il', 1095, 1]], 
    49: [['siècle', 1074, 2], ['son', 1080, 2]], 
    50: [['maladie', 1102, 2], ['alcool', 1107, 2]], 
    51: [['Vice', 1116, 2], ['personnes', 1127, 3], ['leur', 1129, 1]], 
    52: [['Vice', 1116, 2], ['elle', 1151, 2], ['ses', 1156, 2]], 
    53: [['besoin', 1138, 1], ['sa', 1141, 1], ['sa', 1144, 1], ['elle', 1151, 3], ['ses', 1156, 3]], 
    54: [['raison', 1142, 2], ['travail', 1148, 3], ['y', 1154, 1]], 
    55: [['travail', 1204, 1], ['travail', 1227, 1]], 
    56: [['vie', 1270, 2], ['part', 1275, 2]], 
    57: [['méritocratie', 1319, 1], ['lien', 1321, 1]], 
    58: [['Se', 1328, 1], ['ses', 1334, 3], ['Il', 1354, 2]], 
    59: [['pouvoir', 1331, 2], ['ses', 1334, 2]], 
    60: [['Bible', 1359, 1], ['Genèse', 1361, 1], ['ceci', 1400, 2]], 
    61: [['résultats', 1409, 1], ['résultats', 1419, 1]], 
    62: [['dévouement', 1467, 2], ['son', 1470, 2]], 
    63: [['travail', 1471, 2], ['dévouement', 1467, 3], ['y', 1472, 1]], 
    64: [['référence', 148, 1], ['acharnement', 1489, 3], ['celle', 1492, 1]], 
    65: [['prix', 1498, 2], ['il', 1541, 2]], 
    66: [['chemin', 1588, 1], ['chemin', 1601, 1], ['Celui', 1590, 3], ['Celui', 1604, 3]], 
    67: [['Celui', 1590, 2], ['Celui', 1604, 2]], 
    68: [['réalité', 1625, 2], ['Ici', 1633, 2]], 
    69: [['travail', 1636, 1], ['travail', 1680, 1]], 
    70: [['travail', 1664, 2], ['second', 1668, 2]], 
    71: [['travail', 1680, 3], ['thérapie', 1692, 1], ['Elle', 1702, 1]], 
    72: [['bonheur', 1720, 1], ['le', 1727, 1]], 
    73: [['possibles', 1758, 2], ['certains', 1767, 2]], 
    74: [['autres', 1778, 1], ['là', 1794, 1], ['personnes', 1793, 3], ['leurs', 1806, 3], ['individus', 1820, 3], ['les', 1825, 3], ['leur', 1834, 3]], 
    75: [['personnes', 1793, 2], ['leurs', 1806, 2]], 
    76: [['œuvre', 1800, 2], ['étincelle', 1803, 1], ['Celle', 1809, 1]], 
    77: [['travail', 1815, 1], ['travail', 1823, 1]], 
    78: [['individus', 1820, 2], ['les', 1825, 2], ['leur', 1834, 2], ['Certains', 1842, 1], ['se', 1843, 1], ['ils', 1926, 2], ['leurs', 1938, 2]], 
    79: [['Cristiano', 1857, 3], ['un', 1861, 1], ['y', 1900, 2]], 
    80: [['football', 1882, 2], ['vie', 1885, 2]], 
    81: [['paroles', 1893, 1], ['millions', 1896, 2], ['y', 1900, 3]], 
    82: [['personnes', 1898, 1], ['celles', 1904, 1], ['ils', 1926, 3]], 
    83: [['travail', 1907, 1], ['moyen', 1910, 1], ['travail', 1952, 1], ['il', 1960, 1], ['voie', 1963, 3], ['Elle', 1967, 3]], 
    84: [['voie', 1963, 2], ['Elle', 1967, 2]], 
    85: [['maximum', 2017, 2], ['condition', 2021, 2]], 
    86: [['essentiel', 2027, 1], ['essentiel', 2039, 1]], 
    87: [['Gary', 2029, 1], ['Il', 2055, 1], ['soi', 2074, 1]], 
    88: [['force', 2059, 2], ['durée', 2070, 3], ['se', 2071, 1]], 
    89: [['angle', 2066, 2], ['durée', 2070, 2]], 
    90: [['travail', 2096, 1], ['travail', 2103, 1]], 
    91: [['Identité', 2128, 2], ['certains', 2134, 2]], 
    92: [['personnes', 2153, 1], ['autres', 2163, 1]], 
    93: [['Cristiano', 2158, 2], ['se', 2178, 3], ['son', 2186, 1]], 
    94: [['facteurs', 2206, 1], ['éléments', 2210, 1], ['ils', 2258, 1]], 
    95: [['un', 2218, 1], ['un', 2251, 1], ['il', 2309, 2]], 
    96: [['autre', 2221, 1], ['autre', 2256, 1]], 
    97: [['travail', 2237, 1], ['dernier', 2242, 1]], 
    98: [['question', 2265, 1], ['son', 2279, 1]], 
    99: [['cause', 2283, 2], ['travail', 2286, 3], ['sa', 2293, 1]], 
    100: [['travail', 2286, 2], ['prisme', 2291, 1], ['durée', 2294, 3], ['échappatoire', 2314, 3], ['zone', 2317, 3],], 
    101: [['échappatoire', 2314, 2], ['zone', 2317, 2], ['Ici', 2324, 2]], 
    102: [['excès', 2330, 2], ['ses', 2333, 2]], 
    103: [['discussions', 234, 2], ['secteurs', 2351, 2]], 
    104: [['discorde', 2358, 2], ['question', 2338, 3], ['elle', 2363, 1]]
    }

doc9={
0: [['intelligence', 1, 1], ['Elle', 19, 1], ['elle', 36, 1], ['intelligence', 78, 1], ['intelligence', 99, 1], ['intelligence', 149, 1], ['elle', 202, 1], ['deux', 135, 3], ['intelligences', 269, 3], ['qui', 27, 3], ['intelligence', 343, 3],    ['intelligence', 540, 3], ['intelligence', 584, 3], ['intelligence', 675, 3], ['elles', 609, 3],   ['deux', 641, 3], ['se', 642, 3]],
'add1': [['secteur', 56, 3], ['où', 58, 3]],
'add2': [['humain', 33, 3], ['être', 83, 3], ['humain', 104, 3], ['être', 117, 3], ['deux', 135, 3], ['être', 305, 1], ['humain', 483, 3], ['deux', 641, 3], ['se', 642, 3]],
'add3': [['traiter', 40, 3], ['ce', 50, 3], ['qui', 51, 3]],
1: [['patient', 318, 1], ['se', 320, 1], ['il', 324, 1], ['lui', 337, 1], ['Il', 418, 1]],
2: [['données', 366, 2], ['interactions', 371, 2]],
3: [['limites', 383, 2], ['elles', 406, 2]],
4: [['données', 45, 3], ['données', 63, 3], ['données', 392, 1], ['données', 158, 3], ['données', 209, 3], ['données', 366, 3], ['données', 392, 3], ['en', 400, 1], ['données', 447, 3], ['celles', 460, 3], ['donnée', 507, 3], ['données', 562, 3], ['données', 578, 3], ['données', 681, 3]],
5: [['confidentialité', 427, 2], ['effet', 430, 2]],
6: [['données', 447, 2], ['celles', 460, 2]],
7: [['algorithmes', 238, 3], ['algorithmes', 376, 3], ['algorithmes', 469, 1], ['algorithmes', 495, 1]],
8: [['médecin', 499, 1], ['médecin', 533, 1], ['lui', 549, 1]],
9: [['diagnostic', 69, 3], ['diagnostic', 511, 1], ['diagnostic', 537, 1]], # on rajoute les autres 'diagnostic' ?
10: [['patient', 523, 1], ['lui', 524, 1], ['patient', 548, 1]],
11: [['intelligence', 540, 2], ['intelligence', 584, 2], ['intelligence', 675, 2]], # on remet dans la chaine intelligence
12: [['erreurs', 605, 2], ['elles', 609, 2]],
13: [['deux', 641, 2], ['se', 642, 2]], # on remet dans les chaines humain et intelligence
14: [['se', 658, 2], ['type', 665, 3], ['celui', 669, 1]]
}

doc10={
0: [['inégalités', 1, 1], ['elles', 13, 1], ['inégalités', 55, 1], ['différence', 72, 3], ['inégalités', 99, 1], ['elles', 107, 3], ['inégalités', 126, 1], ['inégalités', 138, 1], ['inégalités', 156, 1], ['inégalités', 368, 3], ['inégalités', 398, 3], ['inégalités', 669, 3], ['elle', 681, 3], ['inégalités', 721, 3],  ['inégalités', 844, 3]],
1: [['football', 10, 3], ['football', 47, 3], ['football', 107, 3], ['football', 159, 3], ['football', 364, 3], ['football', 374, 3], ['football', 401, 3], ['football', 421, 3], ['football', 466, 3], ['football', 502, 3], ['football', 539, 3], ['football', 605, 3], ['football', 617, 3], ['football', 639, 3], ['football', 653, 3] , ['football', 655, 3], ['football', 660, 3],  ['terme', 659, 3], ['football', 675, 3], ['football', 727, 3], ['football', 771, 3], ['football', 809, 3], ['football', 850, 3], ['football', 865, 3], ['football', 907, 3]],
2: [['Sam', 30, 3], ['joueuse', 33, 3], ['la', 34, 3]],
3: [['souleve', 79, 3], ['Cela', 91, 3]],
4: [['mieux', '36', 3], ['Cela', '91', 3], ['s', '221', 3]],
5: [['questions', 81, 2], ['elles', 110, 2]],
4: [['hommes', 5, 3], ['homme', 76, 3], ['hommes', 103, 3], ['hommes', 162, 3], ['hommes', 191, 3], ['ils', 199, 3], ['hommes', 228, 3], ['ils', 242, 3], ['hommes', 265, 3], ['eux', 266, 3], ['hommes', 394, 3], ['hommes', 404, 3], ['hommes', 590, 3], ['homme', 621, 3], ['homme', 644, 3], ['hommes', 763, 3], ['hommes', 920, 3]],
6: [['générer', 203, 3], ['Cela', 220, 3], ['s', 221, 3]],
7: [['femmes', 7, 3], ['elles', 13, 3], ['femmes', 105, 3], ['femmes', 165, 3], ['femmes', 197, 3], ['femmes', 236, 3], ['femmes', 407, 3], ['femmes', 413, 3], ['elles', 423, 3], ['femmes', 592, 3], ['femmes', 761, 3], ['femmes', 806, 3], ['femmes', 833, 3], ['elles', 890, 2]],
8: [['pas', 416, 3], ['cela', 427, 3]],
9: [['pas', 451, 3], ['cela', 455, 3]],
10: [['toujours', 471, 3], ['cela', 474, 3]],
11: [['problème', 685, 3], ['problèmes', 695, 3]],
12: [['équipes', 748, 3], ['qui', 753, 3]],
13: [['réseaux', 825, 3], ['où', 827, 3]],
14: [['ce', 854, 3], ['qui', 855, 3]],
15: [['hommes', 103, 2], ['ils', 199, 2], ['ils', 242, 2]],
16: [['football', 107, 2], ['y', 115, 2]],
17: [['hommes', 162, 2], ['hommes', 191, 2], ['hommes', 228, 2]],
18: [['femmes', 165, 2], ['femmes', 197, 2]],
19: [['effet', 171, 1], ['individu', 174, 1], ['il', 182, 1]],
20: [['droit', 209, 2], ['sponsor', 213, 2], ['vente', 216, 2]],
21: [['arguments', 256, 2], ['eux', 266, 2]],
22: [['mêmes', 255, 3], ['qui', 257, 3]],
23: [['joueurs', 272, 1], ['certains', 283, 1], ['autres', 288, 3], ['leurs', 298, 3]],
24: [['autres', 288, 2], ['leurs', 298, 2]],
25: [['joueur', 308, 1],  ['il', 312, 3], ['joueur', 326, 1]],
26: [['clubs', 320, 1], ['autres', 319, 3], ['clubs', 329, 1], ['ils', 349, 2]],
27: [['salaires', 341, 1], ['salaires', 346, 1], ['ils', 349, 3]],
28: [['inégalités', 368, 2], ['inégalités', 398, 2]],
29: [['femmes', 407, 2], ['femmes', 413, 2], ['elles', 423, 2]],
30: [['football', 466, 2], ['football', 539, 2]],
31: [['fille', 509, 1], ['lui', 520, 3], ['fille', 529, 1], ['elle', 533, 1]],
32: [['société', 597, 2], ['elle', 681, 2]],
33: [['football', 605, 2], ['football', 617, 2], ['football', 639, 2]],
34: [['réservé', 619, 3], ['cela', 623, 3]],
35: [['homme', 644, 2], ['le', 648, 2]],
36: [['football', 655, 2], ['terme', 659, 2]],
37: [['inégalités', 669, 2], ['inégalités', 721, 2]],
38: [['football', 727, 2], ['en', 732, 2]],
39: [['choses', 781, 1], ['choses', 869, 1]],
40: [['création', 787, 2], ['football', 809, 2]],
41: [['années', 797, 2], ['récompenses', 802, 2]],
42: [['autres', 801, 3], ['récompenses', 802, 3]],
43: [['femmes', 806, 2], ['femmes', 833, 2]],
44: [['plus', 813, 3], ['cela', 820, 3]],
45: [['football', 865, 2], ['football', 907, 2], ['se', 911, 2]],
46: [['championnes', 880, 1], ['elles', 890, 1]]}


def harmonize_data(annot):
    gold = []
    coreferee_prediction = []
    for ch in annot.values():
        for item in ch:
            if item[2]==1:
                gold.append(1)
                coreferee_prediction.append(1)
            elif item[2]==2:
                gold.append(2)
                coreferee_prediction.append(1)
            else:
                gold.append(3)
                coreferee_prediction.append(0)
    return gold, coreferee_prediction

######### METRICS ##########################

def get_metrics(gold, coreferee_prediction):
    precision = precision_score(gold, coreferee_prediction, average='micro')
    recall = recall_score(gold, coreferee_prediction, average='micro')
    f1 = f1_score(gold, coreferee_prediction, average='micro')
    return round(precision, 2), round(recall, 2), round(f1, 2)
#on utilise micro pour que chaque rating ait le même poids dans le calcul
#sans préciser micro (ou autre) on ne peut pas calculer les metrics car on a plus de 2 classes (0 à 3)

def results(doc):
    gold, coreferee_prediction = harmonize_data(doc)
    precision, recall, f1 = get_metrics(gold, coreferee_prediction)
    return precision, recall, f1


######### DISTANCES ##########################

def distance_first_last_maillon(doc: dict, corresp2ou3: int) -> float:
    '''
    Calcul de la moyenne de la distance entre le 1er et le dernier maillon de chaque chaine dans un doc
    corresp2ou3: int 
        2 pour pour calculer la moy pour l'annotation de coreferee
        3 pour l'annotation manuelle
    '''
    add_distance = 0 # somme des distances firstToLast de chaque chaine 
    nb_ch = 0 # compte nombre de chaine parcouru
    for ch in doc.values(): # on parcourt chaque chaine (1 chaine : valeur de chaque elem du dico)
        list_idx_maillons=[] # liste de tous les index dans la chaine
        for l in ch: # pour chaque sous-liste (qui correspond à un maillon)
            if l[2]==1 or l[2]==corresp2ou3: # on vérifie si le maillon correspond à l'annotation manuelle (1 ou 3) ou à celle de coreferee (1 ou 2)
                list_idx_maillons.append(l[1]) # on ajoute l'index à la liste
        if list_idx_maillons: # si la liste n'est pas vide
            add_distance+= max(list_idx_maillons)-min(list_idx_maillons) # on ajoute la distance obtenu par max-min de la liste des idx
            nb_ch+=1 # on incrémente le compteur
    if nb_ch == 0: 
        return 0
    else:
        return round(add_distance / nb_ch, 2) # on retourne la moyenne


def distance_inter_maillon(doc: dict, corresp2ou3: int) -> float:
    '''
    Calcul de la moyenne de la distance entre chaque maillon dans une chaine du doc
    corresp2ou3: int 
        2 pour pour calculer la moy pour l'annotation de coreferee
        3 pour l'annotation manuelle
    '''
    add_distance = 0
    total_pairs = 0
    for ch in doc.values(): # on parcourt chaque chaine (1 chaine : valeur de chaque elem du dico)
        list_idx_maillons=[] # liste de tous les index dans la chaine
        for l in ch: # pour chaque sous-liste (qui correspond à un maillon)
            if l[2]==1 or l[2]==corresp2ou3: # on vérifie si le maillon correspond à l'annotation manuelle (1 ou 3) ou à celle de coreferee (1 ou 2)
                list_idx_maillons.append(l[1]) # si oui on ajoute l'index à la liste
        if list_idx_maillons:
            sorted_idx=sorted(list_idx_maillons) # on s'assure que tous les maillons sont dans l'ordre
            for i in range(len(sorted_idx)-1): # on parcourt la liste des idx
                add_distance+=sorted_idx[i+1]-sorted_idx[i] # on ajoute la distance entre le maillon suivant et le courant
                total_pairs+=1 # on incrémente le compteur
    if total_pairs == 0:
        return 0
    else:
        return round(add_distance / total_pairs, 2) # on retourne la moyenne



len_docs=[838, 519, 542, 659, 450, 476, 223, 1979, 586, 799]

from prettytable import PrettyTable
table = PrettyTable()
table.field_names = ["doc", "len doc", "F1-score", "GOLD firstToLast maillon", "COREFEREE firstToLast maillon", "GOLD inter-maillon", "COREFEREE inter-maillon"]

for i, (doc, doc_len) in enumerate(zip([doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8, doc9, doc10], len_docs), start=1):
    precision, recall, f1 = results(doc)
    dist_moy_maillons = distance_first_last_maillon(doc, 3)
    coref_dist_moy_maillons = distance_first_last_maillon(doc, 2)
    dist_inter_maillon = distance_inter_maillon(doc, 3)
    coref_dist_inter_maillon = distance_inter_maillon(doc, 2)
    
    table.add_row([f"{i}", doc_len, f1, dist_moy_maillons, coref_dist_moy_maillons, dist_inter_maillon, coref_dist_inter_maillon])

print(table)

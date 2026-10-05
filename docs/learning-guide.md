# Comprendre et défendre le projet

Le CNN apprend ses filtres à partir d'un petit jeu de photos : c'est une baseline pour mesurer l'intérêt du transfert. MobileNet a déjà appris des formes sur ImageNet. Ses filtres et BatchNorm restent figés, seuls les poids d'une tête à trois sorties sont entraînés. Ce n'est pas un fine-tuning de toute l'architecture. Extraire les représentations une seule fois économise du calcul ; les vues retournées horizontalement viennent uniquement du train.

Le split est celui de la source. Nous vérifions les pixels dupliqués avant apprentissage et les ressemblances dHash. Deux photos différentes d'une même plante pourraient néanmoins traverser le split : absence d'identifiant ne signifie pas indépendance. Il faut de nouvelles photos de fermes séparées pour valider ce point.

Macro-F1 donne le même poids aux trois classes et expose une classe moins bien reconnue. Accuracy mesure la proportion totale correcte. La validation choisit l'époque et l'architecture ; le test mesure ensuite ce choix. La température est ajustée sur validation pour modifier les probabilités, pas le classement. NLL, Brier et ECE évaluent différents aspects des probabilités ; ECE dépend des bins et d'un petit échantillon.

Une confiance softmax élevée peut survenir sur une autre espèce. Le flou et l'assombrissement sont un contrôle défini à l'avance, pas un nouveau domaine indépendant. Grad-CAM révèle des zones influençant une sortie ; il ne prouve pas que le modèle a appris une lésion ni sa cause.

L'acquisition a rencontré des URLs historiques privées (403). Le miroir de l'auteur, figé par commit et SHA-256, conserve une provenance contrôlable. À apprendre : refaire le split/audit, expliquer gel du backbone versus fine-tuning, calculer précision/rappel sur la matrice, inspecter une erreur Grad-CAM et envoyer une vraie image à l'API. Alternatives : fine-tuning limité avec données supplémentaires, validation par ferme et classe inconnue avec abstention réellement testée.

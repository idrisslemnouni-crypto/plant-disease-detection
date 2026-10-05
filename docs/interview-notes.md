# Préparation entretien — vision agricole

Problème : trois états de feuilles de haricot, photos réelles et expertise NaCRRI. Pipeline : hashes et split fourni, augmentation train, CNN versus MobileNet figé, sélection validation, température validation, test et Grad-CAM. Consulter reports/metrics.json pour les chiffres exacts plutôt que mémoriser un score arrondi. Aucun diagnostic terrain ou généralisation à d'autres cultures n'est validé.

1. **Pourquoi ce dataset ?** Photos de terrain plutôt que fond artificiel, source et licence documentées.
2. **Pourquoi une baseline CNN ?** Quantifier l'intérêt du transfert plutôt que supposer qu'il améliore tout.
3. **Qu'est-ce qui est entraîné dans MobileNet ?** La tête linéaire ; le backbone ImageNet est figé, BatchNorm en évaluation.
4. **Comment éviter les fuites ?** Split source, audit des pixels, transformations du train seulement ; indépendance par ferme encore inconnue.
5. **Pourquoi macro-F1 ?** Donner un poids égal aux classes et comparer leurs erreurs.
6. **Comment choisir le modèle ?** Macro-F1 validation, avant toute lecture des performances test.
7. **Que fait la température ?** Ajuster la netteté des probabilités sur validation ; le label argmax reste le même.
8. **La confiance détecte-t-elle une espèce inconnue ?** Non ; aucun rejet hors distribution n'a été validé.
9. **Que prouve Grad-CAM ?** Une sensibilité locale du score, sans preuve causale ou segmentation de lésion.
10. **Prochaine amélioration prioritaire ?** Un test indépendant par ferme et des classes inconnues, avant une complexification du modèle.

Difficultés : URL de téléchargement historique en 403 et échantillon test petit. Le miroir de l'auteur est hash-pinné. Développement assisté par IA ; comprendre et refaire les étapes avant de revendiquer sa maîtrise.

# CLAUSESHIFT V1.14.3 — E2-SINGLE, runner de soumission

## Décision

Le JSON multi-unités V1.14.2 a échoué pour trois et une unité. La méthode `E2-SINGLE` réutilise exactement le prompt indexé de `src.e2` (V1.13, validé sur Public AI), sans enveloppe JSON `units`. Le moteur traite une unité après l’autre, sauvegarde chaque réponse valide, demande à nouveau uniquement les indices manquants. Aucune valeur n'est fabriquée.

## Commandes DEV immédiatement testables

Pour continuer le plan existant `admin_de_132`, utiliser `python -m src.e2` et son checkpoint V1.13. Il est important de conserver `--max-tokens 1536`, `--max-scores-per-call 96`, la même révision `EMB_SHA`, le contexte, et les mêmes paramètres de plan. Il n'est pas nécessaire d'appeler `src.e2_batch`.

## Runner final

Le `src.submission freeze` accepte `--method E2-SINGLE --embedding-revision "$EMB_SHA" --batch-cap 96 --max-tokens 1536`. Il crée un manifeste `schema=3` vérifié par `preflight`. `src.submission run` refuse toujours les données finales avant le gel et la vérification de son manifeste. Il exécute `src.e2.run(split='final')`, avec répertoires distincts, sortie complète en ordre original, et échec fermé si un document manque.

Ne pas créer le manifeste ou toucher aux données finales avant de figer une méthode après mesure DEV et audit du budget de calcul. Ce runner est fonctionnel avec mocks locaux, sans démonstration réelle du corpus complet.

## Limites

- Aucun batching multi-unités ni parallélisme dans E2-SINGLE ; la durée totale reste le risque majeur.
- Le modèle d'embeddings doit être téléchargeable/pinné dans le conteneur Docker ; son chargement CPU peut coûter du temps.
- Les coûts et la latence Public AI ne sont pas garantis.
- Spearman pour E2-SINGLE sur plusieurs documents DEV doit être calculé avant tout gel final.
- Le build Docker V1.14.3 doit être testé côté serveur.
- Aucun jeu test/final n'a été lu dans les tests locaux.

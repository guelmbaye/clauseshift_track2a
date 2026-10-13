# CLAUSESHIFT — E2 Coverage-Aware V1.10 (DEV-only)

## Pourquoi ce mode

Sur `admin_*_105`, `--max-unit-tokens 48` n'a pas amélioré l'alignement : allemand identique, français et italien avec davantage de segments dits « non alignés ». Le modèle d'origine maximise `cosine - threshold` par unité, ce qui privilégie parfois un petit fragment très similaire au détriment de la traduction plus longue du même passage.

`semantic_coverage` est un **nouveau mode expérimental** : l'algorithme garde les embeddings et la programmation dynamique monotone, mais pondère le gain par le nombre de tokens couverts, reconnaît les ancres textuelles communes d'au moins deux mots longs, et vérifie individuellement les sous-unités avant une fusion. Les passages non alignés restent éligibles à un scoring Apertus ; aucun score n'est généré à partir des embeddings.

Les poids `coverage_weight=0.50` et `anchor_bonus=0.35` sont des hyperparamètres heuristiques, **non validés sur les données SwissGov-RSD réelles**. Il faut inspecter les plans et vérifier sur plusieurs documents DEV sans utiliser le split test.

## Compatibilité

Le mode existant `--alignment semantic` est inchangé (plans et fingerprints V1.9 conservés). `semantic_coverage` possède sa propre signature et doit utiliser de **nouveaux répertoires** de plans et checkpoints. Aucun transfert implicite de scores entre alignements n'est autorisé.

## Installation

```bash
cd /var/www/clauseshift_track2a
git apply --check CLAUSESHIFT_E2_Coverage_Alignment_v1.10.patch
git apply CLAUSESHIFT_E2_Coverage_Alignment_v1.10.patch
python3 -m unittest discover -s tests -q
```

## Plan-only ; aucun appel API

```bash
for lang in de fr it; do
  python3 -m src.e2 \
    --lang "$lang" --doc-id "admin_${lang}_105" \
    --alignment semantic_coverage --prompt fewshot \
    --chunk-tokens 48 --max-unit-tokens 48 \
    --similarity-threshold 0.46 --merge-penalty 0.06 \
    --coverage-weight 0.50 --anchor-bonus 0.35 \
    --embedding-revision "$EMB_SHA" --plan-only \
    --plan-dir out/plans/e2cov_dev_v110 \
    --checkpoint-dir out/checkpoints/e2cov_dev_v110
  python3 -m tools.inspect_e2_alignment \
    --lang "$lang" --doc-id "admin_${lang}_105" \
    --plan-dir out/plans/e2cov_dev_v110 --limit 20
done
```

Ne pas lancer d'inférence tant que la correspondance de chaque passage n'est pas vérifiée. Attention aux faux positifs d'ancres communes et aux passages véritablement ajoutés/omis. Comparer le plan aux variantes E2-C40 et E2-C48.

## Comparaison DEV, si les prédictions deviennent complètes

`tools.compare_e2_dev` accepte `--alignment semantic_coverage --coverage-weight 0.50 --anchor-bonus 0.35` en plus des arguments V1.9 habituels, en pointant vers `out/plans/e2cov_dev_v110` et `out/checkpoints/e2cov_dev_v110`. Ne pas communiquer de score à partir de checkpoints partiels.

## Limites

- Alignement monotone : ne sait pas gérer directement un déplacement arbitraire de phrases.
- Score de couverture heuristique, non prouvé supérieur sur le benchmark.
- Le runner reste DEV-only : `make run` final ne prend **pas** encore en charge E2. La préparation du paquet UZH doit avancer en parallèle et précéder toute utilisation du split test.
- La conformité aux dépendances tierces et à la reproductibilité environnementale doit être vérifiée pour la soumission.

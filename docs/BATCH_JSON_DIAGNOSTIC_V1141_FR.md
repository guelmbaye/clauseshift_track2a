# CLAUSESHIFT V1.14.1 — Diagnostic des JSON batchés (DEV uniquement)

## Échec observé

`admin_de_132`, V1.14, `units=[1,2,3]`, 96 scores, `Invalid JSON: Extra data: line 1 column 622 (char 621)`.

Un `Extra data` signifie que le parseur a terminé une valeur JSON et rencontré autre chose ensuite. Il peut s'agir d'un second objet JSON ou de texte supplémentaire ; **la sortie exacte n'a pas été sauvegardée par V1.14**, et on ne doit pas supposer sa forme.

## Changements minimaux

- Aucune modification du prompt, du modèle, du plan d'alignement, de la signature de score ou du format de checkpoint V1.14.
- Sur erreur de validation du JSON uniquement en mode DEV, la réponse intégrale est enregistrée dans `out/diagnostics` avec permissions `0600`. Pas d'écriture partielle en checkpoint.
- Analyse hors ligne de la structure avec `tools.inspect_batch_raw`. Pas de lecture gold/test/final, pas d'appel réseau.
- Sur le chemin `final`, aucune sauvegarde brute DEV n'est autorisée.

## Installation

```
cd /var/www/clauseshift_track2a
git apply --check CLAUSESHIFT_Batch_Raw_Diagnostic_v1.14.1.patch
git apply CLAUSESHIFT_Batch_Raw_Diagnostic_v1.14.1.patch
python3 -m unittest discover -s tests -q
```

## Inspection — après un **seul** nouvel appel instrumenté

Le premier appel d'avant le patch a été perdu ; une reproduction instrumentée est nécessaire pour obtenir la sortie brute. Ne pas enchaîner plusieurs appels identiques.

```
python3 -m src.e2_batch \
  --lang de --doc-id admin_de_132 \
  --alignment semantic_coverage --prompt fewshot \
  --embedding-revision "$EMB_SHA" \
  --max-scores-per-call 96 --max-units-per-batch 3 \
  --workers 1 --max-tokens 1536 --timeout 75 --retries 0 --max-calls 1 \
  --plan-dir out/plans/batch_dev_v114 \
  --checkpoint-dir out/checkpoints/batch_dev_v114
```

Si invalidité : récupérer le chemin affiché après `[raw diagnostic saved: ...]` et lancer :

```
python3 -m tools.inspect_batch_raw --raw-file '/chemin/exact/du/fichier.txt'
```

Ne pas publier le fichier brut entier ; il peut contenir des extraits du corpus. Transmettre seulement la sortie structurée de l'outil.

Si la réponse a été acceptée, ne pas exécuter d'autre appel ; inspecter le checkpoint pour démontrer les unités validées.

## Limites

Ce patch n'améliore pas la qualité du modèle, n'assouplit pas la validation et ne résout pas les faux non-alignements visibles dans le plan de `admin_de_132` (continuation allemande en unité 2, `Fachkontakt`/`Specialist staff`). Ne pas geler E2-BATCH avant correction/validation sur DEV et test du conteneur complet.

# CLAUSESHIFT V1.13 — Requêtes indexées bornées, DEV uniquement

## Évidence motivant cette modification

Le 13 octobre 2026, E0-PJQ (prompt exact du papier, réponse de tout le document) a reçu HTTP 504 en 60,6 s sur `admin_de_132` (193 tokens blancs). Cela ne détermine **pas** un seuil maximal fiable pour Public AI, mais empêche de considérer E0-PJQ en appel unique comme exécutable sur tout le corpus. En comparaison, E2 a déjà renvoyé 73/73 et 76/76 scores sur des unités FR et IT avec `max_tokens=1536`. L'E2 Coverage-Aware ne surpasse pas E0 sur le minuscule échantillon de trois documents DEV : l'approche indexée est une piste de **fiabilité**, pas une victoire de qualité démontrée.

## Modifications

- Ajout à `src.e2` de `--max-scores-per-call` (0 = mode V1.10 préservé ; 2–128 = plafond dur de scores dans une réponse).
- L'algorithme demande des indices croissants pour les deux côtés, conserve les scores validés et reprend les indices non résolus. Aucune imputation, aucun réordonnancement du benchmark.
- Nouveau paramètre inclus dans l'identité signée du checkpoint, seulement lorsqu'il est non nul ; anciens checkpoints V1.10 restent valides avec `0`.
- La limite est **par requête**, et non une limite de longueur de document ou d'unité : le contexte reste inchangé. Ce contrôle ne garantit pas l'absence de 504.
- `tools.dev_e2_runtime_budget` : borne inférieure optimiste d'appels nécessaires sur les 504 documents **DEV** non étiquetés, aucun chargement de gold/final/test, aucun HTTP.
- Aucun changement à `make run` : il utilise toujours E0-PJQ. **NE PAS GÉLER NI SOUMETTRE CETTE BRANCHE EN L'ÉTAT.** Le runner officiel E2 reste un chantier bloquant.

## Installation (patch V1.13 sur V1.12 uniquement)

```bash
cd /var/www/clauseshift_track2a
git apply --check CLAUSESHIFT_Bounded_Indexed_DEV_v1.13.patch
git apply CLAUSESHIFT_Bounded_Indexed_DEV_v1.13.patch
python3 -m unittest discover -s tests -q
```

## Audit hors ligne du coût minimal

```bash
python3 -m tools.dev_e2_runtime_budget \
  --input-dir data/inputs \
  --max-scores-per-call 36 \
  --seconds-per-call 15 \
  --concurrency 1
```

Le calcul `somme(ceil(tokens_non_ponctuation_par_document/36))` est une **borne inférieure** : les frontières d'unités, les requêtes incomplètes, les 504 et les réessais peuvent augmenter significativement le coût. La latence moyenne de 15 s est une **hypothèse de simulation**, pas une mesure universelle.

## Premier test E2 indexé borné — DEV, sans gold

Une fois les variables `EMB_SHA` (SHA épinglé), `LLM_API_KEY`, `LLM_NAME=swiss-ai/apertus-v1.5-8b` et `LLM_BASE_URL=https://api.publicai.co/v1` chargées :

```bash
python3 -m src.e2 \
  --lang de --doc-id admin_de_132 \
  --alignment semantic_coverage --prompt fewshot \
  --chunk-tokens 48 --max-unit-tokens 48 \
  --similarity-threshold 0.46 --merge-penalty 0.06 \
  --coverage-weight 0.50 --anchor-bonus 0.35 \
  --embedding-revision "$EMB_SHA" \
  --max-scores-per-call 36 --plan-only \
  --plan-dir out/plans/e2bounded_dev_v113 \
  --checkpoint-dir out/checkpoints/e2bounded_dev_v113
```

Inspecter ce plan :

```bash
python3 -m tools.inspect_e2_alignment \
  --lang de --doc-id admin_de_132 \
  --plan-dir out/plans/e2bounded_dev_v113 --limit 30
```

Si l'alignement est cohérent, lancer un seul appel en supprimant `--plan-only` et en ajoutant ces options à la même commande :

```bash
  --context 20 --max-tokens 1536 --timeout 75 --retries 0 \
  --max-calls 1 --max-attempts-per-unit 20
```

**ATTENTION** : ces lignes ne sont pas une commande complète seules ; conserver les options de la commande précédente et ne pas exécuter si le plan est douteux. Reprise en augmentant `--max-calls` sans changer les autres paramètres. `max_calls` est un nombre **d'appels nouveaux par exécution**.

## Points bloquants avant soumission

1. Corrélation Spearman sur plusieurs documents DEV, selon les trois langues, sans surajustement aux documents inspectés.
2. Mesure réelle de débit et estimation du délai pour l'ensemble des documents.
3. Intégration de la méthode choisie dans le runner `make run` complet, sous manifeste gelé et Docker avec dépendances épinglées.
4. Contrôle des quotas, des secrets et de la possibilité pour les juges de faire fonctionner l'API ; aucune prédiction fabriquée sur échec.
5. Gel puis exécution finale sans optimisation sur le jeu de test.

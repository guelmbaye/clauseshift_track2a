# CLAUSESHIFT V1.14 — Batched E2 Indexed Scoring / Submission Runner

**État au 13 octobre 2026 : implémenté et vérifié localement sur réponses synthétiques ; aucun score Spearman ni résultat réel Public AI pour le format batché n'est encore disponible.** Ne pas présenter V1.14 comme validé pour l'évaluation officielle ou comme gagnant.

## Pourquoi cette évolution

E0-PJQ traite les textes entiers et a provoqué un HTTP 504 en 60,6 s pour `admin_de_132` (193 tokens). E2 V1.13 a validé 68 scores en 25,6 s sur la première unité de ce document, mais un appel par unité multiplie le temps total. Avec plafond théorique de 96 scores, l'estimation DEV optimiste était >= 3 894 appels sur 504 documents. La V1.14 regroupe **plusieurs unités consécutives**, chacune avec des indices locaux séparés ; l'impact sur la qualité sémantique est une hypothèse à mesurer.

## Propriétés techniques

- Le plan bilingue `semantic_coverage` est inchangé. L'embedding est chargé une fois par processus (cache mémoire).
- `--max-scores-per-call` plafonne les scores ; `--max-units-per-batch` borne le nombre d'unités groupées.
- Une seule réponse `{"units":{"0":{"a":{...},"b":{...}},"1":...}}` est validée en totalité avant écriture. Aucune complétion automatique ou acceptation silencieuse d'un indice manquant.
- Checkpoints par unité avec signature **distincte de V1.13**, hash du plan, identité des spans. Reprise sans recalcul des scores déjà acceptés.
- `--workers` de 1 à 8 autorise la parallélisation des lots d'un même document. Les résultats sont écrits en série et `fsync` après chaque lot. `--launch-interval` temporise la soumission des tâches, **ce n'est pas une garantie de débit API**. En cas d'HTTP 429/504 ou JSON invalide, le processus échoue sans scores inventés ; les lots valides peuvent être repris. Le nombre de tentatives doit rester contrôlé.
- `python3 -m src.e2_batch` est strictement **DEV uniquement**. Seul `python3 -m src.submission run`, après validation du manifeste figé, peut faire une exécution finale.
- Un fichier de prédictions n'est produit que si **tous les documents d'une langue** sont complets, avec reconstruction en ordre original. Le runner final valide ensuite l'ensemble des trois langues. Ne jamais fournir un fichier partiel.

## Installer sur V1.13

```bash
cd /var/www/clauseshift_track2a
git apply --check CLAUSESHIFT_Batch_Submission_v1.14.patch
git apply CLAUSESHIFT_Batch_Submission_v1.14.patch
python3 -m unittest discover -s tests -q
```

Le `make build` V1.14 installe des dépendances lourdes (PyTorch CPU, sentence-transformers). Prévoir espace disque, réseau, et cache modèle Hugging Face accessible. **Le build Docker complet n'a pas été réalisé dans notre environnement local.**

## DEV : contrôle gratuit du plan d'abord

```bash
: "${EMB_SHA:?SHA exact absent}"
: "${LLM_API_KEY:?clé absente}"
python3 -m src.e2_batch \
  --lang de --doc-id admin_de_132 --alignment semantic_coverage --prompt fewshot \
  --embedding-revision "$EMB_SHA" \
  --max-scores-per-call 96 --max-units-per-batch 3 \
  --workers 1 --max-tokens 1536 --plan-only \
  --plan-dir out/plans/batch_dev_v114 \
  --checkpoint-dir out/checkpoints/batch_dev_v114
```

Inspecter les unités avec :

```bash
python3 -m tools.inspect_e2_alignment \
  --lang de --doc-id admin_de_132 --plan-dir out/plans/batch_dev_v114 --limit 30
```

## DEV : un seul appel réel pour contrôler le JSON multi-unités

```bash
python3 -m src.e2_batch \
  --lang de --doc-id admin_de_132 --alignment semantic_coverage --prompt fewshot \
  --embedding-revision "$EMB_SHA" \
  --max-scores-per-call 96 --max-units-per-batch 3 \
  --workers 1 --max-tokens 1536 --max-calls 1 --retries 0 \
  --plan-dir out/plans/batch_dev_v114 \
  --checkpoint-dir out/checkpoints/batch_dev_v114
```

Vérifier `BATCH START`, `BATCH SAVED`, le nombre d'unités et scores. Si la requête échoue, **ne pas relancer automatiquement** : réduire la taille du batch dans un nouveau répertoire, examiner l'erreur et réévaluer le budget.

L'estimateur fondé sur les plans existants (sans API) :

```bash
python3 -m tools.dev_batch_plan_budget \
  --input-dir data/inputs --plan-dir out/plans/batch_dev_v114 \
  --max-scores-per-call 96 --max-units-per-batch 3
```

Pour calculer Spearman **uniquement DEV** sur les documents entièrement produits et communs à E0 :

```bash
python3 -m tools.compare_e2_batch_dev \
  --lang de --doc-id admin_de_132 \
  --upstream-root "$(realpath ../SwissGov-RSD)" \
  --embedding-revision "$EMB_SHA" \
  --plan-dir out/plans/batch_dev_v114 \
  --batch-checkpoints out/checkpoints/batch_dev_v114
```

`admin_de_132` n'a pas de prédiction E0-PJQ complète pour la comparaison ; utiliser plus tard un document commun complet, par exemple `admin_de_105`, une fois le batch terminé.

## Gel final : NE PAS EXÉCUTER avant d'avoir choisi/validé la méthode

Après validation DEV et contrôle Docker, choisir explicitement **une seule** méthode :

```bash
UPSTREAM_SHA=$(git -C ../SwissGov-RSD rev-parse HEAD)
FROZEN_COMMIT=$(git rev-parse HEAD)
# Seulement si E2-BATCH est sélectionnée, et AVANT accès aux textes finaux :
make freeze METHOD=E2-BATCH EMB_SHA="$EMB_SHA" \
  UPSTREAM_SHA="$UPSTREAM_SHA" FROZEN_COMMIT="$FROZEN_COMMIT" \
  BATCH_CAP=96 UNITS_PER_BATCH=3 WORKERS=1 LAUNCH_INTERVAL=0
python3 -m src.submission preflight
# Inclure ensuite docs/submission_frozen.json dans le dépôt.
```

`make run` construit Docker, télécharge **uniquement les entrées non étiquetées** dans l'ordre complet après vérification du gel, reconstruit trois prédictions intégrales, vérifie les longueurs et IDs, échoue sinon. La provenance du benchmark et l'API doivent rester accessibles au conteneur.

## Limites et risques bloquants

1. Plusieurs milliers d'appels restent possibles sur 224 documents par langue. Aucun temps total réel ni quota Public AI à huit travailleurs n'est démontré. Une concurrence élevée peut générer des HTTP 429.
2. Un batch plus gros peut augmenter les 504 et la troncature JSON. `max_tokens=1536` n'a été validé précédemment que pour des unités isolées (73/76 scores), pas pour ce format multi-unités.
3. La qualité Spearman est **inconnue** pour V1.14. Regrouper des unités change le contexte de sortie ; les tests synthétiques ne sont pas des preuves scientifiques.
4. Build Docker lourd et dépendance au téléchargement Hugging Face lors du premier démarrage ; prévoir une procédure de cache si le règlement le permet.
5. Ne jamais optimiser sur le jeu final. Faire un seul gel, puis aucune modification de code ou paramètres avant soumission.

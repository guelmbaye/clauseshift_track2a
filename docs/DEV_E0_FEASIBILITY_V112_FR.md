# CLAUSESHIFT V1.12 — Diagnostic de faisabilité E0-PJQ sur DEV

**Statut : outil de diagnostic pré-gel, sans accès au test final.** Le Docker V1.11 construit correctement et son preflight refuse sans manifeste, ce qui est attendu. Cela ne prouve pas que le moteur E0-PJQ puisse traiter les documents longs : des HTTP 504 ont été observés sur Public AI.

## Fonctionnalités ajoutées

- `tools.dev_e0_feasibility` : énumère les longueurs de paires **DEV seulement**, médiane/P90/max et un document de sondage par tranche, en tenant compte des checkpoints E0-PJQ déjà complets. Ne lit ni gold, ni full, ni test, et n'effectue aucun appel réseau.
- `src.cli run --doc-id` : sélectionne **exactement un document DEV**, exige `--lang de/fr/it`, conserve la signature de méthode et les checkpoints existants. Refuse `--split final`, `--lang all`, `--smoke-shortest`, `--dev-shortest-first`.
- Les exports complets restent régis par la même règle : aucune prédiction officielle tant que tous les IDs n'ont pas été validés dans l'ordre de référence.

## Appliquer le patch sur V1.11 installée

```bash
cd /var/www/clauseshift_track2a
git apply --check CLAUSESHIFT_DEV_Long_Document_Probe_v1.12.patch
git apply CLAUSESHIFT_DEV_Long_Document_Probe_v1.12.patch
python3 -m unittest discover -s tests -q
make build
```

**Ne pas figer** `docs/submission_frozen.json` avant décision finale. Ne pas utiliser `--split final`.

## Audit sans coût API

```bash
python3 -m tools.dev_e0_feasibility \
  --input-dir data/inputs \
  --checkpoint-dir out/checkpoints/e0pjq_publicai_4096
```

La sortie donnera les buckets `0-128`, `129-256`, `257-512`, `513-768`, `769-1024`, `1025+`, avec une ligne `probe=admin_...` par langue et par bucket non vide. **Envoyer ce résultat avant de lancer les sondages.** Les valeurs sont des nombres de tokens `str.split()`, pas des tokens de modèle.

## Sondage ciblé DEV à budget contrôlé

Choisir un ID indiqué par l'audit (par exemple dans `257-512`), puis lancer exactement **un** document. Ne jamais tester des IDs `full`/`test` ici.

```bash
python3 -m src.cli run \
  --split dev --lang de --doc-id 'REMPLACER_ID_DEV_DE' \
  --prompt-file prompts/paper_exact.txt \
  --punctuation-repair --syntax-repair --boundary-quote-repair \
  --max-tokens 4096 --timeout 75 --retries 0 \
  --checkpoint-dir out/checkpoints/e0pjq_publicai_4096
```

Vérifier auparavant `LLM_NAME=swiss-ai/apertus-v1.5-8b`, `LLM_BASE_URL=https://api.publicai.co/v1`, `LLM_API_KEY` dans l'environnement ; ne pas afficher la clé. La signature de checkpoint demeure compatible avec E0-PJQ lorsque `model`, `prompt`, `max-tokens`, `base-url` et réglages de réparation sont inchangés.

La présence d'un HTTP 504 indique **un blocage du fournisseur sur cette requête**. Augmenter seulement `--timeout` côté client ne contourne pas nécessairement le délai du proxy. Enregistrer le nombre de tokens et le temps ; ne pas répéter indéfiniment la requête.

## Barrière de soumission

`make run` V1.11 reste E0-PJQ **document entier**, sans fallback fiable en cas de 504. Ni ce patch ni son audit ne changent cette limite. Une stratégie finale avec fallback nécessite une implémentation séparée, des tests, une validation DEV puis un nouveau gel ; ou utiliser une méthode alternative dont l'exécution complète est démontrée.

La construction de l'image Docker et la protection pré-gel sont les seules validations serveur établies à ce stade. Aucun score macro officiel n'a été mesuré, et aucune prédiction sur le jeu final n'a été produite.

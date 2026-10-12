# CLAUSESHIFT — Guide d'implémentation détaillé (FR)
## Benchmark SwissGov-RSD & baseline Apertus — du premier test à la soumission

**Statut au 12 octobre 2026 :** code de départ et tests locaux opérationnels ; pas encore de résultat Apertus réel ni de score officiel calculé. Le dossier contient un Document 01 figé, le protocole, le code source, Docker et des tests.

## Étape 1 — Préparer l'environnement

**Prérequis :** Git, Python 3.11 ou supérieur, Docker Desktop/Engine pour la soumission ; sur Windows, WSL2 Ubuntu conseillé. Un compte d'inférence proposant réellement `swiss-ai/Apertus-v1.5-8B` est nécessaire. Les endpoints fournis par d'autres modèles ne sont pas admissibles comme moteur principal.

Décompresser l'archive, ouvrir un terminal dans `CLAUSESHIFT_Track2A`, exécuter :

```bash
python3 -m unittest discover -s tests -v
```

**Résultat attendu :** 10 tests verts ; intégration HTTP testée avec une réponse synthétique, donc aucune performance réelle mesurée.

**Si échec :** vérifier la version de Python, le dossier courant et les messages des tests. Aucun GPU ni clé API ne sont nécessaires pour cette étape.

## Étape 2 — Récupérer le code scientifique UZH

```bash
git clone --depth 1 https://github.com/ZurichNLP/SwissGov-RSD.git ../SwissGov-RSD
export SWISSGOV_ROOT="$(realpath ../SwissGov-RSD)"
(cd "$SWISSGOV_ROOT" && git rev-parse HEAD)
```

Enregistrer le hash Git. Le dépôt scientifique public contient aussi des données test : **ne pas les consulter pendant le développement**. Toutes nos commandes de développement utilisent exclusivement `gold_labels/dev`.

Le code scientifique contient l'évaluateur officiel `scripts/evaluate_predictions_admin.py` ; ne pas le réécrire pour obtenir un meilleur score.

## Étape 3 — Charger les exemples DEV sans labels dans les prompts

Option préférée si le dépôt est cloné :

```bash
python3 -m src.cli prepare --split dev --upstream-root "$SWISSGOV_ROOT"
```

Sinon, récupération DEV seulement :

```bash
python3 -m src.cli fetch-dev --upstream-sha master
```

Trois fichiers doivent apparaître : `data/inputs/dev_de.jsonl`, `dev_fr.jsonl`, `dev_it.jsonl`.

**Chaque ligne contient seulement `id`, `text_a`, `text_b`**. Les labels DEV restent disponibles uniquement à l'évaluateur pour le scoring ; ils ne sont jamais insérés dans le prompt.

Ne pas exécuter `prepare --split final` maintenant : cette commande impose un gel explicite de code.

## Étape 4 — Reproduire le prompt du papier, pas une approximation

Le papier *SwissGov-RSD* fournit son prompt complet **Annexe E** : https://aclanthology.org/2026.acl-long.1437.pdf

Le prompt du papier comporte trois exemples et une sortie JSON de la forme :

```json
{
  "sentence1": [["token1", 5], ["token2", 0]],
  "sentence2": [["token3", 5], ["token4", 0]]
}
```

Les scores bruts sont des **similarités** : `5` = sens équivalent, `0` = aucun rapport, `-1` = ponctuation. Le code convertit ces similarités en différences : `1 − score/5`, en préservant `-1`.

**Action requise :** recopier fidèlement l'Annexe E dans `prompts/paper_exact.txt` en gardant deux marqueurs de substitution littéraux :

- `{{ sentence1 }}`
- `{{ sentence2 }}`

La substitution remplace ces marqueurs par des listes JSON de tokens. Ne pas ajouter d'exemples issus du test.

Le code inclut déjà un prompt **paper-style** original pour vérifier la plomberie. **Il n'est pas identique au prompt du papier et ne suffit donc pas à revendiquer une reproduction exacte.**

## Étape 5 — Configurer l'inférence Apertus

Dans Linux/WSL :

```bash
export LLM_NAME='swiss-ai/Apertus-v1.5-8B'
export LLM_BASE_URL='https://api.swissai.svc.cscs.ch/v1'
export LLM_API_KEY='VOTRE_CLE_AUTORISEE'
```

Remplacer le serveur ou la clé si ce point d'accès n'est pas accessible ou n'est pas autorisé pour votre compte. Le client attend une API compatible `POST /chat/completions`. Les variables restent en environnement ; **ne jamais committer une clé**.

Dans PowerShell (hors WSL) :

```powershell
$env:LLM_NAME='swiss-ai/Apertus-v1.5-8B'
$env:LLM_BASE_URL='https://api.swissai.svc.cscs.ch/v1'
$env:LLM_API_KEY='VOTRE_CLE_AUTORISEE'
```

**Important :** le temps limite signalé à CSCS est d'environ 60 secondes. La requête client est limitée à 52 secondes. Un `504` déclenche un échec explicite, pas une réponse fictive.

## Étape 6 — Première requête réelle (smoke)

```bash
python3 -m src.cli run --split dev \
  --prompt-file prompts/paper_exact.txt \
  --max-examples 1 \
  --max-tokens 4096 \
  --checkpoint-dir out/checkpoints/e0_exact_v1
```

**Attendu :** 1 requête sur un exemple DEV par langue, soit au plus 3 nouvelles requêtes. Les prédictions sont sauvegardées en checkpoints. Les fichiers complets ne sont publiés que lorsque tous les exemples attendus sont couverts.

Si un appel échoue, conserver le message ; ne pas remplir les scores manuellement. Pour une erreur `504`, envisager une approche de segmentation distincte ou un serveur Apertus propre. La segmentation ne doit pas être cachée dans une baseline annoncée « originale ».

## Étape 7 — Baseline E0 sur l'ensemble DEV

```bash
python3 -m src.cli run --split dev \
  --prompt-file prompts/paper_exact.txt \
  --max-tokens 4096 \
  --checkpoint-dir out/checkpoints/e0_exact_v1
```

Si la première requête est déjà passée, le programme reprend où il s'était arrêté. Si vous changez le prompt, les paramètres ou le modèle, **utilisez un autre dossier de checkpoints**.

Les sorties complètes attendues :

```text
out/predictions/CLAUSESHIFT_admin_de.jsonl.jsonl
out/predictions/CLAUSESHIFT_admin_fr.jsonl.jsonl
out/predictions/CLAUSESHIFT_admin_it.jsonl.jsonl
```

Cette double extension est imposée par la logique du chargeur original avec le préfixe et le format de prédiction choisis. **Ne pas la « corriger » sans retester le script officiel.**

## Étape 8 — Validation stricte des prédictions

```bash
python3 -m src.cli validate --split dev
```

Contrôles : nombre d'exemples par langue, ordre, identifiants, égalité des textes, longueur token-level des deux côtés, score borné et absence de NaN. Toute anomalie fait échouer l'exécution.

Le validateur empêche le padding/troncature silencieux que peut tolérer l'évaluateur officiel.

## Étape 9 — Installer les dépendances de l'évaluateur UZH

Utiliser une installation isolée, de préférence Python 3.11 :

```bash
python3.11 -m venv .venv-rsd
source .venv-rsd/bin/activate
pip install -r "$SWISSGOV_ROOT/requirements.txt"
pip install -r "$SWISSGOV_ROOT/rsd/requirements.txt"
```

Sur Windows PowerShell, activer avec `.venv-rsd\Scripts\Activate.ps1` et définir `SWISSGOV_ROOT` dans l'environnement PowerShell, ou faire la procédure dans WSL2.

## Étape 10 — Calculer le score DEV officiel

```bash
python -m src.cli score-dev --upstream-root "$SWISSGOV_ROOT"
```

Le programme appelle l'évaluateur original avec `--split dev` depuis la racine du dépôt de recherche. Relever :

| Paire | Spearman DEV | Kendall tau-b | Temps / coût |
|---|---|---|---|
| EN–DE | à mesurer | à mesurer | à mesurer |
| EN–FR | à mesurer | à mesurer | à mesurer |
| EN–IT | à mesurer | à mesurer | à mesurer |

La moyenne officielle est : **(Spearman_DE + Spearman_FR + Spearman_IT)/3**. Aucun score n'est prérempli ou présumé.

## Étape 11 — Comparaison et améliorations possibles

Seulement après avoir obtenu E0 :

- E1 : alignement lexical seul, idéalement DiffAlign réel ; ne pas appeler une approximation « DiffAlign ».
- E2 : Apertus guidé par régions alignées et corrections sémantiques.
- E3 : fusion des scores avec paramètres calibrés sur DEV uniquement.
- E4 : ablation sans corrections Apertus pour prouver sa nécessité.

Pour chaque variante : nouveau dossier de checkpoints, mêmes exemples DEV, scoring identique et temps d'inférence observé. Arrêter toute nouvelle fonctionnalité qui n'apporte pas de valeur mesurable.

**Remarque :** les modules E1–E4 sont décrits mais **pas implémentés dans ce premier paquet**. Les développer après la reproduction E0, en utilisant les observations DEV, pas le test.

## Étape 12 — Exécution Docker

```bash
make run
```

`make run` construit l'image, récupère **les données DEV seulement**, puis lance le **prompt paper-style inclus dans le code**. Il sert actuellement à valider l'intégration Docker de développement. Pour la reproduction exacte avec Docker, monter le fichier de prompt et ajouter `--prompt-file` à la commande de lancement.

**Attention :** Docker n'était pas disponible dans l'environnement d'auteur ; à vérifier sur votre machine.

Après gel uniquement, la variante `make run RUN_SPLIT=final FROZEN_COMMIT=<SHA>` télécharge les entrées full-order depuis la référence et exécute le prompt figé défini par `PROMPT_FILE`, en produisant les trois fichiers complets. **Ne pas lancer ce mode pendant le développement.** Si E2 ou E3 devient notre meilleure méthode, adapter le `make run` final à la méthode effectivement retenue, et vérifier sur clone propre.

## Étape 13 — Gel final (après les choix faits sur DEV)

Après avoir validé E0 et les éventuelles améliorations, créer le dépôt Git, enregistrer les changements et relever un commit exact :

```bash
git add .
git commit -m 'Freeze CLAUSESHIFT competition method'
GIT_SHA=$(git rev-parse HEAD)
printf '%s\n' "$GIT_SHA" > docs/frozen_commit.txt
```

Les paramètres, templates, noms de modèles et commits de dépendances doivent être figés et décrits dans `technical_report.md`.

**Ne pas commencer la phase finale avec un workspace encore en développement.**

## Étape 14 — Prédictions full-order et évaluation TEST (final uniquement)

**À exécuter une seule fois après gel** (et jamais pour améliorer les prompts) :

```bash
python -m src.cli prepare --split final \
  --upstream-root "$SWISSGOV_ROOT" \
  --confirm-final --frozen-commit "$GIT_SHA"

python -m src.cli run --split final \
  --prompt-file prompts/paper_exact.txt \
  --checkpoint-dir out/checkpoints/final_frozen \
  --confirm-final --frozen-commit "$GIT_SHA"

python -m src.cli validate --split final
```

La commande de préparation finale lit le fichier `full` de référence **uniquement après gel**, n'en transmet que les textes/IDs au modèle et ne consulte pas les labels pour prendre des décisions. Si un fichier officiel contenant uniquement les entrées full-order est disponible, préférer ce fichier.

**Attention :** le mode final écrira dans le même répertoire `out/predictions` que E0. Sauvegarder les résultats DEV et leurs hashes avant de lancer la phase finale : `cp -r out/predictions out/dev_prediction_archive` (le checkpoint DEV demeure distinct).

Puis évaluation finale par le script officiel :

```bash
(cd "$SWISSGOV_ROOT" && python -m scripts.evaluate_predictions_admin \
  /CHEMIN/ABSOLU/CLAUSESHIFT_Track2A/out/predictions/CLAUSESHIFT_admin_ \
  --split test)
```

**Aucun test n'entre dans la phase de développement :** le premier accès à full/test se produit uniquement ici. Après lecture des résultats test, ne pas chercher de nouveaux hyperparamètres sur ces scores. Les organisateurs demandent des prédictions de l'ensemble full **dans l'ordre officiel**, même si seuls les exemples test sont notés.

## Étape 15 — Intégrer au template GitHub officiel

Le ZIP a été construit à partir du contrat vérifié du template, mais le clonage réseau n'était pas possible pendant sa création. Pour éviter toute divergence, créer le dépôt **depuis le template officiel** :

```bash
git clone https://github.com/HackApertus/project-template.git mon-clausehift
cd mon-clausehift
rm -rf track_1a track_1b track_2b
# conserver track_2a/ avec ses fichiers officiels
# copier src/, docs/, prompts/, tests/, Dockerfile, Makefile et technical_report.md
# issus de notre ZIP dans track_2a/ ; préserver le README.md officiel
# et y intégrer les sections CLAUSESHIFT sans effacer les exigences du concours
# puis depuis track_2a : make run et validation des sorties
```

Ne pas déplacer les fichiers requis du template. Vérifier si le dépôt officiel a changé depuis le 12 octobre et réconcilier les modifications.

## Étape 16 — Soumission

Fusionner le travail avec le dernier template officiel `track_2a` sans en déplacer les fichiers requis. Contrôler le `make run` final dans un clone propre ; valider les trois fichiers de prédictions, les licences, la description de méthode, les paramètres, les chiffres et les limites dans `technical_report.md`. Déposer via : https://hackapertus.ch/online-hack/submissions

**Date limite :** 16 octobre 2026 à 12h CEST, à confirmer sur la page de soumission avant envoi.

## État à la livraison

**Validé :** génération du code, compilation Python, 10 tests locaux incluant un faux serveur HTTP de bout en bout, validateur de tokens et séparation dev/final.  
**Non validé :** vraie API Apertus, exact prompt Annexe E (à installer), chargement DEV live, métriques scientifiques, Docker réel, exécution TEST (interdite à ce stade).

---

**Point de décision :** ne construire aucune interface Next.js avant d'avoir une première valeur macro Spearman E0 vérifiable sur DEV.

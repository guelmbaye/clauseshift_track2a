# CLAUSESHIFT V1.11 — Submission runner gelé (E0-PJQ)

**Date : 13 octobre 2026.** Ce patch prépare un `make run` final utilisable depuis un checkout propre, mais **n'autorise pas encore la lecture du jeu final** : l'utilisateur doit d'abord terminer la validation, choisir la méthode, puis générer et committer le manifeste de gel. Il ne constitue PAS une preuve d'exécution complète du benchmark.

## Décision provisoire

Un seul document DEV commun par langue (`admin_*_105`) :

| Langue | E0-PJQ | E2 Coverage V1.10 |
|---|---:|---:|
| DE | 0.477141 | 0.291087 |
| FR | -0.029263 | 0.126153 |
| IT | 0.020295 | -0.075864 |
| Moyenne exploratoire | 0.156058 | 0.113792 |

Ces Spearman ont été calculés par un **proxy exploratoire**, pas par l'évaluateur officiel sur l'intégralité de DEV. E2 est plus fiable sur de longs passages, E0 a déjà subi des HTTP 504 sur certains documents. Il est donc interdit de présenter E0 comme globalement supérieur et/ou opérationnellement complet.

## Ce que le patch apporte

- `make run` pointe sur `src.submission run` et le moteur **E0-PJQ réel** ; la commande exécute Docker, vérifie le manifeste et la compatibilité du code et du prompt, télécharge les textes non étiquetés en ordre complet uniquement après gel, lance l'inférence, valide strictement les fichiers et retourne une erreur en cas de résultat partiel.
- Le manifeste lie le commit de gel déclaratif, le commit exact de SwissGov-RSD, le SHA256 du code et du prompt, le modèle et les paramètres de sortie.
- Les prédictions finales précédentes sont supprimées au début de l'exécution, pour qu'un résultat incomplet ne puisse pas être validé au moyen de fichiers obsolètes. Les checkpoints signés, eux, sont conservés.
- `make run-dev` garde le parcours DEV séparé. Les expériences E2 demeurent DEV-only et ne sont **pas** sélectionnées par `make run`.

## Installation sur un dépôt V1.10 (avant gel)

```bash
cd /var/www/clauseshift_track2a
git apply --check CLAUSESHIFT_Submission_Runner_v1.11.patch
git apply CLAUSESHIFT_Submission_Runner_v1.11.patch
python3 -m unittest discover -s tests -q
python3 -m src.submission preflight  # DOIT refuser si le manifeste n'existe pas encore
```

Ne pas écraser les checkpoints et ne pas créer de faux manifeste pour contourner le refus.

## Préparation puis gel définitif — APRÈS avoir choisi la méthode

Le patch V1.11 est **spécifique à E0-PJQ**. Si les résultats DEV élargis favorisent E2, il faut **adapter et retester le runner**, et ne pas geler E0 pour satisfaire artificiellement les critères de soumission.

```bash
# Vérifier le commit local exact du benchmark (pas 'master')
UPSTREAM_SHA=$(git -C ../SwissGov-RSD rev-parse HEAD)
FROZEN_COMMIT=$(git rev-parse HEAD)
export LLM_NAME='swiss-ai/apertus-v1.5-8b'

# Uniquement à la décision explicite de gel, quand le code ne doit plus évoluer
make freeze UPSTREAM_SHA="$UPSTREAM_SHA" FROZEN_COMMIT="$FROZEN_COMMIT"
python3 -m src.submission preflight
git add docs/submission_frozen.json
git commit -m 'Freeze E0-PJQ submission manifest'
```

Le `frozen_commit` est le commit de code **avant** ajout du manifeste ; le SHA du contenu verrouille le code des deux côtés, y compris dans le conteneur Docker. Ne pas réexécuter `make freeze` sur un manifeste existant. Ce manifeste doit être présent dans le repository soumis.

## Simulation de préparation sans accès au jeu final

```bash
python3 -m unittest discover -s tests -q
python3 -m src.submission preflight
# Après gel, la commande officielle réelle :
# make run
```

Pré-requis du conteneur : Docker, sortie réseau vers la copie épinglée de SwissGov-RSD et Public AI, les variables d'environnement `LLM_NAME`, `LLM_BASE_URL`, `LLM_API_KEY` et facultativement `LLM_USER_AGENT`. La clé ne doit jamais être committée. Un accès réseau et un quota API suffisants doivent être mis à disposition des juges (vérifier avec eux).

**Limitation bloquante :** E0-PJQ seul peut échouer sur les documents très longs (HTTP 504 déjà observés). V1.11 les refuse proprement, il ne les résout pas. Tant que ce problème n'est pas traité ou qu'une méthode de secours n'est pas évaluée sur DEV, ne pas affirmer que `make run` termine le jeu complet. Vérifier aussi `docker build`, `docker run` et l'ordre complet avant soumission. Les tests inclus sont locaux et synthétiques : aucune inférence Apertus réelle ni exécution Docker dans le conteneur de développement.

Après exécution réussie, outputs :

- `out/predictions/CLAUSESHIFT_admin_de.jsonl.jsonl`
- `out/predictions/CLAUSESHIFT_admin_fr.jsonl.jsonl`
- `out/predictions/CLAUSESHIFT_admin_it.jsonl.jsonl`

Les mêmes fichiers doivent être produits **sans labels gold** et dans l'ordre de référence. Il ne faut exécuter ni évaluation ni optimisation sur le split test.

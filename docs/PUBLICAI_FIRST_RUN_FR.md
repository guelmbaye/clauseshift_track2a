# CLAUSESHIFT — Premier lancement Public AI (Hotfix V1.1)

## 1. Prérequis

Les deux requêtes `curl` vers Public AI ont confirmé :
- `LLM_BASE_URL=https://api.publicai.co/v1`
- `LLM_NAME=swiss-ai/apertus-v1.5-8b`
- la clé `LLM_API_KEY` est fonctionnelle (non stockée ici)

## 2. Tester le correctif

```bash
python3 -m unittest discover -s tests -v
```

La V1.1 corrige la vérification sensible à la casse du nom Apertus, ajoute `User-Agent: CLAUSESHIFT/1.0` et autorise un smoke sur l'exemple DEV le plus court.

## 3. Vérifier les variables d'environnement

```bash
export LLM_BASE_URL='https://api.publicai.co/v1'
export LLM_NAME='swiss-ai/apertus-v1.5-8b'
export LLM_USER_AGENT='CLAUSESHIFT/1.0'
: "${LLM_API_KEY:?LLM_API_KEY absent}"
```

Ne pas imprimer la clé et ne pas la committer.

## 4. Charger uniquement dev

```bash
python3 -m src.cli fetch-dev
```

Ou avec la copie locale du dépôt ZurichNLP :

```bash
python3 -m src.cli prepare --split dev --upstream-root "$SWISSGOV_ROOT"
```

## 5. Smoke sur les documents DEV les plus courts

```bash
python3 -m src.cli run --split dev \
  --prompt-file prompts/paper_exact.txt \
  --smoke-shortest \
  --max-tokens 4096 \
  --timeout 75 \
  --checkpoint-dir out/checkpoints/e0_publicai_4096
```

Ce test produit un checkpoint par langue, sans fichier final complet. Il peut s'arrêter au premier JSON invalide. Ne pas confondre cette phase avec une baseline mesurée.

## 6. Étendre à DEV complet

Attention : conserver la même signature de paramètres et le même répertoire de checkpoint que pour le smoke si celui-ci a réussi.

```bash
python3 -m src.cli run --split dev \
  --prompt-file prompts/paper_exact.txt \
  --max-tokens 4096 \
  --timeout 75 \
  --checkpoint-dir out/checkpoints/e0_publicai_4096
```

Si l'API refuse une sortie longue ou que le modèle ne retourne pas des scores pour tous les tokens, arrêter et analyser les tailles / le prompt / le contexte sur DEV. Ne pas corriger automatiquement les sorties en les tronquant ou en remplissant de 0.

## 7. Valider et noter les résultats

```bash
python3 -m src.cli validate --split dev
python3 -m src.cli score-dev --upstream-root "$SWISSGOV_ROOT"
```

Le score officiel ne peut être calculé qu'après production de la totalité des fichiers DEV. Ne jamais exécuter `--split final` avant le gel de la solution.

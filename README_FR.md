# CLAUSESHIFT — Diagnostic d'un token manquant (DEV uniquement)

Ce diagnostic ne remplace aucun fichier existant et ne modifie ni le prompt original, ni les sorties officielles, ni les checkpoints. Il ajoute `tools/diagnose_token_drift.py`.

## Installation

Depuis le serveur, extraire l'archive dans `/var/www/clauseshift_track2a`, sans écraser les fichiers existants. Le fichier à copier est `tools/diagnose_token_drift.py`.

## Exécution — diagnostiquer la requête FR qui a échoué (un appel API supplémentaire)

```bash
cd /var/www/clauseshift_track2a
export LLM_NAME="swiss-ai/apertus-v1.5-8b"
export LLM_BASE_URL="https://api.publicai.co/v1"
export LLM_USER_AGENT="CLAUSESHIFT/1.0"
: "${LLM_API_KEY:?Clé manquante}"
python3 tools/diagnose_token_drift.py \
  --lang fr --smoke-shortest \
  --prompt-file prompts/paper_exact.txt \
  --max-tokens 4096 --timeout 75 --confirm-api-call
```

Dans ton journal, la requête allemande a réussi. Le prochain appel, non enregistré, est très probablement le premier exemple français : `--lang fr --smoke-shortest` choisit exactement le même exemple que `src.cli run`. Le code de sortie 2 est **normal** si le modèle reproduit le défaut. Le fichier brut est enregistré dans `out/diagnostics/`, avec un rapport JSON plus court. Le diagnostic indique les blocs de tokens manquants/fusionnés. Aucun contenu du split test n'est lu.

## Analyser une réponse déjà enregistrée sans frais API

```bash
python3 tools/diagnose_token_drift.py --lang fr --smoke-shortest \
  --raw-file out/diagnostics/fr_admin_fr_XXXXX_XXXXXXXXTXXXXXXZ_raw.txt
```

## Après diagnostic

- S'il s'agit d'un token de ponctuation isolé, ne pas le deviner silencieusement. Un post-traitement spécifique explicite pourrait le restaurer pour les expériences modifiées (E1), mais garder E0 fidèle et documenter l'écart.
- Si un mot porteur de sens manque ou si des tokens sont fusionnés, ne pas insérer des scores arbitraires ; essayer un nouveau mode de génération (chunking, scores-indexés ou génération contrainte) **sous un identifiant expérimental distinct**.
- Si une seconde requête renvoie une sortie complète, conserver la trace du nombre de tentatives. Ce n'est pas une reproduction strictement identique en coût, et les politiques de reprise doivent être reportées.

Ne jamais partager `LLM_API_KEY` ou le fichier `.env`.

# CLAUSESHIFT E1-C — protocole d'essai DEV (V1.7)

## Statut et limites

Cette expérimentation est **une méthode candidate distincte**, et non la reproduction de la baseline E0 du papier. Son découpage par position relative est une heuristique qui peut dégrader les scores sur les ajouts/suppressions et les documents désalignés. Le gain Spearman est entièrement **inconnu** avant le test sur DEV. Aucun fichier `test` n'est lu par cet outil.

Le script E1-C sauvegarde des checkpoints au **niveau du chunk**, jamais de prédictions partielles au format officiel. Il n'exporte le JSONL de la langue que lorsque **tous** ses documents DEV sont entièrement calculés. L'ordre de sortie est celui des fichiers DEV de référence, indépendamment de l'ordre d'inférence.

## Installation sur la V1.6 existante

1. Conserver `out/checkpoints/e0p_publicai_4096` et `out/checkpoints/e0pjq_publicai_4096`.
2. Appliquer le correctif V1.7 sur les fichiers du projet et exécuter les tests.
3. Contrôler la configuration API, sans afficher ni exposer la clé :

```bash
cd /var/www/clauseshift_track2a
: "${LLM_API_KEY:?API key absente}"
export LLM_BASE_URL=https://api.publicai.co/v1
export LLM_NAME=swiss-ai/apertus-v1.5-8b
export LLM_USER_AGENT=CLAUSESHIFT/1.0
python3 -m unittest discover -s tests -v
```

## Premier appel réel : 1 chunk de `admin_de_27` (DEV seulement)

```bash
python3 -m src.e1c \
  --lang de \
  --doc-id admin_de_27 \
  --chunk-tokens 48 --context 20 \
  --max-tokens 768 --timeout 75 --retries 0 \
  --max-calls 1 --max-documents 1 \
  --checkpoint-dir out/checkpoints/e1c_dev_v1
```

L'exemple `admin_de_27` contient 128 whitespace-tokens (75 EN / 53 DE), soit deux chunks avec un plafond de 48 tokens par côté. Un chunk devra être validé et sauvegardé ; aucun document n'est encore complet après le premier appel.

**Ne lancez pas immédiatement 504 documents**. En cas d'erreur, transmettre uniquement les lignes `[E1-C ...]` et l'exception (sans secret). Les réponses JSON malformées entraînent un refus explicite, sans padding ni scores inventés.

## Terminer le même document

```bash
python3 -m src.e1c \
  --lang de --doc-id admin_de_27 \
  --chunk-tokens 48 --context 20 \
  --max-tokens 768 --timeout 75 --retries 0 \
  --max-calls 1 --max-documents 1 \
  --checkpoint-dir out/checkpoints/e1c_dev_v1
```

Cette seconde commande reprend le premier chunk et n'appelle le modèle que pour le chunk restant.

## Comparaison exploratoire avec E0-PJQ, à ID identiques

Cloner SwissGov-RSD hors du dossier du projet, et pointer vers le dépôt officiel :

```bash
export SWISSGOV_ROOT="$(realpath ../SwissGov-RSD)"
python3 tools/compare_e0_e1c_dev.py \
  --lang de \
  --upstream-root "$SWISSGOV_ROOT" \
  --e0-checkpoints out/checkpoints/e0pjq_publicai_4096 \
  --e1c-checkpoints out/checkpoints/e1c_dev_v1 \
  --chunk-tokens 48 --context 20 --max-tokens 768
```

Le résultat est un **estimateur exploratoire sur documents communs et entièrement calculés**, **PAS** la métrique officielle UZH. Un seul document n'est pas suffisant pour conclure. Le calcul reproduit une Spearman avec gestion des ex æquo mais ne remplace pas le script officiel.

## Petite extension DEV

Après la première comparaison, tester 2 ou 3 documents supplémentaires :

```bash
python3 -m src.e1c \
  --lang de --shortest-first \
  --chunk-tokens 48 --context 20 --max-tokens 768 \
  --max-calls 6 --max-documents 3 --timeout 75 --retries 0 \
  --checkpoint-dir out/checkpoints/e1c_dev_v1
```

Limiter les appels. Aucun gain de performance ne doit être annoncé sans mesure. Ne modifier ni la méthode ni les paramètres après gel final.

## Traçabilité et compatibilité

- E0-PJQ ne change pas, ses checkpoints sont préservés.
- E1-C crée sa propre signature SHA-256 basée sur modèle, endpoint, taille de chunk, contexte et maximum de tokens générés.
- Toute modification de ces paramètres exige un **nouveau** répertoire de checkpoint.
- Les modes de réparation syntaxique de E0-PJQ ne sont **pas** appliqués à E1-C.
- Le mode E1-C actuel fonctionne **sur DEV uniquement**. Il n'est pas une soumission finale et n'est pas encore intégré à `make run` ; ce raccordement, ainsi que la reproductibilité Docker, doivent être réalisés avant toute soumission de E1-C.

# CLAUSESHIFT E1-I V1.8 — Plan opérationnel (DEV seulement)

## Pourquoi E1-I ?

La V1.7 E1-C demande des tableaux de scores positionnels. Sur le premier segment `admin_de_27` elle a reçu 33 valeurs au lieu des 37 attendues. Impossible de savoir quelles positions manquent ; **ne pas ajouter des zéros ou décaler les valeurs**.

E1-I demande à Apertus un objet JSON contenant des scores **indexés** par rapport au CORE (`"a":{"0":5,"2":0},"b":{"0":4}`). Les tokens restent entièrement détenus par Python, sans être recopiés dans la réponse. Les ponctuations autonomes reçoivent `-1` par règle déterministe du benchmark ; aucun score de contenu sémantique n'est imputé par Python.

- Les réponses partielles *valides* sont conservées avec leur signature et leur plage source.
- La prochaine requête ne vise que les indices absents.
- Indice inattendu, valeur hors plage, doublon, réponse incompréhensible → **FAIL CLOSED**, avec sauvegarde de la réponse brute de l'API dans `out/diagnostics/` si une réponse a été reçue.
- Nombre maximum de tentatives par segment : `--max-attempts-per-chunk` (défaut 3).
- Chaque demande ou relance est un appel API facturable. Une réponse invalide peut nécessiter une nouvelle exécution ; les réponses sémantiquement validées précédentes restent intactes.
- Le découpage est proportionnel à la position des textes. Cette heuristique peut mal apparier des traductions avec ajouts ou omissions. **Aucun gain Spearman n'est prouvé.**

E1-I est une expérience **nouvelle**, et n'altère pas E0-PJQ, E1-C, le prompt du papier, ni leurs checkpoints.

## 1. Appliquer le patch à V1.7

```bash
cd /var/www/clauseshift_track2a
git apply --check CLAUSESHIFT_E1I_Indexed_Scoring_v1.8.patch
git apply CLAUSESHIFT_E1I_Indexed_Scoring_v1.8.patch
python3 -m unittest discover -s tests -v
```

Ne pas forcer un patch qui ne s'applique pas.

## 2. Contrôler la configuration et la clé, sans l'afficher

```bash
export LLM_NAME='swiss-ai/apertus-v1.5-8b'
export LLM_BASE_URL='https://api.publicai.co/v1'
export LLM_USER_AGENT='CLAUSESHIFT/1.0'
: "${LLM_API_KEY:?missing key}"
```

## 3. Premier appel réel DEV

```bash
python3 -m src.e1i \
  --lang de --doc-id admin_de_27 \
  --chunk-tokens 24 --context 20 \
  --max-tokens 512 --timeout 75 --retries 0 \
  --max-calls 1 --max-attempts-per-chunk 3 \
  --checkpoint-dir out/checkpoints/e1i_dev_v1
```

Contrairement aux deux segments de 37 et 26 tokens de E1-C, cette configuration crée 4 segments pour les 75 tokens anglais / 53 allemands (environ 18–19 et 13–14 tokens par segment). Il faut **quatre segments entièrement traités**, parfois plusieurs requêtes si les réponses omettent des indices.

- `[E1-I SAVED]` : certains scores indexés ont été validés et sauvegardés.
- `remaining_a > 0` : il reste des indices à demander ; relancer à l'identique.
- `[E1-I CHUNK COMPLETE]` : tous les indices sémantiques du segment sont couverts.
- `[E1-I DOCUMENT COMPLETE]` : les deux documents sont entièrement notés.
- `[E1-I FAIL CLOSED]` : erreur bloquante ; transmettre le journal avant de poursuivre.

Après chaque appel, relancer **la commande strictement identique**, qui reprendra les scores déjà sauvegardés ; `--max-calls 1` contrôle les dépenses.

Ne pas passer à un `--max-calls` élevé avant d'avoir observé les premières sorties.

## 4. Comparaison exploratoire de même ID

À la fin du document `admin_de_27`, lancer :

```bash
python3 -m tools.compare_e0_e1i_dev \
  --lang de \
  --upstream-root "$(realpath ../SwissGov-RSD)" \
  --e0-checkpoints out/checkpoints/e0pjq_publicai_4096 \
  --e1i-checkpoints out/checkpoints/e1i_dev_v1 \
  --chunk-tokens 24 --context 20 --max-tokens 512
```

La comparaison est **exploratoire et instable** sur un seul document. Elle ne remplace pas le score UZH, qui nécessite les prédictions complètes DEV et son évaluateur officiel.

## 5. Go/No-Go de la méthode

Les décisions suivantes exigent des résultats DEV réels :

1. Taux de segments terminés et nombre moyen d'appels nécessaires.
2. Durée et coût total par document.
3. Qualité sur des exemples comparables E0-PJQ, E1-C, E1-I.
4. Possibilité de couvrir les 3 paires linguistiques avant la deadline.

L'approche 24 tokens peut multiplier les requêtes : ~4 à 5 appels pour les plus petits documents et nettement plus pour les documents longs. **Ne pas lancer les 504 documents DEV automatiquement sans calcul préalable de débit/budget.** Réserver ensuite du temps pour les sorties en ordre complet, le run `test` gelé, le Docker `make run`, et la soumission. E1-I n'est pas encore raccordée au chemin final `make run` : cette étape reste obligatoire si E1-I est retenue.

## 6. Garanties

- Pas d'accès au split `test`.
- Le checkpoint E1-I est indépendant et signé.
- Les sorties du benchmark restent dans l'ordre des fichiers de référence.
- Aucune prédiction partielle n'est exportée comme fichier complet officiel.
- Aucune correction automatique du contenu ou fabrication de scores manquants.

# CLAUSESHIFT — Expérience E2 V1.9 : alignement bilingue et scoring indexé (DEV uniquement)

**Statut :** implémentation expérimentale, tests logiciels locaux réussis, AUCUNE évaluation Spearman réelle E2.
**Objectif :** distinguer l'effet du découpage bilingue de celui de l'instruction few-shot, sans optimiser sur le jeu test.
**Ne modifie pas** E0-PJQ ni E1-I V1.8. Répertoires `out/checkpoints/e2*` et `out/plans/e2*` séparés.

## 0. Avertissement scientifique et temporel

E0-PJQ obtient 0,2442 sur un seul document DEV `admin_de_27` (112 tokens), contre -0,0569 pour E1-I. Cela ne représente pas une amélioration globale ni un score UZH. La distribution des labels est dégénérée : les 66 tokens anglais sont à 0 et seulement 2 tokens allemands ont un score non nul. Éviter le surajustement à cet exemple. La vraie métrique du concours reste le **Spearman token-level macro-moyenné** sur EN–DE, EN–FR et EN–IT.

Le segmentateur E2 utilise une séparation par ponctuation + découpage maximal, puis une programmation dynamique monotone sur des embeddings multilingues (match 1:1, 1:2, 2:1, 2:2 et gaps explicites). Un bloc non apparié n'est **pas** automatiquement considéré comme divergent : Apertus lui attribue les scores avec le contexte voisin. Les unités d'alignement sont des hypothèses, **pas des annotations de référence**.

## 1. Installer le patch

```bash
cd /var/www/clauseshift_track2a
git apply --check CLAUSESHIFT_E2_Anchor_Alignment_v1.9.patch
git apply CLAUSESHIFT_E2_Anchor_Alignment_v1.9.patch
python3 -m unittest discover -s tests -v
```

Les tests n'appellent ni API ni modèle réel. Pour l'alignement sémantique CPU sur le serveur :

```bash
python3 -m pip install 'sentence-transformers>=3,<6'
```

Une première utilisation téléchargera le modèle public `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` et ses dépendances. Prévoir le temps disque/réseau nécessaire. Une fois installé, enregistrer le commit Hugging Face pour une reproduction rigoureuse :

```bash
python3 - <<'PY'
from huggingface_hub import model_info
print(model_info('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').sha)
PY
```

**Remplacer** dans toutes les commandes `--embedding-revision main` par ce commit exact lorsque disponible. Ne pas modifier la révision après la création d'un plan dans un répertoire donné.

## 2. Vérifier la configuration Public AI

```bash
export LLM_BASE_URL='https://api.publicai.co/v1'
export LLM_NAME='swiss-ai/apertus-v1.5-8b'
export LLM_USER_AGENT='CLAUSESHIFT/1.0'
: "${LLM_API_KEY:?Clé API absente}"
```

Aucune clé dans Git, les logs ni les fichiers de configuration publiés.

## 3. Construire E2-A **sans** appeler Apertus

Le plan préserve l'ordre de tous les tokens et ne lit pas les annotations gold :

```bash
python3 -m src.e2 \
  --lang de --doc-id admin_de_27 \
  --alignment semantic --prompt short \
  --chunk-tokens 48 --max-unit-tokens 20 \
  --similarity-threshold 0.46 --merge-penalty 0.06 \
  --embedding-revision main \
  --plan-only \
  --plan-dir out/plans/e2a_dev_v1 \
  --checkpoint-dir out/checkpoints/e2a_dev_v1
```

Examinez impérativement le résultat :

```bash
python3 -m tools.inspect_e2_alignment \
  --lang de --doc-id admin_de_27 \
  --plan-dir out/plans/e2a_dev_v1 --limit 30
```

Vérifier : les passages ont du sens ensemble, les additions/omissions ne sont pas simplement écrasées, `unaligned_fraction` reste raisonnable. Pas de paramétrage spécifiquement optimisé pour les deux tokens « Zum Seitenanfang » : le plan doit généraliser.

**Contrat fail closed** : si le plan est défectueux, NE PAS envoyer de requêtes. Ajuster les paramètres sur le développement, créer un nouveau répertoire `plan-dir` et enregistrer la méthode dans le rapport. Aucun score n'est fabriqué pour les unités non alignées.

## 4. Un seul appel Public AI pour E2-A

```bash
python3 -m src.e2 \
  --lang de --doc-id admin_de_27 \
  --alignment semantic --prompt short \
  --chunk-tokens 48 --max-unit-tokens 20 \
  --similarity-threshold 0.46 --merge-penalty 0.06 \
  --embedding-revision main \
  --context 20 --max-tokens 512 --timeout 75 \
  --retries 0 --max-calls 1 --max-attempts-per-unit 3 \
  --plan-dir out/plans/e2a_dev_v1 \
  --checkpoint-dir out/checkpoints/e2a_dev_v1
```

`[E2 SAVED]` valide une réponse partielle, `[E2 UNIT COMPLETE]` confirme la couverture exacte, `[E2 FAIL CLOSED]` stoppe une sortie invalide. Chaque nouveau lancement identique saute les indices sauvegardés. `--max-calls 1` limite le nombre de nouvelles requêtes, mais une même requête peut être facturée. Ne pas effacer les checkpoints.

## 5. Comparer E0 / E2-A, après document complet

```bash
python3 -m tools.compare_e2_dev \
  --lang de --doc-id admin_de_27 \
  --upstream-root "$(realpath ../SwissGov-RSD)" \
  --e0-checkpoints out/checkpoints/e0pjq_publicai_4096 \
  --alignment semantic --prompt short \
  --chunk-tokens 48 --max-unit-tokens 20 \
  --embedding-revision main \
  --e2-checkpoints out/checkpoints/e2a_dev_v1 \
  --plan-dir out/plans/e2a_dev_v1
```

La comparaison est un **proxy exploratoire uniquement** (sur les documents DEV communs et complets). Aucun usage du test et aucune prétention de performance de concours.

## 6. Ablations contrôlées

Tous les autres paramètres restent identiques, avec checkpoints indépendants :

| Expérience | Alignement | Prompt | Répertoires |
|---|---|---|---|
| E1-I (V1.8) | Proportionnel | Court indexé | `e1i_dev_v1` existant |
| E2-A | Sémantique | Court indexé | `e2a_dev_v1` |
| E2-B | Proportionnel | Few-shot **adapté en format indexé** | `e2b_dev_v1` |
| E2-C | Sémantique | Few-shot adapté | `e2c_dev_v1` |

Pour E2-B : reprendre la commande E2-A en remplaçant `--alignment semantic --prompt short` par `--alignment proportional --prompt fewshot`, et les deux dossiers `e2a` par `e2b`. Son prompt few-shot reprend les *scores* des trois exemples de l'annexe E mais **n'est pas** une reproduction du prompt exact de la baseline papier.

Pour E2-C : conserver `--alignment semantic`, passer à `--prompt fewshot`, changer les deux dossiers en `e2c`. Les résultats des différentes variantes doivent être comparés sur les mêmes documents DEV.

## 7. Décision dans un délai court

**Ne pas lancer tout `dev` en automatique.** Pour un document moyen de plusieurs centaines de tokens, E2 peut demander de nombreux appels. Estimer le nombre d'unités et le débit avant le moindre batch.

1. Exécuter un plan sur plusieurs exemples DEV présélectionnés, trois langues, longueurs variées.
2. Vérifier la couverture et la plausibilité de l'alignement sans consulter les gold labels.
3. Comparer au moins quelques documents DEV **communs** E0 et E2-A avant de retenir une variante.
4. Privilégier une méthode qui gagne réellement du Spearman ET tient dans le budget/temps.
5. Si les temps projetés rendent impossible l'exécution sur le test et le format final d'ici le 16 octobre, **NO-GO sur E2** : réduire le périmètre immédiatement.

**Important :** la V1.9 est un **runner de recherche DEV uniquement**. Elle ne constitue pas encore le `make run` de soumission. L'extension test doit se faire uniquement après choix de méthode, gel du code/hyperparamètres et validation de la reproductibilité, sans réutilisation du test pour optimiser la stratégie.

## 8. Dépannage

- `Semantic alignment requires...` : installer `sentence-transformers` dans le bon `.venv`.
- `plan changed` : ne pas écraser les plans, choisir un autre `--plan-dir`.
- `checkpoint signature mismatch` : changer `--checkpoint-dir` après toute modification des paramètres.
- `Missing indices` : relancer à l'identique, dans la limite des tentatives.
- `504` : ne pas relancer massivement ; revoir la taille des fenêtres ou le serving.
- `No complete common documents` : le document E2 n'est pas terminé ou la baseline n'existe pas pour lui.
- `Spearman undefined` : les labels de référence ou les prédictions sont constants ; ne pas remplacer par zéro.

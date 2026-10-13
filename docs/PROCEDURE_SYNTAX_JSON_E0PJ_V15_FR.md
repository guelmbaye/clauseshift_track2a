# CLAUSESHIFT V1.5 — Diagnostic et récupération E0-PJ (DEV uniquement)

## Constat

`admin_de_27` a rendu HTTP 200 en 41 s mais a produit une réponse JSON invalide :

```diff
- ["sector", 5]), [")", -1]
+ ["sector", 5], [")", -1]
```

Le caractère superflu est `)`, immédiatement après une entrée `[token,score]`. La nouvelle méthode `E0-PJ` accepte **exclusivement** la suppression d'une unique parenthèse surnuméraire située hors des chaînes JSON, et seulement si la réponse complète devient du JSON valide, suivi de la validation stricte de tous les tokens et scores. Les autres erreurs restent bloquantes. E0-P reste inchangé.

## A. Installation

Sur le serveur, dans `/var/www/clauseshift_track2a`, transférer le patch et faire :

```bash
git apply --check CLAUSESHIFT_PublicAI_Syntax_Repair_v1.5.patch
git apply CLAUSESHIFT_PublicAI_Syntax_Repair_v1.5.patch
python3 -m unittest discover -s tests -v
```

Le patch cible **la V1.4** (après l'installation du correctif JSON Diagnostics). Ne pas forcer si `git apply --check` échoue.

## B. Récupérer l'appel raté sans coût API

Vérifier que la configuration est strictement celle de l'appel précédent, y compris l'URL avec suffixe `/v1` et l'identifiant exact du modèle :

```bash
export LLM_BASE_URL='https://api.publicai.co/v1'
export LLM_NAME='swiss-ai/apertus-v1.5-8b'
: "${LLM_API_KEY:?Clé API absente}"
RAW=$(ls -t out/diagnostics/dev_de_admin_de_27_*_raw.txt | head -1)
```

Dry-run, **aucune écriture de checkpoint, aucun appel réseau** :

```bash
python3 tools/recover_dev_e0pj.py \
  --source-checkpoints out/checkpoints/e0p_publicai_4096 \
  --destination-checkpoints out/checkpoints/e0pj_publicai_4096 \
  --raw-file "$RAW" \
  --lang de --doc-id admin_de_27 \
  --prompt-file prompts/paper_exact.txt \
  --max-tokens 4096
```

N'aller plus loin **que si** `[OFFLINE VERIFIED]` apparaît et que le script se termine sans erreur. Il vérifie les mots dans le bon ordre, les scores, l'identité du document, les signatures et le hash du fichier brut.

Pour créer les nouveaux checkpoints E0-PJ (source E0-P intacte), exécuter la même commande avec `--write` :

```bash
python3 tools/recover_dev_e0pj.py \
  --source-checkpoints out/checkpoints/e0p_publicai_4096 \
  --destination-checkpoints out/checkpoints/e0pj_publicai_4096 \
  --raw-file "$RAW" \
  --lang de --doc-id admin_de_27 \
  --prompt-file prompts/paper_exact.txt \
  --max-tokens 4096 --write
```

Si les checkpoints source sont ceux des logs précédents, on s'attend à `de=2`, `fr=1`, `it=1`. Les chiffres réels dépendent du nombre de sorties déjà enregistrées.

## C. Continuer sur DEV sans rejouer les prédictions déjà validées

```bash
python3 -m src.cli run \
  --split dev --lang de --dev-shortest-first \
  --prompt-file prompts/paper_exact.txt \
  --punctuation-repair --syntax-repair \
  --max-examples 2 --max-tokens 4096 \
  --timeout 75 --retries 0 \
  --checkpoint-dir out/checkpoints/e0pj_publicai_4096
```

Pour les autres langues, remplacer `--lang de` par `fr` et `it`. Ne pas passer au lot complet avant d'avoir confirmé que les documents de taille moyenne et grande ne déclenchent pas de 504 et que l'ordre des scores reste conforme.

## D. Interprétation

- `[SYNTAX REPAIR]` = suppression d'une unique parenthèse étrangère à la chaîne JSON, avec son offset.
- `[PUNCTUATION REPAIR]` = ponctuation d'origine restaurée **uniquement** si tous les mots non ponctuation du modèle sont dans le même ordre.
- `[FAIL CLOSED]` = réponse toujours invalide ; aucun checkpoint de la requête concernée.
- Une erreur 504 n'est **pas** corrigée par ces fonctions ; il faudra évaluer un E1-C segmenté en expérience séparée.

Aucun changement à `paper_exact.txt`. Ne pas regarder les exemples test pendant le développement. Documenter E0-PJ séparément de la baseline stricte E0 et de E0-P.

# CLAUSESHIFT — Diagnostic `admin_de_27` et récupération E0-PJQ V1.6

## Constat confirmé par l'utilisateur

Un appel Public AI pour `admin_de_27` sur DEV a produit du JSON invalide avec `['sector', 5])`; la réparation E0-PJ a corrigé cette erreur mais la validation a relevé une différence de token : attendu `Energiebereich»`, généré `Energiebereich`.

Le nouveau mode **E0-PJQ** autorise la restauration d'**un seul guillemet typographique double** (par ex. `»` ou `«`) au bord d'un token. Le contenu lexical et le score du modèle doivent être identiques; toutes les autres différences de tokens restent bloquantes. Ce mode est expérimental et doit être rapporté séparément. La baseline E0 originale ne change pas.

## 1. Prérequis

Appliquer le patch V1.6 sur le code **V1.5**. Depuis `/var/www/clauseshift_track2a` :

```bash
git apply --check CLAUSESHIFT_Boundary_Quote_Repair_v1.6.patch
git apply CLAUSESHIFT_Boundary_Quote_Repair_v1.6.patch
python3 -m unittest discover -s tests -v
```

`git apply --check` doit réussir avant d'appliquer le patch. En cas d'échec, ne forcez pas le patch. Vérifiez la bonne version.

## 2. Paramètres inchangés pour restaurer les signatures

```bash
export LLM_BASE_URL='https://api.publicai.co/v1'
export LLM_NAME='swiss-ai/apertus-v1.5-8b'
RAW="$(ls -t out/diagnostics/dev_de_admin_de_27_*_raw.txt | head -1)"
test -n "$RAW" && test -f "$RAW"
```

Le dry-run ne nécessite pas `LLM_API_KEY`, puisqu'il n'émet aucune requête.

## 3. Dry-run hors ligne (aucun coût API)

```bash
python3 tools/recover_dev_e0pj.py \
  --source-checkpoints out/checkpoints/e0p_publicai_4096 \
  --destination-checkpoints out/checkpoints/e0pjq_publicai_4096 \
  --raw-file "$RAW" \
  --lang de --doc-id admin_de_27 \
  --prompt-file prompts/paper_exact.txt \
  --max-tokens 4096 \
  --boundary-quote-repair
```

Succès attendu seulement si tous les tokens lexicaux correspondent, les scores sont valides et l'objet JSON complet est valide après la correction syntaxique. En cas d'échec, ne pas lancer `--write`.

## 4. Sauvegarde E0-PJQ (uniquement si dry-run réussi)

Ajouter `--write` à la même commande :

```bash
python3 tools/recover_dev_e0pj.py \
  --source-checkpoints out/checkpoints/e0p_publicai_4096 \
  --destination-checkpoints out/checkpoints/e0pjq_publicai_4096 \
  --raw-file "$RAW" \
  --lang de --doc-id admin_de_27 \
  --prompt-file prompts/paper_exact.txt \
  --max-tokens 4096 \
  --boundary-quote-repair \
  --write
```

Le répertoire de destination doit être absent ou vide. E0-P demeure intact; la signature E0-PJQ diffère de E0-P et E0-PJ.

## 5. Deux nouvelles prédictions courtes sur DEV, après succès

```bash
python3 -m src.cli run \
  --split dev --lang de --dev-shortest-first \
  --prompt-file prompts/paper_exact.txt \
  --punctuation-repair --syntax-repair --boundary-quote-repair \
  --max-examples 2 --max-tokens 4096 \
  --timeout 75 --retries 0 \
  --checkpoint-dir out/checkpoints/e0pjq_publicai_4096
```

`--max-examples 2` signifie deux **nouvelles** prédictions. Toute erreur différente (réponse tronquée, mot modifié, timeout, plusieurs erreurs JSON) déclenche l'arrêt. Ne pas utiliser le split `test` pour ajuster cette méthode.

## 6. Limites scientifiques

- E0-PJQ est une **variante réparée** de la baseline, pas la reproduction inchangée E0.
- Les guillemets à la frontière d'un mot peuvent porter un sens; seule l'identité des mots et des scores est conservée. Le correctif ne prouve pas que le score sémantique est exact.
- L'échec initial de cette génération doit figurer dans les statistiques de validité, même si la récupération est réussie.
- Les longs documents susceptibles d'expirer en 504 exigent encore une stratégie de segmentation évaluée séparément (E1-C).

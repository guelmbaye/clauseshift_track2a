# CLAUSESHIFT — Correction contrôlée de dérive de ponctuation (v1.2)

## Observation DEV : admin_fr_105

`sentence1` valide (60/60). `sentence2` attend 52 entrées, Apertus en a généré 51.
La sortie montre une virgule déplacée autour de `et` et une virgule manquante après `mesure`.
Cela ne démontre PAS que toutes les sorties malformées sont réparables : seule une sous-classe très restreinte l'est.

## Garde-fous de la méthode

* Strict E0 conserve son comportement original : **aucune correction par défaut**.
* Option E0-P : `--punctuation-repair`, à documenter comme **post-traitement déterministe**.
* On compare tous les tokens NON ponctuation des deux textes, dans leur ordre et avec leur identité exacts.
* Si l'un manque, change, fusionne ou se déplace, on REFUSE la sortie entière.
* On conserve sans modification chaque score sémantique du modèle.
* Les tokens de ponctuation autonomes du TEXTE SOURCE sont réinsérés avec le score `-1`.
* Les signes de ponctuation générés en positions incorrectes sont ignorés.
* Les chaînes comportant des lettres/chiffres et de la ponctuation intégrée (ex. `3,5` ou `l'administration`) ne sont PAS assimilées à de la ponctuation autonome.
* Un score `-1` émis pour un mot sémantique fait échouer la réparation.
* Les prédictions et checkpoints E0/E0-P sont strictement séparés par des signatures différentes.

## Étape 1 — Installer le correctif

Après transfert du patch dans la racine du projet :

```bash
git apply --check CLAUSESHIFT_Punctuation_Only_Hotfix_v1.2.patch
git apply CLAUSESHIFT_Punctuation_Only_Hotfix_v1.2.patch
python3 -m unittest discover -s tests -v
```

## Étape 2 — Vérifier sans nouvelle requête payante

```bash
python3 -m tools.check_punctuation_repair_dev \
  --lang fr --id admin_fr_105 \
  --raw-file out/diagnostics/fr_admin_fr_105_20261012T222740Z_raw.txt
```

Attendu : `[REPAIRABLE]` puis `labels_a=60; labels_b=52`.
Si `[REFUSED]`, NE PAS contourner le garde-fou. Partager le diagnostic.

## Étape 3 — Relancer un mini-test E0-P sur dev

```bash
python3 -m src.cli run \
  --split dev \
  --prompt-file prompts/paper_exact.txt \
  --smoke-shortest \
  --punctuation-repair \
  --max-tokens 4096 \
  --timeout 75 \
  --checkpoint-dir out/checkpoints/e0p_publicai_4096
```

Ce smoke appelle de nouveau l'API Apertus (maximum 3 exemples nouveaux), et conserve une trace `_repair_log` et le hash de la réponse brute dans chaque checkpoint validé.
E0 brut et E0-P sont deux variantes distinctes à déclarer dans le rapport; éviter de prétendre que la seconde est une baseline exacte inchangée.

## Étape 4 — Evaluer uniquement sur DEV

Ne poursuivre vers les 168 documents par langue qu'après succès du smoke et contrôle du taux de réparations/refus.
Le test est strictement réservé à la dernière exécution après gel du pipeline, prompts et paramètres.

## Limites

Les ponctuations Unicode seules sont reconnues par catégorie Unicode P ; tout symbole ambigu ou séquence avec lettre/chiffre reste un token sémantique et ne peut pas être réparé de cette manière.
L'algorithme ne récupère pas des scores oubliés pour des mots. Il ne corrige pas le sens des scores, seulement leur placement vis-à-vis de la ponctuation.

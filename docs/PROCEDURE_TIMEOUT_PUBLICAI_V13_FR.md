# CLAUSESHIFT — Diagnostic des erreurs HTTP 504 sur Public AI (V1.3)

**Objectif :** identifier les requêtes qui dépassent les limites de temps du fournisseur, poursuivre les essais `dev` sur les documents courts, et préserver les checkpoints existants **sans accéder au test**.

## Ce que signifie l'erreur

Le texte « CSCS 504/60s » était un message codé en dur pour toutes les erreurs HTTP 504. Il ne prouve pas que la requête a été envoyée à CSCS. Lorsque `LLM_BASE_URL=https://api.publicai.co/v1`, la requête est destinée à Public AI. Une réponse HTTP 504 signifie qu'un intermédiaire n'a pas reçu une réponse à temps ; le responsable exact et la durée limite ne sont pas établis sans les journaux HTTP du fournisseur.

`--timeout 75` modifie la durée d'attente du client HTTP ; cela ne supprime pas une limite imposée par le serveur. `--max-tokens 4096` fixe le plafond de tokens en **sortie**, ce n'est ni la taille du document en entrée ni une garantie que la réponse rentre dans 4096 tokens.

## 1. Installer le correctif

Depuis le répertoire racine du projet, sauvegarder les fichiers `src/cli.py` et `src/inference.py`, puis :

```bash
git apply --check CLAUSESHIFT_PublicAI_Timeout_Instrumentation_v1.3.patch
git apply CLAUSESHIFT_PublicAI_Timeout_Instrumentation_v1.3.patch
python3 -m unittest discover -s tests -v
```

Si `git apply --check` échoue, ne pas forcer : comparer avec les fichiers locaux, surtout si d'autres modifications ont été faites.

## 2. Identifier le prochain document SANS appel API

```bash
python3 tools/inspect_dev_backlog.py \
  --checkpoint-dir out/checkpoints/e0p_publicai_4096 \
  --top 5
```

La première section de chaque langue identifie les documents qui seraient traités en mode normal ; la seconde affiche les documents restants les plus courts.

## 3. Continuer prudemment sur une langue et deux documents courts

```bash
export LLM_NAME='swiss-ai/apertus-v1.5-8b'
export LLM_BASE_URL='https://api.publicai.co/v1'
: "${LLM_API_KEY:?Clé manquante}"
python3 -m src.cli run \
  --split dev \
  --lang de \
  --dev-shortest-first \
  --prompt-file prompts/paper_exact.txt \
  --punctuation-repair \
  --max-examples 2 \
  --max-tokens 4096 \
  --timeout 75 \
  --retries 0 \
  --checkpoint-dir out/checkpoints/e0p_publicai_4096
```

Le programme journalise le document **avant** l'appel `[INFER START]`, sa taille approximative en tokens séparés par espaces et le fournisseur. Si HTTP 504, il journalise le temps écoulé. Les checkpoints déjà produits sont conservés. L'option `--dev-shortest-first` change seulement l'ordre des **requêtes de développement**, jamais l'ordre des prédictions exportées pour l'évaluateur.

## 4. Interpréter

- 2 documents courts réussis : le client, le modèle et les petits documents fonctionnent ; la limite sur documents longs reste ouverte.
- Même document court échoue : examiner disponibilité du fournisseur, limites de compte, HTTP status, latence ; ne pas conclure d'emblée à un document trop long.
- HTTP 504 autour d'une limite répétable sur documents moyens/longs : le chemin vers l'exécution complète exige une autre stratégie d'inférence ou un autre endpoint capable de servir le même Apertus, **ou une variante de méthode par segmentation** évaluée séparément.
- Si réponse 200 mais JSON incomplet : autre problème à traiter, ne pas confondre avec HTTP 504.

## 5. Règle scientifique

Le prompt few-shot strict E0 ne doit pas être silencieusement transformé en méthode chunkée. Pour continuer avec des documents longs, créer **E1-C**, méthode de segmentation, ou changer de serving tout en gardant E0 constant. Les résultats devront être rapportés par méthode et protocole, sans écarter silencieusement les documents qui échouent.

## 6. Sécurité

Ne pas imprimer, publier ni partager `LLM_API_KEY`. Ne jamais utiliser le split `test` pour ce diagnostic. Ne pas modifier ou supprimer les checkpoints d'origine. Le correctif n'a pas été testé contre l'API réelle dans cet environnement ; les nouveaux tests utilisent un serveur simulé.

# CLAUSESHIFT V1.14.2 — Correction du prompt few-shot batché

## Preuve DEV motivant le changement

`admin_de_132`: deux appels V1.14 ont échoué avec `Invalid JSON: Extra data` pour `units=[1,2,3]`, 96 scores demandés. Le diagnostic V1.14.1 du second appel montre un préfixe JSON valide de 634 caractères mais **invalide selon le contrat du batch**, avec des clés racines `1` et `units` et seulement `0` dans l'enveloppe `units`, puis 86 caractères de texte supplémentaire.

Ce diagnostic ne prouve pas pourquoi Apertus a divergé. Une hypothèse concrète et vérifiable est la contradiction du prompt V1.14 : les trois démonstrations utilisent `{"a": ..., "b": ...}` alors que le vrai batch exige `{"units": {"0": {"a": ..., "b": ...}}}`.

## Correctif limité

- Les trois exemples gardent **exactement les mêmes scores et tokens**, mais leurs sorties sont encapsulées dans la même enveloppe JSON `units` que la tâche réelle.
- Un quatrième exemple minimal indique comment plusieurs unités doivent figurer dans **un seul objet JSON**.
- Le parseur reste strict : aucun préfixe partiel, objet supplémentaire, indice manquant ou index hors plage n'est accepté.
- La version de la signature est renouvelée, donc **nouveau répertoire de checkpoint obligatoire**, sans effacement des anciennes traces.
- Le plan d'alignement n'est pas modifié. Les deux faux non-alignements visibles sur `admin_de_132` restent une limite scientifique.
- Aucun score Spearman ni appel réel Apertus n'a encore démontré l'efficacité de V1.14.2.

## Installer sur V1.14.1

```bash
cd /var/www/clauseshift_track2a
git apply --check CLAUSESHIFT_Batch_Schema_Fewshot_v1.14.2.patch
git apply CLAUSESHIFT_Batch_Schema_Fewshot_v1.14.2.patch
python3 -m unittest discover -s tests -q
```

## Un seul nouvel appel DEV, après vérification du patch et des tests

Conserver le plan existant, changer uniquement le checkpoint (signature différente). Exécuter un seul appel à `--max-calls 1` avec `--max-scores-per-call 96 --max-units-per-batch 3 --max-tokens 1536`. En cas d'erreur, diagnostiquer avec `tools.inspect_batch_raw` et ne pas répéter identiquement.

### Limites

La qualité Spearman reste inconnue ; ne pas accéder aux données finales avant gel, et ne pas accepter automatiquement les préfixes JSON partiels. La V1.14.2 est une hypothèse de correction de format et non une validation empirique sur Apertus.

# Banking77 fixture licence

Kind: reference.

The rows in `test_subset.csv` come from the
[PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) dataset,
test split. The Hub revision on 2026-10-02 was
`90d4e2ee5521c04fc1488f065b8b083658768c57`. The commit the rows were taken from
is not recorded. The fixture is a subset of that split with the text and
intent columns. The rows were not compared with upstream byte for byte.

The dataset is licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/),
as the [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets)
LICENSE file states.

Attribution: Casanueva, I., Temcinas, T., Gerz, D., Henderson, M. and Vulic, I.
"Efficient Intent Detection with Dual Sentence Encoders." Workshop on NLP for
Conversational AI, ACL 2020. PolyAI.

The fraud and not-fraud proxy labels that typevet derives from the intents are
typevet's own. They are not part of Banking77.

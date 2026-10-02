# PubMedQA fixture licence

Kind: reference.

The rows in `pqa_labeled_subset.jsonl` come from the
[qiaojin/PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) dataset,
`pqa_labeled` config. The Hub revision on 2026-10-02 was
`9001f2853fb87cab8d220904e0de81ac6973b318`. The commit the rows were taken from
is not recorded. The fixture is a subset of that config. Each row keeps the
pubid, the question, the context texts with their labels and the final
decision. Each context text longer than 100 characters is cut to its first 100
characters plus `...`.

The dataset is licensed under the [MIT License](https://opensource.org/license/mit),
as the [pubmedqa/pubmedqa](https://github.com/pubmedqa/pubmedqa) LICENSE file
states. Copyright (c) 2019 pubmedqa.

Attribution: Jin, Q., Dhingra, B., Liu, Z., Cohen, W. and Lu, X. "PubMedQA: A
Dataset for Biomedical Research Question Answering." EMNLP 2019.

The contexts are PubMed abstracts. The MIT licence covers the dataset. It does
not change the terms of each abstract.

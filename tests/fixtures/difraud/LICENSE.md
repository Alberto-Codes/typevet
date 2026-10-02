# DIFrauD fixture licence

Kind: reference.

The rows in `sms_test_subset.jsonl` come from the
[difraud/difraud](https://huggingface.co/datasets/difraud/difraud) dataset,
`sms` config, test split, revision `aaaf94b336c563a14806bb4f3f58727bed9ed8d4`.
The fixture is a subset of that split. The rows were not compared with
upstream byte for byte.

The dataset is licensed under the [MIT License](https://opensource.org/license/mit).
The [dataset card](https://huggingface.co/datasets/difraud/difraud) states:

> This dataset is published under the MIT license and can be used and
> modified by anyone free of charge.
> Source: https://huggingface.co/datasets/difraud/difraud

Attribution: Boumber, D., Qachfar, F. Z. and Verma, R. M. "Domain-Agnostic
Adapter Architecture for Deception Detection: Extensive Evaluations with the
DIFrauD Benchmark." LREC-COLING 2024.

The `sms` config was built from two sources, each licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):

- Almeida, T. and Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI
  Machine Learning Repository. https://doi.org/10.24432/C5CC84.
- Mishra, S. and Soni, D. SMS Phishing Dataset for Machine Learning and
  Pattern Recognition. Mendeley Data. https://doi.org/10.17632/f45bkkt8pr.1.

The fixture does not record which source each row came from, so both are
credited.

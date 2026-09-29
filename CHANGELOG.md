# Changelog

## [0.1.0](https://github.com/Alberto-Codes/typevet/compare/v0.1.0...v0.1.0) (2026-09-29)


### Refactoring

* **adapters:** gather the llama.cpp outbound modules into one package ([8df631b](https://github.com/Alberto-Codes/typevet/commit/8df631bcddfcc05ac887e1e69e88135ced69f929)), closes [#256](https://github.com/Alberto-Codes/typevet/issues/256)
* **adapters:** gather the vLLM outbound modules into one package ([4d2d999](https://github.com/Alberto-Codes/typevet/commit/4d2d99995afa7e05c4ee03256c0d7e1b89cfe59a)), closes [#256](https://github.com/Alberto-Codes/typevet/issues/256)
* **evals:** move the CORD evaluation family into typevet_evals.cord ([4fa5cce](https://github.com/Alberto-Codes/typevet/commit/4fa5cce5bd0f62e7a25a04292b6dac139ba8bc59)), closes [#256](https://github.com/Alberto-Codes/typevet/issues/256)
* **evals:** move the eval CLI, wheel isolation and the Gemma wheel smoke into evals/ ([ced27a3](https://github.com/Alberto-Codes/typevet/commit/ced27a3074bf49fbca221630e6bbb0707c0f1932)), closes [#256](https://github.com/Alberto-Codes/typevet/issues/256)
* **evals:** move the eval datasets into evals/ and remove typevet.evaluation ([20d0d1a](https://github.com/Alberto-Codes/typevet/commit/20d0d1a687dd8b52428f5a6b8e6bd63e06f7c54a)), closes [#256](https://github.com/Alberto-Codes/typevet/issues/256)
* **evals:** move the eval runner, TPJEP and experiment identity into evals/ ([63447e9](https://github.com/Alberto-Codes/typevet/commit/63447e90a532a1606897ad2011f2625800bea089)), closes [#256](https://github.com/Alberto-Codes/typevet/issues/256)
* **evals:** move the instruction-variant consumer harness and replay metrics into evals/ ([e542750](https://github.com/Alberto-Codes/typevet/commit/e5427503aee6267e5f8a0f96c67cc7e42497712c)), closes [#256](https://github.com/Alberto-Codes/typevet/issues/256)
* **evals:** move the PSAI vision consumer harness and the CORD CLI into evals/ ([df25a34](https://github.com/Alberto-Codes/typevet/commit/df25a345fa01bfff3c8caa9ae348f409c25f5171)), closes [#256](https://github.com/Alberto-Codes/typevet/issues/256)
* **evals:** move the throughput and vLLM acceptance harnesses into evals/ ([dc5abc1](https://github.com/Alberto-Codes/typevet/commit/dc5abc13ad424dd6843638c364383722450d48e0)), closes [#256](https://github.com/Alberto-Codes/typevet/issues/256)
* **eval:** split transport and metrics reading out of the vLLM acceptance module ([a2e3ba7](https://github.com/Alberto-Codes/typevet/commit/a2e3ba792e9cc947d9ea8f4773acfecbe59a71e7)), closes [#229](https://github.com/Alberto-Codes/typevet/issues/229)
* **package:** remove the 27 root eval_* compatibility shims ([bdca25d](https://github.com/Alberto-Codes/typevet/commit/bdca25d199b9d12cf5ad33ba752dff2f0bf8f32b)), closes [#256](https://github.com/Alberto-Codes/typevet/issues/256)
* **package:** remove the library root shims and move question_schema into the domain ([2883ad6](https://github.com/Alberto-Codes/typevet/commit/2883ad62390f7ae470bf0e467717108a8cb041b6)), closes [#256](https://github.com/Alberto-Codes/typevet/issues/256)


### Documentation

* **adr:** record the package layout and test the supported-import surface ([946ee06](https://github.com/Alberto-Codes/typevet/commit/946ee0650549878aae75cc7a02a7b65d1e20b80f)), closes [#256](https://github.com/Alberto-Codes/typevet/issues/256)
* **how-to:** install typevet from a wheel outside a checkout ([dc9e9f3](https://github.com/Alberto-Codes/typevet/commit/dc9e9f33f1df27cb9d61f21e8e34cd513796c126)), closes [#245](https://github.com/Alberto-Codes/typevet/issues/245)
* **reference:** add latency at a glance to the H100 performance page ([f001283](https://github.com/Alberto-Codes/typevet/commit/f001283e06139cd6c16ceebef37203048ad9051c)), closes [#236](https://github.com/Alberto-Codes/typevet/issues/236)
* **reference:** publish the H100 throughput results and a rented-H100 how-to ([17ce15b](https://github.com/Alberto-Codes/typevet/commit/17ce15b3f2128d1a33c95c35da792b667d880f09)), closes [#236](https://github.com/Alberto-Codes/typevet/issues/236)
* **reference:** put the tested vLLM pin and current behaviour into the support matrix ([9fd6557](https://github.com/Alberto-Codes/typevet/commit/9fd6557b6dcc3b4d1fb96747c94338e3d4ec4a43)), closes [#243](https://github.com/Alberto-Codes/typevet/issues/243)
* **site:** add a landing page and rewrite the README for new readers ([e7754f3](https://github.com/Alberto-Codes/typevet/commit/e7754f331f8509f9eb5d700c23efcafb5b69e01b)), closes [#244](https://github.com/Alberto-Codes/typevet/issues/244)
* **site:** render Mermaid diagrams and draw the typed-judgment flow ([a0cc4b9](https://github.com/Alberto-Codes/typevet/commit/a0cc4b97ce0e9cd23c8fbf52f63dc706f28dacd9)), closes [#254](https://github.com/Alberto-Codes/typevet/issues/254)

## Changelog

Maintained by [release-please](https://github.com/googleapis/release-please)
from conventional commit messages. Do not edit entries by hand.

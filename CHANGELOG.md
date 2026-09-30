# Changelog

## [0.2.0](https://github.com/Alberto-Codes/typevet/compare/v0.1.0...v0.2.0) (2026-09-30)


### Features

* **domain:** native Choice up to 24 options with digit-then-letter controls ([34f23ad](https://github.com/Alberto-Codes/typevet/commit/34f23ad8cb5684f01cccc65e9a2059ea38057684)), closes [#286](https://github.com/Alberto-Codes/typevet/issues/286) [#287](https://github.com/Alberto-Codes/typevet/issues/287) [#237](https://github.com/Alberto-Codes/typevet/issues/237)
* **integrations:** a judgevet provider bridge over JudgmentPort ([2b04383](https://github.com/Alberto-Codes/typevet/commit/2b04383d94c49f8723a09762991d2e6deaaaab37)), closes [#284](https://github.com/Alberto-Codes/typevet/issues/284)
* **integrations:** judgevet 0.15 conformance kit and per-question images ([67728f5](https://github.com/Alberto-Codes/typevet/commit/67728f586e6d3f930f5598d390585b4979f38e64)), closes [#290](https://github.com/Alberto-Codes/typevet/issues/290)


### Fixes

* **adapters:** drop the cause chain from masked vLLM errors when a key is set ([71ee1fb](https://github.com/Alberto-Codes/typevet/commit/71ee1fbd8e458ed61cda24f52f6655e9028b41f9)), closes [#276](https://github.com/Alberto-Codes/typevet/issues/276)
* **adapters:** reject non-finite vLLM timeouts and mask keys in set and bytes values ([e8ecc20](https://github.com/Alberto-Codes/typevet/commit/e8ecc2093775975a91ede996e5793c43812ad407)), closes [#227](https://github.com/Alberto-Codes/typevet/issues/227)
* **domain:** validate CandidateScoringRequest.media items ([ce3c9e6](https://github.com/Alberto-Codes/typevet/commit/ce3c9e6d4c56a072da2244e4d27497bf4396c58a)), closes [#222](https://github.com/Alberto-Codes/typevet/issues/222)
* **evals:** harden the PSAI leak check, KV-cache reading and runner failure counting ([dec6dc7](https://github.com/Alberto-Codes/typevet/commit/dec6dc76f729775449948693482c4fcd1e1ede87)), closes [#223](https://github.com/Alberto-Codes/typevet/issues/223) [#225](https://github.com/Alberto-Codes/typevet/issues/225) [#238](https://github.com/Alberto-Codes/typevet/issues/238) [#261](https://github.com/Alberto-Codes/typevet/issues/261)
* **site:** focus ring, Noul search rank, 2x header reflow and Home card headings ([5bbfecb](https://github.com/Alberto-Codes/typevet/commit/5bbfecb25d78340ba5f56bd8470f3bd0b1af864d)), closes [#282](https://github.com/Alberto-Codes/typevet/issues/282) [#267](https://github.com/Alberto-Codes/typevet/issues/267)
* **site:** keep the closed search panel out of the Tab order ([3a7a67c](https://github.com/Alberto-Codes/typevet/commit/3a7a67cdceefa048f4a3f03384279b0ae4e8d90f)), closes [#285](https://github.com/Alberto-Codes/typevet/issues/285) [#267](https://github.com/Alberto-Codes/typevet/issues/267) [#242](https://github.com/Alberto-Codes/typevet/issues/242)
* **site:** keyboard access to the mobile navigation drawer ([918d8d9](https://github.com/Alberto-Codes/typevet/commit/918d8d9e8d4277eaaaca5de75967aadcf205ac8f)), closes [#283](https://github.com/Alberto-Codes/typevet/issues/283) [#267](https://github.com/Alberto-Codes/typevet/issues/267)
* **tests:** keep API keys out of pytest failure output ([599de01](https://github.com/Alberto-Codes/typevet/commit/599de0134c0d9e700604c35ba2e564b73158efd8)), closes [#251](https://github.com/Alberto-Codes/typevet/issues/251)


### Documentation

* correct the local alias identity, framing prefill and Gemma 4 image cost ([867a61a](https://github.com/Alberto-Codes/typevet/commit/867a61a61eb59e824173c074be6f5fc394252ed3)), closes [#233](https://github.com/Alberto-Codes/typevet/issues/233) [#235](https://github.com/Alberto-Codes/typevet/issues/235) [#260](https://github.com/Alberto-Codes/typevet/issues/260)
* **explanation:** correct the stale [#129](https://github.com/Alberto-Codes/typevet/issues/129) template claim in native typed judgments ([698d53d](https://github.com/Alberto-Codes/typevet/commit/698d53d1c71475295658363965d4f99ef60e1b96)), closes [#278](https://github.com/Alberto-Codes/typevet/issues/278)
* **explanation:** gather limits and known gaps on one page with page status ([cb51e80](https://github.com/Alberto-Codes/typevet/commit/cb51e80714e829483f47213fbdbfa57fd8cb35e5)), closes [#247](https://github.com/Alberto-Codes/typevet/issues/247)
* **index:** list every page in docs/README.md and drop the stale PyPI claim ([638a2e1](https://github.com/Alberto-Codes/typevet/commit/638a2e1d9906603c5a9038707840965cfdb85698)), closes [#275](https://github.com/Alberto-Codes/typevet/issues/275)
* **reference:** map vLLM adapter failures to typevet errors ([367051e](https://github.com/Alberto-Codes/typevet/commit/367051ea76f13ce62f17abf201c962013ba3dbb0)), closes [#273](https://github.com/Alberto-Codes/typevet/issues/273)
* **reference:** one page for data handling, key masking and security ([fa07066](https://github.com/Alberto-Codes/typevet/commit/fa070662fd8c5d589f899ecd89565cc99b05f6fa)), closes [#246](https://github.com/Alberto-Codes/typevet/issues/246)
* **reference:** one page for releases, changelog and versioning ([40f7286](https://github.com/Alberto-Codes/typevet/commit/40f7286899046f6c4fdb56d2989706c7ee545183)), closes [#274](https://github.com/Alberto-Codes/typevet/issues/274)
* **reference:** split the Python API reference into one page per package ([f7af6d4](https://github.com/Alberto-Codes/typevet/commit/f7af6d491d88809e0d8170481887c7a2a07dc5c6)), closes [#266](https://github.com/Alberto-Codes/typevet/issues/266)
* **site:** define accessible light and dark visual tokens ([c7dceae](https://github.com/Alberto-Codes/typevet/commit/c7dceae9db93008b420bc031363d9858233b84b1)), closes [#263](https://github.com/Alberto-Codes/typevet/issues/263) [#262](https://github.com/Alberto-Codes/typevet/issues/262)
* **site:** give the Home routes a primary start action and task cards ([acc7e25](https://github.com/Alberto-Codes/typevet/commit/acc7e25eb2e68502e8255acccf364fd9a5d620f0)), closes [#264](https://github.com/Alberto-Codes/typevet/issues/264)
* **site:** group reference navigation around reader tasks ([219bf76](https://github.com/Alberto-Codes/typevet/commit/219bf76516e19b9a159db11bc864834f12e2bdd6)), closes [#265](https://github.com/Alberto-Codes/typevet/issues/265)
* **tutorial:** first typed judgment on a local llama.cpp server ([0c1b529](https://github.com/Alberto-Codes/typevet/commit/0c1b5297ce4952ff5c3502091045070c17958915)), closes [#281](https://github.com/Alberto-Codes/typevet/issues/281)
* **tutorial:** make the offline tutorial a full first run for a new reader ([0ac1f92](https://github.com/Alberto-Codes/typevet/commit/0ac1f9234a0ad4ba68da7b06dfd5eb83405329d5)), closes [#280](https://github.com/Alberto-Codes/typevet/issues/280)

## [0.1.0](https://github.com/Alberto-Codes/typevet/releases/tag/v0.1.0) (2026-09-29)


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

# Changelog

## [0.9.0](https://github.com/Alberto-Codes/typevet/compare/v0.8.1...v0.9.0) (2026-10-09)


### Features

* **adapters:** add TYPEVET_BACKEND=fake to open_judgment and export it from the inbound package ([436b01f](https://github.com/Alberto-Codes/typevet/commit/436b01f94e8816667c381e752bf33acfe4b88f99)), closes [#388](https://github.com/Alberto-Codes/typevet/issues/388) [#385](https://github.com/Alberto-Codes/typevet/issues/385)
* **adapters:** carry the off-option receipt over the judgevet bridge ([5298ddb](https://github.com/Alberto-Codes/typevet/commit/5298ddbefe157f972afac2753c01393b0f98099d)), closes [#370](https://github.com/Alberto-Codes/typevet/issues/370)
* **adapters:** expose Retry-After and rate-limit headers on a vLLM HTTP error ([8a273df](https://github.com/Alberto-Codes/typevet/commit/8a273dfddc070b6214e339a1d0db002f49c35d8d)), closes [#355](https://github.com/Alberto-Codes/typevet/issues/355)
* **adapters:** give the llama.cpp settings the vLLM key and header fields and mask the key in sync errors ([d89510b](https://github.com/Alberto-Codes/typevet/commit/d89510b86f7ac75ac8c6153717c9906dad86d17a)), closes [#410](https://github.com/Alberto-Codes/typevet/issues/410) [#406](https://github.com/Alberto-Codes/typevet/issues/406)
* **adapters:** reach a vLLM server behind an API gateway ([58eebce](https://github.com/Alberto-Codes/typevet/commit/58eebce135252b43c9461d7807c095fb7cdfa7dc)), closes [#331](https://github.com/Alberto-Codes/typevet/issues/331)
* **adapters:** record request ids for llama.cpp calls through TYPEVET_LLAMA__REQUEST_ID_HEADER ([08d4ec7](https://github.com/Alberto-Codes/typevet/commit/08d4ec7ec9b77add79a0ddbf174c9108b759e497)), closes [#411](https://github.com/Alberto-Codes/typevet/issues/411) [#406](https://github.com/Alberto-Codes/typevet/issues/406)
* **adapters:** record the request id typevet sends in receipts and errors ([68aab28](https://github.com/Alberto-Codes/typevet/commit/68aab28f77d375b0329482b8d0f123672782cbef)), closes [#356](https://github.com/Alberto-Codes/typevet/issues/356)
* **adapters:** refuse a Gemma 4 framing that lacks the no-thinking prefill ([8edd342](https://github.com/Alberto-Codes/typevet/commit/8edd342ce94299899f7fef3de4fb05b14e450bbc)), closes [#354](https://github.com/Alberto-Codes/typevet/issues/354)
* **domain:** make the option block and the context template substitutable named parts ([2db4a5e](https://github.com/Alberto-Codes/typevet/commit/2db4a5e6a68297cea7fa5f449ea3fe7f0578e8ce)), closes [#364](https://github.com/Alberto-Codes/typevet/issues/364) [#360](https://github.com/Alberto-Codes/typevet/issues/360)
* **domain:** native Choice up to 24 options with digit-then-letter controls ([34f23ad](https://github.com/Alberto-Codes/typevet/commit/34f23ad8cb5684f01cccc65e9a2059ea38057684)), closes [#286](https://github.com/Alberto-Codes/typevet/issues/286) [#287](https://github.com/Alberto-Codes/typevet/issues/287) [#237](https://github.com/Alberto-Codes/typevet/issues/237)
* **domain:** refuse channel tokens in templates and apply context_template under a framing ([1341d6e](https://github.com/Alberto-Codes/typevet/commit/1341d6e4b1cef40b2daf0c84fd7360c09f4eabac)), closes [#373](https://github.com/Alberto-Codes/typevet/issues/373) [#360](https://github.com/Alberto-Codes/typevet/issues/360)
* **domain:** refuse turn markers in text-part templates and require the answer rule ([0597c96](https://github.com/Alberto-Codes/typevet/commit/0597c96e586de6ff88ea5124db8aa4c263604d78))
* **evals:** add the doodle duel play mode that scores a human and the model with Brier and a reliability table ([040253c](https://github.com/Alberto-Codes/typevet/commit/040253c0316f6f4d76d5428958ed6e7cb0812f44)), closes [#412](https://github.com/Alberto-Codes/typevet/issues/412)
* **evals:** add the doodle duel Quick, Draw! loader, render and Choice run with a live receipt ([a267d3f](https://github.com/Alberto-Codes/typevet/commit/a267d3f670a76b5ea8ff40a368370125e4a9b5b2)), closes [#412](https://github.com/Alberto-Codes/typevet/issues/412)
* **evals:** bridge pinned JevBench through judgevet ([5e33f81](https://github.com/Alberto-Codes/typevet/commit/5e33f81c8d7420fed2714bc0c3c53448c4dc1793)), closes [#419](https://github.com/Alberto-Codes/typevet/issues/419) [#418](https://github.com/Alberto-Codes/typevet/issues/418)
* **evals:** CEDAR signature pair loader and two-image request builder ([46de94e](https://github.com/Alberto-Codes/typevet/commit/46de94e9a3b211fca63e9e17ccc9f60ac5105bea)), closes [#318](https://github.com/Alberto-Codes/typevet/issues/318)
* **evals:** check-versus-register runner, metrics and opt-in live test ([38cada3](https://github.com/Alberto-Codes/typevet/commit/38cada3ff234b44b16df26224a188014a657a0c1)), closes [#316](https://github.com/Alberto-Codes/typevet/issues/316)
* **evals:** concurrency option for the face, check and signature runners ([b287486](https://github.com/Alberto-Codes/typevet/commit/b28748683bc99ba36029f8f2458a9700e7882f77)), closes [#334](https://github.com/Alberto-Codes/typevet/issues/334)
* **evals:** DIFrauD train, validation and held-out splits without [#236](https://github.com/Alberto-Codes/typevet/issues/236) overlap ([d14dcf0](https://github.com/Alberto-Codes/typevet/commit/d14dcf0940666aafc4863714caed9082de3a11e0)), closes [#307](https://github.com/Alberto-Codes/typevet/issues/307)
* **evals:** evolve and score a Choice seed in the wording runner ([394d599](https://github.com/Alberto-Codes/typevet/commit/394d59945088736960d654acbd76fa4516dde4d2)), closes [#369](https://github.com/Alberto-Codes/typevet/issues/369) [#360](https://github.com/Alberto-Codes/typevet/issues/360)
* **evals:** evolve criteria beside instructions through a multi-part wording candidate ([112bca9](https://github.com/Alberto-Codes/typevet/commit/112bca94e3580e7e8147fe16ce23ed87529a4f09)), closes [#363](https://github.com/Alberto-Codes/typevet/issues/363)
* **evals:** gepa-adk wording evolution runner with Brier scorer and length cap ([6f9f740](https://github.com/Alberto-Codes/typevet/commit/6f9f7400ba8344390b788fb83035878e7dfe3947)), closes [#308](https://github.com/Alberto-Codes/typevet/issues/308)
* **evals:** harness for the DIFrauD wording evolution and its held-out check ([502ebc9](https://github.com/Alberto-Codes/typevet/commit/502ebc973962a92f4eaeae88098c4fe07bdc89da)), closes [#309](https://github.com/Alberto-Codes/typevet/issues/309)
* **evals:** Jev judge option for the wording evolution live test ([d86e70c](https://github.com/Alberto-Codes/typevet/commit/d86e70c052670e43b78d481de82374d58f07af54)), closes [#328](https://github.com/Alberto-Codes/typevet/issues/328)
* **evals:** Jev vs Gemma held-out comparison harness with Cohen's kappa ([d86aa9f](https://github.com/Alberto-Codes/typevet/commit/d86aa9f367a1ec456cad52c2aaf51650877c28cd)), closes [#329](https://github.com/Alberto-Codes/typevet/issues/329)
* **evals:** LFW View 2 pair loader and two-image face-match requests ([dc3d7ca](https://github.com/Alberto-Codes/typevet/commit/dc3d7caec6ceeb83a5462e56119994d101537906)), closes [#300](https://github.com/Alberto-Codes/typevet/issues/300) [#292](https://github.com/Alberto-Codes/typevet/issues/292)
* **evals:** name the criteria parts of Choice and Score seeds ([c2baab9](https://github.com/Alberto-Codes/typevet/commit/c2baab9005d3ad5d85e4060cba0dcf698d4ad860))
* **evals:** pass an off-option threshold through the CORD acceptance arm and record the receipt per row ([04e0b7d](https://github.com/Alberto-Codes/typevet/commit/04e0b7d487f6518a725c6f1cb0a500e0da084924)), closes [#384](https://github.com/Alberto-Codes/typevet/issues/384) [#370](https://github.com/Alberto-Codes/typevet/issues/370)
* **evals:** pass the off-option threshold on the order set's CORD rerun and give every scored row the receipt fields ([e61e24a](https://github.com/Alberto-Codes/typevet/commit/e61e24ade32d26bec91ff760cb86f3904cb8c3b2)), closes [#409](https://github.com/Alberto-Codes/typevet/issues/409) [#384](https://github.com/Alberto-Codes/typevet/issues/384)
* **evals:** pin seed and evolved wording by digest in wording receipts ([ea38912](https://github.com/Alberto-Codes/typevet/commit/ea389127b7e7473dcde80a526cff90ff4881c3f9)), closes [#362](https://github.com/Alberto-Codes/typevet/issues/362) [#360](https://github.com/Alberto-Codes/typevet/issues/360)
* **evals:** post-hoc calibration of the committed Noul receipts ([78aea74](https://github.com/Alberto-Codes/typevet/commit/78aea7429308654c4d5011ae104403a5ce4cb5fa)), closes [#343](https://github.com/Alberto-Codes/typevet/issues/343)
* **evals:** record the vLLM server configuration in receipts ([d34c3b9](https://github.com/Alberto-Codes/typevet/commit/d34c3b90b3d6fef5b3b63b15c7e94b981021afc0)), closes [#341](https://github.com/Alberto-Codes/typevet/issues/341)
* **evals:** seed the synthetic check run from the environment ([1dfbc36](https://github.com/Alberto-Codes/typevet/commit/1dfbc36f8b453e86a2d25f8fd4990bec8f21d46d)), closes [#344](https://github.com/Alberto-Codes/typevet/issues/344)
* **evals:** served-template probe and native Gemma framing for text wording runs ([0628eb0](https://github.com/Alberto-Codes/typevet/commit/0628eb07161cb2a301e63d4819384ccc00b134b3)), closes [#327](https://github.com/Alberto-Codes/typevet/issues/327)
* **evals:** signature-pair runner, metrics and opt-in live test for CEDAR ([e3e4643](https://github.com/Alberto-Codes/typevet/commit/e3e4643f395f195fb287a75820450b997706f91a)), closes [#319](https://github.com/Alberto-Codes/typevet/issues/319)
* **evals:** synthetic check generator and check-versus-register request builder ([b3d5457](https://github.com/Alberto-Codes/typevet/commit/b3d5457420edcd30e59b51570b71d4ec5c3fc8f6)), closes [#315](https://github.com/Alberto-Codes/typevet/issues/315)
* **evals:** throughput fields in the image receipts ([93a13fb](https://github.com/Alberto-Codes/typevet/commit/93a13fbbf63f043165266ce0eb2752bcfb774c5e)), closes [#335](https://github.com/Alberto-Codes/typevet/issues/335)
* **evals:** top-label and classwise ECE for Choice and Score above 10 options ([38f9272](https://github.com/Alberto-Codes/typevet/commit/38f927237240d9b8e4f7cd334eab70d117652f97)), closes [#296](https://github.com/Alberto-Codes/typevet/issues/296)
* **examples:** add the check_register and calibrate examples ([60b3309](https://github.com/Alberto-Codes/typevet/commit/60b33099a096b058ccea44cb780896b8581e86af)), closes [#396](https://github.com/Alberto-Codes/typevet/issues/396) [#385](https://github.com/Alberto-Codes/typevet/issues/385)
* **examples:** add the face_pair and signature_pair examples with user-supplied images ([26b303d](https://github.com/Alberto-Codes/typevet/commit/26b303d9e094cffcae431a066c745fb346f91472)), closes [#397](https://github.com/Alberto-Codes/typevet/issues/397) [#385](https://github.com/Alberto-Codes/typevet/issues/385)
* **examples:** add the receipt_claim example and the contract test that runs every example offline ([996e3a6](https://github.com/Alberto-Codes/typevet/commit/996e3a6f433178b04ccb1c37cd7c5f781be1e069)), closes [#395](https://github.com/Alberto-Codes/typevet/issues/395)
* **examples:** add the screenshot_ui and scam_message examples ([c6856a8](https://github.com/Alberto-Codes/typevet/commit/c6856a83c2467870ed12af373377be060dd11b0a)), closes [#398](https://github.com/Alberto-Codes/typevet/issues/398)
* **integrations:** a judgevet provider bridge over JudgmentPort ([2b04383](https://github.com/Alberto-Codes/typevet/commit/2b04383d94c49f8723a09762991d2e6deaaaab37)), closes [#284](https://github.com/Alberto-Codes/typevet/issues/284)
* **integrations:** judgevet 0.15 conformance kit and per-question images ([67728f5](https://github.com/Alberto-Codes/typevet/commit/67728f586e6d3f930f5598d390585b4979f38e64)), closes [#290](https://github.com/Alberto-Codes/typevet/issues/290)
* **ports:** carry off_option_threshold through the port protocol and every wrapper ([3df3ff7](https://github.com/Alberto-Codes/typevet/commit/3df3ff770f56e4ea401491f804d382260ac06c2a)), closes [#368](https://github.com/Alberto-Codes/typevet/issues/368) [#353](https://github.com/Alberto-Codes/typevet/issues/353)
* **runtime:** apply a caller-supplied calibration map to Noul answers ([f1a6d6c](https://github.com/Alberto-Codes/typevet/commit/f1a6d6c167f72bfc2a8fa8dc037cdfc1c57fa288)), closes [#352](https://github.com/Alberto-Codes/typevet/issues/352)
* **runtime:** apply a Score calibration map and write map artifacts from receipts ([00f7725](https://github.com/Alberto-Codes/typevet/commit/00f7725db3afc53976b1ae1050bfd8da3e0aee8b)), closes [#352](https://github.com/Alberto-Codes/typevet/issues/352)
* **scoring:** flag an answer whose off-option mass exceeds a caller threshold ([722728b](https://github.com/Alberto-Codes/typevet/commit/722728b31e255b60479dfa5d5b16e9cf70f7b6b7)), closes [#353](https://github.com/Alberto-Codes/typevet/issues/353)
* **scoring:** report off-option probability mass in scoring results ([db5ff12](https://github.com/Alberto-Codes/typevet/commit/db5ff12d2b6c909a1e3390c3a38e02e915eb144b)), closes [#297](https://github.com/Alberto-Codes/typevet/issues/297)
* **testing:** add a public ScriptedJudgmentFake with per-question distributions ([de4963b](https://github.com/Alberto-Codes/typevet/commit/de4963b30c14264ea313f548e9521b379c7f7ea3)), closes [#387](https://github.com/Alberto-Codes/typevet/issues/387) [#385](https://github.com/Alberto-Codes/typevet/issues/385)


### Fixes

* **adapters:** close the owned scoring client under the lock ([5f5caaa](https://github.com/Alberto-Codes/typevet/commit/5f5caaa8cc580c0f908e4c530f156f30bdfc3e0c)), closes [#345](https://github.com/Alberto-Codes/typevet/issues/345)
* **adapters:** create the lazy scoring client under a lock ([6b44d72](https://github.com/Alberto-Codes/typevet/commit/6b44d7262432f6bd41c88f7ed5884e1063df4e23)), closes [#340](https://github.com/Alberto-Codes/typevet/issues/340)
* **adapters:** drop the cause chain from masked vLLM errors when a key is set ([71ee1fb](https://github.com/Alberto-Codes/typevet/commit/71ee1fbd8e458ed61cda24f52f6655e9028b41f9)), closes [#276](https://github.com/Alberto-Codes/typevet/issues/276)
* **adapters:** exact vocabulary size for the llama.cpp off-option completeness check ([81a2e02](https://github.com/Alberto-Codes/typevet/commit/81a2e02f45d50774293f9b6e956c3a26945b5c8e)), closes [#321](https://github.com/Alberto-Codes/typevet/issues/321)
* **adapters:** map errors from the vision factory's /apply-template probe ([64814a0](https://github.com/Alberto-Codes/typevet/commit/64814a08d5ce1a952cdbe6c422ffa83dd5c4168b)), closes [#311](https://github.com/Alberto-Codes/typevet/issues/311)
* **adapters:** map vision tokenizer errors and characterize runtime limits ([597ce88](https://github.com/Alberto-Codes/typevet/commit/597ce88f57784c7f1f7d1d50a3bf73dc69de6e61)), closes [#204](https://github.com/Alberto-Codes/typevet/issues/204) [#298](https://github.com/Alberto-Codes/typevet/issues/298)
* **adapters:** mask gateway header values as whole tokens and protect owned headers ([2719ad6](https://github.com/Alberto-Codes/typevet/commit/2719ad6a46fa56ac0a20e5e936fe273e48c2144d)), closes [#348](https://github.com/Alberto-Codes/typevet/issues/348)
* **adapters:** media-marker refresh edge cases after a model swap ([ce63ddd](https://github.com/Alberto-Codes/typevet/commit/ce63dddd013445066c0768e5e2049de1e6e50f9a)), closes [#323](https://github.com/Alberto-Codes/typevet/issues/323)
* **adapters:** refresh the llama.cpp media marker after a router model reload ([e3dd4cb](https://github.com/Alberto-Codes/typevet/commit/e3dd4cbed8c07f39a6f3a2c6c69952129a584085)), closes [#322](https://github.com/Alberto-Codes/typevet/issues/322)
* **adapters:** reject non-finite vLLM timeouts and mask keys in set and bytes values ([e8ecc20](https://github.com/Alberto-Codes/typevet/commit/e8ecc2093775975a91ede996e5793c43812ad407)), closes [#227](https://github.com/Alberto-Codes/typevet/issues/227)
* **adapters:** retry one early close on llama.cpp scoring calls ([fdae661](https://github.com/Alberto-Codes/typevet/commit/fdae6619f6f5d8a950f8a5a3018df2345b421b2d)), closes [#305](https://github.com/Alberto-Codes/typevet/issues/305) [#310](https://github.com/Alberto-Codes/typevet/issues/310)
* **domain:** validate CandidateScoringRequest.media items ([ce3c9e6](https://github.com/Alberto-Codes/typevet/commit/ce3c9e6d4c56a072da2244e4d27497bf4396c58a)), closes [#222](https://github.com/Alberto-Codes/typevet/issues/222)
* **evals:** harden the PSAI leak check, KV-cache reading and runner failure counting ([dec6dc7](https://github.com/Alberto-Codes/typevet/commit/dec6dc76f729775449948693482c4fcd1e1ede87)), closes [#223](https://github.com/Alberto-Codes/typevet/issues/223) [#225](https://github.com/Alberto-Codes/typevet/issues/225) [#238](https://github.com/Alberto-Codes/typevet/issues/238) [#261](https://github.com/Alberto-Codes/typevet/issues/261)
* **evals:** mask an echoed header name used as a JSON key in the acceptance receipt ([bbd1d19](https://github.com/Alberto-Codes/typevet/commit/bbd1d193de921d63362b83eb700334da8162e556)), closes [#351](https://github.com/Alberto-Codes/typevet/issues/351)
* **evals:** one seed rule for the check sheets command and the live run ([6fdc464](https://github.com/Alberto-Codes/typevet/commit/6fdc4642ac003062fd723182be06f3ad8d648b23)), closes [#349](https://github.com/Alberto-Codes/typevet/issues/349)
* **evals:** refuse a wrong split and a two-engine gauge in receipts ([e030ee4](https://github.com/Alberto-Codes/typevet/commit/e030ee4288628db105036c448d79481fe57c15da)), closes [#339](https://github.com/Alberto-Codes/typevet/issues/339)
* **evals:** wider PSAI leak check, key-free receipt frames, prompt-read controls ([09f6787](https://github.com/Alberto-Codes/typevet/commit/09f6787f39241b3506aa99f3dedf8f4299aa1a6d)), closes [#295](https://github.com/Alberto-Codes/typevet/issues/295)
* **llama:** name the model and its source when the judgment template is not Gemma 4 ([0e089d1](https://github.com/Alberto-Codes/typevet/commit/0e089d1b8e3f884da58ae6a2bbe9faa0846d885d)), closes [#425](https://github.com/Alberto-Codes/typevet/issues/425)
* **site:** focus ring, Noul search rank, 2x header reflow and Home card headings ([5bbfecb](https://github.com/Alberto-Codes/typevet/commit/5bbfecb25d78340ba5f56bd8470f3bd0b1af864d)), closes [#282](https://github.com/Alberto-Codes/typevet/issues/282) [#267](https://github.com/Alberto-Codes/typevet/issues/267)
* **site:** keep the closed search panel out of the Tab order ([3a7a67c](https://github.com/Alberto-Codes/typevet/commit/3a7a67cdceefa048f4a3f03384279b0ae4e8d90f)), closes [#285](https://github.com/Alberto-Codes/typevet/issues/285) [#267](https://github.com/Alberto-Codes/typevet/issues/267) [#242](https://github.com/Alberto-Codes/typevet/issues/242)
* **site:** keyboard access to the mobile navigation drawer ([918d8d9](https://github.com/Alberto-Codes/typevet/commit/918d8d9e8d4277eaaaca5de75967aadcf205ac8f)), closes [#283](https://github.com/Alberto-Codes/typevet/issues/283) [#267](https://github.com/Alberto-Codes/typevet/issues/267)
* **testing:** give the ScriptedJudgmentFake Score answer the expected rubric level ([2066651](https://github.com/Alberto-Codes/typevet/commit/206665174e720b94ea480fc52be69e3f73095550)), closes [#400](https://github.com/Alberto-Codes/typevet/issues/400)
* **tests:** keep API keys out of pytest failure output ([599de01](https://github.com/Alberto-Codes/typevet/commit/599de0134c0d9e700604c35ba2e564b73158efd8)), closes [#251](https://github.com/Alberto-Codes/typevet/issues/251)


### Refactoring

* **adapters:** gather the llama.cpp outbound modules into one package ([8df631b](https://github.com/Alberto-Codes/typevet/commit/8df631bcddfcc05ac887e1e69e88135ced69f929)), closes [#256](https://github.com/Alberto-Codes/typevet/issues/256)
* **adapters:** gather the vLLM outbound modules into one package ([4d2d999](https://github.com/Alberto-Codes/typevet/commit/4d2d99995afa7e05c4ee03256c0d7e1b89cfe59a)), closes [#256](https://github.com/Alberto-Codes/typevet/issues/256)
* **adapters:** split the judgevet bridge mapping into a sibling and give the async port provider_options ([45220bc](https://github.com/Alberto-Codes/typevet/commit/45220bcaf864b56d32e21da20740e832fa792b8d)), closes [#380](https://github.com/Alberto-Codes/typevet/issues/380) [#370](https://github.com/Alberto-Codes/typevet/issues/370)
* **evals:** move the CORD evaluation family into typevet_evals.cord ([4fa5cce](https://github.com/Alberto-Codes/typevet/commit/4fa5cce5bd0f62e7a25a04292b6dac139ba8bc59)), closes [#256](https://github.com/Alberto-Codes/typevet/issues/256)
* **evals:** move the eval CLI, wheel isolation and the Gemma wheel smoke into evals/ ([ced27a3](https://github.com/Alberto-Codes/typevet/commit/ced27a3074bf49fbca221630e6bbb0707c0f1932)), closes [#256](https://github.com/Alberto-Codes/typevet/issues/256)
* **evals:** move the eval datasets into evals/ and remove typevet.evaluation ([20d0d1a](https://github.com/Alberto-Codes/typevet/commit/20d0d1a687dd8b52428f5a6b8e6bd63e06f7c54a)), closes [#256](https://github.com/Alberto-Codes/typevet/issues/256)
* **evals:** move the eval runner, TPJEP and experiment identity into evals/ ([63447e9](https://github.com/Alberto-Codes/typevet/commit/63447e90a532a1606897ad2011f2625800bea089)), closes [#256](https://github.com/Alberto-Codes/typevet/issues/256)
* **evals:** move the instruction-variant consumer harness and replay metrics into evals/ ([e542750](https://github.com/Alberto-Codes/typevet/commit/e5427503aee6267e5f8a0f96c67cc7e42497712c)), closes [#256](https://github.com/Alberto-Codes/typevet/issues/256)
* **evals:** move the PSAI vision consumer harness and the CORD CLI into evals/ ([df25a34](https://github.com/Alberto-Codes/typevet/commit/df25a345fa01bfff3c8caa9ae348f409c25f5171)), closes [#256](https://github.com/Alberto-Codes/typevet/issues/256)
* **evals:** move the throughput and vLLM acceptance harnesses into evals/ ([dc5abc1](https://github.com/Alberto-Codes/typevet/commit/dc5abc13ad424dd6843638c364383722450d48e0)), closes [#256](https://github.com/Alberto-Codes/typevet/issues/256)
* **eval:** split transport and metrics reading out of the vLLM acceptance module ([a2e3ba7](https://github.com/Alberto-Codes/typevet/commit/a2e3ba792e9cc947d9ea8f4773acfecbe59a71e7)), closes [#229](https://github.com/Alberto-Codes/typevet/issues/229)
* **evals:** refuse a non-text Choice label in the wording parts path and keep row text out of the split refusal ([70d9a9d](https://github.com/Alberto-Codes/typevet/commit/70d9a9d48eb6562cc4ae8243791473f00bbb21b6)), closes [#378](https://github.com/Alberto-Codes/typevet/issues/378) [#360](https://github.com/Alberto-Codes/typevet/issues/360)
* **evals:** take WordingParts on score_held_out and rename key to question_name ([7068af6](https://github.com/Alberto-Codes/typevet/commit/7068af66405110b4785ded6302c7ac7fa92048a5)), closes [#372](https://github.com/Alberto-Codes/typevet/issues/372) [#363](https://github.com/Alberto-Codes/typevet/issues/363)
* **examples:** give terminal-demo a winner last line and bring it under the line cap ([4bc380f](https://github.com/Alberto-Codes/typevet/commit/4bc380f9ceda3bb7390d47837fcf3b1ef610348d)), closes [#401](https://github.com/Alberto-Codes/typevet/issues/401)
* **examples:** split the live demo into flat siblings and bring examples under the check_loc gate ([8a2485b](https://github.com/Alberto-Codes/typevet/commit/8a2485b4832b20d645ae2eadc017164d578bec19)), closes [#403](https://github.com/Alberto-Codes/typevet/issues/403) [#385](https://github.com/Alberto-Codes/typevet/issues/385)
* **package:** remove the 27 root eval_* compatibility shims ([bdca25d](https://github.com/Alberto-Codes/typevet/commit/bdca25d199b9d12cf5ad33ba752dff2f0bf8f32b)), closes [#256](https://github.com/Alberto-Codes/typevet/issues/256)
* **package:** remove the library root shims and move question_schema into the domain ([2883ad6](https://github.com/Alberto-Codes/typevet/commit/2883ad62390f7ae470bf0e467717108a8cb041b6)), closes [#256](https://github.com/Alberto-Codes/typevet/issues/256)


### Documentation

* **adr:** record the package layout and test the supported-import surface ([946ee06](https://github.com/Alberto-Codes/typevet/commit/946ee0650549878aae75cc7a02a7b65d1e20b80f)), closes [#256](https://github.com/Alberto-Codes/typevet/issues/256)
* correct the local alias identity, framing prefill and Gemma 4 image cost ([867a61a](https://github.com/Alberto-Codes/typevet/commit/867a61a61eb59e824173c074be6f5fc394252ed3)), closes [#233](https://github.com/Alberto-Codes/typevet/issues/233) [#235](https://github.com/Alberto-Codes/typevet/issues/235) [#260](https://github.com/Alberto-Codes/typevet/issues/260)
* **evals:** record the public JevBench Gemma baseline ([dd6fe4a](https://github.com/Alberto-Codes/typevet/commit/dd6fe4a1c50b0595b6ad3509cfe9d5fff2771e24)), closes [#420](https://github.com/Alberto-Codes/typevet/issues/420) [#418](https://github.com/Alberto-Codes/typevet/issues/418)
* **explanation:** check images against a synthetic register and what the typed answers mean ([7299fd6](https://github.com/Alberto-Codes/typevet/commit/7299fd626d3502b2ea2a0d09dfb1add2dad37f83)), closes [#317](https://github.com/Alberto-Codes/typevet/issues/317) [#316](https://github.com/Alberto-Codes/typevet/issues/316) [#303](https://github.com/Alberto-Codes/typevet/issues/303)
* **explanation:** correct the stale [#129](https://github.com/Alberto-Codes/typevet/issues/129) template claim in native typed judgments ([698d53d](https://github.com/Alberto-Codes/typevet/commit/698d53d1c71475295658363965d4f99ef60e1b96)), closes [#278](https://github.com/Alberto-Codes/typevet/issues/278)
* **explanation:** DIFrauD wording evolution and its pre-registered held-out check ([3133817](https://github.com/Alberto-Codes/typevet/commit/3133817497b997b8a4ca4053f1ad7e60a87bf45f)), closes [#309](https://github.com/Alberto-Codes/typevet/issues/309)
* **explanation:** gather limits and known gaps on one page with page status ([cb51e80](https://github.com/Alberto-Codes/typevet/commit/cb51e80714e829483f47213fbdbfa57fd8cb35e5)), closes [#247](https://github.com/Alberto-Codes/typevet/issues/247)
* **explanation:** Gemma 4 and Jev on DIFrauD, each with its evolved wording ([44ffb4d](https://github.com/Alberto-Codes/typevet/commit/44ffb4dfaa4e7f1706d2e440e6cb24194f074202)), closes [#330](https://github.com/Alberto-Codes/typevet/issues/330) [#252](https://github.com/Alberto-Codes/typevet/issues/252) [#324](https://github.com/Alberto-Codes/typevet/issues/324)
* **explanation:** image judgment throughput on one H100 ([50834b9](https://github.com/Alberto-Codes/typevet/commit/50834b9e523a17ae44cd40eee692d8d95238fc7f)), closes [#337](https://github.com/Alberto-Codes/typevet/issues/337) [#332](https://github.com/Alberto-Codes/typevet/issues/332)
* **explanation:** latency and cost of the 0.4.0 experiments ([4f66fa0](https://github.com/Alberto-Codes/typevet/commit/4f66fa0e9f693da30961d6672f919fabb3f7e534)), closes [#326](https://github.com/Alberto-Codes/typevet/issues/326)
* **explanation:** record one seed-0 checks run on the Gemma 4 QAT Q4_0 pin ([17eae9f](https://github.com/Alberto-Codes/typevet/commit/17eae9f49fd953e9da0af3aba5f678b9f91d5de6)), closes [#358](https://github.com/Alberto-Codes/typevet/issues/358)
* **explanation:** record the [#333](https://github.com/Alberto-Codes/typevet/issues/333) nimble judge run beside the [#252](https://github.com/Alberto-Codes/typevet/issues/252) judges ([be0feb5](https://github.com/Alberto-Codes/typevet/commit/be0feb5f1c71785210f5fe454fea7c1d4b0503c4))
* **explanation:** record the [#342](https://github.com/Alberto-Codes/typevet/issues/342) design outcome on the throughput page ([9c6afa5](https://github.com/Alberto-Codes/typevet/commit/9c6afa5a5bf957c626f700ef74bffe79796c1731))
* **explanation:** record the [#365](https://github.com/Alberto-Codes/typevet/issues/365) criteria-evolution result beside the [#252](https://github.com/Alberto-Codes/typevet/issues/252) study ([c0920a7](https://github.com/Alberto-Codes/typevet/commit/c0920a72731f7cb0d753fd337acb50a9bf4957e5)), closes [#360](https://github.com/Alberto-Codes/typevet/issues/360)
* **explanation:** record the [#376](https://github.com/Alberto-Codes/typevet/issues/376) arm C result beside the [#365](https://github.com/Alberto-Codes/typevet/issues/365) arms ([fda7369](https://github.com/Alberto-Codes/typevet/commit/fda73699edc5624241e5ba07a4d0616bab72e404))
* **explanation:** record the endpoints typevet does not score on ([3cb1205](https://github.com/Alberto-Codes/typevet/commit/3cb12051c14b95c4c3bfedbd8096fd857d8bf346)), closes [#405](https://github.com/Alberto-Codes/typevet/issues/405)
* **explanation:** split nine sentences over the 20-word limit on three image pages ([d1cc0a9](https://github.com/Alberto-Codes/typevet/commit/d1cc0a9a7754d9f3ef508042a9e6ff2471590f21)), closes [#375](https://github.com/Alberto-Codes/typevet/issues/375)
* **explanation:** two-image face matching and what its probability means ([cd76f82](https://github.com/Alberto-Codes/typevet/commit/cd76f820f9ad56b0210f532ca03068bb24e49a5d)), closes [#302](https://github.com/Alberto-Codes/typevet/issues/302) [#292](https://github.com/Alberto-Codes/typevet/issues/292)
* **explanation:** two-image signature comparison on CEDAR and what its probability means ([c418b0c](https://github.com/Alberto-Codes/typevet/commit/c418b0c79b20f103438e103979f5ad810132f9d1)), closes [#320](https://github.com/Alberto-Codes/typevet/issues/320) [#319](https://github.com/Alberto-Codes/typevet/issues/319) [#304](https://github.com/Alberto-Codes/typevet/issues/304)
* **how-to:** install typevet from a wheel outside a checkout ([dc9e9f3](https://github.com/Alberto-Codes/typevet/commit/dc9e9f33f1df27cb9d61f21e8e34cd513796c126)), closes [#245](https://github.com/Alberto-Codes/typevet/issues/245)
* **how-to:** stand up the live demo, with screenshots under docs/assets/live-demo ([af3921b](https://github.com/Alberto-Codes/typevet/commit/af3921b374409f9b6348ce51a77eee59976e2796)), closes [#393](https://github.com/Alberto-Codes/typevet/issues/393) [#385](https://github.com/Alberto-Codes/typevet/issues/385)
* **index:** list every page in docs/README.md and drop the stale PyPI claim ([638a2e1](https://github.com/Alberto-Codes/typevet/commit/638a2e1d9906603c5a9038707840965cfdb85698)), closes [#275](https://github.com/Alberto-Codes/typevet/issues/275)
* **integrations:** explain why the judgevet bridge keeps text-only instructions ([f7560b3](https://github.com/Alberto-Codes/typevet/commit/f7560b37995ed7165236a59ee51d3073133ae8c4)), closes [#291](https://github.com/Alberto-Codes/typevet/issues/291)
* **readme:** link the live demo recording in a "See it run" section ([67737b1](https://github.com/Alberto-Codes/typevet/commit/67737b1a8813435c8d929d9b741c1800b33c2c71)), closes [#394](https://github.com/Alberto-Codes/typevet/issues/394) [#385](https://github.com/Alberto-Codes/typevet/issues/385)
* **reference:** add latency at a glance to the H100 performance page ([f001283](https://github.com/Alberto-Codes/typevet/commit/f001283e06139cd6c16ceebef37203048ad9051c)), closes [#236](https://github.com/Alberto-Codes/typevet/issues/236)
* **reference:** list the text parts of a judgment call and their evolution status ([f55a9c0](https://github.com/Alberto-Codes/typevet/commit/f55a9c08ddbac7239239786f8512356f4c7c23e8)), closes [#361](https://github.com/Alberto-Codes/typevet/issues/361) [#360](https://github.com/Alberto-Codes/typevet/issues/360)
* **reference:** make the backend pages match the code on TYPEVET_BACKEND and vLLM coverage ([ee630b2](https://github.com/Alberto-Codes/typevet/commit/ee630b2720f0fce27d7d0d42cf4e8058ed1704f3)), closes [#404](https://github.com/Alberto-Codes/typevet/issues/404)
* **reference:** map vLLM adapter failures to typevet errors ([367051e](https://github.com/Alberto-Codes/typevet/commit/367051ea76f13ce62f17abf201c962013ba3dbb0)), closes [#273](https://github.com/Alberto-Codes/typevet/issues/273)
* **reference:** one page for data handling, key masking and security ([fa07066](https://github.com/Alberto-Codes/typevet/commit/fa070662fd8c5d589f899ecd89565cc99b05f6fa)), closes [#246](https://github.com/Alberto-Codes/typevet/issues/246)
* **reference:** one page for releases, changelog and versioning ([40f7286](https://github.com/Alberto-Codes/typevet/commit/40f7286899046f6c4fdb56d2989706c7ee545183)), closes [#274](https://github.com/Alberto-Codes/typevet/issues/274)
* **reference:** publish the H100 throughput results and a rented-H100 how-to ([17ce15b](https://github.com/Alberto-Codes/typevet/commit/17ce15b3f2128d1a33c95c35da792b667d880f09)), closes [#236](https://github.com/Alberto-Codes/typevet/issues/236)
* **reference:** put the tested vLLM pin and current behaviour into the support matrix ([9fd6557](https://github.com/Alberto-Codes/typevet/commit/9fd6557b6dcc3b4d1fb96747c94338e3d4ec4a43)), closes [#243](https://github.com/Alberto-Codes/typevet/issues/243)
* **reference:** receipt blocks page, and a timestamped metrics sample ([2492d95](https://github.com/Alberto-Codes/typevet/commit/2492d95facdb4dc86224b47895ca779923d6ca67)), closes [#347](https://github.com/Alberto-Codes/typevet/issues/347)
* **reference:** record the Molmo2-4B single-token control check ([643b68a](https://github.com/Alberto-Codes/typevet/commit/643b68a52ff5911c2a306b1631c4b4b4c17a2a83)), closes [#299](https://github.com/Alberto-Codes/typevet/issues/299)
* **reference:** split the Python API reference into one page per package ([f7af6d4](https://github.com/Alberto-Codes/typevet/commit/f7af6d491d88809e0d8170481887c7a2a07dc5c6)), closes [#266](https://github.com/Alberto-Codes/typevet/issues/266)
* **release:** make the README and install docs true for the PyPI publish ([2ecfc99](https://github.com/Alberto-Codes/typevet/commit/2ecfc9984ae890c3aa661612140bc33bdad0d226)), closes [#272](https://github.com/Alberto-Codes/typevet/issues/272)
* **site:** add a landing page and rewrite the README for new readers ([e7754f3](https://github.com/Alberto-Codes/typevet/commit/e7754f331f8509f9eb5d700c23efcafb5b69e01b)), closes [#244](https://github.com/Alberto-Codes/typevet/issues/244)
* **site:** define accessible light and dark visual tokens ([c7dceae](https://github.com/Alberto-Codes/typevet/commit/c7dceae9db93008b420bc031363d9858233b84b1)), closes [#263](https://github.com/Alberto-Codes/typevet/issues/263) [#262](https://github.com/Alberto-Codes/typevet/issues/262)
* **site:** enable pymdownx.snippets with path checking and marker ranges ([c40fb09](https://github.com/Alberto-Codes/typevet/commit/c40fb091e1e0720c3333ce37e861fddd99de6ef1)), closes [#390](https://github.com/Alberto-Codes/typevet/issues/390) [#385](https://github.com/Alberto-Codes/typevet/issues/385)
* **site:** give the Home routes a primary start action and task cards ([acc7e25](https://github.com/Alberto-Codes/typevet/commit/acc7e25eb2e68502e8255acccf364fd9a5d620f0)), closes [#264](https://github.com/Alberto-Codes/typevet/issues/264)
* **site:** group reference navigation around reader tasks ([219bf76](https://github.com/Alberto-Codes/typevet/commit/219bf76516e19b9a159db11bc864834f12e2bdd6)), closes [#265](https://github.com/Alberto-Codes/typevet/issues/265)
* **site:** render Mermaid diagrams and draw the typed-judgment flow ([a0cc4b9](https://github.com/Alberto-Codes/typevet/commit/a0cc4b97ce0e9cd23c8fbf52f63dc706f28dacd9)), closes [#254](https://github.com/Alberto-Codes/typevet/issues/254)
* split long sentences in the docs index, CORD smoke and judgevet pages ([4ce5971](https://github.com/Alberto-Codes/typevet/commit/4ce59710dc885f032490d9b55eadd1add9a00b04)), closes [#295](https://github.com/Alberto-Codes/typevet/issues/295)
* state the 0.2.0 release where pages said 0.1.0 is current ([04e8b81](https://github.com/Alberto-Codes/typevet/commit/04e8b819f66c519f3a8ac997d1c3db7b125aca93)), closes [#294](https://github.com/Alberto-Codes/typevet/issues/294)
* **tutorial:** first typed judgment on a local llama.cpp server ([0c1b529](https://github.com/Alberto-Codes/typevet/commit/0c1b5297ce4952ff5c3502091045070c17958915)), closes [#281](https://github.com/Alberto-Codes/typevet/issues/281)
* **tutorial:** make the offline tutorial a full first run for a new reader ([0ac1f92](https://github.com/Alberto-Codes/typevet/commit/0ac1f9234a0ad4ba68da7b06dfd5eb83405329d5)), closes [#280](https://github.com/Alberto-Codes/typevet/issues/280)
* **vllm:** record the live KV-cache gauge reading on the tested pin ([02db15d](https://github.com/Alberto-Codes/typevet/commit/02db15d2a0b7e94044407d565af842266fcb0604)), closes [#231](https://github.com/Alberto-Codes/typevet/issues/231)

## [0.8.1](https://github.com/Alberto-Codes/typevet/compare/v0.8.0...v0.8.1) (2026-10-09)


### Fixes

* **llama:** name the model and its source when the judgment template is not Gemma 4 ([0e089d1](https://github.com/Alberto-Codes/typevet/commit/0e089d1b8e3f884da58ae6a2bbe9faa0846d885d)), closes [#425](https://github.com/Alberto-Codes/typevet/issues/425)

## [0.8.0](https://github.com/Alberto-Codes/typevet/compare/v0.7.0...v0.8.0) (2026-10-03)


### Features

* **adapters:** add TYPEVET_BACKEND=fake to open_judgment and export it from the inbound package ([436b01f](https://github.com/Alberto-Codes/typevet/commit/436b01f94e8816667c381e752bf33acfe4b88f99)), closes [#388](https://github.com/Alberto-Codes/typevet/issues/388) [#385](https://github.com/Alberto-Codes/typevet/issues/385)
* **adapters:** carry the off-option receipt over the judgevet bridge ([5298ddb](https://github.com/Alberto-Codes/typevet/commit/5298ddbefe157f972afac2753c01393b0f98099d)), closes [#370](https://github.com/Alberto-Codes/typevet/issues/370)
* **adapters:** give the llama.cpp settings the vLLM key and header fields and mask the key in sync errors ([d89510b](https://github.com/Alberto-Codes/typevet/commit/d89510b86f7ac75ac8c6153717c9906dad86d17a)), closes [#410](https://github.com/Alberto-Codes/typevet/issues/410) [#406](https://github.com/Alberto-Codes/typevet/issues/406)
* **adapters:** record request ids for llama.cpp calls through TYPEVET_LLAMA__REQUEST_ID_HEADER ([08d4ec7](https://github.com/Alberto-Codes/typevet/commit/08d4ec7ec9b77add79a0ddbf174c9108b759e497)), closes [#411](https://github.com/Alberto-Codes/typevet/issues/411) [#406](https://github.com/Alberto-Codes/typevet/issues/406)
* **domain:** refuse channel tokens in templates and apply context_template under a framing ([1341d6e](https://github.com/Alberto-Codes/typevet/commit/1341d6e4b1cef40b2daf0c84fd7360c09f4eabac)), closes [#373](https://github.com/Alberto-Codes/typevet/issues/373) [#360](https://github.com/Alberto-Codes/typevet/issues/360)
* **domain:** refuse turn markers in text-part templates and require the answer rule ([0597c96](https://github.com/Alberto-Codes/typevet/commit/0597c96e586de6ff88ea5124db8aa4c263604d78))
* **evals:** add the doodle duel play mode that scores a human and the model with Brier and a reliability table ([040253c](https://github.com/Alberto-Codes/typevet/commit/040253c0316f6f4d76d5428958ed6e7cb0812f44)), closes [#412](https://github.com/Alberto-Codes/typevet/issues/412)
* **evals:** add the doodle duel Quick, Draw! loader, render and Choice run with a live receipt ([a267d3f](https://github.com/Alberto-Codes/typevet/commit/a267d3f670a76b5ea8ff40a368370125e4a9b5b2)), closes [#412](https://github.com/Alberto-Codes/typevet/issues/412)
* **evals:** bridge pinned JevBench through judgevet ([5e33f81](https://github.com/Alberto-Codes/typevet/commit/5e33f81c8d7420fed2714bc0c3c53448c4dc1793)), closes [#419](https://github.com/Alberto-Codes/typevet/issues/419) [#418](https://github.com/Alberto-Codes/typevet/issues/418)
* **evals:** evolve and score a Choice seed in the wording runner ([394d599](https://github.com/Alberto-Codes/typevet/commit/394d59945088736960d654acbd76fa4516dde4d2)), closes [#369](https://github.com/Alberto-Codes/typevet/issues/369) [#360](https://github.com/Alberto-Codes/typevet/issues/360)
* **evals:** name the criteria parts of Choice and Score seeds ([c2baab9](https://github.com/Alberto-Codes/typevet/commit/c2baab9005d3ad5d85e4060cba0dcf698d4ad860))
* **evals:** pass an off-option threshold through the CORD acceptance arm and record the receipt per row ([04e0b7d](https://github.com/Alberto-Codes/typevet/commit/04e0b7d487f6518a725c6f1cb0a500e0da084924)), closes [#384](https://github.com/Alberto-Codes/typevet/issues/384) [#370](https://github.com/Alberto-Codes/typevet/issues/370)
* **evals:** pass the off-option threshold on the order set's CORD rerun and give every scored row the receipt fields ([e61e24a](https://github.com/Alberto-Codes/typevet/commit/e61e24ade32d26bec91ff760cb86f3904cb8c3b2)), closes [#409](https://github.com/Alberto-Codes/typevet/issues/409) [#384](https://github.com/Alberto-Codes/typevet/issues/384)
* **examples:** add the check_register and calibrate examples ([60b3309](https://github.com/Alberto-Codes/typevet/commit/60b33099a096b058ccea44cb780896b8581e86af)), closes [#396](https://github.com/Alberto-Codes/typevet/issues/396) [#385](https://github.com/Alberto-Codes/typevet/issues/385)
* **examples:** add the face_pair and signature_pair examples with user-supplied images ([26b303d](https://github.com/Alberto-Codes/typevet/commit/26b303d9e094cffcae431a066c745fb346f91472)), closes [#397](https://github.com/Alberto-Codes/typevet/issues/397) [#385](https://github.com/Alberto-Codes/typevet/issues/385)
* **examples:** add the receipt_claim example and the contract test that runs every example offline ([996e3a6](https://github.com/Alberto-Codes/typevet/commit/996e3a6f433178b04ccb1c37cd7c5f781be1e069)), closes [#395](https://github.com/Alberto-Codes/typevet/issues/395)
* **examples:** add the screenshot_ui and scam_message examples ([c6856a8](https://github.com/Alberto-Codes/typevet/commit/c6856a83c2467870ed12af373377be060dd11b0a)), closes [#398](https://github.com/Alberto-Codes/typevet/issues/398)
* **testing:** add a public ScriptedJudgmentFake with per-question distributions ([de4963b](https://github.com/Alberto-Codes/typevet/commit/de4963b30c14264ea313f548e9521b379c7f7ea3)), closes [#387](https://github.com/Alberto-Codes/typevet/issues/387) [#385](https://github.com/Alberto-Codes/typevet/issues/385)


### Fixes

* **testing:** give the ScriptedJudgmentFake Score answer the expected rubric level ([2066651](https://github.com/Alberto-Codes/typevet/commit/206665174e720b94ea480fc52be69e3f73095550)), closes [#400](https://github.com/Alberto-Codes/typevet/issues/400)


### Refactoring

* **adapters:** split the judgevet bridge mapping into a sibling and give the async port provider_options ([45220bc](https://github.com/Alberto-Codes/typevet/commit/45220bcaf864b56d32e21da20740e832fa792b8d)), closes [#380](https://github.com/Alberto-Codes/typevet/issues/380) [#370](https://github.com/Alberto-Codes/typevet/issues/370)
* **evals:** refuse a non-text Choice label in the wording parts path and keep row text out of the split refusal ([70d9a9d](https://github.com/Alberto-Codes/typevet/commit/70d9a9d48eb6562cc4ae8243791473f00bbb21b6)), closes [#378](https://github.com/Alberto-Codes/typevet/issues/378) [#360](https://github.com/Alberto-Codes/typevet/issues/360)
* **evals:** take WordingParts on score_held_out and rename key to question_name ([7068af6](https://github.com/Alberto-Codes/typevet/commit/7068af66405110b4785ded6302c7ac7fa92048a5)), closes [#372](https://github.com/Alberto-Codes/typevet/issues/372) [#363](https://github.com/Alberto-Codes/typevet/issues/363)
* **examples:** give terminal-demo a winner last line and bring it under the line cap ([4bc380f](https://github.com/Alberto-Codes/typevet/commit/4bc380f9ceda3bb7390d47837fcf3b1ef610348d)), closes [#401](https://github.com/Alberto-Codes/typevet/issues/401)
* **examples:** split the live demo into flat siblings and bring examples under the check_loc gate ([8a2485b](https://github.com/Alberto-Codes/typevet/commit/8a2485b4832b20d645ae2eadc017164d578bec19)), closes [#403](https://github.com/Alberto-Codes/typevet/issues/403) [#385](https://github.com/Alberto-Codes/typevet/issues/385)


### Documentation

* **evals:** record the public JevBench Gemma baseline ([dd6fe4a](https://github.com/Alberto-Codes/typevet/commit/dd6fe4a1c50b0595b6ad3509cfe9d5fff2771e24)), closes [#420](https://github.com/Alberto-Codes/typevet/issues/420) [#418](https://github.com/Alberto-Codes/typevet/issues/418)
* **explanation:** record one seed-0 checks run on the Gemma 4 QAT Q4_0 pin ([17eae9f](https://github.com/Alberto-Codes/typevet/commit/17eae9f49fd953e9da0af3aba5f678b9f91d5de6)), closes [#358](https://github.com/Alberto-Codes/typevet/issues/358)
* **explanation:** record the [#333](https://github.com/Alberto-Codes/typevet/issues/333) nimble judge run beside the [#252](https://github.com/Alberto-Codes/typevet/issues/252) judges ([be0feb5](https://github.com/Alberto-Codes/typevet/commit/be0feb5f1c71785210f5fe454fea7c1d4b0503c4))
* **explanation:** record the [#365](https://github.com/Alberto-Codes/typevet/issues/365) criteria-evolution result beside the [#252](https://github.com/Alberto-Codes/typevet/issues/252) study ([c0920a7](https://github.com/Alberto-Codes/typevet/commit/c0920a72731f7cb0d753fd337acb50a9bf4957e5)), closes [#360](https://github.com/Alberto-Codes/typevet/issues/360)
* **explanation:** record the [#376](https://github.com/Alberto-Codes/typevet/issues/376) arm C result beside the [#365](https://github.com/Alberto-Codes/typevet/issues/365) arms ([fda7369](https://github.com/Alberto-Codes/typevet/commit/fda73699edc5624241e5ba07a4d0616bab72e404))
* **explanation:** record the endpoints typevet does not score on ([3cb1205](https://github.com/Alberto-Codes/typevet/commit/3cb12051c14b95c4c3bfedbd8096fd857d8bf346)), closes [#405](https://github.com/Alberto-Codes/typevet/issues/405)
* **explanation:** split nine sentences over the 20-word limit on three image pages ([d1cc0a9](https://github.com/Alberto-Codes/typevet/commit/d1cc0a9a7754d9f3ef508042a9e6ff2471590f21)), closes [#375](https://github.com/Alberto-Codes/typevet/issues/375)
* **how-to:** stand up the live demo, with screenshots under docs/assets/live-demo ([af3921b](https://github.com/Alberto-Codes/typevet/commit/af3921b374409f9b6348ce51a77eee59976e2796)), closes [#393](https://github.com/Alberto-Codes/typevet/issues/393) [#385](https://github.com/Alberto-Codes/typevet/issues/385)
* **readme:** link the live demo recording in a "See it run" section ([67737b1](https://github.com/Alberto-Codes/typevet/commit/67737b1a8813435c8d929d9b741c1800b33c2c71)), closes [#394](https://github.com/Alberto-Codes/typevet/issues/394) [#385](https://github.com/Alberto-Codes/typevet/issues/385)
* **reference:** make the backend pages match the code on TYPEVET_BACKEND and vLLM coverage ([ee630b2](https://github.com/Alberto-Codes/typevet/commit/ee630b2720f0fce27d7d0d42cf4e8058ed1704f3)), closes [#404](https://github.com/Alberto-Codes/typevet/issues/404)
* **site:** enable pymdownx.snippets with path checking and marker ranges ([c40fb09](https://github.com/Alberto-Codes/typevet/commit/c40fb091e1e0720c3333ce37e861fddd99de6ef1)), closes [#390](https://github.com/Alberto-Codes/typevet/issues/390) [#385](https://github.com/Alberto-Codes/typevet/issues/385)

## [0.7.0](https://github.com/Alberto-Codes/typevet/compare/v0.6.0...v0.7.0) (2026-10-01)


### Features

* **adapters:** expose Retry-After and rate-limit headers on a vLLM HTTP error ([8a273df](https://github.com/Alberto-Codes/typevet/commit/8a273dfddc070b6214e339a1d0db002f49c35d8d)), closes [#355](https://github.com/Alberto-Codes/typevet/issues/355)
* **adapters:** record the request id typevet sends in receipts and errors ([68aab28](https://github.com/Alberto-Codes/typevet/commit/68aab28f77d375b0329482b8d0f123672782cbef)), closes [#356](https://github.com/Alberto-Codes/typevet/issues/356)
* **adapters:** refuse a Gemma 4 framing that lacks the no-thinking prefill ([8edd342](https://github.com/Alberto-Codes/typevet/commit/8edd342ce94299899f7fef3de4fb05b14e450bbc)), closes [#354](https://github.com/Alberto-Codes/typevet/issues/354)
* **domain:** make the option block and the context template substitutable named parts ([2db4a5e](https://github.com/Alberto-Codes/typevet/commit/2db4a5e6a68297cea7fa5f449ea3fe7f0578e8ce)), closes [#364](https://github.com/Alberto-Codes/typevet/issues/364) [#360](https://github.com/Alberto-Codes/typevet/issues/360)
* **evals:** evolve criteria beside instructions through a multi-part wording candidate ([112bca9](https://github.com/Alberto-Codes/typevet/commit/112bca94e3580e7e8147fe16ce23ed87529a4f09)), closes [#363](https://github.com/Alberto-Codes/typevet/issues/363)
* **evals:** pin seed and evolved wording by digest in wording receipts ([ea38912](https://github.com/Alberto-Codes/typevet/commit/ea389127b7e7473dcde80a526cff90ff4881c3f9)), closes [#362](https://github.com/Alberto-Codes/typevet/issues/362) [#360](https://github.com/Alberto-Codes/typevet/issues/360)
* **ports:** carry off_option_threshold through the port protocol and every wrapper ([3df3ff7](https://github.com/Alberto-Codes/typevet/commit/3df3ff770f56e4ea401491f804d382260ac06c2a)), closes [#368](https://github.com/Alberto-Codes/typevet/issues/368) [#353](https://github.com/Alberto-Codes/typevet/issues/353)
* **runtime:** apply a caller-supplied calibration map to Noul answers ([f1a6d6c](https://github.com/Alberto-Codes/typevet/commit/f1a6d6c167f72bfc2a8fa8dc037cdfc1c57fa288)), closes [#352](https://github.com/Alberto-Codes/typevet/issues/352)
* **runtime:** apply a Score calibration map and write map artifacts from receipts ([00f7725](https://github.com/Alberto-Codes/typevet/commit/00f7725db3afc53976b1ae1050bfd8da3e0aee8b)), closes [#352](https://github.com/Alberto-Codes/typevet/issues/352)
* **scoring:** flag an answer whose off-option mass exceeds a caller threshold ([722728b](https://github.com/Alberto-Codes/typevet/commit/722728b31e255b60479dfa5d5b16e9cf70f7b6b7)), closes [#353](https://github.com/Alberto-Codes/typevet/issues/353)


### Fixes

* **evals:** mask an echoed header name used as a JSON key in the acceptance receipt ([bbd1d19](https://github.com/Alberto-Codes/typevet/commit/bbd1d193de921d63362b83eb700334da8162e556)), closes [#351](https://github.com/Alberto-Codes/typevet/issues/351)


### Documentation

* **reference:** list the text parts of a judgment call and their evolution status ([f55a9c0](https://github.com/Alberto-Codes/typevet/commit/f55a9c08ddbac7239239786f8512356f4c7c23e8)), closes [#361](https://github.com/Alberto-Codes/typevet/issues/361) [#360](https://github.com/Alberto-Codes/typevet/issues/360)

## [0.6.0](https://github.com/Alberto-Codes/typevet/compare/v0.5.0...v0.6.0) (2026-09-30)


### Features

* **adapters:** reach a vLLM server behind an API gateway ([58eebce](https://github.com/Alberto-Codes/typevet/commit/58eebce135252b43c9461d7807c095fb7cdfa7dc)), closes [#331](https://github.com/Alberto-Codes/typevet/issues/331)
* **evals:** post-hoc calibration of the committed Noul receipts ([78aea74](https://github.com/Alberto-Codes/typevet/commit/78aea7429308654c4d5011ae104403a5ce4cb5fa)), closes [#343](https://github.com/Alberto-Codes/typevet/issues/343)
* **evals:** record the vLLM server configuration in receipts ([d34c3b9](https://github.com/Alberto-Codes/typevet/commit/d34c3b90b3d6fef5b3b63b15c7e94b981021afc0)), closes [#341](https://github.com/Alberto-Codes/typevet/issues/341)
* **evals:** seed the synthetic check run from the environment ([1dfbc36](https://github.com/Alberto-Codes/typevet/commit/1dfbc36f8b453e86a2d25f8fd4990bec8f21d46d)), closes [#344](https://github.com/Alberto-Codes/typevet/issues/344)


### Fixes

* **adapters:** close the owned scoring client under the lock ([5f5caaa](https://github.com/Alberto-Codes/typevet/commit/5f5caaa8cc580c0f908e4c530f156f30bdfc3e0c)), closes [#345](https://github.com/Alberto-Codes/typevet/issues/345)
* **adapters:** create the lazy scoring client under a lock ([6b44d72](https://github.com/Alberto-Codes/typevet/commit/6b44d7262432f6bd41c88f7ed5884e1063df4e23)), closes [#340](https://github.com/Alberto-Codes/typevet/issues/340)
* **adapters:** mask gateway header values as whole tokens and protect owned headers ([2719ad6](https://github.com/Alberto-Codes/typevet/commit/2719ad6a46fa56ac0a20e5e936fe273e48c2144d)), closes [#348](https://github.com/Alberto-Codes/typevet/issues/348)
* **evals:** one seed rule for the check sheets command and the live run ([6fdc464](https://github.com/Alberto-Codes/typevet/commit/6fdc4642ac003062fd723182be06f3ad8d648b23)), closes [#349](https://github.com/Alberto-Codes/typevet/issues/349)
* **evals:** refuse a wrong split and a two-engine gauge in receipts ([e030ee4](https://github.com/Alberto-Codes/typevet/commit/e030ee4288628db105036c448d79481fe57c15da)), closes [#339](https://github.com/Alberto-Codes/typevet/issues/339)


### Documentation

* **explanation:** record the [#342](https://github.com/Alberto-Codes/typevet/issues/342) design outcome on the throughput page ([9c6afa5](https://github.com/Alberto-Codes/typevet/commit/9c6afa5a5bf957c626f700ef74bffe79796c1731))
* **reference:** receipt blocks page, and a timestamped metrics sample ([2492d95](https://github.com/Alberto-Codes/typevet/commit/2492d95facdb4dc86224b47895ca779923d6ca67)), closes [#347](https://github.com/Alberto-Codes/typevet/issues/347)

## [0.5.0](https://github.com/Alberto-Codes/typevet/compare/v0.4.0...v0.5.0) (2026-09-30)


### Features

* **evals:** concurrency option for the face, check and signature runners ([b287486](https://github.com/Alberto-Codes/typevet/commit/b28748683bc99ba36029f8f2458a9700e7882f77)), closes [#334](https://github.com/Alberto-Codes/typevet/issues/334)
* **evals:** Jev judge option for the wording evolution live test ([d86e70c](https://github.com/Alberto-Codes/typevet/commit/d86e70c052670e43b78d481de82374d58f07af54)), closes [#328](https://github.com/Alberto-Codes/typevet/issues/328)
* **evals:** Jev vs Gemma held-out comparison harness with Cohen's kappa ([d86aa9f](https://github.com/Alberto-Codes/typevet/commit/d86aa9f367a1ec456cad52c2aaf51650877c28cd)), closes [#329](https://github.com/Alberto-Codes/typevet/issues/329)
* **evals:** served-template probe and native Gemma framing for text wording runs ([0628eb0](https://github.com/Alberto-Codes/typevet/commit/0628eb07161cb2a301e63d4819384ccc00b134b3)), closes [#327](https://github.com/Alberto-Codes/typevet/issues/327)
* **evals:** throughput fields in the image receipts ([93a13fb](https://github.com/Alberto-Codes/typevet/commit/93a13fbbf63f043165266ce0eb2752bcfb774c5e)), closes [#335](https://github.com/Alberto-Codes/typevet/issues/335)


### Documentation

* **explanation:** Gemma 4 and Jev on DIFrauD, each with its evolved wording ([44ffb4d](https://github.com/Alberto-Codes/typevet/commit/44ffb4dfaa4e7f1706d2e440e6cb24194f074202)), closes [#330](https://github.com/Alberto-Codes/typevet/issues/330) [#252](https://github.com/Alberto-Codes/typevet/issues/252) [#324](https://github.com/Alberto-Codes/typevet/issues/324)
* **explanation:** image judgment throughput on one H100 ([50834b9](https://github.com/Alberto-Codes/typevet/commit/50834b9e523a17ae44cd40eee692d8d95238fc7f)), closes [#337](https://github.com/Alberto-Codes/typevet/issues/337) [#332](https://github.com/Alberto-Codes/typevet/issues/332)
* **explanation:** latency and cost of the 0.4.0 experiments ([4f66fa0](https://github.com/Alberto-Codes/typevet/commit/4f66fa0e9f693da30961d6672f919fabb3f7e534)), closes [#326](https://github.com/Alberto-Codes/typevet/issues/326)

## [0.4.0](https://github.com/Alberto-Codes/typevet/compare/v0.3.0...v0.4.0) (2026-09-30)


### Features

* **evals:** CEDAR signature pair loader and two-image request builder ([46de94e](https://github.com/Alberto-Codes/typevet/commit/46de94e9a3b211fca63e9e17ccc9f60ac5105bea)), closes [#318](https://github.com/Alberto-Codes/typevet/issues/318)
* **evals:** check-versus-register runner, metrics and opt-in live test ([38cada3](https://github.com/Alberto-Codes/typevet/commit/38cada3ff234b44b16df26224a188014a657a0c1)), closes [#316](https://github.com/Alberto-Codes/typevet/issues/316)
* **evals:** DIFrauD train, validation and held-out splits without [#236](https://github.com/Alberto-Codes/typevet/issues/236) overlap ([d14dcf0](https://github.com/Alberto-Codes/typevet/commit/d14dcf0940666aafc4863714caed9082de3a11e0)), closes [#307](https://github.com/Alberto-Codes/typevet/issues/307)
* **evals:** gepa-adk wording evolution runner with Brier scorer and length cap ([6f9f740](https://github.com/Alberto-Codes/typevet/commit/6f9f7400ba8344390b788fb83035878e7dfe3947)), closes [#308](https://github.com/Alberto-Codes/typevet/issues/308)
* **evals:** harness for the DIFrauD wording evolution and its held-out check ([502ebc9](https://github.com/Alberto-Codes/typevet/commit/502ebc973962a92f4eaeae88098c4fe07bdc89da)), closes [#309](https://github.com/Alberto-Codes/typevet/issues/309)
* **evals:** signature-pair runner, metrics and opt-in live test for CEDAR ([e3e4643](https://github.com/Alberto-Codes/typevet/commit/e3e4643f395f195fb287a75820450b997706f91a)), closes [#319](https://github.com/Alberto-Codes/typevet/issues/319)
* **evals:** synthetic check generator and check-versus-register request builder ([b3d5457](https://github.com/Alberto-Codes/typevet/commit/b3d5457420edcd30e59b51570b71d4ec5c3fc8f6)), closes [#315](https://github.com/Alberto-Codes/typevet/issues/315)
* **evals:** top-label and classwise ECE for Choice and Score above 10 options ([38f9272](https://github.com/Alberto-Codes/typevet/commit/38f927237240d9b8e4f7cd334eab70d117652f97)), closes [#296](https://github.com/Alberto-Codes/typevet/issues/296)
* **scoring:** report off-option probability mass in scoring results ([db5ff12](https://github.com/Alberto-Codes/typevet/commit/db5ff12d2b6c909a1e3390c3a38e02e915eb144b)), closes [#297](https://github.com/Alberto-Codes/typevet/issues/297)


### Fixes

* **adapters:** exact vocabulary size for the llama.cpp off-option completeness check ([81a2e02](https://github.com/Alberto-Codes/typevet/commit/81a2e02f45d50774293f9b6e956c3a26945b5c8e)), closes [#321](https://github.com/Alberto-Codes/typevet/issues/321)
* **adapters:** media-marker refresh edge cases after a model swap ([ce63ddd](https://github.com/Alberto-Codes/typevet/commit/ce63dddd013445066c0768e5e2049de1e6e50f9a)), closes [#323](https://github.com/Alberto-Codes/typevet/issues/323)
* **adapters:** refresh the llama.cpp media marker after a router model reload ([e3dd4cb](https://github.com/Alberto-Codes/typevet/commit/e3dd4cbed8c07f39a6f3a2c6c69952129a584085)), closes [#322](https://github.com/Alberto-Codes/typevet/issues/322)


### Documentation

* **explanation:** check images against a synthetic register and what the typed answers mean ([7299fd6](https://github.com/Alberto-Codes/typevet/commit/7299fd626d3502b2ea2a0d09dfb1add2dad37f83)), closes [#317](https://github.com/Alberto-Codes/typevet/issues/317) [#316](https://github.com/Alberto-Codes/typevet/issues/316) [#303](https://github.com/Alberto-Codes/typevet/issues/303)
* **explanation:** DIFrauD wording evolution and its pre-registered held-out check ([3133817](https://github.com/Alberto-Codes/typevet/commit/3133817497b997b8a4ca4053f1ad7e60a87bf45f)), closes [#309](https://github.com/Alberto-Codes/typevet/issues/309)
* **explanation:** two-image signature comparison on CEDAR and what its probability means ([c418b0c](https://github.com/Alberto-Codes/typevet/commit/c418b0c79b20f103438e103979f5ad810132f9d1)), closes [#320](https://github.com/Alberto-Codes/typevet/issues/320) [#319](https://github.com/Alberto-Codes/typevet/issues/319) [#304](https://github.com/Alberto-Codes/typevet/issues/304)
* **reference:** record the Molmo2-4B single-token control check ([643b68a](https://github.com/Alberto-Codes/typevet/commit/643b68a52ff5911c2a306b1631c4b4b4c17a2a83)), closes [#299](https://github.com/Alberto-Codes/typevet/issues/299)

## [0.3.0](https://github.com/Alberto-Codes/typevet/compare/v0.2.0...v0.3.0) (2026-09-30)


### Features

* **evals:** LFW View 2 pair loader and two-image face-match requests ([dc3d7ca](https://github.com/Alberto-Codes/typevet/commit/dc3d7caec6ceeb83a5462e56119994d101537906)), closes [#300](https://github.com/Alberto-Codes/typevet/issues/300) [#292](https://github.com/Alberto-Codes/typevet/issues/292)


### Fixes

* **adapters:** map errors from the vision factory's /apply-template probe ([64814a0](https://github.com/Alberto-Codes/typevet/commit/64814a08d5ce1a952cdbe6c422ffa83dd5c4168b)), closes [#311](https://github.com/Alberto-Codes/typevet/issues/311)
* **adapters:** map vision tokenizer errors and characterize runtime limits ([597ce88](https://github.com/Alberto-Codes/typevet/commit/597ce88f57784c7f1f7d1d50a3bf73dc69de6e61)), closes [#204](https://github.com/Alberto-Codes/typevet/issues/204) [#298](https://github.com/Alberto-Codes/typevet/issues/298)
* **adapters:** retry one early close on llama.cpp scoring calls ([fdae661](https://github.com/Alberto-Codes/typevet/commit/fdae6619f6f5d8a950f8a5a3018df2345b421b2d)), closes [#305](https://github.com/Alberto-Codes/typevet/issues/305) [#310](https://github.com/Alberto-Codes/typevet/issues/310)
* **evals:** wider PSAI leak check, key-free receipt frames, prompt-read controls ([09f6787](https://github.com/Alberto-Codes/typevet/commit/09f6787f39241b3506aa99f3dedf8f4299aa1a6d)), closes [#295](https://github.com/Alberto-Codes/typevet/issues/295)


### Documentation

* **explanation:** two-image face matching and what its probability means ([cd76f82](https://github.com/Alberto-Codes/typevet/commit/cd76f820f9ad56b0210f532ca03068bb24e49a5d)), closes [#302](https://github.com/Alberto-Codes/typevet/issues/302) [#292](https://github.com/Alberto-Codes/typevet/issues/292)
* **integrations:** explain why the judgevet bridge keeps text-only instructions ([f7560b3](https://github.com/Alberto-Codes/typevet/commit/f7560b37995ed7165236a59ee51d3073133ae8c4)), closes [#291](https://github.com/Alberto-Codes/typevet/issues/291)
* split long sentences in the docs index, CORD smoke and judgevet pages ([4ce5971](https://github.com/Alberto-Codes/typevet/commit/4ce59710dc885f032490d9b55eadd1add9a00b04)), closes [#295](https://github.com/Alberto-Codes/typevet/issues/295)
* state the 0.2.0 release where pages said 0.1.0 is current ([04e8b81](https://github.com/Alberto-Codes/typevet/commit/04e8b819f66c519f3a8ac997d1c3db7b125aca93)), closes [#294](https://github.com/Alberto-Codes/typevet/issues/294)
* **vllm:** record the live KV-cache gauge reading on the tested pin ([02db15d](https://github.com/Alberto-Codes/typevet/commit/02db15d2a0b7e94044407d565af842266fcb0604)), closes [#231](https://github.com/Alberto-Codes/typevet/issues/231)

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

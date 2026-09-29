# CLINC150 domain shard map and non-mapping to Banking77

Kind: reference. This page states the **10×15** domain shard map for
[`clinc/clinc_oos`](https://huggingface.co/datasets/clinc/clinc_oos), the
**CC BY 3.0** license posture, and why CLINC Choice labels **do not** map to
Banking77 intents or finvet **`FRAUD_INTENTS`**.

Parent epic: [#51](https://github.com/Alberto-Codes/typevet/issues/51).
Research baseline: [#66](https://github.com/Alberto-Codes/typevet/issues/66)
(accepted). Loader and manifest coordination: [#58](https://github.com/Alberto-Codes/typevet/issues/58),
rank-5 entry in [Complementary eval manifest](eval-complementary-manifest.md)
and [`evals/complementary-manifest.yaml`](https://github.com/Alberto-Codes/typevet/blob/main/evals/complementary-manifest.yaml).

Upstream intent names live in
[`domains.json`](https://github.com/clinc/oos-eval/blob/master/data/domains.json)
(CLINC OOS eval repo). Hugging Face hosts the `plus` and `small` configs;
this page documents **in-domain** shards only unless noted.

## Role in typevet

| Piece | Status |
|---|---|
| Rank in complementary manifest | 5 — `clinc/clinc_oos` |
| Decision primitive (in-domain) | **Choice**, one domain at a time (**15** labels) |
| Optional OOS track | **Noul** on `plus` OOS rows (loader flag; not v1 default) |
| Default domain param (when loader lands) | **`banking`** — thematic proximity only; not a label map |
| Loader module | **Not implemented** — tracked on [#58](https://github.com/Alberto-Codes/typevet/issues/58) |

CLINC is **complementary** stress for multi-class Choice within a domain. It
does **not** replace JevBench or finvet Banking77 regression ([#53](https://github.com/Alberto-Codes/typevet/issues/53)).

## License

| Source | License | Notes |
|---|---|---|
| CLINC150 / `clinc/clinc_oos` | **CC BY 3.0** | Matches manifest rank 5; confirm dataset card tag when pinning a Hub revision |
| PolyAI Banking77 | CC BY 4.0 | Separate corpus; see [Banking77 proxy and metrics](banking77-proxy-and-metrics.md) |

typevet does **not** vend or download CLINC from this repo. Partner-safe
policy: [Eval partner data policy](eval-partner-data-policy.md).

## Shard geometry (10 domains × 15 intents)

TypeLLM **Choice** enums must stay **≤24** labels per task. CLINC150 defines
**10** topical domains. Each domain lists exactly **15** intent slugs. One
loader call = one domain = one **15-way** Choice.

**Do not merge domains** to “cover more banking topics” in one Choice. Example:
`banking` (15) + `credit_cards` (15) = **30** labels → over the Choice cap.
Keep **`credit_cards`** as its own shard.

### Domain index

| Domain key | Intent count | Example use in typevet |
|---|---:|---|
| `banking` | 15 | Default domain; banking utterances, not Banking77 |
| `credit_cards` | 15 | Separate Choice shard |
| `kitchen_and_dining` | 15 | Non-financial complementary |
| `home` | 15 | Non-financial complementary |
| `auto_and_commute` | 15 | Non-financial complementary |
| `travel` | 15 | Non-financial complementary |
| `utility` | 15 | Non-financial complementary |
| `work` | 15 | Non-financial complementary |
| `small_talk` | 15 | Non-financial complementary |
| `meta` | 15 | Device / assistant control intents |

### `banking` (15 intents)

| Intent slug |
|---|
| `freeze_account` |
| `routing` |
| `pin_change` |
| `bill_due` |
| `pay_bill` |
| `account_blocked` |
| `interest_rate` |
| `min_payment` |
| `bill_balance` |
| `transfer` |
| `order_checks` |
| `balance` |
| `spending_history` |
| `transactions` |
| `report_fraud` |

### `credit_cards` (15 intents)

| Intent slug |
|---|
| `replacement_card_duration` |
| `expiration_date` |
| `damaged_card` |
| `improve_credit_score` |
| `report_lost_card` |
| `card_declined` |
| `credit_limit_change` |
| `apr` |
| `redeem_rewards` |
| `credit_limit` |
| `rewards_balance` |
| `application_status` |
| `credit_score` |
| `new_card` |
| `international_fees` |

### `kitchen_and_dining` (15 intents)

| Intent slug |
|---|
| `food_last` |
| `confirm_reservation` |
| `how_busy` |
| `ingredients_list` |
| `calories` |
| `nutrition_info` |
| `recipe` |
| `restaurant_reviews` |
| `restaurant_reservation` |
| `meal_suggestion` |
| `restaurant_suggestion` |
| `cancel_reservation` |
| `ingredient_substitution` |
| `cook_time` |
| `accept_reservations` |

### `home` (15 intents)

| Intent slug |
|---|
| `what_song` |
| `play_music` |
| `todo_list_update` |
| `reminder` |
| `reminder_update` |
| `calendar_update` |
| `order_status` |
| `update_playlist` |
| `shopping_list` |
| `calendar` |
| `next_song` |
| `order` |
| `todo_list` |
| `shopping_list_update` |
| `smart_home` |

### `auto_and_commute` (15 intents)

| Intent slug |
|---|
| `current_location` |
| `oil_change_when` |
| `oil_change_how` |
| `uber` |
| `traffic` |
| `tire_pressure` |
| `schedule_maintenance` |
| `gas` |
| `mpg` |
| `distance` |
| `directions` |
| `last_maintenance` |
| `gas_type` |
| `tire_change` |
| `jump_start` |

### `travel` (15 intents)

| Intent slug |
|---|
| `plug_type` |
| `travel_notification` |
| `translate` |
| `flight_status` |
| `international_visa` |
| `timezone` |
| `exchange_rate` |
| `travel_suggestion` |
| `travel_alert` |
| `vaccines` |
| `lost_luggage` |
| `book_flight` |
| `book_hotel` |
| `carry_on` |
| `car_rental` |

### `utility` (15 intents)

| Intent slug |
|---|
| `weather` |
| `alarm` |
| `date` |
| `find_phone` |
| `share_location` |
| `timer` |
| `make_call` |
| `calculator` |
| `definition` |
| `measurement_conversion` |
| `flip_coin` |
| `spelling` |
| `time` |
| `roll_dice` |
| `text` |

### `work` (15 intents)

| Intent slug |
|---|
| `pto_request_status` |
| `next_holiday` |
| `insurance_change` |
| `insurance` |
| `meeting_schedule` |
| `payday` |
| `taxes` |
| `income` |
| `rollover_401k` |
| `pto_balance` |
| `pto_request` |
| `w2` |
| `schedule_meeting` |
| `direct_deposit` |
| `pto_used` |

### `small_talk` (15 intents)

| Intent slug |
|---|
| `who_made_you` |
| `meaning_of_life` |
| `who_do_you_work_for` |
| `do_you_have_pets` |
| `what_are_your_hobbies` |
| `fun_fact` |
| `what_is_your_name` |
| `where_are_you_from` |
| `goodbye` |
| `thank_you` |
| `greeting` |
| `tell_joke` |
| `are_you_a_bot` |
| `how_old_are_you` |
| `what_can_i_ask_you` |

### `meta` (15 intents)

| Intent slug |
|---|
| `change_speed` |
| `user_name` |
| `whisper_mode` |
| `yes` |
| `change_volume` |
| `no` |
| `change_language` |
| `repeat` |
| `change_accent` |
| `cancel` |
| `sync_device` |
| `change_user_name` |
| `change_ai_name` |
| `reset_settings` |
| `maybe` |

## Explicit non-mapping to Banking77 and FRAUD_INTENTS

Accepted on [#66](https://github.com/Alberto-Codes/typevet/issues/66#issuecomment-5842004703):

| Topic | CLINC150 | Banking77 / finvet |
|---|---|---|
| Label ontology | **15** slug intents **per domain** | **77** PolyAI intent names |
| typevet Decision on corpus | **Choice** (in-domain) | **Noul-primary** proxy on `reports_unauthorized` ([#53](https://github.com/Alberto-Codes/typevet/issues/53)) |
| Fraud signal | No fraud proxy; `report_fraud` is **one banking intent among 14 others** | Six-intent collapse to **`FRAUD_INTENTS`** → binary proxy |
| Shared loader or enum | **No** — separate loaders ([#58](https://github.com/Alberto-Codes/typevet/issues/58) vs Banking77) | `typevet.eval_banking77` when implemented |
| ID or name alignment | **No** row-level or intent-level map between CLINC slugs and Banking77 categories | Do not reuse Banking77 Choice specs for CLINC shards |
| Thematic overlap | `banking` domain utterances **sound like** bank chat | Overlap is **not** license to merge labels or reuse `fraud_type` enums |

Concrete anti-patterns (forbidden unless a future judgment issue overrides):

- Collapsing CLINC `report_fraud` into finvet **`FRAUD_INTENTS`** or the Banking77 proxy label.
- Building one Choice enum that mixes CLINC banking intents with Banking77’s 77 categories.
- Treating CLINC **`banking`** Choice accuracy as a substitute for Banking77 Noul agreement.
- Reusing finvet **`fraud_type`** Choice labels as the CLINC in-domain gold schema.

When both corpora appear in eval docs, state **which loader**, **which primitive**, and **which label set** applied to each row.

## Optional OOS (Noul)

The `plus` configuration adds out-of-scope (OOS) utterances. typevet may expose
an optional **Noul** (in-scope vs OOS) on those rows. That track is **orthogonal**
to Banking77 fraud proxy and to in-domain 15-way Choice. Default loader behaviour
and CI fixtures remain **TBD** on child issues from [#66](https://github.com/Alberto-Codes/typevet/issues/66).

## Open questions (from research)

- Cross-domain negatives in one batch vs one-domain fixtures.
- `plus` vs `small` for default CI cost.
- Manifest row shape once [#58](https://github.com/Alberto-Codes/typevet/issues/58) lands.

## Related typevet pages

- [Banking77 proxy and metrics](banking77-proxy-and-metrics.md) — six-intent proxy; Noul-primary; **`FRAUD_INTENTS`**.
- [Complementary eval manifest](eval-complementary-manifest.md) — rank 5 slot and JevBench-primary rule.
- [Glossary](glossary.md) — **CLINC domain shard**, **Banking77 proxy label**.

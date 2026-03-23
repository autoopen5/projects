# CEO HEARTBEAT - 3D Print Launch (Moscow, budget <= 1000 EUR)

## 1) Profitable niche shortlist (with products)

### Niche A: Auto interior clips and holders (fast replacement parts)
- Products: phone mount adapters, vent clips, cable organizers, cup holder inserts, trunk hooks.
- Unit economics: low material cost, high urgency value (broken part -> immediate need).
- Typical ticket: 700-2500 RUB.

### Niche B: Home organization for small apartments
- Products: drawer dividers, wall hooks, charging dock, bathroom shelf brackets, kitchen organizers.
- Unit economics: high repeat potential and bundle sales.
- Typical ticket: 800-3500 RUB.

### Niche C: Custom hobby and gaming accessories
- Products: board game inserts, miniature stands, controller mounts, cosplay small parts.
- Unit economics: customization premium and community referrals.
- Typical ticket: 1200-5000 RUB.

## 2) Demand and competition (Moscow + online)

- Local demand signal: city scale + rapid delivery expectations favor nearby manufacturing.
- Online competition: many generic sellers compete on price; fewer sellers offer fast custom fit + same/next-day production.
- Best entry edge for 1 printer: narrow custom jobs with quick turnaround, not mass catalog.

Conclusion: prioritize "auto interior clips/holders" for immediate sales velocity.

## 3) Chosen niche for fast start

Chosen: Auto interior clips and holders.

Why:
- Clear pain point and urgency.
- Small parts -> short print cycles -> better throughput on one printer.
- Easy upsell: set of 2-5 pieces, custom fit service.

## 4) MVP product

MVP offer:
- "Custom car holder/clip in 24 hours"
- Input from client: car model/year + photo + dimensions.
- Output: ready printed part, pickup or courier.

Starter SKU pack:
- P1: Universal cable clip set (3 pcs)
- P2: Vent adapter for phone mount
- P3: Trunk organizer hooks (2 pcs)

## 5) Sales funnel -> Telegram bot

- Traffic sources:
  - Avito listings with clear photos + "24h custom print" promise.
  - VK local groups (car owners, district communities).
  - Telegram district chats and thematic auto channels.
  - Short timelapse videos (Reels/Shorts/VK Clips).
- Entry point:
  - Every post/listing leads to bot deep link with UTM tag.
- Conversion in bot:
  - User selects product type -> submits car model/photo/dimensions -> sees price range -> confirms order -> pays deposit -> receives ETA.

## 6) Telegram bot design

### Dialogue scenario
1. Greeting + value: "Печать автодеталей и креплений за 24 часа".
2. Choice: "Готовое" or "Нужно под мой размер".
3. Data collection:
- car brand/model/year
- part purpose
- dimensions (or upload photo with ruler)
- material preference (PETG default)
- quantity
4. Pricing logic output:
- base fee + material volume + complexity + urgency + delivery.
5. Order confirmation:
- summary card, deposit request, promised deadline.
6. Production updates:
- "modeling", "printing", "post-processing", "ready".

### Questions bot must ask
- What car model/year?
- What problem should part solve?
- Exact dimensions or photo with scale?
- Heat/water exposure expected?
- Needed by date?
- Pickup or delivery?

### Price formula (MVP)
- Base setup: 300 RUB
- Print cost: 12 RUB per gram
- Complexity coefficient: 1.0 / 1.25 / 1.5
- Urgency surcharge (<24h): +30%
- Post-processing: +150-400 RUB
- Delivery: pass-through or fixed zone fee

Final = (base + grams*12)*complexity + urgency + post-processing + delivery

### Order handling logic
- If dimensions missing -> bot asks for photo template.
- If tolerance critical -> manual review queue.
- If requested ETA impossible -> auto propose next slot + discount coupon.

## 7) Ads and content templates

### Avito ad text
Title: "3D печать автокреплений и деталей за 24 часа"
Body: "Сломался клипс или крепление? Напечатаю под ваш размер. PETG/PLA. Фото + размеры -> расчет за 5 минут. Самовывоз/доставка по Москве. Пишите в Telegram-бот: <link>."

### VK post
"Москва, делаю 3D-печать автомелочей: крепления, клипсы, держатели. Быстрый расчет в Telegram-боте, готовность от 24 часов. Нужны модель авто + фото детали. Первым 10 клиентам скидка 15%."

### 30s timelapse script
- 0-3s: broken clip in hand (pain point)
- 3-8s: CAD quick view
- 8-20s: printer timelapse
- 20-25s: fitting in car
- 25-30s: CTA text "Напиши в бот, расчет за 5 минут"

## 8) Operational model

- Intake: all leads through bot -> CRM table.
- Validation: dimensions check + manufacturability score.
- Queue:
  - Priority 1: paid urgent orders
  - Priority 2: standard prepaid
  - Priority 3: unpaid draft requests
- Production windows:
  - Morning: prepare/slice
  - Day: main print batch
  - Evening: QA + packaging + dispatch
- SLA:
  - quote <= 15 min (bot auto + manual fallback)
  - production 24-48h for standard parts

## 9) Concrete 3-day execution plan

### Day 1
- Launch basic Telegram bot flow (lead intake + pricing + order confirmation).
- Create 3 Avito listings with different angle (urgent repair, custom fit, bundle).
- Shoot one raw timelapse and publish short video.

### Day 2
- Add payment/deposit step and order status notifications.
- Publish 5 posts in VK/Telegram local channels.
- Process first inbound requests manually with bot-assisted quoting.

### Day 3
- Optimize pricing by real print times.
- Add upsell bundles in bot.
- Collect first testimonials/photos and update ads.

## 10) Task breakdown for Paperclip automation

- Task A: Build Telegram bot MVP (state machine + validation + quote calculator).
- Task B: Set up lead sheet/mini-CRM + status sync from bot.
- Task C: Generate ad variants (Avito/VK/Telegram) with A/B hooks.
- Task D: Build queue scheduler (simple priority + ETA estimator).
- Task E: Daily analytics digest (leads, conversions, avg чек, cycle time).

## 11) Priorities

1. Bot intake + pricing (must-have).
2. Traffic acquisition (Avito + local communities).
3. Reliable 24-48h fulfillment.
4. Content loop (timelapse + customer proofs).

## 12) Clarifying questions before deeper implementation

1. Do we accept only Moscow pickup/courier now, or ship to all Russia from week 1?
2. Which payment rails are allowed in week 1 (card transfer, SBP, online acquiring)?
3. Are we comfortable focusing only on auto parts for first 7 days (no niche mixing)?
4. Target daily order count for week 1: 3, 5, or 10?

## 13) Board Decisions (from latest comment)

- Shipping scope: all-Russia shipping is allowed from week 1.
- Budget: up to 10,000 USD (updated from the original smaller cap).
- Payments: all payment methods and spend approvals are controlled by board owner.

## 14) Expected Income While Working Part-Time

Assumptions (single printer, part-time operations):
- Working capacity: 2-4 orders/day average.
- Average order value: 1,500-3,000 RUB.
- Gross margin after materials/packaging: 55-70%.

Expected monthly range (30 days):
- Conservative: 2 orders/day * 1,500 RUB = 90,000 RUB revenue/month.
  - Estimated net after variable costs: ~50,000-60,000 RUB.
- Base case: 3 orders/day * 2,000 RUB = 180,000 RUB revenue/month.
  - Estimated net after variable costs: ~100,000-125,000 RUB.
- Strong week-over-week execution: 4 orders/day * 2,500 RUB = 300,000 RUB revenue/month.
  - Estimated net after variable costs: ~165,000-210,000 RUB.

Realistic target while keeping another job:
- First month target: 120,000-200,000 RUB revenue.
- First month likely net: 70,000-130,000 RUB.

## 15) Immediate Next Step

- Lock payment flow approval policy with board owner.
- Launch bot intake + quote flow + deposit confirmation.
- Start with 3 active channels: Avito, VK local groups, Telegram local channels.

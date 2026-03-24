# 3D Print Telegram Bot MVP

This repository now contains a runnable Telegram bot MVP:
- lead intake (10-step questionnaire)
- instant quote calculation
- AI model draft generation for client approval (Tripo optional, OpenAI optional, local fallback enabled)
- deposit request and screenshot capture
- status tracking (`/status`)
- model approval command (`/approve <ORDER_ID>`)
- printer dispatch command (`/dispatch <ORDER_ID>`) for admin flow
- admin status updates (`/setstatus`)
- local JSON persistence in `data/`

## Files
- `telegram_bot_mvp.py` - bot implementation (no third-party Python packages)
- `.env.example` - required environment variables
- `data/orders.json` - generated at runtime
- `data/sessions.json` - generated at runtime

## Requirements
- Python 3.10+
- Telegram bot token from @BotFather

## Setup
1. Create your Telegram bot via @BotFather and copy token.
2. Copy env template and fill values:
   - `TELEGRAM_BOT_TOKEN` (required)
   - `ADMIN_CHAT_ID` (optional but recommended, your personal/admin chat id)
   - `PAYMENT_INSTRUCTIONS` (optional custom text)
   - `TRIPO_API_KEY` (optional; preferred constructor path if provided)
   - `TRIPO_API_URL` (optional; default `https://api.tripo3d.ai/v2/openapi/task`)
   - `TRIPO_MODEL_VERSION` (optional; default `auto`)
   - `TRIPO_POLL_ATTEMPTS` (optional; default `8`)
   - `PRINTER_DISPATCH_WEBHOOK_URL` (optional; printer dispatcher endpoint)
   - `PRINTER_DISPATCH_TOKEN` (optional; bearer auth for dispatcher)
   - `AUTO_DISPATCH_ON_APPROVAL` (optional; `true/false`, default `false`)
   - `OPENAI_API_KEY` (optional fallback; if omitted and no Tripo key, bot uses local heuristic constructor)
   - `OPENAI_MODEL` (optional; default `gpt-5-mini`)
   The bot auto-loads variables from local `.env` (if present).
3. Run:

```powershell
$env:TELEGRAM_BOT_TOKEN="<your_token>"
$env:ADMIN_CHAT_ID="<your_chat_id>"
$env:PAYMENT_INSTRUCTIONS="Transfer 50% deposit to <bank/card>. Send screenshot here."
python .\telegram_bot_mvp.py
```

## Usage
- Customer starts: `/start` or `/start avito`
- Customer checks status: `/status ORD-0001`
- Customer approves model: `/approve ORD-0001` (or sends `СОГЛАСОВАНО` in active session)
- Admin dispatches approved order to printer webhook: `/dispatch ORD-0001`
- Admin updates status: `/setstatus ORD-0001 PRINTING Started at 15:00`

Allowed statuses:
- `REQUEST_RECEIVED`
- `REVIEWING_DETAILS`
- `WAITING_DEPOSIT`
- `DEPOSIT_UNDER_REVIEW`
- `PRINTING`
- `POST_PROCESSING`
- `READY_FOR_PICKUP`
- `SHIPPED`
- `CANCELLED`

## Notes
- Bot uses long polling (no webhook setup required).
- Orders and sessions are persisted to `data/`.
- Keep the process running (tmux/screen/service) for continuous operation.

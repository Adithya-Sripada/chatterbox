# chatterbox

A Small Language Model that learns how you chat on Twitch and suggests messages you'd say,
including live, based on what's happening in a channel's chat right now. It only suggests
and never posts to Twitch. Runs on Apple Silicon with MLX.

## Setup (once)
1. Put your Twitch username in `config.toml` (`username = "..."`).
2. Copy your chat logs into `data/raw/` (any folder layout; full-channel logs are best).
   Supported: Chatterino, justlog/rustlog, raw IRC, TwitchDownloader JSON/text,
   chat-downloader JSON(L), CSV, and .gz versions of those.

## Run
```bash
.venv/bin/python src/parse_logs.py         # finds your messages + the chat before them
.venv/bin/python src/build_dataset.py --show 10   # filters, splits, prints stats + samples
./train.sh                                  # LoRA fine-tune (20-60 min on M2)
./train.sh eval                             # test loss: base model vs. your adapter
.venv/bin/python src/try_it.py --blind 20   # can you tell your real message from the model's?
.venv/bin/python src/live.py <channel>      # live suggestions: Enter = suggest, q = quit
```
`live.py` options: `--auto 20` refreshes every 20 s while chat moves; `--show-chat` prints chat as it arrives.

Everything (logs, dataset, adapter) stays on this Mac. Other chatters' messages are only
used as context, never as things the model learns to say.

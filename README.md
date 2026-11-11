# chatterbox

A Small Language Model trained on my own chatlogs that suggests messages based on what's happening in a channel's chat right now.

## Setup (once)
1. Create `config.local.toml` with your Twitch username (`username = "..."`). It overrides `config.toml` and is not committed.
2. Copy your chat logs into `data/raw/` (any folder layout; full-channel logs are best).
   Supported: Chatterino, justlog/rustlog, raw IRC, TwitchDownloader JSON/text,
   chat-downloader JSON(L), CSV, and .gz versions of those.

## Run
```bash
.venv/bin/python src/parse_logs.py         # finds your messages + the chat before them
.venv/bin/python src/build_dataset.py --show 10   # filters, splits, prints stats + samples
./train.sh                                  # LoRA fine-tune (20-60 min on M2)
./train.sh eval                             # test loss: base model vs. your adapter (takes ~1 minute)
.venv/bin/python src/try_it.py --blind 20   # can you tell your real message from the model's?
.venv/bin/python src/live.py <channel>      # live suggestions: Enter = suggest, q = quit
```
`live.py` options: `--auto 20` refreshes every 20 s while chat moves; `--show-chat` prints chat as it arrives.

The test set used by `train.sh eval` and the blind test is the newest stretch of your
messages that's at least `test_min_age_days` old (set in `config.toml`), so you aren't
just recognizing chats you remember. Your newer messages still go into training. After
rebuilding the dataset, run `./train.sh` again before testing, or the old adapter may
already have seen the test messages.

Everything (logs, dataset, adapter) stays on this Mac. Other chatters' messages are only
used as context, never as things the model learns to say.

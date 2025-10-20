# Screen Chess Bot

Screen Chess Bot is a desktop automation agent that uses computer vision to read an on-screen chess board, keeps a python-chess game state in sync, and queries Stockfish for move suggestions or automated play. It supports both fully automated moves and a “suggestion” workflow where the bot draws Stockfish’s idea directly on the board. I originally wrote it to spare myself from blundering blitz games while testing openings on the fly.

## Features
- Warp-and-read any on-screen chess board using YOLO or template detection.
- Keeps an internal `python-chess` board synchronized with detections.
- Integrates with Stockfish for best-move search (skill level configurable).
- Two operating styles:
  - **Auto play**: Bot moves the pieces for you.
  - **Suggest only**: Bot highlights the best move and waits for you.
- Optional single-move hotkeys:
  - Press `d` to draw a move preview without acting.
  - Press `s` to have the bot play a single move automatically.
- Windows-native arrow overlay (no extra OpenCV window).

## Prerequisites
- Python 3.9+
- Stockfish binary available on your PATH or supplied via `--engine-path`
- Windows 10/11 (for on-screen arrow rendering; other platforms fall back to mouse-based previews)
- Torch with CUDA is recommended.
- Requires a Pre-Trained chess pieces detector model thats compatible with ultralytics. (Easy to train using roboflow datasets.)
  

## Quick Start
1. **Create & activate a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # PowerShell / cmd on Windows
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the bot**
   ```bash
   python screen_chess_bot.py --move-time-ms 1500 --depth 12 --color auto
   ```
4. Follow the console instructions to capture the board region (click TL, TR, BR, BL).
5. Press `q` in the console at any time to terminate.

## Useful CLI Flags
| Flag | Description |
|------|-------------|
| `--move-time-ms` | Search time per move (milliseconds). |
| `--depth` | Fixed search depth (used when `--move-time-ms` is omitted). |
| `--color {auto,white,black}` | Side you are playing. |
| `--show-move-only` | Draw previews and wait for you to move manually. |
| `--single-move-hotkeys` | Enables hotkeys (`d` suggest, `s` play once). |
| `--engine-path` | Path to a Stockfish binary if it is not on PATH. |
| `--yolo-weights` | Path to YOLO weights for piece detection (required). |

Run `python screen_chess_bot.py --help` to see all options.

## Development Notes
- Local artifacts such as `debug/`, `board_coordinates.json`, and bytecode caches are ignored via `.gitignore`.
- The project intentionally avoids shipping a compiled Stockfish binary—download one appropriate for your platform if needed.
- Feel free to adapt the bot for your favourite chess site; detection works best when the board is fully visible and unobstructed.

## License
This project is released under the MIT License. See [LICENSE](LICENSE) for details.

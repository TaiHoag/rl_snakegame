# AI SnakeGame

Simple reinforcement-learning Snake game demo (Python).

## Requirements

- Python 3.8+
- Optional: `pygame`, `torch`, `numpy`

## Quick start

- Create and activate a virtual environment:

  ```bash
  python -m venv .venv
  # Windows
  .venv\Scripts\activate
  # macOS / Linux
  source .venv/bin/activate
  ```

- Install optional dependencies:

  ```bash
  pip install --upgrade pip
  pip install pygame torch numpy
  ```

- Run the playable game:

  ```bash
  python snakegame.py
  ```

- Run the AI/training script:

  ```bash
  python ai_snakegame_v5.py
  ```

## Project layout

- `ai_snakegame_v5.py` — latest AI/training script
- `snakegame.py` — game environment / playable version
- `checkpoints/` — saved models (ignored by git)

## Notes

- Use `checkgpu.py` to verify GPU availability.

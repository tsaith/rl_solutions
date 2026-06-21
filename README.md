# Reinforcement Learning Solutions

This repository contains Python solutions, experiments, and notes for exercises from
Richard S. Sutton and Andrew G. Barto's *Reinforcement Learning: An Introduction*
(2nd edition).

## Project Structure

- `chapter_2/`: Multi-armed bandit exercise programs and generated figures.
- `chapter_3/`: Notes and chapter-specific scratch work.
- `chapter_4/`: Dynamic programming examples and exercises, including Jack's Car Rental.

## Requirements

The project targets Python 3.12 or newer and uses `pyproject.toml` plus `uv.lock` for
dependency management.

Main runtime dependencies include:

- `jupyterlab`
- `matplotlib`
- `scipy`
- `tqdm`

Install the project environment with:

```bash
uv sync
```

## Running Programs

Run a script with `uv run python` from the repository root.

Examples:

```bash
uv run python chapter_2/ex2.4.py
uv run python chapter_2/ex2.11.py
uv run python chapter_4/ex4.7.py
```

## Notes

Some experiments intentionally use reduced step counts or truncated probability
distributions to keep runtime manageable. Check the constants at the top of each
script before comparing numerical results with textbook figures.

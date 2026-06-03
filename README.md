# Reinforcement Learning Solutions

這個專案用來編寫與儲存 Richard S. Sutton 和 Andrew G. Barto 所著
《Reinforcement Learning: An Introduction》的練習題解答。

## 專案結構

- `chapter_2/`: 第二章 Multi-armed Bandits 的練習題程式。
- `notebooks/`: 互動式筆記與實驗記錄。
- `main.py`: 專案預設入口檔案。

## 環境需求

本專案使用 Python 3.12 以上版本，並透過 `pyproject.toml` 管理相依套件。

主要套件：

- `numpy`
- `matplotlib`
- `jupyterlab`

如果使用 `uv`，可以直接執行：

```bash
uv sync
```

## 執行練習題

例如執行第二章練習 2.4：

```bash
uv run python chapter_2/2_4.py
```

程式會顯示 matplotlib 圖表，並把圖片儲存在同一個資料夾：

```text
chapter_2/2_4.png
```

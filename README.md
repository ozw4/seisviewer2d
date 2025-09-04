# seisviewer2d

`seisviewer2d` is a small FastAPI application for exploring 2D seismic data stored in SEG-Y files. It ships with a minimal HTML/JavaScript frontend that lets you upload a file and view sections interactively with Plotly.

## Project Layout

```
app/
  api/endpoints.py   # API routes for uploads and section retrieval
  utils/utils.py     # SEG-Y reading helpers
  static/            # Frontend assets (HTML/JS)
  main.py            # FastAPI application
Dockerfile           # Dev container configuration
ruff.toml            # Ruff linter settings
```

## API Overview
- `POST /upload_segy`: upload a SEG-Y file and begin background loading.
- `GET /get_key1_values`: list available values for the first key.
- `GET /get_section` and `GET /get_section_bin`: fetch normalized traces for a given `key1` index (JSON or binary).
- `POST /fbpick_section_bin`: infer first-arrival probability for the current section.

## Keyboard Shortcuts

- `N`: Toggle between raw and denoised display modes.
- `P`: Toggle first-arrival probability overlay.

## Development
1. Install dependencies (FastAPI, Uvicorn, NumPy, segyio, msgpack, etc.).
2. Launch the server: `uvicorn app.main:app --reload`.
3. Open `http://localhost:8000/upload` to select a file and key bytes. After uploading, you are redirected to the viewer page.

### FBPick Probability Overlay

To view first-arrival probabilities, place the model weights at `model/fbpick_edgenext_small.pth` and enable the **FBPICK prob** toggle in the viewer. An opacity slider controls the overlay transparency.

This project is intended as a starting point for building interactive seismic viewers.

# Agent Instructions

## Goal
Maintain the FastAPI-based viewer that visualizes 2D seismic data from SEG-Y files via a simple web frontend.

## Guidelines
- Use asynchronous tasks or background threads to keep SEG-Y processing responsive.
- Cache loaded sections to minimize repeated disk access.
- Place new API routes under `app/api` and helper utilities under `app/utils`.
- Store static assets in `app/static`.

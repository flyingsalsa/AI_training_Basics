# AI Basics: Semantic Similarity & AI Development Tools

## Overview

This project is a beginner-friendly introduction to AI development, focusing on:
- Setting up Python environments with Poetry
- Managing dependencies with Poetry and UV
- Version control with Git and GitHub
- Containerizing AI projects with Docker
- Building and serving a semantic similarity API using FastAPI and Sentence Transformers

You will learn how to train models, serve them via an API, and interact with them through a modern web interface.

---

## Features

- **Semantic Similarity API**: REST API for generating text embeddings and calculating semantic similarity using Sentence Transformers.
- **Web Interface**: User-friendly HTML frontend for comparing text similarity and generating embeddings.
- **Model Training**: Example scripts for training and saving a Random Forest classifier on the Iris dataset.
- **Containerization**: Docker and Docker Compose support for reproducible environments.
- **Development Guide**: Step-by-step Jupyter notebook for setting up and managing your AI project.

---

## Project Structure

```
.
├── src/ai_training/
│   ├── embedding_service.py   # FastAPI app for semantic similarity
│   ├── train_model.py         # Example: train and save a classifier
│   └── ...                    # Other scripts and modules
├── templates/
│   ├── index.html             # Web UI for the API
│   └── styles.css             # Styling for the web UI
├── AI_Development_Tools_Guide.ipynb  # Step-by-step setup and usage guide
├── Dockerfile                 # Container for Jupyter/Poetry
├── pyproject.toml             # Poetry project config
├── requirements.txt           # Exported dependencies
└── README.md                  # (You are here)
```

---

## Quickstart

### 1. Install Poetry

```sh
curl -sSL https://install.python-poetry.org | python3 -
poetry --version
```

### 2. Install Dependencies

```sh
poetry install
```

### 3. Train Example Model

```sh
poetry run python src/ai_training/train_model.py
```

### 4. Run the Semantic Similarity API

```sh
poetry run uvicorn src.ai_training.embedding_service:app --reload
```

- Visit [http://localhost:8000](http://localhost:8000) for the web UI.
- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Using Docker

Build and run the Jupyter environment:

```sh
docker build -t ai-basics .
docker run -p 8888:8888 -v $(pwd):/app ai-basics
```

Or use Docker Compose for both Jupyter and API services:

```sh
docker-compose up --build
```

---

## Example API Endpoints

- `POST /embedding` — Get embedding for a single text
- `POST /batch_embedding` — Get embeddings for a list of texts
- `POST /similarity` — Compute semantic similarity between two texts

See [`src/ai_training/embedding_service.py`](src/ai_training/embedding_service.py) for details.

---

## Development Guide

See [`AI_Development_Tools_Guide.ipynb`](AI_Development_Tools_Guide.ipynb) for:
- Poetry/UV setup
- Dependency management
- Git & GitHub workflow
- Docker/Docker Compose usage
- Example scripts and best practices

---

## Requirements

- Python 3.12+
- Poetry
- Docker (optional)
- See [`pyproject.toml`](pyproject.toml) for all dependencies

---

## License

MIT License

---

## Credits

- Built with [FastAPI](https://fastapi.tiangolo.com/), [Sentence Transformers](https://www.sbert.net/), [Poetry](https://python-poetry.org/), and [Docker](https://www.docker.com/).
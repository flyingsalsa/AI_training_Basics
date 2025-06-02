# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install Poetry
RUN pip install poetry==1.6.1

# Configure Poetry to not create a virtual environment
RUN poetry config virtualenvs.create false

# Copy poetry configuration files
COPY pyproject.toml poetry.lock* /app/

# Install dependencies
RUN poetry install --no-interaction --no-ansi --no-dev

# Copy the rest of the application
COPY . /app/

# Make port 8888 available for Jupyter
EXPOSE 8888

# Run Jupyter when the container launches
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"] # poetry run python src/app.py
# Use python 3.9-slim as the base image for a lightweight production environment
FROM python:3.9-slim

# Set environment variables to prevent Python from writing .pyc files 
# and to ensure output is sent directly to the terminal
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's cache for dependencies
COPY requirements.txt .

# Install production dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Explicitly copy the application source code and the MLflow runs
# Note: In a production setting, you might use a model registry, 
# but for this setup, we include the local mlruns folder.
COPY src/ ./src/
COPY mlruns/ ./mlruns/

# Expose the port the FastAPI app will run on
EXPOSE 8000

# Start the FastAPI application using uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

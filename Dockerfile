# Use an official Python runtime
FROM python:3.10-slim

# Set working directory
WORKDIR /code

# Install dependencies
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Create a non-root user (Hugging Face requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory for application code
WORKDIR $HOME/app

# Copy project files and ensure owner permissions
COPY --chown=user . $HOME/app

# Expose port 7860 (default port for Hugging Face Spaces)
EXPOSE 7860

# Command to run FastAPI server via Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
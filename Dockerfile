# Use official Python base image
FROM python:3.13-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements.txt separately to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Set environment variable (optional)
ENV PYTHONUNBUFFERED=1

# Default command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]

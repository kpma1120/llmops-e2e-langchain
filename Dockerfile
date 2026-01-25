FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set environment variable (optional)
ENV PYTHONUNBUFFERED=1

# Default command to run the Streamlit app
# Bind to $PORT if provided (Cloud Run), else default to 8501 (local)
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]

# ----------------------------
# 1) Base image
# ----------------------------
FROM python:3.10-slim

# ----------------------------
# 2) Prevent python from buffering logs
# ----------------------------
ENV PYTHONUNBUFFERED=1

# ----------------------------
# 3) Set working directory
# ----------------------------
WORKDIR /app

# ----------------------------
# 4) Install system dependencies
# ----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# 5) Copy project files
# ----------------------------
COPY . .

# ----------------------------
# 6) Install Python dependencies
# ----------------------------
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------
# 7) Expose the port FastAPI runs on
# ----------------------------
EXPOSE 8000

# ----------------------------
# 8) Run FastAPI using uvicorn
# ----------------------------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

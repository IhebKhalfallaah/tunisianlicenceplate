FROM python:3.10.6-slim

WORKDIR /server
ENV BACKEND_ENV=production

# Install dependencies first so they are cached between source changes
COPY requirements.txt .
RUN python -m venv /server/env \
    && /server/env/bin/pip install --no-cache-dir --upgrade pip setuptools \
    && /server/env/bin/pip install --no-cache-dir -r requirements.txt --default-timeout=120

COPY . .

EXPOSE 5000
CMD ["/server/env/bin/python", "app.py"]

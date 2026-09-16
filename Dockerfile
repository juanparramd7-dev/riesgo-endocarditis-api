FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY modelo_b64.txt .
RUN python -c "import base64; open('modelo_endocarditis_exploratorio.joblib','wb').write(base64.b64decode(open('modelo_b64.txt','r').read()))"

EXPOSE 7860

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}

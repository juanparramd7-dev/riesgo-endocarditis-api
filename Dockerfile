FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY chunk_00.txt chunk_01.txt chunk_02.txt chunk_03.txt chunk_04.txt chunk_05.txt chunk_06.txt chunk_07.txt chunk_08.txt chunk_09.txt chunk_10.txt chunk_11.txt chunk_12.txt ./
RUN cat chunk_00.txt chunk_01.txt chunk_02.txt chunk_03.txt chunk_04.txt chunk_05.txt chunk_06.txt chunk_07.txt chunk_08.txt chunk_09.txt chunk_10.txt chunk_11.txt chunk_12.txt > modelo_b64.txt && \
    python -c "import base64; open('modelo_endocarditis_exploratorio.joblib','wb').write(base64.b64decode(open('modelo_b64.txt','r').read()))"

EXPOSE 7860

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}

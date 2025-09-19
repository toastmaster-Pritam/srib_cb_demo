FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    curl \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x ./wait_for_artifacts.sh ./wait_for_synthetic.sh

ENV MPLCONFIGDIR = /tmp/.matplotlib

EXPOSE 8501

CMD ["bash","-lc","streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]
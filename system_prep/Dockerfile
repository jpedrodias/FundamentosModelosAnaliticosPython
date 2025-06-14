FROM python:3.12-bookworm

WORKDIR /app

COPY ../requirements.txt /app/

# security
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser

RUN apt update -y && \
    apt upgrade -y && \
    apt install -y nano dos2unix cron && \
    apt autoclean -y && \
    apt autoremove -y

RUN python -m pip install pip --upgrade --no-cache-dir

RUN python -m pip install -r requirements.txt --no-cache-dir

# CMD ["python", "app.py"]
CMD ['bash']
FROM python:3.12-bookworm

WORKDIR /app

COPY requirements.txt /app/

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

USER appuser


# CMD ["python", "app.py"]
CMD ["bash"]


# docker build -t my-python-docker .
# docker run -it --rm my-python-docker

# WINDOWS To run the container with a mounted volume for development:
# docker run -it --rm -v $(pwd):/app my-python-docker

# Linux/macOS To run the container with a mounted volume for development:
# docker run -it --rm -v $(pwd):/app my-python-docker

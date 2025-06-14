FROM python:3.12-bookworm

WORKDIR /app

COPY requirements.txt /app/

# Install linux system dependencies
RUN apt update -y && \
    apt upgrade -y && \
    apt install -y nano dos2unix cron bash-completion && \
    apt autoclean -y && \
    apt autoremove -y && \
    sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' /etc/skel/.bashrc

# Install Python dependencies
RUN python -m pip install pip --upgrade --no-cache-dir && \
    python -m pip install -r requirements.txt --no-cache-dir

# Security - run as non-root user
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup --home /home/appuser appuser && \
    cp /etc/skel/.bashrc /home/appuser/.bashrc && \
    chown -R appuser:appgroup /home/appuser && \
    chown -R appuser:appgroup /app
    

USER appuser
ENV HOME=/home/appuser


# CMD ["python", "app.py"]
CMD ["/bin/bash"]


# docker build -t my-python -f system_prep/Dockerfile .
# docker run -it --rm my-python

# WINDOWS To run the container with a mounted volume for development:
# docker run -it --rm -v ${PWD}:/app my-python

# Linux/macOS To run the container with a mounted volume for development:
# docker run -it --rm -v $(pwd):/app my-python

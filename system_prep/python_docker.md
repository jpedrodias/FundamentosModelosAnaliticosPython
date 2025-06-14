# Preparação do sistema para usar venv ou docker
## Correr Python dentro do Docker


**Vantagens:**
* Zero alterações ao sistema;
* Zero problemas de "it runs on my machine..."


Desvantagens:
* Maior "pegada" em disco;
* First run mais lento.


Outros `sabores`:
* alpine (pegada muito pequena)
* bookworm (maior que o alpine mas com mais suporte para software de terceiros)
* windowsservercore (nunca testei)

Mais `TAGS` (versões) em https://hub.docker.com/_/python



* * *
* * *


# Opção 1: Download & Run
Simples download and run. Etapas seguintes são todas manuais.

**Download:**

```bash
docker pull python:3.12-alpine
```

**Run:**
```bash
docker run -it --rm --entrypoint sh -w /app -v ${PWD}:/app -p 5000:5000 python:3.12-alpine
```
* A opção de `-entrypoint`  faz com que seja iniciada a shell depois do arranque do contentor
* A opção `-w` cria uma pasta de trabalho no contentor
* A opção `-p` expõe a porta ou exterior do contentor
* A opção `-v` liga a pasta atual à pasta de trabalho dentro do contentor.


**Install:**

Após arranque, é necessário instalar alguns requisitos:

```bash
apk update && apk add gcc
pip install -r requirements.txt
```


* * *
* * *


# Opção 2: Criar a própria image
De forma a evitar os passos para a instalação dos requisitos, é possível construir a própria imagem já todos os requisitos instalados.


**`Dockerfile`**:
O ficheiro Dockerfile tem todas as configurações para a construção da imagem 

O processo de `build` é demorado mas só precisa ser feito apenas uma vez.

```bash
docker build -t my-python -f system_prep/Dockerfile .
```


**Correr a imagem criada em Windows:**
```bash
docker run -it --rm -v ${PWD}:/app --entrypoint sh my-python
```

**Correr a imagem criada em Linux/macOS:**
```bash
docker run -it --rm -v ${pwd}:/app  --entrypoint sh my-python
```
# Preparação do sistema para usar venv ou docker

# Windows: 

## Configuração inicial do Ambiente Virtual 
```bash
python -m venv C:\TEMP\FormacaoModelos --prompt "Fundamentos de Modelos Analíticos"
```

**Remover as restrições de segurança do powershell**
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine
```

## Ativação do ambiente virtual

**em PowerShell**:
```bash
C:\TEMP\FormacaoModelos\Scripts\Activate.ps1
```

**em cmd** (linha de comandos):
```bash
C:\TEMP\FormacaoModelos\Scripts\activate.bat
```

* * * 

# Linux/macOS: 

## Configuração de um ambiente virtual
```bash
python3 -m venv /tmp/FormacaoModelos --prompt "Fundamentos de Python"
```

## Ativação do ambiente virtual
```bash
source /tmp/FormacaoModelos/bin/activate
```


* * * 

# Instalação de prerequesitos
```bash
pip install -r requirements.txt
```
* ipython - Um interpretador interativo mais avançado que o default do Python;
* jupyterlab - Um ambiente de trabalho interativo para criação de Blocos de notas;
* numpy - Biblioteca para cálculos numéricos eficientes, como matrizes e álgebra linear;
* pandas - Biblioteca para manipulação e análise de dados em tabelas;
* seaborn - Biblioteca para visualização de dados que extende o MapPlotLib.
* ...

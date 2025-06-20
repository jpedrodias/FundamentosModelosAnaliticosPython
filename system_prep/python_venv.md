# Preparação do sistema para usar venv ou docker

# Windows: 

## Configuração inicial do ambiente virtual 
```bash
python -m venv C:\TEMP\venvs\FormacaoModelos --prompt "Modelos Analiticos"
```

## Ativação do ambiente virtual

**em PowerShell**:
```bash
C:\TEMP\venvs\FormacaoModelos\Scripts\Activate.ps1
```

Se esta última instrução não correr por faltar de permissões de execussão de scripts powershell em windows, então é possível remover essa restrição com a seguinte instrução


**Remover as restrições de segurança do powershell (admin mode):**
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine
```
PS: Essa instrução precisa ser executada em powershell em mode Admin


Em alternativa, abrir o terminal
**em cmd** (linha de comandos):
```bash
C:\TEMP\venvs\FormacaoModelos\Scripts\activate.bat
```

* * * 

# Linux/macOS: 

## Configuração de um ambiente virtual
```bash
python3 -m venv /tmp/FormacaoModelos --prompt "Modelos Analiticos"
```

## Ativação do ambiente virtual
```bash
source /tmp/FormacaoModelos/bin/activate
```


* * * 
* * * 

# Instalação de requesitos
```bash
pip install -r requirements.txt
```

* ipython - Um interpretador interativo mais avançado que o default do Python;
* jupyterlab - Um ambiente de trabalho interativo para criação de Blocos de notas;
* numpy - Biblioteca para cálculos numéricos eficientes, como matrizes e álgebra linear;
* pandas - Biblioteca para manipulação e análise de dados em tabelas;
* seaborn - Biblioteca para visualização de dados que extende o MapPlotLib.
* scikit-learn - Biblioteca para machine learning e modelagem preditiva.
* xgboost - Biblioteca otimizada para boosting de árvores de decisão, amplamente utilizada para tarefas de classificação e regressão em machine learning.
* ...


# IDE Visual Studio Code
* download
* correr dentro do ambiente virtual

```bash
code .
```
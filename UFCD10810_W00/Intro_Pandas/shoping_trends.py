# Importação das bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Passo 1: Carregar os Dados
# Carregar o arquivo CSV
df = pd.read_csv('../data/shopping_trends.csv')

# Passo 2: Verificação de Tipos de Dados
# Exibir os tipos de dados para garantir que as colunas estão corretamente tipadas
print("Tipos de dados:\n", df.dtypes)

# Passo 3: Verificação de Valores Ausentes
# Identificar se há valores ausentes em qualquer coluna
missing_values = df.isnull().sum()
print("\nValores ausentes por coluna:\n", missing_values)

# Verificar a porcentagem de valores ausentes
missing_percentage = (df.isnull().mean() * 100).round(2)
print("\nPorcentagem de valores ausentes por coluna:\n", missing_percentage)

# Passo 4: Verificação de Duplicados
# Identificar e remover linhas duplicadas
duplicates = df.duplicated().sum()
print("\nNúmero de linhas duplicadas:", duplicates)

# Remover duplicados
df.drop_duplicates(inplace=True)

# Passo 5: Conversão de Variáveis Categóricas para Variáveis Dummy (One-Hot Encoding)
# As colunas 'Gender' e 'Category' são categóricas, então vamos convertê-las para variáveis dummy
df = pd.get_dummies(df, columns=['Gender', 'Category'], drop_first=True)

# Passo 6: Estatísticas Descritivas
# Exibir estatísticas descritivas para variáveis numéricas
print("\nEstatísticas descritivas:\n", df.describe())

# Passo 7: Visualizações - Distribuição de Variáveis Numéricas
# Visualizar a distribuição da variável 'Purchase Amount (USD)'
plt.figure(figsize=(10, 6))
df['Purchase Amount (USD)'].hist(bins=20, color='skyblue', edgecolor='black')
plt.title('Distribuição do Valor das Compras (USD)')
plt.xlabel('Valor da Compra (USD)')
plt.ylabel('Frequência')
plt.show()

# Passo 8: Visualizações - Boxplot para Identificar Outliers
# Boxplot para a variável 'Purchase Amount (USD)' para identificar outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Purchase Amount (USD)'], color='orange')
plt.title('Boxplot do Valor das Compras (USD)')
plt.xlabel('Valor da Compra (USD)')
plt.show()

# Passo 9: Análise de Correlação entre Variáveis Numéricas
# Filtrando apenas as colunas numéricas
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Calcular a matriz de correlação para as variáveis numéricas
correlation_matrix = numeric_df.corr()
print("\nMatriz de Correlação:\n", correlation_matrix)

# Visualização da Matriz de Correlação com Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlação entre Variáveis Numéricas')
plt.show()

# Passo 10: Verificação de Outliers Usando Intervalo Interquartil (IQR)
# Detectar e remover outliers na variável 'Purchase Amount (USD)' utilizando o IQR
Q1 = df['Purchase Amount (USD)'].quantile(0.25)
Q3 = df['Purchase Amount (USD)'].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# Remover outliers
df = df[(df['Purchase Amount (USD)'] >= lower_limit) & (df['Purchase Amount (USD)'] <= upper_limit)]

# Passo 11: Criação de Faixas Etárias (Novo Grupo)
# Criar uma nova variável para faixa etária (faixa etária por grupos)
bins = [0, 18, 40, 60, 100]
labels = ['Jovem', 'Adulto', 'Meia-Idade', 'Idoso']
df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels)

# Visualizar a nova variável 'Age Group'
print("\nDistribuição das faixas etárias:\n", df['Age Group'].value_counts())

# Passo 12: Visualização das Frequências de Compra
# Exibir a distribuição das frequências de compra dos clientes
purchase_frequency = df['Frequency of Purchases'].value_counts()
print("\nDistribuição das Frequências de Compras:\n", purchase_frequency)

# Gráfico de barras para mostrar a distribuição de frequências de compras
plt.figure(figsize=(10, 6))
purchase_frequency.plot(kind='bar', color='lightgreen', edgecolor='black')
plt.title('Distribuição das Frequências de Compras')
plt.xlabel('Frequência de Compras')
plt.ylabel('Número de Clientes')
plt.show()

# Passo 13: Identificação de Relação entre 'Age' e 'Purchase Amount (USD)'
# Gráfico de dispersão entre 'Age' e 'Purchase Amount (USD)'
plt.figure(figsize=(10, 6))
plt.scatter(df['Age'], df['Purchase Amount (USD)'], alpha=0.5, color='blue')
plt.title('Relação entre Idade e Valor da Compra (USD)')
plt.xlabel('Idade')
plt.ylabel('Valor da Compra (USD)')
plt.show()

# Passo 14: Verificação de Valores Ausentes Após Limpeza
# Recalcular valores ausentes após os tratamentos
missing_values_after_cleaning = df.isnull().sum()
print("\nValores ausentes após limpeza:\n", missing_values_after_cleaning)

# Passo 15: Verificação de Duplicados Após Limpeza
# Verificar se há duplicados após a remoção
duplicates_after_cleaning = df.duplicated().sum()
print("\nNúmero de linhas duplicadas após limpeza:", duplicates_after_cleaning)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carregar os dados do arquivo CSV
data = pd.read_csv('../data/shopping_trends.csv')

# Configurar o estilo dos gráficos
plt.style.use('ggplot')

# ---------------- Gráfico 1: Distribuição Etária dos Clientes ----------------
plt.figure(figsize=(8, 5))
data['Age'].hist(bins=10, color='blue', edgecolor='black')
plt.title("Distribuição Etária dos Clientes")
plt.xlabel("Idade")
plt.ylabel("Frequência")
plt.tight_layout()  # Ajusta o layout para evitar sobreposições
plt.savefig("distribuicao_etaria.png", dpi=300)  # Salvar o gráfico
plt.show()

# ---------------- Gráfico 2: Compras por Categoria ----------------
plt.figure(figsize=(8, 5))
category_counts = data['Category'].value_counts()
category_counts.plot(kind='bar', color='purple', edgecolor='black')
plt.title("Compras por Categoria")
plt.xlabel("Categoria")
plt.ylabel("Número de Compras")
plt.tight_layout()
plt.savefig("compras_por_categoria.png", dpi=300)
plt.show()

# ---------------- Gráfico 3: Média de Valores de Compra por Estação ----------------
plt.figure(figsize=(8, 5))
season_avg = data.groupby('Season')['Purchase Amount (USD)'].mean()
season_avg.plot(kind='bar', color='green', edgecolor='black')
plt.title("Média de Valor de Compras por Estação")
plt.xlabel("Estação")
plt.ylabel("Média de Valor de Compras (USD)")
plt.tight_layout()
plt.savefig("media_valor_compras_estacao.png", dpi=300)
plt.show()

# ---------------- Gráfico 4: Distribuição das Avaliações dos Produtos ----------------
plt.figure(figsize=(8, 5))
data['Review Rating'].hist(bins=10, color='orange', edgecolor='black')
plt.title("Distribuição das Avaliações dos Produtos")
plt.xlabel("Avaliação")
plt.ylabel("Frequência")
plt.tight_layout()
plt.savefig("distribuicao_avaliacoes.png", dpi=300)
plt.show()

# ---------------- Gráfico 5: Top 10 Localidades por Número de Compras ----------------
plt.figure(figsize=(10, 6))
location_counts = data['Location'].value_counts().head(10)  # Top 10 localidades
location_counts.plot(kind='bar', color='red', edgecolor='black')
plt.title("Top 10 Localidades por Número de Compras")
plt.xlabel("Localidade")
plt.ylabel("Número de Compras")
plt.tight_layout()
plt.savefig("top10_localidades.png", dpi=300)
plt.show()

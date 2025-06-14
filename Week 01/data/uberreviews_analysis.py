import pandas as pd

# Carregar o ficheiro CSV
file_path = 'uber_reviews_without_reviewid.csv'
df = pd.read_csv(file_path)

# 1. Explorar a estrutura inicial do DataFrame
print("\nInformações do DataFrame:")
print(df.info())
print("\nPrimeiras 5 linhas do DataFrame:")
print(df.head())

# 2. Limpeza de dados
# Remover a coluna 'userImage' (todos os valores são nulos)
df_cleaned = df.drop(columns=['userImage'])

# Preencher valores ausentes nas colunas 'reviewCreatedVersion' e 'appVersion' com 'Desconhecida'

df_cleaned.fillna({'reviewCreatedVersion': 'Desconhecida'}, inplace=True)
df_cleaned.fillna({'appVersion': 'Desconhecida'}, inplace=True)

df_cleaned['reviewCreatedVersion'].fillna('Desconhecida', inplace=True)
df_cleaned['appVersion'].fillna('Desconhecida', inplace=True)

# 3. Análise descritiva
# Contagem de valores por avaliação (score)
print("\nDistribuição das Avaliações (score):")
print(df_cleaned['score'].value_counts())

# Resumo da distribuição de "thumbsUpCount" (gostos)
print("\nResumo da Distribuição de 'thumbsUpCount':")
print(df_cleaned['thumbsUpCount'].describe())

# 4. Distribuição temporal das análises
df_cleaned['at'] = pd.to_datetime(df_cleaned['at'])  # Converter para datetime
print("\nDistribuição Temporal (primeiros 5 dias):")
print(df_cleaned['at'].dt.date.value_counts().sort_index().head())

# 5. Resumo das respostas do Uber
print("\nExemplos de Respostas do Uber (primeiros 5 registos):")
print(df_cleaned[['replyContent', 'repliedAt']].dropna().head())

# 6. Correlação entre 'score' e 'thumbsUpCount'
print("\nCorrelação entre 'score' e 'thumbsUpCount':")
print(df_cleaned[['score', 'thumbsUpCount']].corr())

# 7. Visualização de insights adicionais
# Contagem de avaliações por score
df_cleaned['score'].value_counts().plot(kind='bar', title='Distribuição de Avaliações (Score)', xlabel='Score', ylabel='Contagem')

# Salvar o DataFrame limpo (se necessário)
df_cleaned.to_csv('uber_reviews_cleaned.csv', index=False)

print("\nAnálise concluída. DataFrame limpo salvo como 'uber_reviews_cleaned.csv'.")
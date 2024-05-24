# Documentação para o Trecho de Código de Configuração

## Visão Geral
Este trecho de código é projetado para configurar um ambiente de processamento de documentos e modelos de linguagem. Ele define caminhos, threads, configurações do Chroma, carregadores de documentos e especificações de modelos para um fluxo de trabalho de aprendizado de máquina e processamento de linguagem natural (NLP).

## Módulos e Imports

```python
import os
from chromadb.config import Settings
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader
from langchain.document_loaders import UnstructuredFileLoader, UnstructuredMarkdownLoader
```

### Descrição:
- `os`: Fornece uma maneira de usar funcionalidades dependentes do sistema operacional.
- `Settings` de `chromadb.config`: Usado para configurar as configurações do banco de dados Chroma.
- Carregadores de documentos de `langchain.document_loaders`: Carregam vários tipos de documentos como CSV, PDF, texto, Excel, Docx e Markdown.

## Configuração do Ambiente

### Constantes:

```python
MODELS_PATH = "./models"
INGEST_THREADS = os.cpu_count() or 8
CONTEXT_WINDOW_SIZE = 4096
MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE
N_GPU_LAYERS = 100
N_BATCH = 512
```

### Descrição:
- `MODELS_PATH`: Caminho para o diretório onde os modelos são armazenados.
- `INGEST_THREADS`: Número de threads usadas para ingerir documentos, padronizando para o número de núcleos da CPU ou 8 se indeterminado.
- `CONTEXT_WINDOW_SIZE`: Tamanho da janela de contexto para o modelo de linguagem.
- `MAX_NEW_TOKENS`: Número máximo de novos tokens gerados pelo modelo, configurado igual ao `CONTEXT_WINDOW_SIZE`.
- `N_GPU_LAYERS`: Número de camadas de GPU usadas no modelo.
- `N_BATCH`: Tamanho do lote para processamento.

## Configurações do Chroma

```python
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)
```

### Descrição:
- `anonymized_telemetry`: Desativa a telemetria anonimizada.
- `is_persistent`: Habilita a persistência para as configurações do Chroma.

## Carregadores de Documentos

```python
DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".py": TextLoader,
    ".pdf": UnstructuredFileLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}
```

### Descrição:
Um dicionário que mapeia extensões de arquivos para seus respectivos carregadores. Isso permite que o sistema processe diferentes tipos de documentos usando carregadores apropriados.

## Configuração do Modelo de Embedding

```python
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"
```

### Descrição:
Especifica o modelo de embedding padrão a ser usado. Modelos alternativos estão comentados com descrições de seu uso de VRAM e precisão.

### Modelos Alternativos:
- `"hkunlp/instructor-xl"`: Usa 5 GB de VRAM, mais preciso.
- `"intfloat/e5-large-v2"`: Usa 1.5 GB de VRAM, um pouco menos preciso.
- `"intfloat/e5-base-v2"`: Usa 0.5 GB de VRAM, adequado para GPUs com menor VRAM.
- `"all-MiniLM-L6-v2"`: Usa 0.2 GB de VRAM, menos preciso mas mais rápido.

### Modelos Multilíngues:
- `"intfloat/multilingual-e5-large"`: Usa 2.5 GB de VRAM.
- `"intfloat/multilingual-e5-base"`: Usa 1.2 GB de VRAM.

## Configuração do Modelo de Linguagem

### Descrição Geral:
Configurações para diferentes tipos de Modelos de Linguagem Grande (LLMs) incluindo modelos GGUF, HF (Hugging Face) e GPTQ. Essas configurações são projetadas para várias capacidades de VRAM da GPU e casos de uso específicos.

### Exemplo de Configuração:

```python
MODEL_ID = "TheBloke/CodeLlama-7B-Instruct-GGUF"
MODEL_BASENAME = "codellama-7b-instruct.q4_K_M.gguf"
```

### Descrição:
- `MODEL_ID`: Identificador para o modelo selecionado.
- `MODEL_BASENAME`: Nome base do arquivo do modelo.

### Notas:
- O código contém numerosas configurações comentadas para outros modelos baseados em seus requisitos de VRAM e características de desempenho.
- As configurações dos modelos GGUF, HF e GPTQ atendem a diferentes capacidades de VRAM, garantindo compatibilidade com vários hardwares de GPU.

## Configurações Específicas para Textos e Documentos em Português Brasil

Para tratar textos e documentos em Português Brasil, podemos usar modelos multilíngues que têm um bom desempenho no idioma. Aqui estão algumas configurações específicas que foram comentadas no código original e são recomendadas para trabalhar com documentos em Português Brasil.

### Modelos de Embedding Multilíngues

#### Modelo Usado

```python
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
```

### Descrição:
- `"intfloat/multilingual-e5-large"`: Este modelo usa 2.5 GB de VRAM e é adequado para tarefas de embedding multilíngue, incluindo Português Brasil.

#### Alternativa

```python
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
```

### Descrição:
- `"intfloat/multilingual-e5-base"`: Este modelo usa 1.2 GB de VRAM e é uma boa alternativa se você tiver limitações de VRAM.

### Configuração do Modelo de Linguagem para Português Brasil

#### Modelo Usado

```python
MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
MODEL_BASENAME = "mistral-7b-instruct-v0.1.Q8_0.gguf"
```

### Descrição:
- `MODEL_ID`: Identificador para o modelo Mistral-7B Instruct, que foi usado para processar textos em Português Brasil.
- `MODEL_BASENAME`: Nome base do arquivo do modelo.

### Notas:
- Este modelo foi testado e mostrou bom desempenho em tarefas de geração e entendimento de texto em Português Brasil.

### Exemplo de Uso:

1. **Instale as Dependências**:
   Certifique-se de ter todas as bibliotecas necessárias instaladas, incluindo `chromadb` e `langchain`.

2. **Configure os Caminhos**:
   Modifique `MODELS_PATH` e outros caminhos conforme necessário para corresponder à sua estrutura de diretórios.

3. **Selecione os Modelos**:
   Defina `EMBEDDING_MODEL_NAME`, `MODEL_ID` e `MODEL_BASENAME` com base nas capacidades do seu hardware e no desempenho desejado.

4. **Execute o Código**:
   Execute seu script para inicializar o ambiente com as configurações especificadas, agora ajustadas para trabalhar com documentos em Português Brasil.

Esta documentação fornece uma visão geral e instruções para configurar um ambiente de processamento de documentos e NLP usando o trecho de código fornecido, com ênfase em configurações específicas para textos em Português Brasil. Personalize as configurações conforme necessário para seus requisitos específicos e capacidades de hardware.
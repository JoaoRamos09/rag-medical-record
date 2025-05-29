# RAG-MEDICAL-RECORD

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-green.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT4-purple.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

</div>

## 🌟 Introdução

### Contexto da Saúde Digital

A área da saúde está passando por uma transformação digital significativa, com o aumento exponencial de dados médicos e a necessidade de processamento eficiente dessas informações. Prontuários médicos, que são documentos cruciais para o cuidado do paciente, frequentemente contêm informações não estruturadas e em diferentes formatos, dificultando sua análise e utilização efetiva.

### O Desafio

Profissionais de saúde enfrentam diariamente o desafio de:
- Processar grandes volumes de prontuários médicos
- Extrair informações relevantes de textos não estruturados
- Manter a consistência na documentação
- Realizar buscas eficientes em registros históricos
- Garantir a qualidade e precisão das informações

### Large Language Models (LLMs) na Saúde

Large Language Models (LLMs) representam um avanço significativo na capacidade de processamento de linguagem natural. Estes modelos, treinados em vastos conjuntos de dados, são capazes de:

- **Compreensão Contextual**: Entender o contexto e nuances do texto médico
- **Geração de Texto**: Criar conteúdo coerente e relevante
- **Análise Semântica**: Identificar relações e padrões em textos médicos
- **Adaptação a Diferentes Formatos**: Processar diversos tipos de documentação médica

### RAG-MEDICAL-RECORD: A Solução

O RAG-MEDICAL-RECORD é um projeto inovador que combina o poder dos LLMs com técnicas avançadas de processamento de linguagem natural para transformar a gestão de prontuários médicos. Nossa plataforma oferece:

1. **Formatação Inteligente**
   - Estruturação automática de prontuários
   - Padronização de documentos
   - Adaptação a diferentes templates

2. **Resumo Automático**
   - Extração de informações-chave
   - Geração de resumos concisos
   - Manutenção do contexto clínico

3. **Busca Semântica Avançada**
   - Sistema RAG (Retrieval Augmented Generation)
   - Busca contextual em múltiplos prontuários
   - Respostas precisas e relevantes

### Benefícios para a Saúde

- **Eficiência**: Redução do tempo de processamento de documentos
- **Precisão**: Minimização de erros na documentação
- **Acessibilidade**: Facilidade na busca e recuperação de informações
- **Padronização**: Consistência na documentação médica
- **Análise**: Melhor compreensão do histórico do paciente

## 🚀 Tecnologias

- **Backend**: Python 3.11+
- **Framework**: FastAPI
- **IA**: 
  - LangChain
  - Langgraph
  - OpenAI
  - ChromaDB


## 🛠️ Instalação

### Pré-requisitos

- Python 3.11 ou superior
- pip (gerenciador de pacotes Python)
- Chave de API da OpenAI
- Verificar arquivo .env_example

### Configuração

1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/rag-medical-record.git
cd rag-medical-record
```

2. Instale as dependências
```bash
pip install -r requirements.txt
```

3. Configure as variáveis de ambiente
```bash
cp .env.example .env
# Edite o arquivo .env com suas configurações
```

4. Execute o projeto
```bash
python main.py
```

## 📚 Documentação da API

### Endpoints

#### 1. Formatação de Prontuário
```http
POST /medical-records/format
Content-Type: application/json

{
    "medical_record": "string",
    "medical_record_template": "string"
}
```

#### 2. Resumo de Prontuário
```http
POST /medical-records/summary
Content-Type: application/json

{
    "medical_record_markdown": "string",
    "summary_max_words": "integer"
}
```

#### 3. Busca Semântica
```http
POST /medical-records/search
Content-Type: application/json

{
    "search": "string",
    "medical_records_source_list": [
        {
            "id": "integer",
            "medical_record": "string"
        }
    ]
}
```
## 🧠 Arquitetura do Sistema

### Componentes Principais

1. **API Layer**
   - FastAPI para endpoints RESTful
   - Validação com Pydantic
   - Tratamento de erros

2. **Service Layer**
   - Orquestração de serviços
   - Gerenciamento de estados
   - Tratamento de exceções

3. **AI Layer**
   - Integração com LLMs
   - Sistema RAG
   - Processamento de linguagem natural
   - Fluxo de nós utilizando Langgraph

### Sistema RAG

O sistema implementa um RAG (Retrieval Augmented Generation) em tempo real:

1. **Chunking**
   - Divisão do texto em chunks de 120 tokens
   - Overlap de 25 tokens

2. **Embeddings**
   - Geração via OpenAI
   - Armazenamento temporário no ChromaDB

3. **Retrieval**
   - Busca por similaridade
   - Seleção de chunks relevantes

4. **Generation**
   - Resposta contextualizada
   - Formatação estruturada

## 📊 Monitoramento

- Logs estruturados com `coloredlogs`
- Métricas de performance
- Rastreamento de requisições

## 🔐 Segurança

- Validação de dados com Pydantic
- Processamento em memória
- Limpeza automática do ChromaDB
- Sem persistência de dados sensíveis

## 🚀 Performance

- Chunking eficiente
- Limpeza automática de recursos

## 🤝 Contribuição

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 👥 Autores

- **João Vitor Ramos** - *Desenvolvimento* - [joaoramos09](https://github.com/joaoramos09)

---

<div align="center">
  <sub>Built with ❤️ by <a href="https://github.com/joaoramos09">joaoramos09</a></sub>
</div>



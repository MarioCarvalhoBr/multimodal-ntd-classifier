
# Multimodal Vision-Language for Image Classification of Neglected Tropical Diseases

![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-ffcc00)
![Poetry](https://img.shields.io/badge/Poetry-Package%20Manager-cyan)
![License](https://img.shields.io/badge/license-MIT-green)

Uma Prova de Conceito (PoC) para triagem e classificação de Doenças Tropicais Negligenciadas (DTNs) utilizando modelos Vision-Language (VLMs) de estado da arte, como CLIP e SigLIP. 

Projeto desenvolvido e submetido como pesquisa para o **I Workshop de Computação Aplicada às Doenças Tropicais Negligenciadas (CADTN)**.

---

## 📖 Visão Geral

Este projeto propõe um **Framework Unificado de Triagem Multimodal** desenhado para operar em ambientes com recursos limitados, como o Sistema Único de Saúde (SUS) do Brasil. 

Devido à heterogeneidade dos dados médicos de DTNs, a arquitetura foi desenhada para lidar com duas ramificações de entrada:
1. **Imagens Clínicas Macroscópicas:** (ex: Lesões de Hanseníase). Utiliza pré-processamento avançado (Filtro Morfológico Black-Hat + Inpainting TELEA) para remoção de artefatos visuais (pelos).
2. **Imagens de Microscopia Laboratorial:** (ex: Doença de Chagas, Esquistossomose, Parasitas). Passagem direta para extração de features para preservar as estruturas celulares (caudas, flagelos).

A classificação é realizada através de **Linear Probing** sobre os *backbones* congelados dos modelos multimodais, garantindo baixo custo computacional e preservando o conhecimento semântico-visual dos modelos pré-treinados.

## 🗂 Estrutura do Projeto

O código foi construído seguindo os princípios **SOLID** e padrões de **Clean Code**.

```text
multimodal-ntd-classifier/
├── data/
│   ├── raw/                 # Cache de download do Kaggle
│   └── processed/           # Dataset unificado e estratificado (Train/Val/Test)
├── src/
│   ├── data/
│   │   ├── dataset.py       # PyTorch Dataset customizado
│   │   └── make_dataset.py  # Coleta, unificação e split dos dados
│   ├── features/
│   │   └── preprocessors.py # Filtros morfológicos e inpainting
│   ├── models/
│   │   ├── classifier.py    # Factory Pattern e Linear Probing (CLIP/SigLIP)
│   │   └── trainer.py       # Loop de treinamento e métricas
│   └── run_experiment.py    # Orquestrador de testes dos modelos
├── .env.example             # Template de configuração de ambiente
├── pyproject.toml           # Dependências do Poetry
└── README.md                # Documentação do projeto
```

---

## 🚀 Como Configurar o Ambiente

O projeto utiliza o gerenciador de pacotes [Poetry](https://python-poetry.org/) para garantir reprodutibilidade estrita no Ubuntu 24.04 (ou ambientes compatíveis).

### 1. Clonar o Repositório
```bash
git clone https://github.com/MarioCarvalhoBr/multimodal-ntd-classifier.git
cd multimodal-ntd-classifier
```

### 2. Instalar Dependências
Certifique-se de ter o Poetry instalado no seu sistema. Em seguida, instale as dependências:
```bash
poetry install
```

### 3. Configurar Credenciais do Kaggle
O download dos datasets é automatizado via `kagglehub`. Para isso:
1. Acesse o [Kaggle](https://www.kaggle.com/), vá em *Settings* e clique em **Create New Token**.
2. Abra o arquivo `kaggle.json` baixado.
3. Crie um arquivo chamado `.env` na raiz do projeto e insira suas credenciais:

```env
KAGGLE_USERNAME="seu_usuario_aqui"
KAGGLE_KEY="sua_chave_longa_aqui"
PROCESSED_DATA_DIR="dataset/processed/Dataset-NTD-V1"
```

---

## 🧪 Como Executar a Pipeline

### Fase 1: Coleta e Engenharia de Dados
Este script baixa os datasets de Doenças Tropicais Negligenciadas, realiza o *Stratified Split* (70% Treino / 15% Validação / 15% Teste) e organiza as imagens no formato `ImageFolder`.

```bash
poetry run python src/data/make_dataset.py
```

### Fase 2 e 3: Treinamento e Avaliação (Linear Probing)
Este orquestrador inicializa os modelos (SigLIP e variações do CLIP), aplica a remoção de pelos apenas para as imagens clínicas (Hanseníase) e executa o *Linear Probing*. O melhor modelo (com base no *validation loss*) é salvo e avaliado contra o conjunto de teste.

```bash
poetry run python src/run_experiment.py
```

Os resultados finais (Acurácia, Precision, Recall e F1-Score Macro) serão gerados e exibidos diretamente no terminal para a compilação do artigo.

---

## 🤖 Métodos: Vision-Language Models (VLM)

Esta pesquisa utiliza a técnica de **Linear Probing** sobre modelos multimodais pré-treinados. Diferente das CNNs tradicionais, os VLMs alinham espaços vetoriais de visão e linguagem, permitindo uma generalização superior em domínios médicos com poucos dados rotulados.

### Arquiteturas Avaliadas

| Modelo (Hugging Face) | Dimensão | Vantagens Metodológicas |
| :--- | :---: | :--- |
| `google/siglip-base-patch16-224` | 768 | Utiliza a loss *Sigmoid* para aprendizado imagem-texto, apresentando o estado da arte em eficiência e compreensão semântica. |
| `openai/clip-vit-base-patch32` | 512 | Alta eficiência computacional, ideal para dispositivos móveis em missões de triagem em campo (SUS). |
| `openai/clip-vit-base-patch16` | 512 | Equilíbrio entre performance e detalhamento visual (patches de 16x16), capturando texturas finas em microscopia. |
| `openai/clip-vit-large-patch14` | 768 | Máxima capacidade de extração de atributos globais e locais, servindo como teto de performance do experimento. |

### Estratégia de Transfer Learning: Linear Probing

Para garantir uma comparação justa e baixo custo computacional, adotamos o seguinte pipeline:
1. **Backbone Freeze:** Os pesos dos Vision Transformers (ViT) dos modelos CLIP/SigLIP são congelados para preservar o conhecimento prévio.
2. **Feature Extraction:** O modelo extrai um vetor (pooled output) que representa as características patológicas da imagem.
3. **Linear Head:** Uma camada linear customizada é treinada para mapear essas características para as classes de Doenças Tropicais Negligenciadas (DTNs).

### Referências (VLM)
```bibtex
@misc{radford2021learning,
  title={Learning Transferable Visual Models From Natural Language Supervision}, 
  author={Alec Radford and Jong Wook Kim and Chris Hallacy and others},
  year={2021},
  eprint={2103.00020},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@misc{zhai2023sigmoid,
  title={Sigmoid Loss for Language-Image Pre-training}, 
  author={Xiaohua Zhai and Basil Mustafa and Alexander Kolesnikov and Lucas Beyer},
  year={2023},
  eprint={2303.15343},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
---

## 📝 Citação e Licença
Se utilizar este código em sua pesquisa, por favor, cite nosso trabalho submetido ao I CADTN.

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 👨‍💻 Autor

**Mário de Araújo Carvalho**
* Estudante de PhD em Ciência da Computação.
* Especialista em Visão Computacional, Machine Learning e Deep Active Learning aplicados à Agropecuária de Precisão e Geoprocessamento.
* Entusiasta do Software Livre.
* [GitHub Profile](https://github.com/MarioCarvalhoBr)
* [LinkedIn Profile](https://www.linkedin.com/in/mariodearaujocarvalho/)


## 📚 Referências (Datasets)

Se for utilizar as imagens compiladas por este framework, certifique-se de referenciar adequadamente os criadores originais dos datasets:
```bibtex
@misc{orvile_leprosy_2023,
  author = {Orvile},
  title = {Leprosy Chronic Wound Images (CO2Wounds-V2)},
  year = {2023},
  publisher = {Kaggle},
  howpublished = {\url{[https://www.kaggle.com/datasets/orvile/leprosy-chronic-wound-images-co2wounds-v2](https://www.kaggle.com/datasets/orvile/leprosy-chronic-wound-images-co2wounds-v2)}},
  note = {Acessado em: 2024}
}

@misc{ahmedxc4_parasite_2022,
  author = {Ahmedxc4},
  title = {Parasite Dataset (Leishmania, Plasmodium, Trypanosome, etc.)},
  year = {2022},
  publisher = {Kaggle},
  howpublished = {\url{[https://www.kaggle.com/datasets/ahmedxc4/parasite-dataset](https://www.kaggle.com/datasets/ahmedxc4/parasite-dataset)}},
  note = {Acessado em: 2024}
}

@misc{pereira_chagas_2021,
  author = {Andr\'{e} Pereira},
  title = {Trypanosoma cruzi Microscopy Detection Dataset},
  year = {2021},
  publisher = {Kaggle},
  howpublished = {\url{[https://www.kaggle.com/datasets/andrpereira157/trypanosoma-cruzi-microscopy-detection-dataset](https://www.kaggle.com/datasets/andrpereira157/trypanosoma-cruzi-microscopy-detection-dataset)}},
  note = {Acessado em: 2024}
}

@misc{mohaliy_katokatz_2023,
  author = {Mohaliy2016},
  title = {Kato-Katz STH & S. mansoni Dataset},
  year = {2023},
  publisher = {Kaggle},
  howpublished = {\url{[https://www.kaggle.com/datasets/mohaliy2016/ai4ntd-p1-5v2](https://www.kaggle.com/datasets/mohaliy2016/ai4ntd-p1-5v2)}},
  note = {Acessado em: 2024}
}
```
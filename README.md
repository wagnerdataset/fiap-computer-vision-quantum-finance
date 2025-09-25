[![Projeto](https://img.shields.io/badge/Projeto-Computer%20Vision(Reconhecimento%20Facial)-blue)](#)
[![Idiomas](https://img.shields.io/badge/Autenticação-Haar%20%7C%20DNN-brightgreen)](#)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.9-3776AB)](#)
[![SO](https://img.shields.io/badge/SO-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)](#)
[![Status](https://img.shields.io/badge/Status-Funcional-success)](#)
[![Tipo](https://img.shields.io/badge/Tipo-Acad%C3%AAmico-orange)](#)

---

# Desafio Computer Vision -  Quantum Finance Autenticação Facial

O setor de fraudes apontou que existem clientes que se queixaram de não contratar serviços específicos, como o crédito pessoal. No entanto, os protocolos de segurança da senha foram realizados em conformidade, cada cliente autenticou com sua própria senha.​ Em função disso, o banco precisa arcar com reembolsos e medidas de contenção para evitar processos judiciais, pois os clientes alegam terem sido invadidos por hackers ou algo similar.​

Além da senha, podemos implementar formas de autenticação complementares, a depender do serviço, que utilizasse uma verificação e identificação facial. Caso o cliente não seja autenticado, ele será atendido por uma esteira dedicada e as evidências da não identificação serão encaminhadas para a área de IA para validação dos parâmetros e limiares para aperfeiçoamento do modelo.

Será necessário construir:​

- Detector de faces​
- Identificação de faces​
- Detecção de vivacidade (liveness) para evitar que um fraudador utilize uma foto estática.

Grave um vídeo da aplicação em execução e envie-o pelo sistema na íntegra, ou compartilhe um link do vídeo armazenado em um drive ou publicado no YouTube.

---

## Verificação Facial: Detecção + Autenticação (Haar / DNN SSD-ResNet10)

> Notebook (**Trabalho_Visao_Computacional_Verificacao_Facial**) para **detecção de faces** e **autenticação** (1:1 e 1:N) com opção de **dois detectores** (Haar ou DNN SSD-ResNet10), **enrollment** guiado por câmera, teste de **liveness** e **avaliação offline** com **matriz de confusão**.  

> Inclui **download automático** do modelo DNN e suporte a **dataset automático**, **dataset default** e **dataset remoto (URL)**.

---

## ✨ Principais recursos

- **Detecta ambiente Colab/Local**:  
  - `IN_COLAB` (True/False)
- **Detectores de face**:  
  - `haar` (OpenCV CascadeClassifier)  
  - `dnn_ssd_resnet10` (OpenCV DNN com Caffe)
- **Runner interativo**:
  - **Inclusão (novo)**: coleta amostras de um novo usuário e treina LBPH  
  - **Autenticação (auth)**: modos **1:1** (usuário esperado) e **1:N** (identificação)
  - **Liveness**: checagem simples de energia do sinal
- **Avaliação offline** renovada:
  - Suporta **.zip / .tar / .tar.gz / .tgz** e **extração recursiva** (ex.: ZIP contendo TAR.GZ).
  - **Remoto default** (POSITIVES): **Caltech Face 1999** (`faces.tar`, CaltechDATA).
  - **Fallback NEGATIVES**: **Caltech-101** (extrai `BACKGROUND_Google`; se ausente, usa outras categorias ≠ “face”).
  - **Câmera também em Remoto/Default** quando `negatives/` estiver vazio (opção interativa).
  - Funciona com **single-class**: converte todas as imagens extraídas em `positives/` e completa `negatives/` automaticamente.
- **Métricas e gráficos**: matriz de confusão, accuracy, precision, recall, F1, tempo médio; tabela comparativa (se `pandas` disponível).
- **Dicas de tuning**: `conf_threshold` para DNN, `minNeighbors/scaleFactor/minSize` para Haar, filtros pós-detecção (razão w/h e tamanho).
- **Downloads automáticos**:
  - `deploy.prototxt` e `res10_300x300_ssd_iter_140000.caffemodel` (DNN SSD-ResNet10)
  - Dataset remoto padrão (CBCL/MIT) reorganizado em `positives/` e `negatives/`

---

## 📁 Estrutura de pastas

```text
cv_colab_data/
  enroll/                 # imagens de cadastro por usuário (criado pelo Runner)
  evidence/               # evidências/auxiliares
  models/                 # DNN: deploy.prototxt, res10_300x300_ssd_iter_140000.caffemodel
  negatives/              # negativos persistentes (reuso)
  dataset_auto/           # montado a partir do ENROLL_DIR + negativos
  dataset_default/        # dataset local fixo (positives/ negatives/)
  dataset_remote/         # dataset baixado e reorganizado (positives/ negatives/)
```

---

## 🔧 Requisitos

- Python 3.x  
- OpenCV (`cv2`), NumPy, Matplotlib, scikit-learn, (opcional) pandas  
- Acesso à **câmera** para enrollment/autenticação e (opcional) captura de negativos  

> Em Colab:  
> `pip install opencv-python-headless numpy matplotlib scikit-learn pandas`

---

## 🚀 Fluxo de uso (resumo)

1. **Célula de diretórios & parâmetros**  
   Define `DATA_DIR`, `ENROLL_DIR`, `EVIDENCE_DIR`, `SERVICE_THRESHOLD`, `LIVENESS_MIN_ENERGY`, etc.

2. **Célula 5 — Detecção de Faces (Haar/DNN)**  
   Expõe:
   - `DETECTION_MODEL` (`haar` ou `dnn_ssd_resnet10`)
   - `detect_faces(image_bgr, conf_threshold=0.5)`
   - Download automático dos **pesos DNN** para `cv_colab_data/models/`  

3. **Runner principal**  
   - Pergunta **detector** (Haar ou DNN)  
   - Modo **novo** (enrollment + treino LBPH + calibração threshold)  
   - Modo **auth** 1:1 / 1:N  
   - **Liveness** (opcional) antes da autenticação  
   - Ao final, opção de **rodar avaliação offline** (ver abaixo)

4) **Avaliação offline (pós-Runner)**  
   Modos de dataset:
   - **Automático**: `positives/` a partir do **enrollment** (usuário alvo ou todos) + `negatives/` persistentes; se faltar, captura por câmera.  
   - **Default**: usa `cv_colab_data/dataset_default/positives` e `.../negatives` (você popula).  
   - **Remoto**: baixa e extrai (**.zip/.tar/.tar.gz/.tgz**) com **extração recursiva**.  
     - **POSITIVES default**: **Caltech Face 1999** (`faces.tar`, CaltechDATA).  
     - **NEGATIVES fallback**: **Caltech-101**; tenta `BACKGROUND_Google`, senão outras categorias ≠ “face”.  
     - Se ainda faltar classe, usa **câmera** (opcional) para completar `negatives/`.
   - Perguntas interativas: `use_camera`, `negatives_src_dir` (pasta extra), `eval_max_images`, `conf_threshold` (DNN).

---

## 🧪 Métricas e matriz de confusão
- **Matriz 2×2** (linhas = verdadeiro, colunas = predito): TN/FP/FN/TP.
- **Métricas**: accuracy, precision, recall, F1, tempo médio por imagem.  
- **Apenas 1 classe**: o notebook segue (avisa e calcula o possível), mas **não** é comparável — inclua `negatives/` para avaliação real.

---

## 🧠 Dicas de ajuste
**Haar**  
- `minNeighbors` ↑ → menos FPs; `scaleFactor` e `minSize` também ajudam.  
**DNN (SSD ResNet10)**  
- Ajuste `conf_threshold` (ex.: 0.5 → 0.6/0.7) para reduzir FPs.  
- (Opcional) Filtro pós-detecção: descarte caixas muito pequenas/alongadas (razão w/h atípica).  
**Dataset**  
- Use **negativos variados** (fundos internos/externos, texturas, objetos).  
- Para comparar detectores, mantenha o **mesmo dataset** e reporte a tabela comparativa.

---

## 🧠 Escolha do detector

- Via Runner: escolha no prompt inicial.  
- Via variável/ambiente (alternativo):  
  ```python
  os.environ["DETECTION_MODEL"] = "dnn_ssd_resnet10"  # ou "haar"
  ```
- **Arquivos DNN** (baixados automaticamente):
  - `cv_colab_data/models/deploy.prototxt`
  - `cv_colab_data/models/res10_300x300_ssd_iter_140000.caffemodel`

---

## ▶️ Runner — modos e parâmetros

- **Novo (enrollment)**  
  - Captura `N_SAMPLES` imagens com intervalo `FRAME_DELAY_MS`  
  - Treina LBPH e realiza **calibração** do `SERVICE_THRESHOLD`  
- **Auth**  
  - **1:1**: valida usuário esperado  
  - **1:N**: identifica entre usuários cadastrados  
- **Liveness**  
  - `DO_LIVENESS_TEST=True` executa uma checagem simples (energia mínima)

**Exemplo de variáveis**:
```python
N_SAMPLES = 30
FRAME_DELAY_MS = 100
LBPH_NEIGHBORS = 16
DO_LIVENESS_TEST = True
```

---

## 🔗 Conexão Runner × Avaliação
- A Avaliação reutiliza **o mesmo detector** da **célula 5** escolhido no Runner.  
- **LBPH, thresholds de autenticação e liveness não são usados** na Avaliação (apenas detecção “face vs. no_face”).

---

## 📊 Avaliação offline (matriz de confusão)

- Calcula **accuracy, precision, recall, F1** e **tempo médio por imagem**.  
- Funciona mesmo com **apenas 1 classe** (exibe aviso e métricas possíveis).

**Parâmetros úteis**
- `eval_max_images`: limita nº de imagens por classe  
- `conf_threshold`: limiar do DNN (ex.: 0.5)  
- `use_camera`: usa câmera para completar negativos se faltar  
- `negatives_src_dir`: pasta extra com negativos

---

## 🌐 Dataset remoto default

- O utilitário embutido pode baixar um dataset público (CBCL/MIT faces/non-faces) e reorganizar para `positives/` e `negatives/`.  
- Se faltar alguma classe, o pipeline completa a partir do **enrollment** e/ou **câmera**.

---

## ⚙️ Variáveis de ambiente (opcionais)

- `DETECTION_MODEL`: `"haar"` | `"dnn_ssd_resnet10"`  
- `DNN_PROTO_PATH`, `DNN_WEIGHTS_PATH`: caminhos customizados para os arquivos do DNN  
- `DATA_DIR`: raiz dos dados (padrão `cv_colab_data`)

---

## 🧪 Troubleshooting

- **FileNotFoundError (DNN)**: garanta que `deploy.prototxt` e `res10_300x300_ssd_iter_140000.caffemodel` estão em `cv_colab_data/models/`. O notebook tenta baixar automaticamente.  
- **Sem câmera**: a avaliação ainda roda; se faltar `negatives`, use **Default** ou **Remoto**.  
- **Apenas 1 classe**: o relatório completo não se aplica; a avaliação mostra as métricas possíveis e a matriz parcial.  
- **Baixa precisão/recall**: aumente `eval_max_images`, garanta diversidade de negativos e ajuste `conf_threshold` (DNN).

---

## 📄 Licença

Este notebook utiliza modelos e datasets públicos para fins educacionais.
Verifique as licenças específicas dos modelos/datasets antes de uso comercial.

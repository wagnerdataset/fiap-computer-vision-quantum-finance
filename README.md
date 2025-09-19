# QuantumFinance —  Verificação Facial: Detecção + Autenticação (Haar / DNN SSD-ResNet10)

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
- **Avaliação offline** (matriz de confusão + métricas):
  - **Automático**: positives do **ENROLL_DIR**, negatives persistidos (e câmera se faltar)  
  - **Default**: usa `cv_colab_data/dataset_default/{positives,negatives}`  
  - **Remoto**: baixa um **dataset público** e reorganiza (faces vs não-faces)
- **Downloads automáticos**:
  - `deploy.prototxt` e `res10_300x300_ssd_iter_140000.caffemodel` (DNN SSD-ResNet10)
  - Dataset remoto padrão (CBCL/MIT) reorganizado em `positives/` e `negatives/`

---

## 📁 Estrutura de pastas

```text
cv_colab_data/
  enroll/                 # imagens de cadastro (enrollment) por usuário (criado pelo Runner)
  evidence/               # evidências/saídas auxiliares
  models/                 # arquivos do detector DNN (.prototxt, .caffemodel)
  negatives/              # negativos persistentes (capturados e reutilizados)
  dataset_auto/           # dataset gerado automaticamente p/ avaliação
  dataset_default/
    positives/
    negatives/
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

4. **Avaliação offline (pós-Runner)**  
   Escolha a fonte do dataset:
   - **Automático**: usa **enrollment** como `positives` e **negatives persistentes/câmera**  
   - **Default**: usa `cv_colab_data/dataset_default`  
   - **Remoto**: baixa um **dataset público** e reorganiza automaticamente

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

## 📌 Changelog (resumo)

- Adicionado **detector DNN SSD-ResNet10** com **download automático**  
- **Runner** com seleção de detector e modos **novo/auth 1:1/1:N**  
- **Avaliação offline**: **Automático**, **Default**, **Remoto**  
- **Dataset remoto default** com reorganização automática  
- Robustez para datasets com **apenas 1 classe**

---

## 📄 Licença

Este notebook utiliza modelos e datasets públicos para fins educacionais.
Verifique as licenças específicas dos modelos/datasets antes de uso comercial.

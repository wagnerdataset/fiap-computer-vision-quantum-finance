# Desafio Computer Vision -  Quantum Finance Autenticação Facial

[![Projeto](https://img.shields.io/badge/Projeto-Computer%20Vision(Reconhecimento%20Facial)-blue)](#)
[![Idiomas](https://img.shields.io/badge/Autenticação-Haar%20%7C%20DNN-brightgreen)](#)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.9-3776AB)](#)
[![SO](https://img.shields.io/badge/SO-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)](#)
[![Status](https://img.shields.io/badge/Status-Funcional-success)](#)
[![Tipo](https://img.shields.io/badge/Tipo-Acad%C3%AAmico-orange)](#)

---

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

## 🚀 Visão Geral por Célula — Verificação Facial (Haar / DNN SSD-ResNet10)

> Este documento resume, em formato de README, o que **cada célula** do notebook final faz e encerra com um **resumo dos resultados** observados.

---

## 1) Imports, Diretórios e Parâmetros Globais
- Importa bibliotecas principais: `opencv-python (cv2)`, `numpy`, `matplotlib`, `scikit-learn`, (opcional) `pandas`.
- Define a estrutura de pastas base: `DATA_DIR`, `ENROLL_DIR`, `EVIDENCE_DIR`, `models/`.
- Inicializa parâmetros de serviço: `SERVICE_THRESHOLD`, `LIVENESS_MIN_ENERGY`.
- Funções simples de visualização: ex.: `show_bgr(img, title)`.

## 2) Câmera e Utilidades de E/S
- Abertura/fechamento de câmera, captura de um frame de teste.
- Helpers de arquivos: listar/copiar imagens, criar dataset (`positives/`, `negatives/`), limpeza e diagnóstico.

## 3) Enrollment (Coleta de Amostras do Usuário)
- `enroll_user_with_preview(user_id, n_samples, interval_ms)`: coleta N imagens do rosto via câmera.
- Salva recortes normalizados em `ENROLL_DIR/<user_id>/` para treinar o reconhecedor.
- Mostra progresso e feedback visual durante a coleta.

## 4) Reconhecedor (LBPH) e Treino
- `get_recognizer(neighbors, force_retrain)`: carrega/treina **LBPH** com as pastas de `enroll/`.
- Mapeia `labels ↔ usuários` e mantém tudo em memória para autenticação rápida.
- `calibrate_threshold(...)` (ver célula 8) ajusta o limiar de decisão com base em amostras reais.

## 5) Detecção de Faces (Haar / DNN SSD-ResNet10)
- Função central: `detect_faces(image_bgr, conf_threshold=0.5)` → retorna caixas `(x,y,w,h,score)`.
- Suporta dois detectores:
  - **Haar** (`cv2.CascadeClassifier`): simples e rápido, configurable com `scaleFactor`, `minNeighbors`, `minSize`.
  - **DNN SSD-ResNet10** (OpenCV DNN/Caffe): mais robusto e com threshold ajustável (`conf_threshold`).
- Seleção do modelo por `DETECTION_MODEL` (`"haar"` ou `"dnn_ssd_resnet10"`) ou variável de ambiente.
- Baixa automaticamente `deploy.prototxt` e `res10_300x300_ssd_iter_140000.caffemodel` para `models/` quando necessário.
- (Opcional) Pós-filtro de tamanho e razão de aspecto para reduzir falsos positivos.
- Helper `draw_faces(...)` para visualização.

## 6) Liveness (Verificação de “Vida”)
- Mede a “energia” (variação entre frames) em curto intervalo.
- Compara com `LIVENESS_MIN_ENERGY`; se abaixo, marca possível spoof e bloqueia autenticação (se ativado).

## 7) Autenticação 1:1 e 1:N (com Pré-visualização)
- **1:1**: `authenticate_1v1_preview(expected_user, neighbors, require_liveness)` compara o rosto com um usuário específico.
- **1:N**: `authenticate_1vN_preview(...)` identifica entre todos os cadastrados.
- Usa LBPH + `SERVICE_THRESHOLD`; pode aplicar liveness antes.
- Retorna `display_result` com status (aceito/recusado), scores e informações auxiliares.

## 8) Calibração de Limiar (Threshold)
- `calibrate_threshold(samples, neighbors)`: coleta distâncias intra-usuário (mesma pessoa) e define `SERVICE_THRESHOLD` (ex.: p95 + margem).
- Reduz falsos rejeitos mantendo segurança — recomendável após novos enrollments.

## 9) Runner — Fluxo Interativo Final
- Mantém a base original (modo **`novo`** ou **`auth`**; em `auth`: **1:1** ou **1:N**).
- **Agora pergunta o detector** a usar (Haar ou DNN) e persiste a escolha para a sessão.
- `novo`: enrollment → re-treino LBPH → calibração do threshold → autenticação.
- `auth`: garante modelo carregado e autentica (com liveness se habilitado).
- Libera câmera/recursos no `finally`. Retorna/mostra `display_result`.

## 10) Avaliação Offline (Pós-Runner) — Dataset e Métricas
- **Objetivo**: medir **somente detecção** (face vs no_face) usando a mesma `detect_faces()` da célula 5.
- **Fontes de dataset**:
  1. **Automático**: `positives/` = imagens de **enrollment** (alvo ou todos); `negatives/` = persistentes/extra/câmera.
  2. **Default**: usa `cv_colab_data/dataset_default/{positives,negatives}` (você popula).
  3. **Remoto/Local**: baixa **.zip/.tar/.tar.gz/.tgz** com **extração recursiva**; reorganiza em `positives/` e `negatives/`.
     - POSITIVES default remoto: **Caltech Face 1999** (`faces.tar`, CaltechDATA).
     - NEGATIVES fallback: **Caltech-101** (tenta `BACKGROUND_Google`; se ausente, usa outras categorias ≠ “face”).
     - **Também pode capturar pela câmera** em Remoto/Default se `negatives/` estiver vazio.
- **Métricas**: matriz de confusão (2×2), `accuracy`, `precision`, `recall`, `f1`, `avg_time/img`.
- **Classe ausente**: segue com aviso (mas o resultado não representa avaliação completa).

## 11) (Opcional) Varredura de Thresholds (DNN) e Comparações
- Varre `conf_threshold` do DNN (ex.: 0.50 → 0.80); gera tabela e curvas (Accuracy × Threshold, Precision–Recall, F1 × Threshold).
- Permite escolher o ponto de operação ideal para o seu caso (ex.: maximizar precision vs recall).

---

# Resultado Final do Trabalho (com base nos testes realizados)

**Haar (CascadeClassifier)**  
- **Accuracy** ≈ **0,9466**  
- **Precision (face)** ≈ **0,9067**  
- **Recall (face)** ≈ **0,9933**  
- **F1** ≈ **0,948**  
- **Tempo** ≈ **0,19 s/img**  
**Interpretação**: altíssimo recall (quase não perde faces) com alguns falsos positivos.

**DNN SSD-ResNet10** (threshold 0,50)  
- **Accuracy** ≈ **0,8975**  
- **Precision (face)** ≈ **0,8272**  
- **Recall (face)** = **1,00** (FN = 0)  
- **F1** ≈ **0,905**  
- **Tempo** ≈ **0,056 s/img**  
**Interpretação**: extremamente sensível (captura todas as faces), mas com mais falsos positivos nesse threshold; recomenda-se subir `conf_threshold` (ex.: 0,6–0,7) e/ou aplicar pós-filtro para reduzir FPs.

**Conclusão**  
- Para **não perder faces** (prioridade em recall), DNN é preferível com threshold ajustado.  
- Para **menos falsos positivos** sem grande tuning, Haar já oferece bom equilíbrio.  
- Recomenda-se varrer thresholds do DNN e escolher o ponto de operação conforme o custo de FP vs FN no seu caso.

---

## 🔗 Conexão Runner × Avaliação
- A Avaliação usa **o mesmo detector** configurado na célula 5 (seleção feita no Runner).  
- **LBPH/threshold de autenticação/liveness não entram** na Avaliação — ali medimos apenas **detecção**.

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
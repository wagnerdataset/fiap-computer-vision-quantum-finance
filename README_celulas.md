# Visão Geral por Célula — Verificação Facial (Haar / DNN SSD-ResNet10)

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

## Conexão Runner × Avaliação
- A Avaliação usa **o mesmo detector** configurado na célula 5 (seleção feita no Runner).  
- **LBPH/threshold de autenticação/liveness não entram** na Avaliação — ali medimos apenas **detecção**.

---

## Requisitos e Observações
- Python 3.x; `opencv-python`, `numpy`, `matplotlib`, `scikit-learn` (opcional `pandas`).  
- Acesso à câmera para enrollment/autenticação e, se desejado, para completar `negatives/` nos modos **Automático/Default/Remoto**.  
- Mantenha `positives/` e `negatives/` com **quantidade e diversidade** suficientes para avaliações confiáveis.

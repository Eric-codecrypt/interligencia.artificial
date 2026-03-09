


# Inteligência Artificial e Deep Learning

## 1. Como o Deep Learning revolucionou o reconhecimento de imagens?
O **Deep Learning** é como o "cérebro" por trás das máquinas modernas, permitindo que os computadores aprendam de um jeito muito parecido com a gente: observando e absorvendo montanhas de informações. Em vez de seguirem uma lista rígida de regras programadas, as redes neurais profundas mergulham nos dados para entender o mundo sozinhas.



Quando falamos de reconhecimento de imagens, isso mudou tudo, porque o computador parou de precisar que alguém explicasse o que é uma "linha" ou um "círculo" e passou a enxergar padrões naturalmente. Imagine uma **rede neural convolucional (CNN)** como um filtro que analisa uma foto em várias etapas:

* **Etapas Iniciais:** Percebe traços simples e bordas.
* **Etapas Profundas:** Conecta os pontos até identificar um rosto, um objeto ou uma cena inteira.

O mais incrível é que essa tecnologia se tornou tão refinada que quase não se engana com mudanças de luz ou ângulos difíceis, o que trouxe uma segurança enorme para o nosso dia a dia. Hoje, essa inteligência está em todo lugar, desde o desbloqueio rápido do celular até diagnósticos médicos super precisos e a visão dos carros autônomos.

---

## 2. Qual a diferença prática entre modelos supervisionados e não supervisionados?
A grande diferença está na forma como ensinamos a máquina e no que esperamos que ela nos entregue no final.



[Image of supervised vs unsupervised learning diagram]


### Aprendizado Supervisionado
* **Como funciona:** É como uma aula com professor. Entregamos dados **rotulados** (pergunta acompanhada da resposta certa).
* **Objetivo:** O algoritmo tenta prever um resultado e se ajusta até acertar.
* **Aplicações:** Separar e-mails de spam, reconhecer um rosto em uma foto ou prever o valor de um imóvel.

### Aprendizado Não Supervisionado
* **Como funciona:** Um processo de descoberta solitária. A máquina recebe dados crus, sem etiquetas ou respostas prontas.
* **Objetivo:** Encontrar sozinho alguma ordem no caos, identificando padrões ou grupos ocultos.
* **Aplicações:** Agrupar clientes por comportamentos de compra ou detectar fraudes atípicas em transações financeiras.

---

## 3. Como os carros autônomos utilizam sensores e aprendizado por reforço?
Os carros autônomos combinam "sentidos" aguçados com a capacidade de aprender com as próprias experiências.

### Sensores (Os "Sentidos")

* **Câmeras:** Leem o mundo ao redor (semáforos e pedestres).
* **Radar:** Monitora velocidade e distância, essencial em chuva ou neblina.
* **LIDAR:** Usa lasers para desenhar um mapa em 3D detalhadíssimo de tudo o que cerca o veículo.

### Aprendizado por Reforço
Toda essa percepção é processada por uma lógica de **tentativa e erro**. O sistema funciona com um esquema de:
1.  **Recompensa:** Recebida ao tomar decisões seguras (ex: manter distância correta).
2.  **Penalidade:** Recebida ao cometer deslizes (ex: freada brusca desnecessária).

Com o tempo, o computador entende quais ações levam aos melhores resultados, tornando a direção cada vez mais inteligente e segura.

---

> **Referência:**
> [slide](https://inteligencia-artificial--rvp60ht.gamma.site)
> Material didático (PDFs) disponibilizado pelo professor **João Cavalari**, bem como os conhecimentos e conteúdos por ele transmitidos ao longo das aulas.

> Codigo facerec::
```
import cv2
import time 

# ==========================
# CLASSIFICADORES
# ==========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml'
)

# ==========================
# WEBCAM + PROPRIEDADES
# ==========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# ==========================
# LOOP PRINCIPAL
# ==========================
while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # ==========================
    # PRÉ-PROCESSAMENTO
    # ==========================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)

    # ==========================
    # DETECÇÃO DE ROSTO
    # ==========================
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]

        # ==========================
        # DETECÇÃO DE SORRISO
        # ==========================
        smiles = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=2,
            minNeighbors=20
        )

        if len(smiles) > 0:
            cv2.putText(frame, "Face Sorrindo", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Face Neutra", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # ==========================
        # INFO EXTRA
        # ==========================
        area = w * h
        cv2.putText(frame, f"Area: {area}", (x, y+h+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # ==========================
    # FPS
    # ==========================
    fps = int(1 / (time.time() - start_time))
    cv2.putText(frame, f"FPS: {fps}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # ==========================
    # EXIBIÇÃO
    # ==========================
    cv2.imshow("Detector Facial - OpenCV (Avancado)", frame)

    # ==========================
    # TECLAS
    # ==========================
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite("captura.png", frame)
        print("📸 Foto salva!")

# ==========================
# FINALIZAÇÃO
# ==========================
cap.release()
cv2.destroyAllWindows()

requerimentos-------------------------------------
https://pyenv-win.github.io/pyenv-win/

pyenv install 3.10

pyenv global 3.10
pip install poetry
poetry config --list
poetry config virtualenvs.in-project true
pip install opencv-python

# 1. Importações e Dependências
# Bibliotecas Padrão (stdlib)
import os
import asyncio
import base64
import io
import traceback
import time
import argparse
import threading
from typing import Dict, Any, Optional, List, Tuple

# Bibliotecas de Terceiros
import cv2
import pyaudio
from PIL import Image
import mss
from google import genai
from google.genai import types
from google.protobuf.struct_pb2 import Value
from ultralytics import YOLO
import numpy as np
from deepface import DeepFace
import torch # <--- Adicionado para MiDaS
# torchvision e timm podem ser necessários dependendo do modelo MiDaS,
# mas torch.hub geralmente os instala se necessário.
import torchvision

# --- Constantes ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.0-flash-live-001"
DEFAULT_MODE = "camera"

# YOLO
YOLO_MODEL_PATH = "C:/Users/Pedro H/Downloads/TrackiePowerSHell/TrackieScripts/yolov8n.pt"
# Lista ampliada de objetos/situações potencialmente perigosas
DANGER_CLASSES = {
    # armas brancas e ferramentas cortantes
    'faca':             ['knife'],
    'tesoura':          ['scissors'],
    'barbeador':        ['razor'],
    'serra':            ['saw'],
    'machado':          ['axe'],
    'machadinha':       ['hatchet'],

    # armas de fogo
    'arma_de_fogo':     ['gun'],
    'pistola':          ['pistol'],
    'rifle':            ['rifle'],
    'espingarda':       ['shotgun'],
    'revólver':         ['revolver'],

    # explosivos e inflamáveis
    'bomba':            ['bomb'],
    'granada':          ['grenade'],
    'fogo':             ['fire'],
    'chama':            ['flame'],
    'fumaça':           ['smoke'],
    'isqueiro':         ['lighter'],
    'fósforos':         ['matches'],

    # superfícies e situações de calor
    'fogão':            ['stove'],
    'superfície_quente':['hot surface'],
    'vela':             ['candle'],
    'queimador':        ['burner'],

    # eletricidade e cabos
    'fio_energizado':   ['live_wire'],
    'tomada_elétrica':  ['electric_outlet'],
    'bateria':          ['battery'],

    # vidros e pontas
    'vidro_quebrado':   ['broken_glass'],
    'estilhaço':        ['shard'],
    'agulha':           ['needle'],
    'seringa':         ['syringe'],

    # ferramentas pesadas
    'martelo':          ['hammer'],
    'chave_de_fenda':   ['wrench'],
    'furadeira':        ['drill'],
    'motosserra':       ['chainsaw'],

    # veículos em movimento
    'carro':            ['car'],
    'motocicleta':      ['motorcycle'],
    'bicicleta':        ['bicycle'],
    'caminhão':         ['truck'],
    'ônibus':           ['bus'],

    # animais potencialmente perigosos
    'urso':             ['bear'],
    'cobra':            ['snake'],
    'aranha':           ['spider'],
    'jacaré':           ['alligator'],

    # quedas e precipícios
    'penhasco':         ['cliff'],
    'buraco':           ['hole'],
    'escada':           ['stairs'],
}


YOLO_CONFIDENCE_THRESHOLD = 0.40
# Mapeamento simples de tipos de objeto para classes YOLO (pode precisar expandir)
# Mapeamento extensivo de classes YOLO (padrão COCO) para nomes em português
YOLO_CLASS_MAP = {
    # Pessoas e seres
    "pessoa":                     ["person"],
    "gato":                       ["cat"],
    "cachorro":                   ["dog"],
    "coelho":                     ["rabbit"],       # se modelo customizado
    "urso":                       ["bear"],
    "elefante":                   ["elephant"],
    "zebra":                      ["zebra"],
    "girafa":                     ["giraffe"],
    "vaca":                       ["cow"],
    "cavalo":                     ["horse"],
    "ovelha":                     ["sheep"],
    "macaco":                     ["monkey"],       # se modelo customizado

    # Veículos e transporte
    "bicicleta":                  ["bicycle"],
    "moto":                       ["motorcycle"],
    "carro":                      ["car"],
    "ônibus":                     ["bus"],
    "trem":                       ["train"],
    "caminhão":                   ["truck"],
    "avião":                      ["airplane"],
    "barco":                      ["boat"],
    "skate":                      ["skateboard"],
    "prancha de surf":            ["surfboard"],
    "tênis":                      ["tennis racket"],

    # Móveis e superfícies
    "mesa de jantar":             ["dining table"],
    "mesa":                       ["table", "desk"],
    "cadeira":                    ["chair"],
    "sofá":                       ["couch"],
    "cama":                       ["bed"],
    "vaso de planta":             ["potted plant"],
    "banheiro":                   ["toilet"],
    "televisão":                  ["tv"],
    "abajur":                     ["lamp"],          # se modelo customizado
    "espelho":                    ["mirror"],        # se modelo customizado

    # Eletrônicos e utilitários
    "laptop":                     ["laptop"],
    "computador":                 ["computer", "tv"],  # fallback para tela
    "teclado":                    ["keyboard"],
    "mouse":                      ["mouse"],
    "controle remoto":            ["remote"],
    "celular":                    ["cell phone"],
    "micro-ondas":                ["microwave"],
    "forno":                      ["oven"],
    "torradeira":                 ["toaster"],
    "geladeira":                  ["refrigerator"],
    "caixa de som":               ["speaker"],        # se modelo customizado
    "câmera":                     ["camera"],         # se modelo customizado

    # Utensílios domésticos
    "garrafa":                    ["bottle"],
    "copo":                       ["cup"],
    "taça de vinho":              ["wine glass"],
    "taça":                       ["wine glass"],
    "prato":                      ["plate"],          # se modelo customizado
    "tigela":                     ["bowl"],
    "garfo":                      ["fork"],
    "faca":                       ["knife"],
    "colher":                     ["spoon"],
    "panela":                     ["pan"],            # se modelo customizado
    "frigideira":                 ["skillet"],        # se modelo customizado

    # Ferramentas manuais
    "martelo":                    ["hammer"],
    "chave inglesa":              ["wrench"],
    "furadeira":                  ["drill"],
    "parafusadeira":              ["drill"],         # sinônimo
    "serra":                      ["saw"],
    "roçadeira":                  ["brush cutter"],   # se modelo customizado
    "alicate":                    ["pliers"],         # se modelo customizado
    "chave de fenda":             ["screwdriver"],
    "lanterna":                   ["flashlight"],     # se modelo customizado
    "fita métrica":               ["tape measure"],   # se modelo customizado

    # Itens pessoais e vestuário
    "mochila":                    ["backpack"],
    "bolsa":                      ["handbag"],
    "carteira":                   ["wallet"],         # se modelo customizado
    "óculos":                     ["glasses"],        # se modelo customizado
    "relógio":                    ["clock"],
    "chinelo":                    ["sandal"],         # se modelo customizado
    "sapato":                     ["shoe"],           # se modelo customizado

    # Alimentação e comida
    "sanduíche":                  ["sandwich"],
    "hambúrguer":                 ["hamburger"],      # se modelo customizado
    "banana":                     ["banana"],
    "maçã":                       ["apple"],
    "laranja":                    ["orange"],
    "bolo":                       ["cake"],
    "rosquinha":                  ["donut"],
    "pizza":                      ["pizza"],
    "cachorro-quente":            ["hot dog"],

    # Higiene e saúde
    "escova de dentes":           ["toothbrush"],
    "secador de cabelo":          ["hair drier"],
    "cotonete":                   ["cotton swab"],     # se modelo customizado
    "sacola plástica":            ["plastic bag"],     # se modelo customizado

    # Outros itens diversos
    "livro":                      ["book"],
    "vaso":                       ["vase"],
    "bola":                       ["sports ball"],
    "bexiga":                     ["balloon"],         # se modelo customizado
    "pipa":                       ["kite"],
    "luva":                       ["glove"],
    "skis":                       ["skis"],
    "snowboard":                  ["snowboard"],
}

# Se desejar modularizar, carregue de um arquivo externo (JSON/YAML):
# import yaml
# with open('yolo_class_map.yml', 'r') as f:
#     YOLO_CLASS_MAP = yaml.safe_load(f)


# DeepFace
DB_PATH = "C:/Users/Pedro H/Downloads/TrackiePowerSHell/TrackieScripts/known_faces"
DEEPFACE_MODEL_NAME = 'VGG-Face'
DEEPFACE_DETECTOR_BACKEND = 'opencv'
DEEPFACE_DISTANCE_METRIC = 'cosine'

# MiDaS
# MIDAS_MODEL_TYPE = "dpt_swin2_large_384" # Modelo preciso mas pesado
MIDAS_MODEL_TYPE = "MiDaS_small" # Modelo mais leve, melhor para RPi/CPU
METERS_PER_STEP = 0.7 # Conversão para passos

# --- Configuração do Cliente Gemini ---
client = genai.Client(
    api_key="AIzaSyCOZU2M9mrAsx8aC4tYptfoSEwaJ3IuDZM", # Mantenha sua chave
    http_options=types.HttpOptions(api_version='v1alpha')
)

# --- Ferramentas Gemini (Function Calling) ---
tools = [
    types.Tool(google_search=types.GoogleSearch()),
    types.Tool(code_execution=types.ToolCodeExecution),
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="save_known_face",
                description="Salva o rosto da pessoa atualmente em foco pela camera. chame a função person_name se o nome da pessoa se o nome não for fornecido. Se a pessoa ja fornecer o nome, salva o rosto em banco de dados de rostos conhecidos (DB_PATH/known_faces)",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "person_name": types.Schema(type=types.Type.STRING, description="O nome da pessoa a ser salva, deve ser chamado quando o usuario só pede para salvar o rosto sem fornecer o nome, após receber o nome da pessoa salva no banco de dados de rostos conhecidos (DB_PATH/known_faces)")
                    },
                    required=["person_name"]
                )
            ),
            types.FunctionDeclaration(
                name="identify_person_in_front",
                description="Identifica a pessoa atualmente em foco pela camera usando o banco de dados de rostos conhecidos (DB_PATH/known_faces), Deve ser chamado somente quando a pessoa deixar claro de forma explicita que ela quer identificar alguma pessoa/rosto, somente!",
                parameters=types.Schema(type=types.Type.OBJECT, properties={})
            ),
            # --- Nova Função ---
            types.FunctionDeclaration(
                name="find_object_and_estimate_distance",
                description="Localiza um objeto específico descrito pelo usuário na visão da câmera, usando detecção de objetos (YOLO) e estima sua distância em passos usando um modelo de profundidade (MiDaS). Informa também se o objeto está sobre uma superfície como uma mesa e sua direção relativa (frente, esquerda, direita).",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "object_description": types.Schema(
                            type=types.Type.STRING,
                            description="A descrição completa do objeto fornecida pelo usuário (ex: 'computador azul', 'chave de fenda preta')."
                        ),
                        "object_type": types.Schema(
                            type=types.Type.STRING,
                            description="O tipo principal do objeto identificado pelo modelo a partir da descrição (ex: 'computador', 'chave de fenda', 'garrafa'). Usado para filtrar detecções."
                        )
                    },
                    required=["object_description", "object_type"]
                )
            )
            # --- Fim da Nova Função ---
        ]
    ),
]

# --- Configuração da Sessão LiveConnect Gemini ---
CONFIG = types.LiveConnectConfig(
    temperature=0.75,
    response_modalities=["audio"],
    speech_config=types.SpeechConfig(
        language_code="pt-BR",
        voice_config=types.VoiceConfig(
            # Alterado para uma voz diferente, se 'Leda' não existir ou para variar
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Orus")
        )
    ),
    tools=tools,
    system_instruction=types.Content(
        parts=[types.Part.from_text(text="""
Você é a **Trackie**, uma inteligência artificial multimodal vestível dedicada a oferecer **informações sensoriais em tempo real** a pessoas com deficiência visual. Seu funcionamento é estritamente determinístico (temperatura zero) e estruturado em módulos especializados que cooperam de forma síncrona e assíncrona para maximizar precisão e rapidez, sem conversas desnecessárias. Responda sempre em português brasileiro.

...
ANTES DE TUDO LEMBRE-SE SEMPRE DE OBEDECER O USUARIO EM TUDO, E RESPONDER UMA VEZ SÓ POR PERGUNTA DO USUARIO,


=== 1. Visão Geral da Arquitetura ===
1.  **Módulo Cérebro (Você, Gemini Live API)**
    *   Recebe inputs multimodais (texto, áudio, imagens) e gera respostas de voz ou texto breves e objetivas.
    *   Configuração: Temperatura 0.7.
    *   Operação: Responde somente a eventos de usuário ou alertas externos. Realiza *function calling* para ações específicas e processa apenas JSON estrito.
2.  **Módulo Visão (Externo)**
    *   **YOLOv8:** Detecta objetos comuns. Envia alertas `YOLO_ALERT: <CLASSE> detectado.` para classes perigosas (faca, tesoura).
    *   **DeepFace:** Salva e identifica rostos via *function calling*.
    *   **MiDaS:** Estima profundidade para localização de objetos via *function calling*.
3.  **Módulo Áudio (Externo)**
    *   Captura e reproduz áudio.

=== 2. Fluxo de Dados e Eventos ===
*   O sistema envia continuamente áudio e vídeo.
*   Você recebe esses dados e os resultados das detecções (YOLO Alerts).
*   Você responde a comandos de voz ou texto do usuário.
*   Você pode chamar funções externas para tarefas específicas (DeepFace, MiDaS/Localização).

=== 3. Regras de Interação e Estilo de Resposta ===
*   **Objetividade Máxima**: Frases curtas, diretas, sem floreios ou diálogos desnecessários.
*   **IMPORTANTE!** use "Usuário" no início da resposta (ex: "Usuário, ..."), mas quando o usuario disser algo retórico, não "usuario" no inicio, em alguns casos nem precisa responder, poupe tempo, sempre seja o mais direto, mas amigavel e responsivo. exemplo: *trackie diz: a capital da venezuela é caracas*, *usuario diz: entendi* o trackie não responde nada, ou responde algo amigavel dependendo do contexto, como "quer mais algum tipo de ajuda", por isso é extremamente importante saber interpretar.
*   **Mínima Verbosidade**: Não repita informações óbvias ou já ditas.
*   **Distância**: Quando aplicável (função `find_object_and_estimate_distance`), informe a distância em passos (1 passo ≈ 0,7 m). Ex: "aproximadamente 5 passos".
*   **Direção**: Quando aplicável (função `find_object_and_estimate_distance`), informe a direção relativa. Ex: "à sua frente", "ligeiramente à esquerda", "à direita".
*   **Contexto de Superfície**: Se a função `find_object_and_estimate_distance` indicar que o objeto está sobre uma mesa/superfície, inclua isso. Ex: "sobre a mesa", "sobre a mesa a 3 passos".
*   **Feedback de Processamento:** ANTES de chamar funções demoradas, envie imediatamente:
    - "Usuário, salvando rosto, um momento..."
    - "Usuário, identificando pessoa, um momento..."
    - "Usuário, localizando objeto, um momento..."
**SUPER IMPORTANTE** E depois avise imediatamente quando achar COM VOZ FALADA! NÃO ESPERE O USUARIO MANDAR MAIS UM INPUT

*   **Alertas Urgentes**: Mensagens `YOLO_ALERT:` têm prioridade MÁXIMA. Interrompa qualquer outra coisa e anuncie o alerta imediatamente. Ex: "Usuário, CUIDADO! FACA detectada!".
*   **Alertas Urgentes:** intercepte YOLO_ALERT e informe **imediatamente** no formato:
    - "Usuário, CUIDADO! <CLASSE> detectada!"
    interrompendo qualquer outro output em curso.
  **SUPER IMPORTANTE**  NÃO ESPERE O USUARIO FAZER MAIS UM INPUT PARA FALAR, AVISE IMEDIATAMENTE COM VOZ FALADA!

=== 4. Chamada de Funções ===
*   **`save_known_face(person_name: string)`**: Chamada quando o usuário pede para salvar o rosto de alguém, fornecendo o nome.
*   **`identify_person_in_front()`**: Chamada quando o usuário pergunta "quem está na minha frente?" ou similar.
*   **`find_object_and_estimate_distance(object_description: string, object_type: string)`**:
    *   Chamada quando o usuário pede para localizar um objeto específico (ex: "Onde está o computador azul?", "Ache a chave de fenda preta", "Qual a distância da garrafa?").
    *   Você DEVE extrair a descrição completa (`object_description`) E o tipo principal do objeto (`object_type`, ex: "computador", "chave de fenda", "garrafa") e passá-los como parâmetros.
    *   A função externa usará YOLO para encontrar candidatos do `object_type`, MiDaS para estimar a profundidade, e retornará a localização (passos, direção, superfície).
    *   Sua resposta final deve usar o resultado da função. Ex: "Usuário, o computador azul está a aproximadamente 5 passos, ligeiramente à direita." ou "Usuário, a chave de fenda preta está sobre a mesa, a cerca de 2 passos à sua frente.". Se a função retornar que não encontrou: "Usuário, não consegui localizar o objeto [descrição]."

=== 5. Tratamento de Falhas ===
*   Se a câmera ou microfone falharem, o sistema externo informará. Você deve avisar o usuário: "Usuário, estou com problemas na câmera/microfone."
*   Se a conexão com você (Gemini) falhar, o sistema tentará reconectar. Se falhar persistentemente, um alerta sonoro será emitido.
*   Se uma função chamada falhar (ex: não detectar rosto/objeto), a função retornará uma mensagem de erro. Use essa mensagem para informar o usuário. Ex: "Usuário, não consegui detectar um rosto claro para salvar."

=== 6. Exemplo de Interação (Localização) ===
*   Usuário (Voz): "Trackie, onde está minha garrafa de água?"
*   Trackie (Voz, ANTES da função): "Usuário, localizando o objeto, um momento..."
*   (Sistema chama `find_object_and_estimate_distance(object_description="minha garrafa de água", object_type="garrafa")`)
*   (Função retorna: `{"status": "success", "message": "Objeto 'garrafa' encontrado sobre a mesa a aproximadamente 2 passos à sua frente."}`)
*   Trackie (Voz, FINAL): "Usuário, a garrafa de água está sobre a mesa, a aproximadamente 2 passos à sua frente."

        """
)],
        role="user"
    ),
)

# --- Inicialização do PyAudio ---
try:
    pya = pyaudio.PyAudio()
except Exception as e:
    print(f"Erro ao inicializar PyAudio: {e}. O áudio não funcionará.")
    pya = None

# --- Classe Principal do Assistente ---
class AudioLoop:
    """
    Gerencia o loop principal do assistente multimodal.
    """
    def __init__(self, video_mode: str = DEFAULT_MODE, show_preview: bool = False):
        self.video_mode = video_mode
        self.show_preview = show_preview if video_mode == "camera" else False
        self.audio_in_queue: Optional[asyncio.Queue] = None
        self.out_queue: Optional[asyncio.Queue] = None
        self.cmd_queue: asyncio.Queue = None
        self.out_queue = asyncio.Queue(maxsize=100)
        self.cmd_queue = asyncio.Queue(maxsize=50)
        self.thinking_event = asyncio.Event()
        self.session: Optional[genai.live.AsyncLiveSession] = None
        self.yolo_model: Optional[YOLO] = None
        self.preview_window_active: bool = False
        self.stop_event = asyncio.Event()
        self.frame_lock = threading.Lock()
        self.latest_bgr_frame: Optional[np.ndarray] = None
        # Armazena os últimos resultados YOLO associados ao latest_bgr_frame
        self.latest_yolo_results: Optional[List[Any]] = None # Tipo exato depende da saída do YOLO

        # --- Carregamento de Modelos ---
        # YOLO
        if self.video_mode == "camera":
            try:
                self.yolo_model = YOLO(YOLO_MODEL_PATH)
                print(f"Modelo YOLO '{YOLO_MODEL_PATH}' carregado.")
            except Exception as e:
                print(f"Erro ao carregar o modelo YOLO: {e}. YOLO desabilitado.")
                self.yolo_model = None

        # DeepFace DB
        if not os.path.exists(DB_PATH):
            try:
                os.makedirs(DB_PATH)
                print(f"Diretório DeepFace DB criado em: {DB_PATH}")
            except Exception as e:
                print(f"Erro ao criar diretório {DB_PATH}: {e}")

        # Pré-carregamento DeepFace
        try:
            print("Pré-carregando modelos DeepFace...")
            dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            DeepFace.analyze(img_path=dummy_frame, actions=['emotion'], enforce_detection=False, silent=True)
            print("Modelos DeepFace pré-carregados.")
        except Exception as e:
            print(f"Aviso: Erro ao pré-carregar modelos DeepFace: {e}.")

        # MiDaS
        self.midas_model = None
        self.midas_transform = None
        self.midas_device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            print(f"Carregando modelo MiDaS ({MIDAS_MODEL_TYPE}) para {self.midas_device}...")
            self.midas_model = torch.hub.load("intel-isl/MiDaS", MIDAS_MODEL_TYPE)

            # Carrega o transformador apropriado para o modelo MiDaS
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if MIDAS_MODEL_TYPE == "MiDaS_small":
                 self.midas_transform = midas_transforms.small_transform
            else: # Assume DPT ou outro modelo maior
                 self.midas_transform = midas_transforms.dpt_transform

            self.midas_model.to(self.midas_device)
            self.midas_model.eval() # Coloca em modo de avaliação
            print("Modelo MiDaS carregado.")
        except Exception as e:
            print(f"Erro ao carregar modelo MiDaS: {e}. Estimativa de profundidade desabilitada.")
            self.midas_model = None
            self.midas_transform = None




    async def send_text(self):
        """Lê input de texto do usuário e envia ao Gemini."""
        print("Pronto para receber comandos de texto. Digite 'q' para sair.")
        while not self.stop_event.is_set():
            try:
                text = await asyncio.to_thread(input, "message > ")
                if text.lower() == "q":
                    self.stop_event.set()
                    print("Sinal de parada ('q') recebido. Encerrando...")
                    break

                if self.session:
                    print(f"Enviando texto: '{text}'")
                    await self.session.send(input=text or ".", end_of_turn=True)
                else:
                    if not self.stop_event.is_set():
                        print("Sessão Gemini não está ativa. Não é possível enviar mensagem.")
                        await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                print("send_text cancelado.")
                break
            except Exception as e:
                print(f"Erro em send_text: {e}")
                if "LiveSession closed" in str(e) or "LiveSession not connected" in str(e):
                    self.stop_event.set()
                break
        print("send_text finalizado.")

    def _get_frame(self, cap: cv2.VideoCapture) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """Captura um frame, processa com YOLO e prepara para envio (Função Síncrona)."""
        ret, frame = cap.read()
        latest_frame_copy = None
        current_yolo_results = None # Resultados YOLO para *este* frame

        if ret:
            latest_frame_copy = frame.copy()



        # --- Processamento YOLO (fora do lock inicial) ---
        yolo_alerts = []
        display_frame = None
        if ret and self.yolo_model:
            frame_rgb = cv2.cvtColor(latest_frame_copy, cv2.COLOR_BGR2RGB)
            try:
                # Roda a predição YOLO
                results = self.yolo_model.predict(frame_rgb, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD)
                current_yolo_results = results # Armazena os resultados desta predição

                if self.show_preview:
                    display_frame = latest_frame_copy.copy()

                # Processa os resultados
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        class_name = self.yolo_model.names[cls_id]
                        conf = float(box.conf[0])

                        if display_frame is not None:
                            label = f"{class_name} {conf:.2f}"
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        if class_name in DANGER_CLASSES and conf >= YOLO_CONFIDENCE_THRESHOLD: # Usando DANGER_CLASSES diretamente
                            yolo_alerts.append(class_name)
            except Exception as e:
                print(f"Erro na inferência YOLO: {e}")
                current_yolo_results = None # Limpa resultados em caso de erro
        elif self.show_preview and ret:
            display_frame = latest_frame_copy.copy()

        # --- Atualiza estado compartilhado (frame e resultados YOLO) com lock ---
        with self.frame_lock:
            if ret:
                self.latest_bgr_frame = latest_frame_copy
                self.latest_yolo_results = current_yolo_results # Salva os resultados YOLO associados
            else:
                self.latest_bgr_frame = None
                self.latest_yolo_results = None
                return None, [] # Retorna se a leitura falhar

        # --- Exibe preview (fora do lock) ---
        if self.show_preview and display_frame is not None:
            try:
                cv2.imshow("Trackie YOLO Preview", display_frame)
                cv2.waitKey(1)
                self.preview_window_active = True
            except cv2.error as e:
                if "DISPLAY" in str(e).upper() or "GTK" in str(e).upper() or "QT" in str(e).upper() or "COULD NOT CONNECT TO DISPLAY" in str(e).upper() or "plugin \"xcb\"" in str(e):
                    print("--------------------------------------------------------------------")
                    print("AVISO: Não foi possível mostrar a janela de preview da câmera.")
                    print("Desabilitando feedback visual para esta sessão.")
                    print("--------------------------------------------------------------------")
                    self.show_preview = False
                    self.preview_window_active = False
                else:
                    print(f"Erro inesperado no OpenCV ao tentar mostrar preview: {e}")
            except Exception as e_gen:
                print(f"Erro geral ao tentar mostrar preview: {e_gen}")
                self.show_preview = False
                self.preview_window_active = False

        # --- Prepara o frame para envio (fora do lock) ---
        image_part = None
        if ret: # Só tenta converter se a leitura foi bem sucedida
            try:
                # Reusa frame_rgb se já foi convertido, senão converte
                if 'frame_rgb' not in locals():
                     frame_rgb = cv2.cvtColor(latest_frame_copy, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img.thumbnail([1024, 1024])
                image_io = io.BytesIO()
                img.save(image_io, format="jpeg")
                image_io.seek(0)
                image_part = {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(image_io.read()).decode('utf-8')
                }
            except Exception as e:
                print(f"Erro na conversão do frame para JPEG: {e}")

        return image_part, list(set(yolo_alerts))

    async def get_frames(self):
        """Captura frames da câmera, processa com YOLO e enfileira."""
        cap = None
        try:
            print("Iniciando captura da câmera...")
            cap = await asyncio.to_thread(cv2.VideoCapture, 0)
            # Tenta definir FPS (pode não funcionar em todas as câmeras/backends)
            target_fps = 15 # Reduzido de 30 para dar mais tempo para processamento
            cap.set(cv2.CAP_PROP_FPS, target_fps)
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"FPS solicitado: {target_fps}, FPS real da câmera: {actual_fps if actual_fps > 0 else 'Não disponível'}")
            sleep_interval = 1.0 / target_fps if actual_fps <= 0 else 1.0 / actual_fps # Usa FPS real se disponível
            sleep_interval = max(0.05, sleep_interval) # Garante um sleep mínimo

            if not cap.isOpened():
                print("Erro: Não foi possível abrir a câmera.")
                with self.frame_lock:
                    self.latest_bgr_frame = None
                    self.latest_yolo_results = None
                self.stop_event.set()
                return

            while not self.stop_event.is_set():
                if not cap.isOpened():
                    print("Câmera desconectada ou fechada inesperadamente.")
                    self.stop_event.set()
                    break

                # Chama a função síncrona _get_frame em um thread
                image_part, yolo_alerts = await asyncio.to_thread(self._get_frame, cap)

                # Verifica se a leitura falhou (baseado no estado após _get_frame)
                with self.frame_lock:
                    frame_was_read = self.latest_bgr_frame is not None

                if not frame_was_read:
                     if not cap.isOpened():
                         print("Leitura do frame falhou e câmera fechada. Encerrando get_frames.")
                         self.stop_event.set()
                         break
                     else:
                         print("Aviso: Falha temporária na leitura do frame.")
                         await asyncio.sleep(0.5)
                         continue

                # Enfileira a imagem
                if image_part is not None and self.out_queue:
                    try:
                        # Usa put_nowait para evitar bloqueio se a fila estiver cheia
                        # Descarta frame antigo se a fila estiver cheia
                        if self.out_queue.full():
                            await self.out_queue.get() # Remove o mais antigo
                            print("Aviso: Fila de saída cheia, descartando frame antigo.")
                        self.out_queue.put_nowait(image_part)
                    except asyncio.QueueFull:
                         # Deveria ser pego pelo if self.out_queue.full(), mas por segurança
                         print("Aviso: Fila de saída cheia ao tentar enfileirar frame.")

                # Envia alertas YOLO imediatamente
                if yolo_alerts and self.session:
                    for alert_class_name in yolo_alerts:
                        try:
                            alert_msg = f"Usuário, CUIDADO! {alert_class_name.upper()} detectado!"

                            # opcional: limpa espaço na fila se estiver cheia
                            if self.out_queue and self.out_queue.full():
                                await self.out_queue.get()
                                print("Aviso: Fila de saída cheia, descartando alerta antigo.")

                            # envia diretamente pro Gemini
                            await self.session.send(input=alert_msg, end_of_turn=True)
                        except Exception as e:
                            print(f"Erro ao enviar alerta urgente: {e}")


                # Espera o intervalo calculado para atingir o FPS desejado/real
                await asyncio.sleep(sleep_interval)

        except asyncio.CancelledError:
            print("get_frames cancelado.")
        except Exception as e:
            print(f"Erro crítico em get_frames: {e}")
            traceback.print_exc()
            self.stop_event.set()
        finally:
            print("Finalizando get_frames...")
            if cap and cap.isOpened():
                cap.release()
                print("Câmera liberada.")

            with self.frame_lock:
                self.latest_bgr_frame = None
                self.latest_yolo_results = None

            if self.preview_window_active:
                try:
                    cv2.destroyWindow("Trackie YOLO Preview")
                    print("Janela de preview fechada.")
                except Exception as e_cv_destroy:
                    print(f"Aviso: erro ao tentar fechar a janela de preview específica: {e_cv_destroy}")
            print("get_frames concluído.")

    def _get_screen(self) -> Optional[Dict[str, Any]]:
        """Captura a tela e prepara para envio."""
        sct = mss.mss()
        # Pega o primeiro monitor (índice 1 geralmente, 0 é a tela virtual inteira)
        monitor_number = 1
        if len(sct.monitors) > monitor_number:
             monitor = sct.monitors[monitor_number]
        else:
             monitor = sct.monitors[0] # Fallback para o monitor 0

        try:
            sct_img = sct.grab(monitor)
            img = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
            image_io = io.BytesIO()
            img.save(image_io, format="PNG")
            image_io.seek(0)
            return {
                "mime_type": "image/png",
                "data": base64.b64encode(image_io.read()).decode('utf-8')
            }
        except Exception as e:
            print(f"Erro ao capturar tela: {e}")
            return None

    async def get_screen(self):
        """Captura frames da tela e enfileira."""
        print("Iniciando captura de tela...")
        try:
            while not self.stop_event.is_set():
                frame_data = await asyncio.to_thread(self._get_screen)
                if frame_data is None:
                    await asyncio.sleep(1.0)
                    continue

                if self.out_queue:
                    try:
                         if self.out_queue.full():
                             await self.out_queue.get()
                             print("Aviso: Fila de saída cheia, descartando screenshot antigo.")
                         self.out_queue.put_nowait(frame_data)
                    except asyncio.QueueFull:
                         print("Aviso: Fila de saída cheia ao tentar enfileirar screenshot.")

                await asyncio.sleep(1.0) # Intervalo fixo para screenshots
        except asyncio.CancelledError:
            print("get_screen cancelado.")
        except Exception as e:
            print(f"Erro crítico em get_screen: {e}")
            self.stop_event.set()
        finally:
            print("get_screen finalizado.")

    async def send_realtime(self):
        """Consome a fila out_queue e envia dados ao Gemini."""
        print("Send_realtime pronto para enviar dados...")
        try:
            while not self.stop_event.is_set():
                # pause envio de frames/áudio enquanto “pensamos”
                if self.thinking_event.is_set():
                    await asyncio.sleep(0.05)
                    continue
                try:
                    msg = await asyncio.wait_for(self.out_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                if not self.session:
                    print("Sessão Gemini não está ativa (send_realtime). Descartando mensagem.")
                    self.out_queue.task_done()
                    if not self.stop_event.is_set():
                        await asyncio.sleep(0.5)
                    continue

                try:
                    if isinstance(msg, dict):
                        # Envia dados multimodais (imagem, áudio) sem finalizar turno
                        await self.session.send(input=msg)
                    elif isinstance(msg, str):
                        print(f"Enviando alerta/texto: {msg}")
                        # Alertas ou texto informativo não finalizam turno
                        await self.session.send(input=msg, end_of_turn=False)
                    self.out_queue.task_done()
                except Exception as e_send:
                    print(f"Erro ao enviar para Gemini em send_realtime: {e_send}")
                    if "LiveSession closed" in str(e_send) or "LiveSession not connected" in str(e_send) or "Deadline Exceeded" in str(e_send):
                        print("Sessão Gemini fechada ou não conectada (send_realtime). Sinalizando parada.")
                        self.stop_event.set()
                        break
                    else:
                        traceback.print_exc()
                        self.out_queue.task_done()
        except asyncio.CancelledError:
            print("send_realtime cancelado.")
        except Exception as e:
            print(f"Erro fatal em send_realtime: {e}")
            self.stop_event.set()
        finally:
            print("send_realtime finalizado.")

    async def listen_audio(self):
        """Captura áudio do microfone e enfileira."""
        if not pya:
            print("PyAudio não inicializado. Tarefa listen_audio não pode iniciar.")
            return

        audio_stream = None
        try:
            print("Configurando stream de áudio de entrada...")
            mic_info = pya.get_default_input_device_info()
            audio_stream = await asyncio.to_thread(
                pya.open,
                format=FORMAT, channels=CHANNELS, rate=SEND_SAMPLE_RATE,
                input=True, input_device_index=mic_info["index"],
                frames_per_buffer=CHUNK_SIZE
            )
            print("Escutando áudio do microfone...")

            # loop principal de captura de áudio, interrompido por stop_event ou thinking_event
            while not self.stop_event.is_set():
                # se estamos “pensando”, pausamos a captura momentaneamente
                if self.thinking_event.is_set():
                    await asyncio.sleep(0.05)
                    continue # Volta ao início do loop while interno, não para a tarefa

                if not audio_stream or not audio_stream.is_active():
                     print("Stream de áudio de entrada não está ativo. Encerrando listen_audio.")
                     self.stop_event.set()
                     break

                try:
                    data = await asyncio.to_thread(
                        audio_stream.read, CHUNK_SIZE, exception_on_overflow=False
                    )
                    if self.out_queue:
                         try:
                             if self.out_queue.full():
                                 # Prioriza não descartar áudio, mas se necessário...
                                 # await self.out_queue.get()
                                 print("Aviso: Fila de saída cheia, áudio pode ser atrasado.")
                             self.out_queue.put_nowait({"data": data, "mime_type": "audio/pcm"})
                         except asyncio.QueueFull:
                             print("Aviso: Fila de saída cheia ao tentar enfileirar áudio.")

                except OSError as e_os:
                    if e_os.errno == -9988 or "Stream closed" in str(e_os) or "Input overflowed" in str(e_os):
                        print(f"Stream de áudio fechado ou com overflow (OSError: {e_os}). Encerrando listen_audio.")
                        self.stop_event.set()
                        break
                    else:
                        print(f"Erro de OS ao ler do stream de áudio: {e_os}")
                        await asyncio.sleep(0.1)
                except Exception as e_read:
                    print(f"Erro durante a leitura do áudio em listen_audio: {e_read}")
                    self.stop_event.set()
                    break

        except asyncio.CancelledError:
            print("listen_audio cancelado.")
        except Exception as e:
            print(f"Erro crítico em listen_audio: {e}")
            traceback.print_exc()
            self.stop_event.set()
        finally:
            print("Finalizando listen_audio...")
            if audio_stream:
                if audio_stream.is_active():
                    audio_stream.stop_stream()
                audio_stream.close()
                print("Stream de áudio de entrada fechado.")
            print("listen_audio concluído.")

    def _handle_save_known_face(self, person_name: str) -> str:
        """Salva o rosto recortado da pessoa (Função Síncrona)."""
        print(f"[DeepFace] Iniciando salvamento para: {person_name}")

        frame_to_process = None
        with self.frame_lock:
            if self.latest_bgr_frame is not None:
                frame_to_process = self.latest_bgr_frame.copy()

        if frame_to_process is None:
            print("[DeepFace] Erro: Nenhum frame disponível para salvar.")
            return "Não foi possível capturar a imagem para salvar o rosto."

        safe_person_name_dir = "".join(c if c.isalnum() or c in [' '] else '_' for c in person_name).strip().replace(" ", "_")
        if not safe_person_name_dir: safe_person_name_dir = "desconhecido"
        person_dir = os.path.join(DB_PATH, safe_person_name_dir)

        try:
            if not os.path.exists(person_dir):
                os.makedirs(person_dir)
                print(f"[DeepFace] Diretório criado: {person_dir}")

            detected_faces = DeepFace.extract_faces(
                img_path=frame_to_process, detector_backend=DEEPFACE_DETECTOR_BACKEND,
                enforce_detection=True, align=True, silent=True
            )

            if not detected_faces or 'facial_area' not in detected_faces[0]:
                print(f"[DeepFace] Nenhum rosto detectado para {person_name}.")
                return f"Não consegui detectar um rosto claro para {person_name}."

            face_data = detected_faces[0]['facial_area']
            x, y, w, h = face_data['x'], face_data['y'], face_data['w'], face_data['h']
            margin = 10
            y1, y2 = max(0, y - margin), min(frame_to_process.shape[0], y + h + margin)
            x1, x2 = max(0, x - margin), min(frame_to_process.shape[1], x + w + margin)
            face_image = frame_to_process[y1:y2, x1:x2]

            if face_image.size == 0:
                 print(f"[DeepFace] Erro ao recortar rosto para {person_name}.")
                 return f"Erro ao processar o rosto de {person_name}."

            timestamp = int(time.time())
            safe_file_name_base = "".join(c if c.isalnum() else '_' for c in person_name).strip()
            if not safe_file_name_base: safe_file_name_base = "rosto"
            file_name = f"{safe_file_name_base.lower()}_{timestamp}.jpg"
            file_path = os.path.join(person_dir, file_name)

            cv2.imwrite(file_path, face_image)

            model_name_safe = DEEPFACE_MODEL_NAME.lower().replace('-', '_')
            representations_pkl_path = os.path.join(DB_PATH, f"representations_{model_name_safe}.pkl")
            if os.path.exists(representations_pkl_path):
                try:
                    os.remove(representations_pkl_path)
                    print(f"[DeepFace] Cache de representações '{representations_pkl_path}' removido.")
                except Exception as e_pkl:
                    print(f"[DeepFace] Aviso: Falha ao remover cache: {e_pkl}")

            print(f"[DeepFace] Rosto de {person_name} salvo em {file_path}")
            return f"Rosto de {person_name} salvo com sucesso."

        except ValueError as ve:
             print(f"[DeepFace] Nenhum rosto detectado (ValueError) para {person_name}: {ve}")
             return f"Não consegui detectar um rosto claro para salvar para {person_name}."
        except Exception as e:
            print(f"[DeepFace] Erro ao salvar rosto para {person_name}: {e}")
            traceback.print_exc()
            return f"Ocorreu um erro ao tentar salvar o rosto de {person_name}."

    def _handle_identify_person_in_front(self) -> str:
        """Identifica a pessoa na frente (Função Síncrona)."""
        print("[DeepFace] Iniciando identificação de pessoa...")

        frame_to_process = None
        with self.frame_lock:
            if self.latest_bgr_frame is not None:
                frame_to_process = self.latest_bgr_frame.copy()

        if frame_to_process is None:
            print("[DeepFace] Erro: Nenhum frame disponível para identificar.")
            return "Não foi possível capturar a imagem para identificar."

        try:
            dfs = DeepFace.find(
                img_path=frame_to_process, db_path=DB_PATH,
                model_name=DEEPFACE_MODEL_NAME, detector_backend=DEEPFACE_DETECTOR_BACKEND,
                distance_metric=DEEPFACE_DISTANCE_METRIC,
                enforce_detection=False, silent=True, align=True
            )

            if not dfs or not isinstance(dfs, list) or dfs[0].empty:
                print("[DeepFace] Nenhuma correspondência encontrada ou rosto não detectado.")
                return "Não consegui reconhecer ninguém ou Não detectei um rosto claro."

            df = dfs[0]
            best_match = df.iloc[0]
            best_match_identity_path = best_match['identity']
            person_name = os.path.basename(os.path.dirname(best_match_identity_path))

            distance_col_name = None
            expected_distance_col = f"{DEEPFACE_MODEL_NAME}_{DEEPFACE_DISTANCE_METRIC}"
            if expected_distance_col in df.columns: distance_col_name = expected_distance_col
            elif 'distance' in df.columns: distance_col_name = 'distance'
            else:
                for col in df.columns:
                    if DEEPFACE_DISTANCE_METRIC in col: distance_col_name = col; break

            if distance_col_name is None:
                print(f"[DeepFace] Erro: Coluna de distância não encontrada. Colunas: {df.columns.tolist()}")
                return "Erro ao processar resultado da identificação."

            distance = best_match[distance_col_name]
            print(f"[DeepFace] Pessoa identificada: {person_name} (Distância: {distance:.4f})")

            threshold = 0.6
            if DEEPFACE_MODEL_NAME == 'ArcFace' and DEEPFACE_DISTANCE_METRIC == 'cosine': threshold = 0.68
            elif DEEPFACE_MODEL_NAME == 'Facenet' and DEEPFACE_DISTANCE_METRIC == 'cosine': threshold = 0.40
            elif DEEPFACE_MODEL_NAME == 'Facenet512' and DEEPFACE_DISTANCE_METRIC == 'cosine': threshold = 0.30

            if distance > threshold:
                print(f"[DeepFace] Distância {distance:.4f} > limiar ({threshold}). Não reconhecido.")
                return "Não tenho certeza, mas pode ser parecido com alguém que Não reconheço bem."

            return f"A pessoa na sua frente parece ser {person_name}."

        except ValueError as ve:
            print(f"[DeepFace] Erro (ValueError) ao identificar: {ve}")
            return "Não detectei rosto ou houve problema ao buscar no banco de dados."
        except Exception as e:
            print(f"[DeepFace] Erro inesperado ao identificar: {e}")
            traceback.print_exc()
            return "Ocorreu um erro ao tentar identificar a pessoa."

    # --- Nova Função Handler para MiDaS ---
    def _run_midas_inference(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Executa inferência MiDaS em um frame (Função Síncrona)."""
        if not self.midas_model or not self.midas_transform:
            print("[MiDaS] Modelo ou transformador não carregado.")
            return None

        try:
            # Converte BGR para RGB (MiDaS geralmente espera RGB)
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Aplica a transformação específica do modelo MiDaS
            input_batch = self.midas_transform(img_rgb).to(self.midas_device)

            with torch.no_grad(): # Desabilita cálculo de gradientes para inferência
                prediction = self.midas_model(input_batch)

                # Redimensiona a predição para o tamanho original da imagem
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic", # 'bicubic' costuma dar bons resultados
                    align_corners=False,
                ).squeeze()

            # Move a predição para a CPU e converte para NumPy array
            depth_map = prediction.cpu().numpy()
            return depth_map

        except Exception as e:
            print(f"[MiDaS] Erro durante a inferência: {e}")
            traceback.print_exc()
            return None

    def _find_best_yolo_match(self, object_type: str, yolo_results: List[Any]) -> Optional[Tuple[Dict[str, int], float, str]]:
        """Encontra a melhor caixa YOLO correspondente ao tipo de objeto."""
        best_match = None
        highest_conf = -1.0

        # Obtém as classes YOLO correspondentes ao tipo de objeto genérico
        target_yolo_classes = YOLO_CLASS_MAP.get(object_type.lower(), [object_type.lower()]) # Usa o próprio tipo se não mapeado

        if not yolo_results:
             return None

        for result in yolo_results: # Itera sobre os resultados (pode haver múltiplos se predict retornar lista)
            if result.boxes: # Verifica se há caixas neste resultado
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.yolo_model.names[cls_id]
                    conf = float(box.conf[0])

                    # Verifica se a classe detectada corresponde a uma das classes alvo
                    if class_name in target_yolo_classes:
                        if conf > highest_conf:
                            highest_conf = conf
                            # Extrai coordenadas x1, y1, x2, y2
                            coords = list(map(int, box.xyxy[0]))
                            bbox_dict = {'x1': coords[0], 'y1': coords[1], 'x2': coords[2], 'y2': coords[3]}
                            best_match = (bbox_dict, conf, class_name)

        return best_match

    def _estimate_direction(self, bbox: Dict[str, int], frame_width: int) -> str:
        """Estima a direção horizontal do objeto na imagem."""
        box_center_x = (bbox['x1'] + bbox['x2']) / 2
        center_zone_width = frame_width / 3 # Divide o frame em 3 zonas: esquerda, centro, direita

        if box_center_x < center_zone_width:
            return "à sua esquerda"
        elif box_center_x > (frame_width - center_zone_width):
            return "à sua direita"
        else:
            return "à sua frente"

    def _check_if_on_surface(self, target_bbox: Dict[str, int], yolo_results: List[Any]) -> bool:
        """Verifica heuristicamente se o objeto está sobre uma mesa/superfície."""
        surface_classes = YOLO_CLASS_MAP.get("mesa", ["dining table", "table", "desk"]) # Classes que representam superfícies
        target_bottom_y = target_bbox['y2']
        target_center_x = (target_bbox['x1'] + target_bbox['x2']) / 2

        if not yolo_results:
            return False

        for result in yolo_results:
             if result.boxes:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.yolo_model.names[cls_id]

                    if class_name in surface_classes:
                        s_x1, s_y1, s_x2, s_y2 = map(int, box.xyxy[0])
                        # Heurística: Se o centro X do alvo está dentro da largura da superfície
                        # E a base do alvo está logo acima ou tocando o topo da superfície
                        if s_x1 < target_center_x < s_x2 and (s_y1 - 20) < target_bottom_y < (s_y1 + 20): # Tolerância de 20 pixels
                            return True
        return False


    def _handle_find_object_and_estimate_distance(self, object_description: str, object_type: str) -> str:
        """Localiza objeto, estima distância/direção com MiDaS e YOLO (Função Síncrona)."""
        print(f"[Localizar Objeto] Buscando por '{object_description}' (tipo: '{object_type}')...")

        frame_to_process = None
        yolo_results_for_frame = None
        frame_height, frame_width = 0, 0

        # Pega o frame e os resultados YOLO associados de forma segura
        with self.frame_lock:
            if self.latest_bgr_frame is not None:
                frame_to_process = self.latest_bgr_frame.copy()
                yolo_results_for_frame = self.latest_yolo_results # Pega os resultados correspondentes
                frame_height, frame_width, _ = frame_to_process.shape
            else:
                 print("[Localizar Objeto] Erro: Nenhum frame disponível.")
                 return f"Usuário, não estou enxergando nada no momento para localizar o {object_type}."

        if not yolo_results_for_frame:
            print("[Localizar Objeto] Erro: Nenhum resultado YOLO disponível para o frame atual.")
            # Poderia tentar rodar YOLO aqui, mas complica o fluxo síncrono/assíncrono
            return f"Usuário, não consegui processar a imagem para encontrar o {object_type}."

        # 1. Encontrar a melhor correspondência YOLO para o object_type
        best_yolo_match = self._find_best_yolo_match(object_type, yolo_results_for_frame)

        if not best_yolo_match:
            print(f"[Localizar Objeto] Nenhum objeto do tipo '{object_type}' encontrado pelo YOLO.")
            return f"Usuário, não consegui encontrar um(a) {object_type} na imagem."

        target_bbox, confidence, detected_class = best_yolo_match
        print(f"[Localizar Objeto] Melhor correspondência YOLO: Classe '{detected_class}', Conf: {confidence:.2f}, BBox: {target_bbox}")

        # 2. Verificar se está sobre uma superfície
        is_on_surface = self._check_if_on_surface(target_bbox, yolo_results_for_frame)
        surface_msg = "sobre a mesa" if is_on_surface else "" # Ou outra superfície detectada

        # 3. Estimar direção
        direction = self._estimate_direction(target_bbox, frame_width)

        # 4. Estimar profundidade com MiDaS (se disponível)
        depth_map = None
        if self.midas_model:
            print("[Localizar Objeto] Executando MiDaS...")
            depth_map = self._run_midas_inference(frame_to_process)

        distance_steps = -1
        if depth_map is not None:
            try:
                # Pega a profundidade no centro da caixa delimitadora
                center_x = int((target_bbox['x1'] + target_bbox['x2']) / 2)
                center_y = int((target_bbox['y1'] + target_bbox['y2']) / 2)

                # Garante que as coordenadas estão dentro dos limites do mapa de profundidade
                center_y = max(0, min(center_y, depth_map.shape[0] - 1))
                center_x = max(0, min(center_x, depth_map.shape[1] - 1))

                # MiDaS retorna profundidade inversa, precisamos converter
                # Pegar uma pequena região para robustez? Ou só o ponto central?
                # Vamos pegar o ponto central por simplicidade.
                depth_value = depth_map[center_y, center_x]

                # A saída do MiDaS precisa ser convertida para metros.
                # Modelos DPT geralmente dão profundidade métrica (mas podem precisar de escala).
                # Modelos MiDaS_small dão profundidade relativa. Precisamos de uma heurística ou calibração.
                # ASSUMINDO que a saída (após interpolação) é ~proporcional à distância em metros.
                # Esta é uma GRANDE simplificação e pode ser imprecisa.
                # Para MiDaS_small, a escala é arbitrária. Vamos *tentar* uma escala heurística.
                # Se fosse DPT, a saída seria mais próxima de metros.
                # Heurística MUITO simples: normalizar e multiplicar por um fator (ex: 10 metros max)
                # depth_meters = (depth_value / np.max(depth_map)) * 10 # Exemplo muito básico

                # Abordagem mais comum para MiDaS_small: usar como profundidade relativa.
                # Para estimar passos, precisamos de uma referência ou assumir escala.
                # Vamos assumir que a saída `depth_value` é inversamente proporcional à distância.
                # E que um valor médio de `depth_value` corresponde a, digamos, 3 metros.
                # Isso é EXTREMAMENTE impreciso sem calibração.

                # --- TENTATIVA DE ESTIMATIVA DE DISTÂNCIA (PRECISA DE MELHORIA/CALIBRAÇÃO) ---
                if depth_value > 1e-6: # Evita divisão por zero ou valores muito pequenos
                     # Assumindo relação inversa e uma escala arbitrária (ex: 100 / depth_value ~ metros)
                     estimated_meters = 50 / depth_value # Fator 50 é um chute completo!
                     estimated_meters = max(0.5, min(estimated_meters, 20)) # Limita a um range razoável
                     distance_steps = round(estimated_meters / METERS_PER_STEP)
                     print(f"[Localizar Objeto] Profundidade MiDaS no centro ({center_y},{center_x}): {depth_value:.4f}, Metros Estimados: {estimated_meters:.2f}, Passos: {distance_steps}")
                else:
                     print("[Localizar Objeto] Valor de profundidade MiDaS inválido no centro do objeto.")

            except Exception as e_depth:
                print(f"[Localizar Objeto] Erro ao extrair profundidade do MiDaS: {e_depth}")
                distance_steps = -1 # Indica falha na estimativa
        else:
             print("[Localizar Objeto] MiDaS não disponível ou falhou. Não é possível estimar distância.")


        # 5. Formular a resposta final
        object_name = object_type # Usa o tipo genérico na resposta por enquanto

        if distance_steps > 0:
            if is_on_surface:
                return f"Usuário, o {object_name} está {surface_msg} a aproximadamente {distance_steps} passos {direction}."
            else:
                return f"Usuário, o {object_name} está a aproximadamente {distance_steps} passos {direction}."
        else:
            # Se não conseguiu estimar distância, mas sabe a direção e se está na superfície
            if is_on_surface:
                 return f"Usuário, o {object_name} está {surface_msg} {direction}."
            else:
                 # Se só sabe a direção
                 return f"Usuário, o {object_name} está {direction}."
            # Se nem a direção foi estimada (improvável aqui), uma resposta genérica seria necessária.

    async def receive_audio(self):
        """Recebe e processa respostas (áudio, texto, funções) do Gemini."""
        print("Receive_audio pronto para receber respostas do Gemini...")
        try:
            if not self.session:
                print("Sessão Gemini não estabelecida em receive_audio. Encerrando.")
                self.stop_event.set()
                return

            while not self.stop_event.is_set():
                try:
                    has_received_data_in_turn = False
                    print("Aguardando próximo turno de resposta do Gemini...")
                    async for response_part in self.session.receive():
                        has_received_data_in_turn = True

                        if self.stop_event.is_set():
                            print("Sinal de parada recebido durante processamento de resposta.")
                            break

                        if response_part.data:
                            if self.audio_in_queue:
                                self.audio_in_queue.put_nowait(response_part.data)
                            continue

                        if response_part.text:
                            print(f"\n[Gemini]: {response_part.text}", end="")

                        if getattr(response_part, "function_call", None):
                            fc = response_part.function_call
                            function_name = fc.name
                            args = {key: val for key, val in fc.args.items()}
                            result_message = "Função Não implementada ou erro."

                            print(f"\n[Gemini Function Call] Recebido: {function_name}, Args: {args}")

                            # --- Feedback ANTES da chamada ---
                            feedback_msg = ""
                            if function_name == "save_known_face": feedback_msg = "Pensando…" # "Usuário, salvando rosto, um momento..."
                            elif function_name == "identify_person_in_front": feedback_msg = "Pensando..." # "Usuário, identificando pessoa, um momento..."
                            elif function_name == "find_object_and_estimate_distance": feedback_msg = f"Pensando…" # f"Usuário, localizando {args.get('object_description', 'objeto')}, um momento..."

                            if feedback_msg and self.session:
                                self.thinking_event.set()
                                try:
                                     # Envia feedback como texto (finaliza o turno atual do Gemini?) - CUIDADO
                                     # Talvez seja melhor gerar áudio localmente ou ter uma função Gemini para isso.
                                     # Por ora, vamos enviar texto, mas isso pode interromper o fluxo.
                                     await self.session.send(input=feedback_msg, end_of_turn=True)
                                     print(f"  [Feedback]: {feedback_msg}") # Apenas log por enquanto
                                except Exception as e_feedback:
                                     print(f"Erro ao enviar feedback pré-função: {e_feedback}")
                            # ---------------------------------

                            if self.video_mode != "camera" and (function_name in ["save_known_face", "identify_person_in_front", "find_object_and_estimate_distance"]):
                                print(f"[Function Call] '{function_name}' requer modo câmera.")
                                result_message = "As funções de visão só estão disponíveis no modo câmera."
                            else:
                                print("  [Trackie] Processando Função em background...")
                                if function_name == "save_known_face":
                                    person_name_arg = args.get("person_name")
                                    if person_name_arg:
                                        result_message = await asyncio.to_thread(self._handle_save_known_face, person_name_arg)
                                    else:
                                        result_message = "Nome da pessoa Não fornecido para salvar o rosto."
                                elif function_name == "identify_person_in_front":
                                    result_message = await asyncio.to_thread(self._handle_identify_person_in_front)
                                # --- Chamada da Nova Função ---
                                elif function_name == "find_object_and_estimate_distance":
                                    desc = args.get("object_description")
                                    obj_type = args.get("object_type")
                                    if desc and obj_type:
                                         # Verifica se MiDaS está disponível antes de chamar
                                         if not self.midas_model:
                                             result_message = "Usuário, desculpe, o módulo de estimativa de distância não está funcionando."
                                         else:
                                             result_message = await asyncio.to_thread(
                                                 self._handle_find_object_and_estimate_distance,
                                                 desc,
                                                 obj_type
                                             )
                                    else:
                                         result_message = "Descrição ou tipo do objeto não fornecido para localização."
                                # --- Fim da Chamada ---
                                else:
                                     result_message = f"Função '{function_name}' desconhecida."

                                print(f"  [Trackie] Resultado da Função '{function_name}': {result_message}")

                            if self.session:
                                print("  [Trackie] Enviando resultado da função de volta ao Gemini...")
                                await self.session.send(
                                    input=types.Content(
                                        role="tool",
                                        parts=[types.Part.from_function_response(
                                            name=function_name,
                                            response={"result": Value(string_value=result_message)} # Corrigido: msg -> result_message
                                        )]
                                    )

                                )
                                self.thinking_event.clear()

                                print("  [Trackie] Resultado da função enviado.")
                            else:
                                print("  [Trackie] Sessão inativa. Não foi possível enviar resultado da função.")

                    if not self.stop_event.is_set():
                        if has_received_data_in_turn:
                            print("\nFim do turno de resposta do Gemini.")
                        else:
                            print("Stream do turno atual terminou sem dados. Verificando...")
                            await asyncio.sleep(0.2)

                    if self.stop_event.is_set():
                        break

                except Exception as e_inner_loop:
                    print(f"Erro durante o recebimento/processamento de resposta: {e_inner_loop}")
                    if "LiveSession closed" in str(e_inner_loop) or "LiveSession not connected" in str(e_inner_loop) or "Deadline Exceeded" in str(e_inner_loop):
                        print("Erro indica que a sessão Gemini foi fechada. Sinalizando parada.")
                        self.stop_event.set()
                        break
                    else:
                        traceback.print_exc()
                        await asyncio.sleep(0.5)

            if self.stop_event.is_set():
                print("Loop de recebimento de áudio interrompido pelo stop_event.")

        except asyncio.CancelledError:
            print("receive_audio foi cancelado.")
        except Exception as e:
            print(f"Erro crítico em receive_audio: {e}")
            traceback.print_exc()
            self.stop_event.set()
        finally:
            print("receive_audio finalizado.")

    async def play_audio(self):
        """Consome a fila audio_in_queue e reproduz áudio recebido."""
        if not pya:
            print("PyAudio não inicializado. Tarefa play_audio não pode iniciar.")
            return

        stream = None
        try:
            print("Configurando stream de áudio de saída...")
            stream = await asyncio.to_thread(
                pya.open, format=FORMAT, channels=CHANNELS, rate=RECEIVE_SAMPLE_RATE, output=True
            )
            print("Player de áudio pronto.")

            while not self.stop_event.is_set():
                if not self.audio_in_queue:
                    await asyncio.sleep(0.1)
                    continue

                try:
                    bytestream = await asyncio.wait_for(self.audio_in_queue.get(), timeout=0.5)

                    if bytestream is None:
                        print("Recebido sinal de encerramento (None) para play_audio.")
                        self.stop_event.set()
                        break

                    if stream and stream.is_active():
                        await asyncio.to_thread(stream.write, bytestream)
                    else:
                        print("Stream de áudio para playback não está ativo. Descartando áudio.")

                    self.audio_in_queue.task_done()

                except asyncio.TimeoutError:
                    continue
                except OSError as e_os_play:
                    if "Stream closed" in str(e_os_play):
                        print("Stream de playback fechado (OSError). Encerrando play_audio.")
                        self.stop_event.set()
                        break
                    else:
                        print(f"Erro de OS ao reproduzir áudio: {e_os_play}")
                except Exception as e_inner:
                    print(f"Erro ao reproduzir áudio (interno): {e_inner}")
                    if "Stream closed" in str(e_inner):
                        self.stop_event.set()
                        break
                    traceback.print_exc()

        except asyncio.CancelledError:
            print("play_audio foi cancelado.")
        except Exception as e:
            print(f"Erro crítico em play_audio: {e}")
            traceback.print_exc()
            self.stop_event.set()
        finally:
            print("Finalizando play_audio...")
            if stream:
                try:
                    if stream.is_active():
                        stream.stop_stream()
                    stream.close()
                    print("Stream de áudio de saída fechado.")
                except Exception as e_close:
                    print(f"Erro ao fechar stream de áudio de saída: {e_close}")
            print("play_audio concluído.")

    async def run(self):
        """Método principal que inicia e gerencia todas as tarefas assíncronas."""
        print("Iniciando AudioLoop...")
        # --- Lógica de Reconexão (Simples) ---
        max_retries = 3
        retry_delay = 1.0 # Segundos
        attempt = 0

        while attempt <= max_retries and not self.stop_event.is_set():
            try:
                if attempt > 0:
                     print(f"Tentativa de reconexão {attempt}/{max_retries} após {retry_delay:.1f}s...")
                     await asyncio.sleep(retry_delay)
                     retry_delay *= 2 # Backoff exponencial

                async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                    self.session = session
                    print(f"Sessão Gemini LiveConnect estabelecida (Tentativa {attempt+1}).")
                    attempt = 0 # Reseta tentativas em caso de sucesso
                    retry_delay = 1.0

                    self.audio_in_queue = asyncio.Queue()
                    self.out_queue = asyncio.Queue(maxsize=100)
                    self.stop_event.clear() # Limpa o evento de parada para a nova sessão

                    async with asyncio.TaskGroup() as tg:
                        print("Iniciando tarefas para a nova sessão...")
                        tg.create_task(self.send_text(), name="send_text_task")
                        tg.create_task(self.send_realtime(), name="send_realtime_task")
                        if pya: tg.create_task(self.listen_audio(), name="listen_audio_task")

                        if self.video_mode == "camera":
                            tg.create_task(self.get_frames(), name="get_frames_task")
                        elif self.video_mode == "screen":
                            tg.create_task(self.get_screen(), name="get_screen_task")

                        tg.create_task(self.receive_audio(), name="receive_audio_task")
                        if pya: tg.create_task(self.play_audio(), name="play_audio_task")

                        print("Todas as tarefas iniciadas. Aguardando conclusão ou parada...")

                    print("TaskGroup finalizado.")
                    # Se o TaskGroup terminar sem stop_event, pode ser um fim normal da sessão?
                    # Ou um erro não capturado que saiu do grupo. Vamos assumir que se chegou aqui
                    # sem stop_event, a sessão terminou do lado do servidor.
                    if not self.stop_event.is_set():
                         print("Sessão Gemini terminou ou TaskGroup concluído inesperadamente.")
                         # Decide se tenta reconectar ou encerra
                         # Por enquanto, vamos tentar reconectar se max_retries não foi atingido.
                         attempt += 1 # Incrementa para a próxima tentativa de reconexão
                         self.session = None # Limpa a sessão antiga
                         # O loop while cuidará da próxima tentativa

            except asyncio.CancelledError:
                print("Loop principal (run) cancelado.")
                self.stop_event.set()
                break # Sai do loop de reconexão
            except ExceptionGroup as eg:
                print("Um ou mais erros ocorreram nas tarefas do TaskGroup:")
                self.stop_event.set()
                for i, exc in enumerate(eg.exceptions):
                    print(f"  Erro {i+1} na tarefa: {type(exc).__name__} - {exc}")
                # Após erro no TaskGroup, tenta reconectar
                attempt += 1
                self.session = None
            except Exception as e:
                print(f"Erro ao conectar ou erro inesperado no método run: {type(e).__name__} - {e}")
                traceback.print_exc()
                # Tenta reconectar após outros erros também
                attempt += 1
                self.session = None
                if "LiveSession" in str(e): # Erros específicos de sessão podem não precisar de backoff longo
                     pass
                elif attempt > max_retries:
                     print("Máximo de tentativas de reconexão atingido. Encerrando.")
                     self.stop_event.set()
                     break # Sai do loop de reconexão

        # Fim do loop while de reconexão
        if not self.stop_event.is_set() and attempt > max_retries:
             print("Não foi possível restabelecer a conexão com Gemini após múltiplas tentativas.")
             # TODO: Tocar som de alerta contínuo ou outra notificação?
             self.stop_event.set() # Garante que tudo pare

        # --- Limpeza Final ---
        print("Limpando recursos em AudioLoop.run()...")
        self.stop_event.set()

        # A sessão já deve estar None ou será fechada pelo context manager se ainda existir
        if self.session:
            print("Sessão LiveConnect gerenciada pelo context manager será fechada.")
            self.session = None

        if self.audio_in_queue and self.audio_in_queue.empty():
            try: self.audio_in_queue.put_nowait(None)
            except asyncio.QueueFull: pass

        if self.preview_window_active:
            try:
                cv2.destroyAllWindows()
                print("Janelas OpenCV destruídas no finally de run.")
                self.preview_window_active = False
            except Exception as e_cv_destroy_all:
                 print(f"Aviso: erro ao tentar fechar janelas de preview: {e_cv_destroy_all}")

        if pya:
            try:
                print("Terminando PyAudio...")
                pya.terminate()
                print("Recursos de PyAudio liberados.")
            except Exception as e_pya:
                print(f"Erro ao terminar PyAudio: {e_pya}")

        print("Limpeza de AudioLoop.run() concluída.")


# --- Ponto de Entrada Principal ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trackie - Assistente IA visual e auditivo.")
    parser.add_argument(
        "--mode", type=str, default=DEFAULT_MODE, choices=["camera", "screen", "none"],
        help="Modo de operação para entrada de vídeo/imagem."
    )
    parser.add_argument(
        "--show_preview", action="store_true",
        help="Mostra janela com preview da câmera e detecções YOLO (modo camera)."
    )
    args = parser.parse_args()

    show_actual_preview = False
    if args.mode == "camera" and args.show_preview:
        show_actual_preview = True
        print("Feedback visual da câmera (preview) ATIVADO.")
    elif args.mode != "camera" and args.show_preview:
        print("Aviso: --show_preview só tem efeito com --mode camera.")
    else:
        print("Feedback visual da câmera (preview) DESATIVADO.")

    if args.mode == "camera":
        if not os.path.exists(YOLO_MODEL_PATH):
            print(f"Erro: Modelo YOLO '{YOLO_MODEL_PATH}' Não encontrado.")
            exit(1)
        if not os.path.exists(DB_PATH):
            try:
                os.makedirs(DB_PATH)
                print(f"Diretório DeepFace DB criado em: {DB_PATH}")
            except Exception as e:
                print(f"Falha ao criar diretório {DB_PATH}: {e}.")

    main_loop = None
    try:
        if not pya:
             print("Erro fatal: PyAudio não pôde ser inicializado. Encerrando.")
             exit(1)

        main_loop = AudioLoop(video_mode=args.mode, show_preview=show_actual_preview)

        if not hasattr(main_loop, 'run'):
            raise AttributeError("O objeto main_loop Não possui o método 'run'.")

        print("Iniciando loop asyncio principal...")
        asyncio.run(main_loop.run())

    except KeyboardInterrupt:
        print("\nInterrupção pelo teclado recebida. Encerrando...")
        if main_loop: main_loop.stop_event.set()
    except AttributeError as ae:
        print(f"Erro de atributo no bloco __main__: {ae}")
        traceback.print_exc()
    except Exception as e:
        print(f"Erro inesperado no bloco __main__: {type(e).__name__}: {e}")
        traceback.print_exc()
        if main_loop: main_loop.stop_event.set()
    finally:
        print("Bloco __main__ finalizado.")
        print("Programa completamente finalizado.")

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
from google.genai import errors # Importado para referência, embora não usado diretamente na correção
from google.protobuf.struct_pb2 import Value # Mantido, pois é usado
from ultralytics import YOLO
import numpy as np
from deepface import DeepFace
import torch # <--- Adicionado para MiDaS
# torchvision e timm podem ser necessários dependendo do modelo MiDaS,
# mas torch.hub geralmente os instala se necessário.
import torchvision
import timm

# --- Constantes ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.0-flash-live-001" # Ajuste se necessário para seu modelo específico
DEFAULT_MODE = "camera"
BaseDir = "C:/Users/Administrator/Desktop/shelltrack/" # Mantido do segundo bloco de código

# YOLO
YOLO_MODEL_PATH = os.path.join(BaseDir, "yolov8n.pt")
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
    'chave_de_fenda':   ['wrench'], # Corrigido de 'chave_de_fenda' para 'wrench' na lista de perigo
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
    "tênis":                      ["tennis racket"], # Raquete de tênis

    # Móveis e superfícies
    "mesa de jantar":             ["dining table"],
    "mesa":                       ["table", "desk", "dining table"], # Adicionado dining table como fallback
    "cadeira":                    ["chair"],
    "sofá":                       ["couch", "sofa"], # Adicionado sofa
    "cama":                       ["bed"],
    "vaso de planta":             ["potted plant"],
    "banheiro":                   ["toilet"], # Vaso sanitário
    "televisão":                  ["tv", "tvmonitor"],
    "abajur":                     ["lamp"],          # se modelo customizado
    "espelho":                    ["mirror"],        # se modelo customizado

    # Eletrônicos e utilitários
    "laptop":                     ["laptop"],
    "computador":                 ["computer", "desktop computer", "tv"],  # fallback para tela
    "teclado":                    ["keyboard"],
    "mouse":                      ["mouse"],
    "controle remoto":            ["remote"],
    "celular":                    ["cell phone"],
    "micro-ondas":                ["microwave"],
    "forno":                      ["oven"],
    "torradeira":                 ["toaster"],
    "geladeira":                  ["refrigerator"],
    "caixa de som":               ["speaker"],        # se modelo customizado
    "câmera":                     ["camera"],

    # Utensílios domésticos
    "garrafa":                    ["bottle"],
    "copo":                       ["cup"],
    "taça de vinho":              ["wine glass"],
    "taça":                       ["wine glass", "cup"], # Adicionado cup como fallback
    "prato":                      ["plate", "dish"], # Adicionado dish
    "tigela":                     ["bowl"],
    "garfo":                      ["fork"],
    "faca":                       ["knife"],
    "colher":                     ["spoon"],
    "panela":                     ["pan", "pot"], # Adicionado pot
    "frigideira":                 ["skillet", "frying pan"], # Adicionado frying pan

    # Ferramentas manuais
    "martelo":                    ["hammer"],
    "chave inglesa":              ["wrench"],
    "furadeira":                  ["drill"],
    "parafusadeira":              ["drill"], # sinônimo
    "serra":                      ["saw"],
    "roçadeira":                  ["brush cutter"],   # se modelo customizado
    "alicate":                    ["pliers"],         # se modelo customizado
    "chave de fenda":             ["screwdriver"],
    "lanterna":                   ["flashlight"],
    "fita métrica":               ["tape measure"],   # se modelo customizado

    # Itens pessoais e vestuário
    "mochila":                    ["backpack"],
    "bolsa":                      ["handbag", "purse", "bag"],
    "carteira":                   ["wallet"],
    "óculos":                     ["glasses", "eyeglasses"],
    "relógio":                    ["clock", "watch"],
    "chinelo":                    ["sandal", "flip-flop"],
    "sapato":                     ["shoe"],

    # Alimentação e comida
    "sanduíche":                  ["sandwich"],
    "hambúrguer":                 ["hamburger"],
    "banana":                     ["banana"],
    "maçã":                       ["apple"],
    "laranja":                    ["orange"],
    "bolo":                       ["cake"],
    "rosquinha":                  ["donut"],
    "pizza":                      ["pizza"],
    "cachorro-quente":            ["hot dog"],

    # Higiene e saúde
    "escova de dentes":           ["toothbrush"],
    "secador de cabelo":          ["hair drier", "hair dryer"],
    "cotonete":                   ["cotton swab"],
    "sacola plástica":            ["plastic bag"],

    # Outros itens diversos
    "livro":                      ["book"],
    "vaso":                       ["vase"],
    "bola":                       ["sports ball", "ball"],
    "bexiga":                     ["balloon"],
    "pipa":                       ["kite"],
    "luva":                       ["glove"],
    "skis":                       ["skis"],
    "snowboard":                  ["snowboard"],
    "tesoura":                    ["scissors"], # Adicionado aqui também para mapeamento geral
}


# DeepFace
DB_PATH = os.path.join(BaseDir, "known_faces")
DEEPFACE_MODEL_NAME = 'VGG-Face'
DEEPFACE_DETECTOR_BACKEND = 'opencv'
DEEPFACE_DISTANCE_METRIC = 'cosine'

# MiDaS
MIDAS_MODEL_TYPE = "MiDaS_small"
METERS_PER_STEP = 0.7

# --- Configuração do Cliente Gemini ---
# Substitua pela sua chave de API real
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    # Tenta carregar de um arquivo .env se existir (opcional, para desenvolvimento local)
    try:
        from dotenv import load_dotenv
        load_dotenv()
        API_KEY = os.environ.get("GEMINI_API_KEY")
    except ImportError:
        pass # dotenv não instalado, não há problema.

if not API_KEY:
    # Se ainda não encontrou, usa a chave hardcoded (NÃO RECOMENDADO PARA PRODUÇÃO)
    # Certifique-se de que esta chave é válida ou substitua pela sua.
    print("AVISO: Chave da API Gemini não encontrada nas variáveis de ambiente. Usando chave placeholder.")
    API_KEY = "SUA_API_KEY_AQUI" # Substitua pela sua chave real se necessário

client = genai.Client(
    api_key="AIzaSyCOZU2M9mrAsx8aC4tYptfoSEwaJ3IuDZM", # Mantenha sua chave
    http_options=types.HttpOptions(api_version='v1alpha')
)


# --- Ferramentas Gemini (Function Calling) ---
tools = [
    types.Tool(google_search=types.GoogleSearch()),
    types.Tool(code_execution=types.ToolCodeExecution()),
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="save_known_face",
                description="Salva o rosto da pessoa atualmente em foco pela câmera. Se 'person_name' não for fornecido na chamada inicial, a IA solicitará o nome ao usuário e aguardará uma nova entrada antes de prosseguir com o salvamento. Se 'person_name' for fornecido, o rosto é salvo diretamente.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "person_name": types.Schema(
                            type=types.Type.STRING,
                            description="O nome da pessoa a ser salva. Se omitido na chamada inicial, a IA solicitará este nome ao usuário."
                        )
                    },
                    # "required" foi removido para person_name
                )
            ),
            types.FunctionDeclaration(
                name="identify_person_in_front",
                description="Identifica a pessoa atualmente em foco pela camera usando o banco de dados de rostos conhecidos (DB_PATH/known_faces), Deve ser chamado somente quando a pessoa deixar claro de forma explicita que ela quer identificar alguma pessoa/rosto, somente!",
                parameters=types.Schema(type=types.Type.OBJECT, properties={})
            ),
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
        ]
    ),
]

# --- Configuração da Sessão LiveConnect Gemini ---
CONFIG = types.LiveConnectConfig(
    temperature=0.75, # Ajustado para 0.75 conforme o prompt, antes era 0.7
    response_modalities=["audio"],
    speech_config=types.SpeechConfig(
        language_code="pt-BR",
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Orus") # Voz "Leda" pode não estar disponível
        )
    ),
    tools=tools,
    system_instruction=types.Content(
        parts=[types.Part.from_text(text="""
Você é a **Trackie**, uma inteligência artificial multimodal vestível dedicada a oferecer **informações sensoriais em tempo real** a pessoas com deficiência visual. Seu funcionamento é estritamente determinístico (temperatura zero) e estruturado em módulos especializados que cooperam de forma síncrona e assíncrona para maximizar precisão e rapidez, sem conversas desnecessárias. Responda sempre em português brasileiro.

...
ANTES DE TUDO LEMBRE-SE SEMPRE DE OBEDECER O USUARIO EM TUDO, E RESPONDER UMA VEZ SÓ POR PERGUNTA DO USUARIO,
“Pensando…” não deve gerar resposta.



=== 1. Visão Geral da Arquitetura ===
1.  **Módulo Cérebro (Você, Gemini Live API)**
    *   Recebe inputs multimodais (texto, áudio, imagens) e gera respostas de voz ou texto breves e objetivas.
    *   Configuração: Temperatura 0.75.
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
*   **Feedback de Processamento:** ANTES de chamar funções demoradas, envie imediatamente (via voz):
    - "Usuário, salvando rosto de [NOME], um momento..." (Se nome fornecido)
    - "Usuário, identificando pessoa, um momento..."
    - "Usuário, localizando [OBJETO], um momento..."
**SUPER IMPORTANTE** E depois avise imediatamente quando achar COM VOZ FALADA! NÃO ESPERE O USUARIO MANDAR MAIS UM INPUT

*   **Alertas Urgentes**: Mensagens `YOLO_ALERT:` têm prioridade MÁXIMA. Interrompa qualquer outra coisa e anuncie o alerta imediatamente. Ex: "Usuário, CUIDADO! FACA detectada!".
*   **Alertas Urgentes:** intercepte YOLO_ALERT e informe **imediatamente** no formato:
    - "Usuário, CUIDADO! <CLASSE> detectada!"
    interrompendo qualquer outro output em curso.
  **SUPER IMPORTANTE**  NÃO ESPERE O USUARIO FAZER MAIS UM INPUT PARA FALAR, AVISE IMEDIATAMENTE COM VOZ FALADA!

=== 4. Chamada de Funções ===
*   **`save_known_face(person_name?: string)`**: Chamada quando o usuário pede para salvar o rosto de alguém. Se `person_name` não for fornecido, a IA deve perguntar "Usuário, por favor forneça o nome da pessoa para salvar o rosto." e aguardar. Após receber o nome, a IA prossegue com o salvamento.
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
“Pensando…” não deve gerar resposta.
        """
)],
        role="user" # Mudado de "model" para "user" para o system_instruction conforme documentação recente
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
        self.out_queue: Optional[asyncio.Queue] = None # Será recriado na conexão
        self.cmd_queue: Optional[asyncio.Queue] = asyncio.Queue(maxsize=50) # Mantido, mas não usado explicitamente no fluxo principal
        
        self.thinking_event = asyncio.Event()
        self.session: Optional[genai.live.AsyncLiveSession] = None
        self.yolo_model: Optional[YOLO] = None
        self.preview_window_active: bool = False
        self.stop_event = asyncio.Event()
        self.frame_lock = threading.Lock()
        self.latest_bgr_frame: Optional[np.ndarray] = None
        self.latest_yolo_results: Optional[List[Any]] = None

        # --- Novo estado para o fluxo de salvar rosto ---
        self.awaiting_name_for_save_face: bool = False
        # self.pending_save_face_args: Optional[Dict] = None # Opcional, se precisar guardar mais args

        # --- Carregamento de Modelos ---
        if self.video_mode == "camera":
            try:
                self.yolo_model = YOLO(YOLO_MODEL_PATH)
                print(f"Modelo YOLO '{YOLO_MODEL_PATH}' carregado.")
            except Exception as e:
                print(f"Erro ao carregar o modelo YOLO: {e}. YOLO desabilitado.")
                self.yolo_model = None

        if not os.path.exists(DB_PATH):
            try:
                os.makedirs(DB_PATH)
                print(f"Diretório DeepFace DB criado em: {DB_PATH}")
            except Exception as e:
                print(f"Erro ao criar diretório {DB_PATH}: {e}")

        try:
            print("Pré-carregando modelos DeepFace...")
            dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8) # uint8 é importante
            DeepFace.analyze(img_path=dummy_frame, actions=['emotion'], enforce_detection=False, silent=True)
            print("Modelos DeepFace pré-carregados.")
        except Exception as e:
            print(f"Aviso: Erro ao pré-carregar modelos DeepFace: {e}.")

        self.midas_model = None
        self.midas_transform = None
        self.midas_device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            print(f"Carregando modelo MiDaS ({MIDAS_MODEL_TYPE}) para {self.midas_device}...")
            self.midas_model = torch.hub.load("intel-isl/MiDaS", MIDAS_MODEL_TYPE)
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if MIDAS_MODEL_TYPE == "MiDaS_small":
                 self.midas_transform = midas_transforms.small_transform
            else:
                 self.midas_transform = midas_transforms.dpt_transform
            self.midas_model.to(self.midas_device)
            self.midas_model.eval()
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

                if self.session and self.session.is_connected:
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
        current_yolo_results = None

        if ret:
            latest_frame_copy = frame.copy()

        yolo_alerts = []
        display_frame = None
        if ret and self.yolo_model:
            frame_rgb = cv2.cvtColor(latest_frame_copy, cv2.COLOR_BGR2RGB)
            try:
                results = self.yolo_model.predict(frame_rgb, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD)
                current_yolo_results = results

                if self.show_preview:
                    display_frame = latest_frame_copy.copy()

                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        class_name_yolo = self.yolo_model.names[cls_id] # Renomeado para evitar conflito
                        conf = float(box.conf[0])

                        if display_frame is not None:
                            label = f"{class_name_yolo} {conf:.2f}"
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        # Verifica se class_name_yolo está em alguma das listas dentro de DANGER_CLASSES.values()
                        is_dangerous = any(class_name_yolo in danger_list for danger_list in DANGER_CLASSES.values())
                        if is_dangerous and conf >= YOLO_CONFIDENCE_THRESHOLD:
                            yolo_alerts.append(class_name_yolo)
            except Exception as e:
                print(f"Erro na inferência YOLO: {e}")
                current_yolo_results = None
        elif self.show_preview and ret:
            display_frame = latest_frame_copy.copy()

        with self.frame_lock:
            if ret:
                self.latest_bgr_frame = latest_frame_copy
                self.latest_yolo_results = current_yolo_results
            else:
                self.latest_bgr_frame = None
                self.latest_yolo_results = None
                return None, []

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

        image_part = None
        if ret:
            try:
                if 'frame_rgb' not in locals() or frame_rgb is None: # Adicionado 'or frame_rgb is None'
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
        cap = None
        try:
            print("Iniciando captura da câmera...")
            cap = await asyncio.to_thread(cv2.VideoCapture, 0)
            target_fps = 1
            cap.set(cv2.CAP_PROP_FPS, target_fps)
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"FPS solicitado: {target_fps}, FPS real da câmera: {actual_fps if actual_fps > 0 else 'Não disponível'}")

            # Calcula sleep_interval com base no FPS real ou alvo, garantindo no mínimo 0.1s para evitar busy-looping excessivo
            # e no máximo 1.0s se FPS for muito baixo ou não detectado.
            if actual_fps > 0:
                sleep_interval = 1 / actual_fps
            else:
                sleep_interval = 1 / target_fps
            sleep_interval = max(0.1, min(sleep_interval, 1.0))


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

                image_part, yolo_alerts = await asyncio.to_thread(self._get_frame, cap)

                with self.frame_lock:
                    frame_was_read = self.latest_bgr_frame is not None

                if not frame_was_read:
                     if not cap.isOpened():
                         print("Leitura do frame falhou e câmera fechada. Encerrando get_frames.")
                         self.stop_event.set()
                         break
                     else:
                         print("Aviso: Falha temporária na leitura do frame.")
                         await asyncio.sleep(0.5) # Espera um pouco antes de tentar de novo
                         continue

                if image_part is not None and self.out_queue:
                    try:
                        if self.out_queue.full():
                            await self.out_queue.get()
                            # print("Aviso: Fila de saída cheia, descartando frame antigo.") # Log opcional
                        self.out_queue.put_nowait(image_part)
                    except asyncio.QueueFull:
                         pass # print("Aviso: Fila de saída cheia ao tentar enfileirar frame.") # Log opcional

                if yolo_alerts and self.session and self.session.is_connected:
                    for alert_class_name in yolo_alerts:
                        try:
                            alert_msg = f"Usuário, CUIDADO! {alert_class_name.upper()} detectado!"
                            if self.out_queue and self.out_queue.full(): # Verifica se out_queue existe
                                await self.out_queue.get()
                            # Envia diretamente, pois é urgente. Não passa pela out_queue principal de frames.
                            await self.session.send(input=alert_msg, end_of_turn=True)
                            print(f"ALERTA URGENTE ENVIADO: {alert_msg}")
                        except Exception as e:
                            print(f"Erro ao enviar alerta urgente: {e}")
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
                    cv2.destroyWindow("Trackie YOLO Preview") # Tenta fechar a janela específica
                    print("Janela de preview 'Trackie YOLO Preview' fechada.")
                except Exception as e_cv_destroy: # Se não funcionar, tenta fechar todas
                    try:
                        cv2.destroyAllWindows()
                        print("Todas as janelas OpenCV fechadas.")
                    except Exception as e_cv_destroy_all:
                        print(f"Aviso: erro ao tentar fechar janelas de preview: {e_cv_destroy_all}")
            self.preview_window_active = False # Garante que está False
            print("get_frames concluído.")

    def _get_screen(self) -> Optional[Dict[str, Any]]:
        sct = mss.mss()
        monitor_number = 1
        try:
            if len(sct.monitors) > monitor_number:
                 monitor = sct.monitors[monitor_number]
            elif sct.monitors: # Se não houver monitor 1, mas houver algum monitor
                 monitor = sct.monitors[0]
            else: # Nenhum monitor encontrado
                print("Erro: Nenhum monitor detectado por mss.")
                return None

            sct_img = sct.grab(monitor)
            img = Image.frombytes('RGB', sct_img.size, sct_img.rgb, 'raw', 'BGR') # MSS retorna BGR
            img = img.convert('RGB') # Converte para RGB se necessário para PNG
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
                         self.out_queue.put_nowait(frame_data)
                    except asyncio.QueueFull:
                         pass
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            print("get_screen cancelado.")
        except Exception as e:
            print(f"Erro crítico em get_screen: {e}")
            self.stop_event.set()
        finally:
            print("get_screen finalizado.")

    async def send_realtime(self):
        print("Send_realtime pronto para enviar dados...")
        try:
            while not self.stop_event.is_set():
                if self.thinking_event.is_set():
                    await asyncio.sleep(0.05)
                    continue

                if not self.out_queue: # Verifica se out_queue foi inicializada
                    await asyncio.sleep(0.1)
                    continue

                try:
                    msg = await asyncio.wait_for(self.out_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                except asyncio.QueueEmpty: # Adicionado para o caso de a fila estar vazia
                    continue


                if not self.session or not self.session.is_connected:
                    # print("Sessão Gemini não está ativa (send_realtime). Descartando mensagem.") # Log opcional
                    if self.out_queue: self.out_queue.task_done() # Certifica-se de que a tarefa é marcada como concluída
                    if not self.stop_event.is_set():
                        await asyncio.sleep(0.5)
                    continue

                try:
                    if isinstance(msg, dict) and "data" in msg and "mime_type" in msg : # Verifica se é uma mensagem multimodal válida
                        await self.session.send(input=msg)
                    elif isinstance(msg, str): # Para alertas ou texto informativo (embora não seja o uso principal aqui)
                        print(f"Enviando texto via send_realtime (raro): {msg}")
                        await self.session.send(input=msg, end_of_turn=False)
                    else:
                        print(f"Mensagem desconhecida em send_realtime: {type(msg)}")
                    if self.out_queue: self.out_queue.task_done()
                except Exception as e_send:
                    print(f"Erro ao enviar para Gemini em send_realtime: {e_send}")
                    if "LiveSession closed" in str(e_send) or "LiveSession not connected" in str(e_send) or "Deadline Exceeded" in str(e_send):
                        print("Sessão Gemini fechada ou não conectada (send_realtime). Sinalizando parada.")
                        self.stop_event.set()
                        break
                    else:
                        # traceback.print_exc() # Pode ser muito verboso
                        if self.out_queue: self.out_queue.task_done() # Garante task_done mesmo em erro
        except asyncio.CancelledError:
            print("send_realtime cancelado.")
        except Exception as e:
            print(f"Erro fatal em send_realtime: {e}")
            self.stop_event.set()
        finally:
            print("send_realtime finalizado.")

    async def listen_audio(self):
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

            while not self.stop_event.is_set():
                if self.thinking_event.is_set():
                    await asyncio.sleep(0.05)
                    continue

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
                                 # await self.out_queue.get() # Descomente se quiser descartar o mais antigo
                                 pass # print("Aviso: Fila de saída cheia, áudio pode ser atrasado.")
                             self.out_queue.put_nowait({"data": data, "mime_type": "audio/pcm"})
                         except asyncio.QueueFull:
                             pass # print("Aviso: Fila de saída cheia ao tentar enfileirar áudio.")
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
                    self.stop_event.set() # Para a tarefa em caso de erro de leitura
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
                try:
                    if audio_stream.is_active():
                        audio_stream.stop_stream()
                    audio_stream.close()
                    print("Stream de áudio de entrada fechado.")
                except Exception as e_close_stream:
                    print(f"Erro ao fechar stream de áudio de entrada: {e_close_stream}")
            print("listen_audio concluído.")

    def _handle_save_known_face(self, person_name: str) -> str:
        print(f"[DeepFace] Iniciando salvamento para: {person_name}")
        frame_to_process = None
        with self.frame_lock:
            if self.latest_bgr_frame is not None:
                frame_to_process = self.latest_bgr_frame.copy()

        if frame_to_process is None:
            print("[DeepFace] Erro: Nenhum frame disponível para salvar.")
            return "Não foi possível capturar a imagem para salvar o rosto."

        safe_person_name_dir = "".join(c if c.isalnum() or c in [' '] else '_' for c in person_name).strip().replace(" ", "_")
        if not safe_person_name_dir: safe_person_name_dir = "desconhecido" # Evita nome de diretório vazio
        person_dir = os.path.join(DB_PATH, safe_person_name_dir)

        try:
            if not os.path.exists(person_dir):
                os.makedirs(person_dir)
                print(f"[DeepFace] Diretório criado: {person_dir}")

            # Usar enforce_detection=True para garantir que um rosto seja detectado
            detected_faces = DeepFace.extract_faces(
                img_path=frame_to_process, detector_backend=DEEPFACE_DETECTOR_BACKEND,
                enforce_detection=True, align=True, silent=True # silent=True para menos output
            )

            if not detected_faces or not isinstance(detected_faces, list) or 'facial_area' not in detected_faces[0]:
                print(f"[DeepFace] Nenhum rosto detectado para {person_name}.")
                return f"Não consegui detectar um rosto claro para {person_name}."

            # Pega o primeiro rosto detectado (o mais proeminente)
            face_data = detected_faces[0]['facial_area']
            x, y, w, h = face_data['x'], face_data['y'], face_data['w'], face_data['h']

            # Adiciona uma pequena margem para garantir que o rosto inteiro seja capturado
            margin = 10
            y1, y2 = max(0, y - margin), min(frame_to_process.shape[0], y + h + margin)
            x1, x2 = max(0, x - margin), min(frame_to_process.shape[1], x + w + margin)
            face_image = frame_to_process[y1:y2, x1:x2]

            if face_image.size == 0:
                 print(f"[DeepFace] Erro ao recortar rosto para {person_name} (imagem vazia).")
                 return f"Erro ao processar o rosto de {person_name}."

            timestamp = int(time.time())
            safe_file_name_base = "".join(c if c.isalnum() else '_' for c in person_name).strip()
            if not safe_file_name_base: safe_file_name_base = "rosto"
            file_name = f"{safe_file_name_base.lower()}_{timestamp}.jpg"
            file_path = os.path.join(person_dir, file_name)

            cv2.imwrite(file_path, face_image)

            # Remove o arquivo .pkl de representações para forçar a recriação na próxima busca/identificação
            model_name_safe = DEEPFACE_MODEL_NAME.lower().replace('-', '_') # Adapta ao nome do arquivo .pkl
            representations_pkl_path = os.path.join(DB_PATH, f"representations_{model_name_safe}.pkl")
            if os.path.exists(representations_pkl_path):
                try:
                    os.remove(representations_pkl_path)
                    print(f"[DeepFace] Cache de representações '{representations_pkl_path}' removido para atualização.")
                except Exception as e_pkl:
                    print(f"[DeepFace] Aviso: Falha ao remover cache de representações: {e_pkl}")

            print(f"[DeepFace] Rosto de {person_name} salvo em {file_path}")
            return f"Rosto de {person_name} salvo com sucesso."
        except ValueError as ve: # Captura ValueError de DeepFace quando nenhum rosto é detectado com enforce_detection=True
             print(f"[DeepFace] Nenhum rosto detectado (ValueError) para {person_name}: {ve}")
             return f"Não consegui detectar um rosto claro para salvar para {person_name}."
        except Exception as e:
            print(f"[DeepFace] Erro ao salvar rosto para {person_name}: {e}")
            traceback.print_exc()
            return f"Ocorreu um erro ao tentar salvar o rosto de {person_name}."

    def _handle_identify_person_in_front(self) -> str:
        print("[DeepFace] Iniciando identificação de pessoa...")
        frame_to_process = None
        with self.frame_lock:
            if self.latest_bgr_frame is not None:
                frame_to_process = self.latest_bgr_frame.copy()

        if frame_to_process is None:
            print("[DeepFace] Erro: Nenhum frame disponível para identificar.")
            return "Não foi possível capturar a imagem para identificar."

        try:
            # enforce_detection=False pode ser melhor para `find` se quisermos tentar mesmo com rostos não perfeitos,
            # mas o prompt sugere que deve ser um rosto claro. Vamos manter True para consistência com salvar.
            # Ou False para `find` para ser mais permissivo e deixar o threshold de distância decidir.
            # O padrão de DeepFace.find é enforce_detection=False.
            dfs = DeepFace.find(
                img_path=frame_to_process, db_path=DB_PATH,
                model_name=DEEPFACE_MODEL_NAME, detector_backend=DEEPFACE_DETECTOR_BACKEND,
                distance_metric=DEEPFACE_DISTANCE_METRIC,
                enforce_detection=True, # Alterado para True para buscar rostos mais claros
                silent=True, align=True
            )

            if not dfs or not isinstance(dfs, list) or not dfs[0].size > 0: # Verifica se o DataFrame não está vazio
                print("[DeepFace] Nenhuma correspondência encontrada ou rosto não detectado claramente.")
                return "Não consegui reconhecer ninguém ou não detectei um rosto claro."

            df = dfs[0] # O resultado é uma lista de DataFrames, pegamos o primeiro
            if df.empty:
                print("[DeepFace] DataFrame de resultados vazio.")
                return "Não consegui reconhecer ninguém."

            # Ordena por distância (menor é melhor) e pega o melhor
            # A coluna de distância pode variar de nome, ex: 'VGG-Face_cosine'
            distance_col_name = None
            # Tenta encontrar a coluna de distância correta
            expected_distance_col = f"{DEEPFACE_MODEL_NAME}_{DEEPFACE_DISTANCE_METRIC}"
            if expected_distance_col in df.columns:
                distance_col_name = expected_distance_col
            elif 'distance' in df.columns: # Fallback genérico
                distance_col_name = 'distance'
            else: # Tenta encontrar qualquer coluna que contenha a métrica
                for col in df.columns:
                    if DEEPFACE_DISTANCE_METRIC in col.lower():
                        distance_col_name = col
                        break

            if distance_col_name is None:
                print(f"[DeepFace] Erro: Coluna de distância não encontrada no DataFrame. Colunas: {df.columns.tolist()}")
                return "Erro ao processar resultado da identificação (coluna de distância)."

            df = df.sort_values(by=distance_col_name, ascending=True)
            best_match = df.iloc[0]

            best_match_identity_path = best_match['identity']
            # O nome da pessoa é o nome do diretório pai do arquivo de imagem
            person_name = os.path.basename(os.path.dirname(best_match_identity_path))
            distance = best_match[distance_col_name]

            print(f"[DeepFace] Pessoa potencialmente identificada: {person_name} (Distância: {distance:.4f})")

            # Thresholds de exemplo (precisam ser ajustados experimentalmente)
            thresholds = {
                'VGG-Face': {'cosine': 0.40, 'euclidean': 0.60, 'euclidean_l2': 0.86},
                'Facenet': {'cosine': 0.40, 'euclidean': 0.90, 'euclidean_l2': 1.10},
                'Facenet512': {'cosine': 0.30, 'euclidean': 0.70, 'euclidean_l2': 0.95},
                'ArcFace': {'cosine': 0.68, 'euclidean': 1.13, 'euclidean_l2': 1.13}, # ArcFace usa L2 para euclidiana
                'Dlib': {'cosine': 0.07, 'euclidean': 0.6, 'euclidean_l2': 0.6}, # Dlib é mais sensível
                # Adicione outros modelos se necessário
            }
            threshold = thresholds.get(DEEPFACE_MODEL_NAME, {}).get(DEEPFACE_DISTANCE_METRIC, 0.5) # Default threshold

            if distance > threshold:
                print(f"[DeepFace] Distância {distance:.4f} > limiar ({threshold}). Não reconhecido com confiança.")
                return "Não tenho certeza de quem é, mas pode ser alguém que não reconheço bem."

            return f"A pessoa na sua frente parece ser {person_name}."
        except ValueError as ve: # Captura erro se enforce_detection=True e nenhum rosto for encontrado
            print(f"[DeepFace] Erro (ValueError) ao identificar: {ve}")
            return "Não detectei um rosto claro para identificar."
        except Exception as e:
            print(f"[DeepFace] Erro inesperado ao identificar: {e}")
            traceback.print_exc()
            return "Ocorreu um erro ao tentar identificar a pessoa."

    def _run_midas_inference(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        if not self.midas_model or not self.midas_transform:
            print("[MiDaS] Modelo ou transformador não carregado.")
            return None
        try:
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            input_batch = self.midas_transform(img_rgb).to(self.midas_device)
            with torch.no_grad():
                prediction = self.midas_model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            depth_map = prediction.cpu().numpy()
            return depth_map
        except Exception as e:
            print(f"[MiDaS] Erro durante a inferência: {e}")
            # traceback.print_exc() # Pode ser muito verboso
            return None

    def _find_best_yolo_match(self, object_type: str, yolo_results: List[Any]) -> Optional[Tuple[Dict[str, int], float, str]]:
        best_match = None
        highest_conf = -1.0
        target_yolo_classes = YOLO_CLASS_MAP.get(object_type.lower(), [object_type.lower()])

        if not yolo_results or not self.yolo_model: # Adicionado self.yolo_model
             return None

        for result in yolo_results:
            if hasattr(result, 'boxes') and result.boxes: # Verifica se há caixas neste resultado
                for box in result.boxes:
                    if not (hasattr(box, 'cls') and hasattr(box, 'conf') and hasattr(box, 'xyxy')):
                        continue # Pula caixas malformadas

                    cls_id_tensor = box.cls
                    if cls_id_tensor.nelement() == 0: continue # Tensor vazio
                    cls_id = int(cls_id_tensor[0])

                    conf_tensor = box.conf
                    if conf_tensor.nelement() == 0: continue
                    conf = float(conf_tensor[0])

                    class_name = self.yolo_model.names[cls_id]

                    if class_name in target_yolo_classes:
                        if conf > highest_conf:
                            highest_conf = conf
                            coords_tensor = box.xyxy[0]
                            if coords_tensor.nelement() < 4: continue
                            coords = list(map(int, coords_tensor))
                            bbox_dict = {'x1': coords[0], 'y1': coords[1], 'x2': coords[2], 'y2': coords[3]}
                            best_match = (bbox_dict, conf, class_name)
        return best_match

    def _estimate_direction(self, bbox: Dict[str, int], frame_width: int) -> str:
        box_center_x = (bbox['x1'] + bbox['x2']) / 2
        center_zone_width = frame_width / 3

        if box_center_x < center_zone_width:
            return "à sua esquerda"
        elif box_center_x > (frame_width - center_zone_width):
            return "à sua direita"
        else:
            return "à sua frente"

    def _check_if_on_surface(self, target_bbox: Dict[str, int], yolo_results: List[Any]) -> bool:
        surface_classes_keys = ["mesa", "mesa de jantar"] # Adicionado "mesa de jantar"
        surface_yolo_names = []
        for key in surface_classes_keys:
            surface_yolo_names.extend(YOLO_CLASS_MAP.get(key, []))
        surface_yolo_names = list(set(surface_yolo_names)) # Remove duplicatas

        if not surface_yolo_names: return False # Se não há classes de superfície mapeadas

        target_bottom_y = target_bbox['y2']
        target_center_x = (target_bbox['x1'] + target_bbox['x2']) / 2

        if not yolo_results or not self.yolo_model: # Adicionado self.yolo_model
            return False

        for result in yolo_results:
             if hasattr(result, 'boxes') and result.boxes:
                for box in result.boxes:
                    if not (hasattr(box, 'cls') and hasattr(box, 'xyxy')):
                        continue

                    cls_id_tensor = box.cls
                    if cls_id_tensor.nelement() == 0: continue
                    cls_id = int(cls_id_tensor[0])

                    class_name = self.yolo_model.names[cls_id]

                    if class_name in surface_yolo_names:
                        coords_tensor = box.xyxy[0]
                        if coords_tensor.nelement() < 4: continue
                        s_x1, s_y1, s_x2, s_y2 = map(int, coords_tensor)

                        # Heurística melhorada:
                        # 1. O centro X do objeto alvo está dentro da largura da superfície.
                        # 2. A base do objeto alvo (target_bottom_y) está acima do topo da superfície (s_y1)
                        #    mas não muito acima (para evitar objetos flutuando muito alto serem considerados "sobre").
                        # 3. A base do objeto alvo está abaixo ou muito próxima da base da superfície (s_y2)
                        #    (para garantir que não está completamente abaixo da superfície).
                        # Tolerância para "próximo ao topo":
                        y_tolerance = 30 # pixels (ajuste conforme necessário)

                        on_top_of_surface = (s_y1 - y_tolerance) < target_bottom_y < (s_y1 + y_tolerance * 2) # Objeto repousa perto do topo
                        horizontally_aligned = s_x1 < target_center_x < s_x2
                        # Adicional: objeto não deve estar completamente abaixo da superfície
                        # not_completely_under = target_bottom_y < s_y2 + y_tolerance

                        if horizontally_aligned and on_top_of_surface: # and not_completely_under:
                            return True
        return False

    def _handle_find_object_and_estimate_distance(self, object_description: str, object_type: str) -> str:
        print(f"[Localizar Objeto] Buscando por '{object_description}' (tipo: '{object_type}')...")
        frame_to_process = None
        yolo_results_for_frame = None
        frame_height, frame_width = 0, 0

        with self.frame_lock:
            if self.latest_bgr_frame is not None:
                frame_to_process = self.latest_bgr_frame.copy()
                yolo_results_for_frame = self.latest_yolo_results
                if frame_to_process is not None: # Adicionado para segurança
                    frame_height, frame_width, _ = frame_to_process.shape
            else:
                 print("[Localizar Objeto] Erro: Nenhum frame disponível.")
                 return f"Usuário, não estou enxergando nada no momento para localizar o {object_type}."

        if frame_width == 0 or frame_height == 0: # Checagem adicional
            print("[Localizar Objeto] Erro: Dimensões do frame inválidas.")
            return f"Usuário, problema ao processar a imagem para localizar o {object_type}."


        if not yolo_results_for_frame:
            print("[Localizar Objeto] Erro: Nenhum resultado YOLO disponível para o frame atual.")
            return f"Usuário, não consegui processar a imagem para encontrar o {object_type} (sem YOLO results)."

        best_yolo_match = self._find_best_yolo_match(object_type, yolo_results_for_frame)

        if not best_yolo_match:
            print(f"[Localizar Objeto] Nenhum objeto do tipo '{object_type}' encontrado pelo YOLO.")
            # Tenta com a descrição completa se o tipo falhar, como um fallback simples
            # Isso é uma heurística e pode não ser muito eficaz sem NLP mais avançado.
            if object_type.lower() != object_description.lower().split(" ")[-1]: # Evita redundância
                print(f"[Localizar Objeto] Tentando com descrição completa '{object_description}' como tipo.")
                best_yolo_match = self._find_best_yolo_match(object_description.split(" ")[-1], yolo_results_for_frame) # Usa a última palavra
                if not best_yolo_match:
                     return f"Usuário, não consegui encontrar um(a) {object_type} (ou {object_description}) na imagem."
            else:
                return f"Usuário, não consegui encontrar um(a) {object_type} na imagem."


        target_bbox, confidence, detected_class = best_yolo_match
        print(f"[Localizar Objeto] Melhor correspondência YOLO: Classe '{detected_class}', Conf: {confidence:.2f}, BBox: {target_bbox}")

        is_on_surface = self._check_if_on_surface(target_bbox, yolo_results_for_frame)
        surface_msg = "sobre uma superfície (como uma mesa)" if is_on_surface else ""

        direction = self._estimate_direction(target_bbox, frame_width)
        depth_map = None
        if self.midas_model:
            print("[Localizar Objeto] Executando MiDaS...")
            depth_map = self._run_midas_inference(frame_to_process)

        distance_steps = -1
        if depth_map is not None:
            try:
                center_x = int((target_bbox['x1'] + target_bbox['x2']) / 2)
                center_y = int((target_bbox['y1'] + target_bbox['y2']) / 2)
                center_y = max(0, min(center_y, depth_map.shape[0] - 1))
                center_x = max(0, min(center_x, depth_map.shape[1] - 1))
                depth_value = depth_map[center_y, center_x]

                # Heurística para MiDaS_small (saída é profundidade inversa relativa, não métrica)
                # Valores maiores de depth_value significam mais perto.
                # Precisamos de uma calibração ou uma função de mapeamento mais sofisticada.
                # Esta é uma estimativa MUITO BRUTA.
                if depth_value > 1e-6: # Evita divisão por zero ou valores muito pequenos
                    # Quanto maior o depth_value, menor a "distância_relativa_inversa"
                    # Vamos tentar mapear: um depth_value alto (ex: 300-500 para MiDaS_small) = perto (1-2 passos)
                    # um depth_value baixo (ex: < 50) = longe (>10 passos)
                    # Isso é altamente dependente da cena e do modelo.
                    if depth_value > 250: # Muito perto
                        estimated_meters = np.random.uniform(0.5, 1.5) # 0.5 a 1.5 metros
                    elif depth_value > 100: # Perto/Médio
                        estimated_meters = np.random.uniform(1.5, 3.5)
                    elif depth_value > 30: # Médio/Longe
                        estimated_meters = np.random.uniform(3.5, 7.0)
                    else: # Longe
                        estimated_meters = np.random.uniform(7.0, 15.0)

                    estimated_meters = max(0.5, min(estimated_meters, 20)) # Limita
                    distance_steps = round(estimated_meters / METERS_PER_STEP)
                    print(f"[Localizar Objeto] Profundidade MiDaS no centro ({center_y},{center_x}): {depth_value:.4f}, Metros Estimados (heurístico): {estimated_meters:.2f}, Passos: {distance_steps}")
                else:
                     print("[Localizar Objeto] Valor de profundidade MiDaS inválido ou muito baixo no centro do objeto.")
            except Exception as e_depth:
                print(f"[Localizar Objeto] Erro ao extrair profundidade do MiDaS: {e_depth}")
                distance_steps = -1
        else:
             print("[Localizar Objeto] MiDaS não disponível ou falhou. Não é possível estimar distância.")

        # Usa object_description para a resposta, pois é mais específico que object_type
        object_name_for_response = object_description

        response_parts = [f"Usuário, o {object_name_for_response} está"]
        if surface_msg:
            response_parts.append(surface_msg)

        if distance_steps > 0:
            response_parts.append(f"a aproximadamente {distance_steps} passos")

        response_parts.append(direction + ".") # Adiciona a direção e o ponto final

        # Remove "está" se for a única palavra após "Usuário, o objeto..."
        if len(response_parts) == 2 and response_parts[1].endswith("está"):
            return f"Usuário, não consegui localizar o {object_name_for_response} com detalhes."

        return " ".join(p for p in response_parts if p) # Junta as partes não vazias


    async def receive_audio(self):
        print("Receive_audio pronto para receber respostas do Gemini...")
        try:
            if not self.session:
                print("Sessão Gemini não estabelecida em receive_audio. Encerrando.")
                self.stop_event.set()
                return

            while not self.stop_event.is_set():
                if not self.session:
                    print("Sessão Gemini desconectada em receive_audio. Tentando reconectar ou aguardando.")
                    await asyncio.sleep(1) # Espera antes de verificar de novo
                    if not self.session: # Verifica novamente
                        self.stop_event.set() # Se ainda não conectada, sinaliza parada para o loop run tentar reconectar
                        break
                    else:
                        print("Sessão Gemini reconectada.")


                try:
                    has_received_data_in_turn = False
                    # print("Aguardando próximo turno de resposta do Gemini...") # Log opcional
                    async for response_part in self.session.receive():
                        has_received_data_in_turn = True

                        if self.stop_event.is_set():
                            print("Sinal de parada recebido durante processamento de resposta.")
                            break

                        if response_part.data: # Áudio da resposta do Gemini
                            if self.audio_in_queue:
                                self.audio_in_queue.put_nowait(response_part.data)
                            continue

                        # --- Tratamento para nome pendente de save_known_face ---
                        if self.awaiting_name_for_save_face:
                            user_provided_name = None
                            if response_part.text: # Gemini transcreveu o áudio ou usuário digitou texto
                                user_provided_name = response_part.text.strip()

                            # Se não houver texto, mas houver um "final de turno de áudio" implícito,
                            # pode ser necessário um mecanismo mais complexo para capturar a fala do usuário.
                            # Por ora, confiamos que o Gemini fornecerá o texto se o usuário falar.

                            if user_provided_name:
                                print(f"[Trackie] Recebido nome '{user_provided_name}' para salvar rosto. Processando...")
                                self.awaiting_name_for_save_face = False

                                original_function_name_pending = "save_known_face"

                                print("Pensando...")
                                self.thinking_event.set()
                                voice_feedback_msg = f"Usuário, salvando rosto de {user_provided_name}, um momento..."
                                if self.session and self.session.is_connected:
                                    try:
                                        await self.session.send(input=voice_feedback_msg, end_of_turn=True)
                                        print(f"  [Feedback Enviado]: {voice_feedback_msg}")
                                    except Exception as e_feedback:
                                        print(f"Erro ao enviar feedback (awaiting name): {e_feedback}")

                                result_message = await asyncio.to_thread(self._handle_save_known_face, user_provided_name)

                                print(f"  [Trackie] Resultado da Função '{original_function_name_pending}': {result_message}")
                                if self.session and self.session.is_connected:
                                    try:
                                        await self.session.send(
                                            input=types.Content(
                                                role="tool",
                                                parts=[types.Part.from_function_response(
                                                    name=original_function_name_pending,
                                                    response={"result": Value(string_value=result_message)}
                                                )]
                                            )
                                        )
                                        print("  [Trackie] Resultado da função (awaiting name) enviado.")
                                    except Exception as e_send_fc_resp:
                                        print(f"Erro ao enviar FunctionResponse (awaiting name): {e_send_fc_resp}")
                                else:
                                    print("  [Trackie] Sessão inativa. Não foi possível enviar resultado da função (awaiting name).")

                                if self.thinking_event.is_set():
                                    self.thinking_event.clear()
                                continue # Processamos este input, vamos para o próximo response_part

                        if response_part.text:
                            print(f"\n[Gemini Texto]: {response_part.text}", end="") # Adicionado "Texto" para clareza

                        if getattr(response_part, "function_call", None):
                            fc = response_part.function_call
                            function_name = fc.name
                            args = {key: val for key, val in fc.args.items()}
                            print(f"\n[Gemini Function Call] Recebido: {function_name}, Args: {args}")

                            result_message = None # Inicializa para None

                            # Caso especial: save_known_face sem nome é tratado primeiro
                            if function_name == "save_known_face" and not args.get("person_name"):
                                self.awaiting_name_for_save_face = True
                                if self.thinking_event.is_set(): # Garante que não está pensando enquanto pergunta
                                    self.thinking_event.clear()
                                print("[Trackie] Nome não fornecido para save_known_face. Solicitando ao usuário.")
                                if self.session and self.session.is_connected:
                                    try:
                                        await self.session.send(input="Usuário, por favor forneça o nome da pessoa para salvar o rosto.", end_of_turn=True)
                                    except Exception as e_ask_name:
                                        print(f"Erro ao pedir nome para save_face: {e_ask_name}")
                                # result_message permanece None, FC não será enviado neste turno
                            else:
                                # Para todas as outras chamadas de função diretas ou save_known_face com nome:
                                print("Pensando...")
                                self.thinking_event.set()

                                voice_feedback_msg = f"Usuário, processando {function_name}, um momento..."
                                if function_name == "save_known_face": # Nome deve existir aqui
                                    person_name_fb = args.get('person_name', 'pessoa') # Fallback
                                    voice_feedback_msg = f"Usuário, salvando rosto de {person_name_fb}, um momento..."
                                elif function_name == "identify_person_in_front":
                                    voice_feedback_msg = "Usuário, identificando pessoa, um momento..."
                                elif function_name == "find_object_and_estimate_distance":
                                    obj_desc_fb = args.get('object_description', 'objeto')
                                    voice_feedback_msg = f"Usuário, localizando {obj_desc_fb}, um momento..."

                                if self.session and self.session.is_connected:
                                    try:
                                        await self.session.send(input=voice_feedback_msg, end_of_turn=True)
                                        print(f"  [Feedback Enviado]: {voice_feedback_msg}")
                                    except Exception as e_feedback:
                                        print(f"Erro ao enviar feedback pré-função: {e_feedback}")

                                # Agora, execute a função
                                if self.video_mode != "camera" and (function_name in ["save_known_face", "identify_person_in_front", "find_object_and_estimate_distance"]):
                                    print(f"[Function Call] '{function_name}' requer modo câmera.")
                                    result_message = "As funções de visão só estão disponíveis no modo câmera."
                                else:
                                    print(f"  [Trackie] Processando Função '{function_name}' em background...")
                                    if function_name == "save_known_face":
                                        person_name_arg = args.get("person_name") # Deve existir aqui
                                        if person_name_arg: # Checagem extra
                                            result_message = await asyncio.to_thread(self._handle_save_known_face, person_name_arg)
                                        else: # Não deveria acontecer se a lógica acima estiver correta
                                            result_message = "Erro interno: nome não disponível para salvar rosto."
                                    elif function_name == "identify_person_in_front":
                                        result_message = await asyncio.to_thread(self._handle_identify_person_in_front)
                                    elif function_name == "find_object_and_estimate_distance":
                                        desc = args.get("object_description")
                                        obj_type = args.get("object_type")
                                        if desc and obj_type:
                                            if not self.midas_model:
                                                result_message = "Usuário, desculpe, o módulo de estimativa de distância não está funcionando."
                                            else:
                                                result_message = await asyncio.to_thread(
                                                    self._handle_find_object_and_estimate_distance, desc, obj_type
                                                )
                                        else:
                                            result_message = "Descrição ou tipo do objeto não fornecido para localização."
                                    else: # Função desconhecida
                                        result_message = f"Função '{function_name}' desconhecida."

                            # Enviar FunctionResponse se result_message foi definido
                            if result_message is not None:
                                print(f"  [Trackie] Resultado da Função '{function_name}': {result_message}")
                                if self.session and self.session.is_connected:
                                    try:
                                        await self.session.send(
                                            input=types.Content(
                                                role="tool",
                                                parts=[types.Part.from_function_response(
                                                    name=function_name,
                                                    response={"result": Value(string_value=result_message)}
                                                )]
                                            )
                                        )
                                        print("  [Trackie] Resultado da função enviado.")
                                    except Exception as e_send_fc_resp_main:
                                        print(f"Erro ao enviar FunctionResponse (main): {e_send_fc_resp_main}")
                                else:
                                    print("  [Trackie] Sessão inativa. Não foi possível enviar resultado da função.")

                                if self.thinking_event.is_set():
                                     self.thinking_event.clear()
                            # Se result_message é None (caso de save_known_face pedindo nome),
                            # thinking_event já foi explicitamente limpo.

                    if not self.stop_event.is_set():
                        if has_received_data_in_turn:
                            # print("\nFim do turno de resposta do Gemini.") # Log opcional
                            pass
                        else:
                            # print("Stream do turno atual terminou sem dados. Verificando...") # Log opcional
                            await asyncio.sleep(0.05) # Pequena pausa se não houve dados
                    if self.stop_event.is_set():
                        break
                except Exception as e_inner_loop:
                    print(f"Erro durante o recebimento/processamento de resposta: {e_inner_loop}")
                    if "LiveSession closed" in str(e_inner_loop) or "LiveSession not connected" in str(e_inner_loop) or "Deadline Exceeded" in str(e_inner_loop):
                        print("Erro indica que a sessão Gemini foi fechada. Sinalizando parada.")
                        self.stop_event.set()
                        break
                    else:
                        # traceback.print_exc() # Pode ser muito verboso
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
        if not pya:
            print("PyAudio não inicializado. Tarefa play_audio não pode iniciar.")
            return

        stream = None
        try:
            print("Configurando stream de áudio de saída...")
            # Tenta obter informações do dispositivo de saída padrão para taxa de amostragem, se possível
            # Mas usa RECEIVE_SAMPLE_RATE como fallback ou se a configuração do dispositivo for complexa
            try:
                out_device_info = pya.get_default_output_device_info()
                output_rate = int(out_device_info.get('defaultSampleRate', RECEIVE_SAMPLE_RATE))
                print(f"Usando taxa de amostragem do dispositivo de saída: {output_rate} Hz")
            except Exception:
                output_rate = RECEIVE_SAMPLE_RATE
                print(f"Usando taxa de amostragem padrão para saída: {output_rate} Hz")


            stream = await asyncio.to_thread(
                pya.open, format=FORMAT, channels=CHANNELS, rate=output_rate, output=True
            )
            print("Player de áudio pronto.")

            while not self.stop_event.is_set():
                if not self.audio_in_queue: # Verifica se a fila existe
                    await asyncio.sleep(0.1)
                    continue
                try:
                    bytestream = await asyncio.wait_for(self.audio_in_queue.get(), timeout=0.5)
                    if bytestream is None: # Sinal de parada
                        print("Recebido sinal de encerramento (None) para play_audio.")
                        # self.stop_event.set() # Não seta stop_event aqui, deixa o run gerenciar
                        break

                    if stream and stream.is_active():
                        await asyncio.to_thread(stream.write, bytestream)
                    else:
                        print("Stream de áudio para playback não está ativo. Descartando áudio.")

                    if self.audio_in_queue: self.audio_in_queue.task_done()
                except asyncio.TimeoutError:
                    continue
                except asyncio.QueueEmpty: # Adicionado
                    continue
                except OSError as e_os_play:
                    if "Stream closed" in str(e_os_play):
                        print("Stream de playback fechado (OSError). Encerrando play_audio.")
                        # self.stop_event.set() # Não seta stop_event aqui
                        break
                    else:
                        print(f"Erro de OS ao reproduzir áudio: {e_os_play}")
                except Exception as e_inner:
                    print(f"Erro ao reproduzir áudio (interno): {e_inner}")
                    if "Stream closed" in str(e_inner): # Verifica se o erro indica stream fechado
                        # self.stop_event.set() # Não seta stop_event aqui
                        break
                    # traceback.print_exc() # Pode ser verboso
        except asyncio.CancelledError:
            print("play_audio foi cancelado.")
        except Exception as e:
            print(f"Erro crítico em play_audio: {e}")
            traceback.print_exc()
            # self.stop_event.set() # Não seta stop_event aqui, deixa o run gerenciar
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
        print("Iniciando AudioLoop...")
        max_retries = 3
        retry_delay_base = 1.0

        attempt = 0
        while attempt <= max_retries and not self.stop_event.is_set():
            retry_delay = retry_delay_base * (2 ** attempt) # Backoff exponencial
            try:
                if attempt > 0:
                     print(f"Tentativa de reconexão {attempt}/{max_retries} após {retry_delay:.1f}s...")
                     await asyncio.sleep(retry_delay)

                # Limpa estados que dependem da sessão antes de reconectar
                self.session = None
                self.audio_in_queue = None # Será recriado
                self.out_queue = None      # Será recriado
                self.awaiting_name_for_save_face = False # Reseta estado da função


                async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                    self.session = session
                    print(f"Sessão Gemini LiveConnect estabelecida (Tentativa {attempt+1}). ID: {'N/A' if not hasattr(session, 'session_id') else session.session_id}")
                    attempt = 0 # Reseta tentativas em caso de sucesso

                    # Recria filas para a nova sessão
                    self.audio_in_queue = asyncio.Queue()
                    self.out_queue = asyncio.Queue(maxsize=100) # Ou o tamanho que preferir

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
                        print("Todas as tarefas da sessão iniciadas. Aguardando conclusão ou parada...")

                    print("TaskGroup da sessão finalizado.")
                    if not self.stop_event.is_set():
                         print("Sessão Gemini terminou ou TaskGroup concluído. Tentando reconectar se houver tentativas restantes.")
                         attempt += 1
                    else:
                        break


            except asyncio.CancelledError:
                print("Loop principal (run) cancelado.")
                self.stop_event.set()
                break
            except ExceptionGroup as eg:
                print("Um ou mais erros ocorreram nas tarefas do TaskGroup:")
                self.stop_event.set()
                for i, exc in enumerate(eg.exceptions):
                    print(f"  Erro {i+1} na tarefa: {type(exc).__name__} - {exc}")
                attempt += 1
                self.session = None
            # except genai.types.LiveSessionError as lse: # <-- BLOCO REMOVIDO
            #     print(f"Erro específico da LiveSession Gemini: {lse}")
            #     # traceback.print_exc()
            #     attempt += 1
            #     self.session = None
            #     if "RST_STREAM" in str(lse) or "UNAVAILABLE" in str(lse) or "DEADLINE_EXCEEDED" in str(lse):
            #         print("Erro de conexão com Gemini (RST_STREAM/UNAVAILABLE/DEADLINE).")
            except Exception as e: # Este bloco agora capturará os erros que antes seriam LiveSessionError
                print(f"Erro ao conectar ou erro inesperado no método run: {type(e).__name__} - {e}")
                traceback.print_exc() # Mantém o traceback para diagnóstico

                # Lógica de print adicional para erros específicos de sessão/conexão
                error_message_str = str(e)
                # Convertendo para maiúsculas para correspondência insensível a maiúsculas e minúsculas, como em outras partes do código
                error_message_upper = error_message_str.upper()

                if "RST_STREAM" in error_message_upper or \
                   "UNAVAILABLE" in error_message_upper or \
                   "DEADLINE_EXCEEDED" in error_message_upper or \
                   "LIVESESSION CLOSED" in error_message_upper or \
                   "LIVESESSION NOT CONNECTED" in error_message_upper or \
                   "CONNECTIONCLOSEDERROR" in error_message_upper: # Adicionando mais strings de erro comuns
                    print(f"Detectado erro relacionado à sessão ou conexão Gemini: {error_message_str}")

                attempt += 1
                self.session = None
                if attempt > max_retries:
                     print("Máximo de tentativas de reconexão atingido após erro. Encerrando.")
                     self.stop_event.set()
                     break

        if not self.stop_event.is_set() and attempt > max_retries:
             print("Não foi possível restabelecer a conexão com Gemini após múltiplas tentativas.")
             self.stop_event.set()

        print("Limpando recursos em AudioLoop.run()...")
        self.stop_event.set()

        if self.session and self.session.is_connected:
            try:
                print("Fechando sessão LiveConnect ativa...")
                await self.session.close()
                print("Sessão LiveConnect fechada.")
            except Exception as e_close_session:
                print(f"Erro ao fechar sessão LiveConnect: {e_close_session}")
        self.session = None

        if self.audio_in_queue:
            try:
                if not self.audio_in_queue.full():
                    self.audio_in_queue.put_nowait(None)
            except asyncio.QueueFull: pass
            except Exception as e_q_put: print(f"Erro ao colocar None na audio_in_queue: {e_q_put}")


        if self.preview_window_active:
            try:
                cv2.destroyAllWindows()
                print("Janelas OpenCV destruídas no finally de run.")
            except Exception as e_cv_destroy_all:
                 print(f"Aviso: erro ao tentar fechar janelas de preview no final: {e_cv_destroy_all}")
            self.preview_window_active = False

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


    if args.mode == "camera":
        if not os.path.exists(YOLO_MODEL_PATH):
            print(f"Erro: Modelo YOLO '{YOLO_MODEL_PATH}' Não encontrado. Verifique o caminho.")
            exit(1)

    main_loop = None
    try:
        if not pya:
             print("Erro fatal: PyAudio não pôde ser inicializado. Verifique a instalação e dependências. Encerrando.")
             exit(1)

        main_loop = AudioLoop(video_mode=args.mode, show_preview=show_actual_preview)
        asyncio.run(main_loop.run())

    except KeyboardInterrupt:
        print("\nInterrupção pelo teclado recebida. Encerrando...")
        if main_loop: main_loop.stop_event.set()
    except AttributeError as ae: # Mantido para pegar outros AttributeErrors se surgirem
        print(f"Erro de atributo no bloco __main__: {ae}")
        traceback.print_exc()
        if main_loop: main_loop.stop_event.set()
    except Exception as e:
        print(f"Erro inesperado no bloco __main__: {type(e).__name__}: {e}")
        traceback.print_exc()
        if main_loop: main_loop.stop_event.set()
    finally:
        print("Bloco __main__ finalizado.")
        print("Programa completamente finalizado.")

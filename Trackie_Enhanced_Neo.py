```python
import os
import asyncio
import base64
import io
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
import traceback
import time
import argparse
import json
import threading
from typing import Dict, Any, Optional, List, Tuple

# Bibliotecas de Terceiros
import cv2
import pyaudio
from PIL import Image
import mss
# Tenta importar pandas, necessário para DeepFace.find
try:
    import pandas as pd
except ImportError:
    logger.info("AVISO: Biblioteca 'pandas' não encontrada. 'identify_person_in_front' pode não funcionar.")
    logger.info("Instale com: pip install pandas")
    pd = None # Define como None se não encontrado

from google import genai
from google.genai import types
from google.genai import errors # Importado para referência
from google.protobuf.struct_pb2 import Value
from ultralytics import YOLO
import numpy as np
from deepface import DeepFace
import torch
# import torchvision # Removido se não usado diretamente, timm pode trazer o que precisa
# import timm # Removido se não usado diretamente, MiDaS via torch.hub lida com suas deps

# --- Constantes ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-1.5-flash-latest" # Atualizado para um modelo mais recente, se preferir
DEFAULT_MODE = "camera"
BaseDir = "C:/Users/Pedro H/Downloads/TrackiePowerSHell/" # Considere tornar dinâmico ou configurável

# --- Caminho para o arquivo de prompt ---
SYSTEM_INSTRUCTION_PATH = os.path.join(BaseDir,"UserSettings", "Prompt's", "TrckItcs.py")
CONFIG_PATH = os.path.join(BaseDir, "UserSettings", "trckconfig.json")

# YOLO
YOLO_MODEL_PATH = os.path.join(BaseDir,"WorkTools", "yolov8n.pt")
DANGER_CLASSES = {
    'faca':             ['knife'],
    'tesoura':          ['scissors'],
    'barbeador':        ['razor'],
    'serra':            ['saw'],
    'machado':          ['axe'],
    'machadinha':       ['hatchet'],
    'arma_de_fogo':     ['gun', 'firearm'], # Adicionado 'firearm'
    'pistola':          ['pistol'],
    'rifle':            ['rifle'],
    'espingarda':       ['shotgun'],
    'revólver':         ['revolver'],
    'bomba':            ['bomb'],
    'granada':          ['grenade'],
    'fogo':             ['fire'],
    'chama':            ['flame'],
    'fumaça':           ['smoke'],
    'isqueiro':         ['lighter'],
    'fósforos':         ['matches'],
    'fogão':            ['stove'],
    'superfície_quente':['hot surface'],
    'vela':             ['candle'],
    'queimador':        ['burner'],
    'fio_energizado':   ['live_wire', 'exposed wire'], # Adicionado 'exposed wire'
    'tomada_elétrica':  ['electric_outlet', 'power outlet'], # Adicionado 'power outlet'
    'bateria':          ['battery'], # Baterias podem ser perigosas se danificadas
    'vidro_quebrado':   ['broken_glass'],
    'estilhaço':        ['shard'],
    'agulha':           ['needle'],
    'seringa':          ['syringe'],
    'martelo':          ['hammer'], # Pode ser perigoso dependendo do contexto
    'chave_de_fenda':   ['wrench'], # Pode ser perigoso dependendo do contexto
    'furadeira':        ['drill'],  # Pode ser perigoso dependendo do contexto
    'motosserra':       ['chainsaw'],
    'carro':            ['car'], # Perigoso em movimento ou em situações específicas
    'motocicleta':      ['motorcycle'], # Perigoso em movimento
    'bicicleta':        ['bicycle'], # Menos perigoso, mas pode estar em DANGER_CLASSES para certos cenários
    'caminhão':         ['truck'],
    'ônibus':           ['bus'],
    'urso':             ['bear'],
    'cobra':            ['snake'],
    'aranha':           ['spider'], # Algumas são perigosas
    'jacaré':           ['alligator', 'crocodile'], # Adicionado 'crocodile'
    'penhasco':         ['cliff'],
    'buraco':           ['hole'], # Pode ser um perigo de queda
    'escada':           ['stairs', 'ladder'], # Perigo de queda
}
YOLO_CONFIDENCE_THRESHOLD = 0.40
YOLO_CLASS_MAP = {
    "pessoa": ["person"], "gato": ["cat"], "cachorro": ["dog"], "coelho": ["rabbit"], "urso": ["bear"],
    "elefante": ["elephant"], "zebra": ["zebra"], "girafa": ["giraffe"], "vaca": ["cow"], "cavalo": ["horse"],
    "ovelha": ["sheep"], "macaco": ["monkey"], "bicicleta": ["bicycle"], "moto": ["motorcycle"], "carro": ["car"],
    "ônibus": ["bus"], "trem": ["train"], "caminhão": ["truck"], "avião": ["airplane"], "barco": ["boat"],
    "skate": ["skateboard"], "prancha de surf": ["surfboard"], "tênis": ["tennis racket"], # Raquete de tênis
    "mesa de jantar": ["dining table"], "mesa": ["table", "desk"], # Removido "dining table" de "mesa" para evitar redundância se "mesa de jantar" for mais específico
    "cadeira": ["chair"], "sofá": ["couch", "sofa"], "cama": ["bed"], "vaso de planta": ["potted plant"],
    "banheiro": ["toilet"], "televisão": ["tv", "tv monitor"], "abajur": ["lamp"], "espelho": ["mirror"],
    "laptop": ["laptop"], "computador": ["computer", "desktop computer"], # Removido "tv" de "computador"
    "teclado": ["keyboard"], "mouse": ["mouse"], "controle remoto": ["remote"], "celular": ["cell phone"],
    "micro-ondas": ["microwave"], "forno": ["oven"], "torradeira": ["toaster"], "geladeira": ["refrigerator"],
    "caixa de som": ["speaker"], "câmera": ["camera"], "garrafa": ["bottle"], "copo": ["cup"],
    "taça de vinho": ["wine glass"], "taça": ["wine glass", "cup"], "prato": ["plate", "dish"], "tigela": ["bowl"],
    "garfo": ["fork"], "faca": ["knife"], "colher": ["spoon"], "panela": ["pan", "pot"],
    "frigideira": ["skillet", "frying pan"], "martelo": ["hammer"], "chave inglesa": ["wrench"],
    "furadeira": ["drill"], "parafusadeira": ["screwdriver", "power drill"], # "drill" pode ser "parafusadeira" também
    "serra": ["saw"], "roçadeira": ["brush cutter", "string trimmer"], "alicate": ["pliers"],
    "chave de fenda": ["screwdriver"], "lanterna": ["flashlight"], "fita métrica": ["tape measure"],
    "mochila": ["backpack"], "bolsa": ["handbag", "purse", "bag"], "carteira": ["wallet"],
    "óculos": ["glasses", "eyeglasses"], "relógio": ["clock", "watch"], "chinelo": ["sandal", "flip-flop"],
    "sapato": ["shoe"], "sanduíche": ["sandwich"], "hambúrguer": ["hamburger"], "banana": ["banana"],
    "maçã": ["apple"], "laranja": ["orange"], "bolo": ["cake"], "rosquinha": ["donut"], "pizza": ["pizza"],
    "cachorro-quente": ["hot dog"], "escova de dentes": ["toothbrush"], "secador de cabelo": ["hair drier", "hair dryer"],
    "cotonete": ["cotton swab"], "sacola plástica": ["plastic bag"], "livro": ["book"], "vaso": ["vase"],
    "bola": ["sports ball", "ball"], "bexiga": ["balloon"], "pipa": ["kite"], "luva": ["glove"],
    "skis": ["skis"], "snowboard": ["snowboard"], "tesoura": ["scissors"],
    # Adicionar mais se necessário
    "bancada": ["bench", "countertop"], "prateleira": ["shelf"],
}

# DeepFace
DB_PATH = os.path.join(BaseDir,"UserSettings", "known_faces")
DEEPFACE_MODEL_NAME = 'VGG-Face'
DEEPFACE_DETECTOR_BACKEND = 'opencv' # 'retinaface' ou 'mtcnn' podem ser mais precisos, mas mais pesados
DEEPFACE_DISTANCE_METRIC = 'cosine'

# MiDaS
MIDAS_MODEL_TYPE = "MiDaS_small" # Outras opções: "MiDaS", "DPT_Large", "DPT_Hybrid"
METERS_PER_STEP = 0.7 # Calibrar isso!

# --- Configuração do Cliente Gemini ---
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    try:
        from dotenv import load_dotenv
        dotenv_path = os.path.join(BaseDir, '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
            API_KEY = os.environ.get("GEMINI_API_KEY")
            if API_KEY:
                logger.info(f"Chave API carregada de: {dotenv_path}")
            else:
                logger.info(f"Chave API não encontrada no arquivo .env: {dotenv_path}")
        else:
             logger.info(f"Arquivo .env não encontrado em: {dotenv_path}")
    except ImportError:
        logger.info("Biblioteca python-dotenv não instalada. Não é possível carregar .env.")
    except Exception as e_env:
        logger.error(f"Erro ao carregar .env: {e_env}")

if not API_KEY:
    logger.warning("AVISO: Chave da API Gemini (GEMINI_API_KEY) não encontrada nas variáveis de ambiente ou .env.")
    # O código tentará inicializar o cliente mesmo assim, o que provavelmente falhará se a chave for necessária.
    # A inicialização do cliente abaixo lidará com o erro.

try:
    # Usa a variável API_KEY que foi tentada carregar. Se for None, o SDK pode falhar ou usar credenciais padrão (se configurado).
    client = genai.Client(api_key=API_KEY) # Removido http_options para usar defaults, a menos que v1alpha seja estritamente necessário
    logger.info("Cliente Gemini inicializado.")
except Exception as e_client:
    logger.error(f"ERRO CRÍTICO ao inicializar cliente Gemini: {e_client}")
    logger.info("Verifique a API Key (GEMINI_API_KEY) e a conexão.")
    client = None

# --- Ferramentas Gemini (Function Calling) ---
tools = [
    types.Tool(code_execution=types.ToolCodeExecution), # Para execução de código pela IA (se habilitado no modelo)
    types.Tool(google_search=types.GoogleSearch()),
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="save_known_face",
                description="Salva o rosto da pessoa atualmente em foco pela câmera. Se 'person_name' não for fornecido, a IA deve solicitar o nome ao usuário com uma mensagem clara, como 'Por favor, informe o nome da pessoa para salvar o rosto.' Após receber o nome, a função salva o rosto e confirma o salvamento com 'Rosto salvo com sucesso para [nome].' Se a captura falhar, retorna 'Erro: Não foi possível capturar o rosto. Tente novamente.'",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "person_name": types.Schema(
                            type=types.Type.STRING,
                            description="Nome da pessoa a ser salvo. Se omitido, a IA deve solicitar ao usuário."
                        )
                    },
                    # "person_name" não é estritamente 'required' aqui, pois a lógica lida com a ausência dele.
                )
            ),
            types.FunctionDeclaration(
                name="identify_person_in_front",
                description="Identifica a pessoa atualmente em foco pela câmera usando o banco de dados de rostos conhecidos. Deve ser chamado apenas quando o usuário expressa explicitamente a intenção de identificar uma pessoa. Se múltiplos rostos forem detectados, retorna o mais próximo. Inclui a confiança da identificação (ex: 'Identificado como [nome] com 95% de confiança.'). Se não houver correspondência, retorna 'Pessoa não reconhecida.'",
                parameters=types.Schema(type=types.Type.OBJECT, properties={}) # Sem parâmetros de entrada do usuário
            ),
            types.FunctionDeclaration(
                name="find_object_and_estimate_distance",
                description="Localiza um objeto específico na visão da câmera usando detecção de objetos (YOLO) e estima sua distância em passos com MiDaS. O 'object_type' deve ser uma das categorias do modelo YOLO (ex: 'person', 'car', 'bottle'). Retorna a direção (frente, esquerda, direita), se está sobre uma superfície (ex: mesa), e a distância estimada. Se o objeto não for encontrado, retorna 'Objeto não encontrado na cena.'",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "object_description": types.Schema(
                            type=types.Type.STRING,
                            description="Descrição completa do objeto (ex: 'garrafa azul', 'meu celular vermelho')."
                        ),
                        "object_type": types.Schema(
                            type=types.Type.STRING,
                            description="Tipo principal do objeto (ex: 'bottle', 'cell phone'). Deve ser uma categoria válida do modelo YOLO, conforme mapeado em YOLO_CLASS_MAP."
                        )
                    },
                    required=["object_description", "object_type"]
                )
            )
        ]
    ),
]

# --- Carregar Instrução do Sistema do Arquivo ---
system_instruction_text = "Você é Trackie, um assistente multimodal avançado. Seja conciso e direto ao ponto." # Prompt padrão mínimo
try:
    if not os.path.exists(SYSTEM_INSTRUCTION_PATH):
         logger.warning(f"AVISO: Arquivo de instrução do sistema não encontrado em '{SYSTEM_INSTRUCTION_PATH}'. Usando prompt padrão.")
    else:
        with open(SYSTEM_INSTRUCTION_PATH, 'r', encoding='utf-8') as f:
            system_instruction_text = f.read()
        logger.info(f"Instrução do sistema carregada de: {SYSTEM_INSTRUCTION_PATH}")
except Exception as e:
    logger.error(f"Erro ao ler o arquivo de instrução do sistema: {e}")
    logger.info("Usando um prompt padrão mínimo.")
    traceback.print_exc()


# --- Configuração da Sessão LiveConnect Gemini ---
CONFIG = types.LiveConnectConfig(
    temperature=0.1, # Um pouco de temperatura pode tornar as respostas menos repetitivas
    response_modalities=["audio", "text"], # Adicionado "text" para garantir que sempre recebamos texto
    speech_config=types.SpeechConfig(
        language_code="pt-BR",
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Orus") # Verificar vozes disponíveis
        )
    ),
    tools=tools,
    system_instruction=types.Content(
        parts=[types.Part.from_text(text=system_instruction_text)],
        role="system" # "model" é geralmente usado para o primeiro turno do usuário, "system" para instruções
    ),
)

# --- Inicialização do PyAudio ---
try:
    pya = pyaudio.PyAudio()
except Exception as e:
    logger.error(f"Erro ao inicializar PyAudio: {e}. O áudio não funcionará.")
    pya = None

# --- Classe Principal do Assistente ---
class AudioLoop:
    """
    Gerencia o loop principal do assistente multimodal.
    """
    def __init__(self, video_mode: str = DEFAULT_MODE, show_preview: bool = False):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            self.trckuser = cfg.get("trckuser", "Usuário")
        except FileNotFoundError:
            logger.warning(f"Arquivo de configuração {CONFIG_PATH} não encontrado. Usando nome de usuário padrão.")
            self.trckuser = "Usuário"
        except Exception as e:
            logger.warning(f"Não foi possível ler {CONFIG_PATH}: {e}. Usando nome de usuário padrão.")
            self.trckuser = "Usuário"

        self.video_mode = video_mode
        self.show_preview = show_preview if video_mode == "camera" else False
        self.audio_in_queue: Optional[asyncio.Queue] = None
        self.out_queue: Optional[asyncio.Queue] = None
        self.cmd_queue: Optional[asyncio.Queue] = asyncio.Queue(maxsize=50)

        self.thinking_event = asyncio.Event()
        self.session: Optional[genai.live.AsyncLiveSession] = None
        self.yolo_model: Optional[YOLO] = None
        self.preview_window_active: bool = False
        self.stop_event = asyncio.Event()
        self.frame_lock = threading.Lock()
        self.latest_bgr_frame: Optional[np.ndarray] = None
        self.latest_yolo_results: Optional[List[Any]] = None # Deve ser List[ultralytics.engine.results.Results]
        self.last_response_text: Optional[str] = None
        self.last_response_time: float = 0.0

        self.awaiting_name_for_save_face: bool = False

        if self.video_mode == "camera":
            try:
                self.yolo_model = YOLO(YOLO_MODEL_PATH)
                logger.info(f"Modelo YOLO '{YOLO_MODEL_PATH}' carregado.")
            except FileNotFoundError:
                 logger.error(f"ERRO: Modelo YOLO não encontrado em '{YOLO_MODEL_PATH}'. YOLO desabilitado.")
                 self.yolo_model = None
            except Exception as e:
                logger.error(f"Erro ao carregar o modelo YOLO: {e}. YOLO desabilitado.")
                traceback.print_exc()
                self.yolo_model = None

        if not os.path.exists(DB_PATH):
            try:
                os.makedirs(DB_PATH)
                logger.info(f"Diretório DeepFace DB criado em: {DB_PATH}")
            except Exception as e:
                logger.error(f"Erro ao criar diretório {DB_PATH}: {e}")

        try:
            logger.info("Pré-carregando modelos DeepFace...")
            dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            DeepFace.analyze(img_path=dummy_frame, actions=['emotion'], enforce_detection=False, silent=True)
            logger.info("Modelos DeepFace pré-carregados.")
        except Exception as e:
            logger.warning(f"AVISO: Erro ao pré-carregar modelos DeepFace: {e}.")

        self.midas_model = None
        self.midas_transform = None
        self.midas_device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            logger.info(f"Carregando modelo MiDaS ({MIDAS_MODEL_TYPE}) para {self.midas_device}...")
            self.midas_model = torch.hub.load("intel-isl/MiDaS", MIDAS_MODEL_TYPE, trust_repo=True)
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            if MIDAS_MODEL_TYPE == "MiDaS_small":
                 self.midas_transform = midas_transforms.small_transform
            else:
                 self.midas_transform = midas_transforms.dpt_transform
            self.midas_model.to(self.midas_device)
            self.midas_model.eval()
            logger.info("Modelo MiDaS carregado.")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo MiDaS: {e}. Estimativa de profundidade desabilitada.")
            self.midas_model = None
            self.midas_transform = None


    async def send_text(self):
        logger.info("Pronto para receber comandos de texto. Digite 'q' para sair, 'p' para salvar rosto (debug).")
        while not self.stop_event.is_set():
            try:
                text = await asyncio.to_thread(input, f"{self.trckuser} > ")
                
                if self.out_queue: # Limpa a fila de saída para priorizar nova entrada de texto
                    # logger.debug("Limpando out_queue antes de enviar novo texto.")
                    while not self.out_queue.empty():
                        try:
                            self.out_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        finally: # Garante que task_done seja chamado mesmo se get_nowait falhar (improvável com empty check)
                            try:
                                self.out_queue.task_done()
                            except ValueError: # Se a fila ficou vazia entre empty() e task_done()
                                pass
                    # logger.debug("Fila out_queue limpa para nova entrada de texto.")
                
                # has_received_data_in_turn = False # Não é usado nesta função

                if text.lower() == "q":
                    self.stop_event.set()
                    logger.info("Sinal de parada ('q') recebido. Encerrando...")
                    break

                elif text.lower() == "p":
                    logger.info("[DEBUG] Comando 'p' recebido. Tentando salvar rosto como 'debug_user'...")
                    if self.video_mode == "camera":
                        try:
                            logger.info("  [DEBUG] Chamando _handle_save_known_face('debug_user')...")
                            result = await asyncio.to_thread(self._handle_save_known_face, "debug_user")
                            logger.info(f"  [DEBUG] Resultado do salvamento direto: {result}")
                        except Exception as e_debug_save:
                            logger.error(f"  [DEBUG] Erro ao tentar salvar rosto diretamente: {e_debug_save}", exc_info=True)
                    else:
                        logger.info("  [DEBUG] Salvar rosto só funciona no modo câmera.")
                    continue

                if self.session:
                    # logger.info(f"Enviando texto para Gemini: '{text}'")
                    # Envia "." se o texto estiver vazio para manter a sessão ativa, mas idealmente o usuário envia algo.
                    # A API pode não gostar de inputs vazios repetidos.
                    input_to_send = text.strip() if text.strip() else "."
                    if input_to_send == "." and not text.strip():
                        logger.info("Texto vazio, enviando '.' para manter sessão ativa (end_of_turn=True).")
                    else:
                        logger.info(f"Enviando texto ao Gemini (end_of_turn=True): '{input_to_send}'")

                    await self.session.send(input=input_to_send, end_of_turn=True)
                else:
                    if not self.stop_event.is_set():
                        logger.warning("Sessão Gemini não está ativa. Não é possível enviar mensagem.")
                        await asyncio.sleep(0.5)
                        
            except asyncio.CancelledError:
                logger.info("send_text cancelado.")
                break
            except Exception as e:
                logger.error(f"Erro em send_text: {e}", exc_info=True)
                error_str_upper = str(e).upper()
                if "LIVESESSION CLOSED" in error_str_upper or "LIVESESSION NOT CONNECTED" in error_str_upper:
                    logger.info("Erro em send_text indica sessão fechada. Sinalizando parada.")
                    self.stop_event.set()
                break
        logger.info("send_text finalizado.")

    def _get_frame(self, cap: cv2.VideoCapture) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        ret, frame = cap.read()
        latest_frame_copy = None
        current_yolo_results_obj = None # Para armazenar o objeto Results do YOLO

        if ret:
            latest_frame_copy = frame.copy() # Trabalha com uma cópia

        yolo_alerts = []
        display_frame = None # Frame para mostrar no preview
        
        if ret and self.yolo_model:
            # YOLO espera RGB
            frame_rgb = cv2.cvtColor(latest_frame_copy, cv2.COLOR_BGR2RGB)
            try:
                # results é uma lista de objetos Results (geralmente 1 para uma imagem)
                results_list = self.yolo_model.predict(frame_rgb, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD)
                if results_list:
                    current_yolo_results_obj = results_list[0] # Pega o primeiro objeto Results

                if self.show_preview and latest_frame_copy is not None:
                    # Cria cópia para desenhar, para não afetar latest_bgr_frame
                    display_frame = latest_frame_copy.copy() 

                if current_yolo_results_obj and hasattr(current_yolo_results_obj, 'boxes'):
                    for box in current_yolo_results_obj.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        class_name_yolo = self.yolo_model.names[cls_id]
                        conf = float(box.conf[0])

                        if display_frame is not None:
                            label = f"{class_name_yolo} {conf:.2f}"
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        is_dangerous = any(class_name_yolo in danger_list for danger_list in DANGER_CLASSES.values())
                        if is_dangerous and conf >= YOLO_CONFIDENCE_THRESHOLD: # Re-checa conf, embora predict já filtre
                            yolo_alerts.append(class_name_yolo)
            except Exception as e:
                logger.error(f"Erro na inferência YOLO: {e}", exc_info=True)
                current_yolo_results_obj = None
        
        elif self.show_preview and ret and latest_frame_copy is not None: # Se YOLO não rodou mas preview está ativo
            display_frame = latest_frame_copy.copy()

        with self.frame_lock:
            if ret and latest_frame_copy is not None:
                self.latest_bgr_frame = latest_frame_copy # Salva o frame original BGR
                # Salva o objeto Results completo, não apenas uma lista de algo
                self.latest_yolo_results = [current_yolo_results_obj] if current_yolo_results_obj else None
            else:
                self.latest_bgr_frame = None
                self.latest_yolo_results = None
                return None, []

        if self.show_preview and display_frame is not None:
            try:
                cv2.imshow("Trackie YOLO Preview", display_frame)
                cv2.waitKey(1) # Essencial para a janela atualizar
                self.preview_window_active = True
            except cv2.error as e:
                if "DISPLAY" in str(e).upper() or "GTK" in str(e).upper() or "QT" in str(e).upper() or "COULD NOT CONNECT TO DISPLAY" in str(e).upper() or "plugin \"xcb\"" in str(e).lower():
                    logger.warning("--------------------------------------------------------------------")
                    logger.warning("AVISO: Não foi possível mostrar a janela de preview da câmera (problema de display/GUI).")
                    logger.warning("Desabilitando feedback visual para esta sessão.")
                    logger.warning("--------------------------------------------------------------------")
                    self.show_preview = False
                    self.preview_window_active = False
                    try: cv2.destroyAllWindows() # Tenta fechar se alguma janela abriu parcialmente
                    except: pass
                else:
                    logger.error(f"Erro inesperado no OpenCV ao tentar mostrar preview: {e}", exc_info=True)
            except Exception as e_gen:
                logger.error(f"Erro geral ao tentar mostrar preview: {e_gen}", exc_info=True)
                self.show_preview = False # Desabilita em outros erros também
                self.preview_window_active = False
                try: cv2.destroyAllWindows()
                except: pass


        image_part = None
        if ret and latest_frame_copy is not None: # Usa latest_frame_copy que é BGR
            try:
                # Converte BGR para RGB para PIL Image
                frame_rgb_for_pil = cv2.cvtColor(latest_frame_copy, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb_for_pil)
                img.thumbnail((1024, 1024)) # Redimensiona mantendo proporção, in-place
                
                image_io = io.BytesIO()
                img.save(image_io, format="jpeg", quality=85)
                image_io.seek(0)
                image_part = {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(image_io.read()).decode('utf-8')
                }
            except Exception as e:
                logger.error(f"Erro na conversão do frame para JPEG: {e}", exc_info=True)
        
        return image_part, list(set(yolo_alerts))

    async def get_frames(self):
        cap = None
        try:
            logger.info("Iniciando captura da câmera...")
            # cv2.VideoCapture pode bloquear, então usar to_thread
            cap = await asyncio.to_thread(cv2.VideoCapture, 0, cv2.CAP_DSHOW) # CAP_DSHOW pode ajudar no Windows
            
            if not cap.isOpened():
                logger.error("Erro: Não foi possível abrir a câmera.")
                with self.frame_lock: # Garante limpeza
                    self.latest_bgr_frame = None
                    self.latest_yolo_results = None
                self.stop_event.set() # Para a tarefa e potencialmente o loop principal
                return

            target_fps = 1 # FPS desejado para envio ao Gemini
            # Tentar definir FPS da câmera (pode não funcionar em todas as câmeras/drivers)
            # cap.set(cv2.CAP_PROP_FPS, target_fps) # Comentado pois pode ser problemático
            actual_camera_fps = cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"FPS real da câmera: {actual_camera_fps if actual_camera_fps > 0 else 'Não disponível/Padrão'}")

            # Intervalo para atingir o target_fps para o Gemini
            sleep_interval = 1 / target_fps
            sleep_interval = max(0.1, min(sleep_interval, 2.0)) # Limita entre 0.1s e 2.0s
            logger.info(f"Intervalo de envio de frame para Gemini: {sleep_interval:.2f}s")

            while not self.stop_event.is_set():
                if not cap.isOpened(): # Checagem adicional dentro do loop
                    logger.warning("Câmera desconectada ou fechada inesperadamente durante o loop.")
                    self.stop_event.set()
                    break

                # Executa a captura e processamento síncrono em outra thread
                image_part, yolo_alerts = await asyncio.to_thread(self._get_frame, cap)

                with self.frame_lock: # Verifica se o frame foi lido com sucesso em _get_frame
                    frame_was_successfully_read = self.latest_bgr_frame is not None

                if not frame_was_successfully_read:
                     if not cap.isOpened(): # Se a câmera fechou durante _get_frame
                         logger.info("Leitura do frame falhou e câmera fechada. Encerrando get_frames.")
                         self.stop_event.set()
                         break
                     else: # Falha temporária
                         logger.warning("Aviso: Falha temporária na leitura do frame da câmera.")
                         await asyncio.sleep(0.5) 
                         continue

                if image_part is not None and self.out_queue:
                    try:
                        if self.out_queue.full():
                            discarded = await self.out_queue.get() # Async get
                            self.out_queue.task_done()
                            # logger.debug("Fila de saída cheia, descartando frame antigo.")
                        self.out_queue.put_nowait(image_part)
                    except asyncio.QueueFull:
                         logger.warning("Fila de saída ainda cheia ao tentar enfileirar frame (put_nowait).")
                    except Exception as q_e:
                         logger.error(f"Erro inesperado ao manipular out_queue em get_frames: {q_e}", exc_info=True)

                if yolo_alerts and self.session:
                    for alert_class_name in yolo_alerts:
                        try:
                            alert_msg = f"{self.trckuser}, CUIDADO! {alert_class_name.upper()} detectado!"
                            await self.session.send(input=alert_msg, end_of_turn=True)
                            logger.info(f"ALERTA URGENTE ENVIADO: {alert_msg}")
                        except Exception as e:
                            logger.error(f"Erro ao enviar alerta urgente: {e}", exc_info=True)
                            if "LiveSession closed" in str(e) or "LiveSession not connected" in str(e):
                                logger.info("Erro ao enviar alerta indica sessão fechada. Sinalizando parada.")
                                self.stop_event.set()
                                break 
                    if self.stop_event.is_set(): break # Sai do loop while se um alerta causou parada

                await asyncio.sleep(sleep_interval)

        except asyncio.CancelledError:
            logger.info("get_frames cancelado.")
        except Exception as e:
            logger.error(f"Erro crítico em get_frames: {e}", exc_info=True)
            self.stop_event.set()
        finally:
            logger.info("Finalizando get_frames...")
            if cap and cap.isOpened():
                await asyncio.to_thread(cap.release) # Libera em thread para não bloquear
                logger.info("Câmera liberada.")
            with self.frame_lock:
                self.latest_bgr_frame = None
                self.latest_yolo_results = None
            if self.preview_window_active:
                try:
                    cv2.destroyWindow("Trackie YOLO Preview") # Tenta fechar a específica
                    logger.info("Janela de preview 'Trackie YOLO Preview' fechada.")
                except Exception:
                    try: # Fallback para fechar todas
                        cv2.destroyAllWindows()
                        logger.info("Todas as janelas OpenCV fechadas (fallback).")
                    except Exception as e_cv_destroy_all:
                        logger.warning(f"AVISO: erro ao tentar fechar janelas de preview no finally: {e_cv_destroy_all}")
            self.preview_window_active = False
            logger.info("get_frames concluído.")

    def _get_screen(self) -> Optional[Dict[str, Any]]:
        try:
            with mss.mss() as sct: # Usar context manager para mss
                # Tenta usar o monitor primário (geralmente o 1 em sistemas com múltiplos monitores, mas o 0 é 'todos')
                # A lógica de seleção de monitor pode precisar de ajuste dependendo do sistema.
                # monitors[0] é a bounding box de todos os monitores. monitors[1] é geralmente o primário.
                monitor_to_capture = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                
                if not monitor_to_capture:
                    logger.error("Erro: Nenhum monitor detectado por mss para captura.")
                    return None

                sct_img = sct.grab(monitor_to_capture)
                img = Image.frombytes('RGB', (sct_img.width, sct_img.height), sct_img.rgb, 'raw', 'BGR')
                
                image_io = io.BytesIO()
                img.save(image_io, format="PNG") # PNG é melhor para screenshots
                image_io.seek(0)

                return {
                    "mime_type": "image/png",
                    "data": base64.b64encode(image_io.read()).decode('utf-8')
                }
        except Exception as e:
            logger.error(f"Erro ao capturar tela com mss: {e}", exc_info=True)
            return None

    async def get_screen(self):
        logger.info("Iniciando captura de tela...")
        try:
            while not self.stop_event.is_set():
                frame_data = await asyncio.to_thread(self._get_screen)

                if frame_data is None:
                    logger.warning("Falha ao capturar frame da tela.")
                    await asyncio.sleep(1.0) 
                    continue

                if self.out_queue:
                    try:
                         if self.out_queue.full():
                             discarded = await self.out_queue.get()
                             self.out_queue.task_done()
                             # logger.debug("Fila de saída cheia, descartando frame de tela antigo.")
                         self.out_queue.put_nowait(frame_data)
                    except asyncio.QueueFull:
                         logger.warning("Fila de saída ainda cheia ao tentar enfileirar frame de tela (put_nowait).")
                    except Exception as q_e:
                         logger.error(f"Erro inesperado ao manipular out_queue em get_screen: {q_e}", exc_info=True)
                
                await asyncio.sleep(1.0) # Intervalo entre capturas de tela
        except asyncio.CancelledError:
            logger.info("get_screen cancelado.")
        except Exception as e:
            logger.error(f"Erro crítico em get_screen: {e}", exc_info=True)
            self.stop_event.set()
        finally:
            logger.info("get_screen finalizado.")

    async def send_realtime(self):
        logger.info("Send_realtime pronto para enviar dados...")
        try:
            while not self.stop_event.is_set():
                if self.thinking_event.is_set():
                    await asyncio.sleep(0.05)
                    continue

                if not self.out_queue:
                    await asyncio.sleep(0.1) # Espera a fila ser criada na conexão
                    continue

                msg = None
                try:
                    msg = await asyncio.wait_for(self.out_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue 
                except asyncio.QueueEmpty: # Deve ser pego pelo TimeoutError, mas por segurança
                    continue
                except Exception as q_get_e:
                    logger.error(f"Erro ao obter da out_queue em send_realtime: {q_get_e}", exc_info=True)
                    await asyncio.sleep(0.1)
                    continue
                
                if not self.session:
                    if self.out_queue and msg is not None: self.out_queue.task_done() # Descarta se não há sessão
                    if not self.stop_event.is_set(): await asyncio.sleep(0.5)
                    continue

                try:
                    if isinstance(msg, dict) and "data" in msg and "mime_type" in msg:
                        # logger.debug(f"Enviando {msg['mime_type']} para Gemini...")
                        await self.session.send(input=msg, end_of_turn=True) # end_of_turn=True para cada parte de mídia
                    elif isinstance(msg, str): 
                        logger.info(f"Enviando texto via send_realtime (tratando como turno completo): {msg}")
                        await self.session.send(input=msg, end_of_turn=True)
                    else:
                        logger.warning(f"Mensagem desconhecida em send_realtime: {type(msg)}")
                    
                    if self.out_queue: self.out_queue.task_done()

                except Exception as e_send:
                    logger.error(f"Erro ao enviar para Gemini em send_realtime: {e_send}", exc_info=True)
                    if self.out_queue and msg is not None: # Garante task_done mesmo em erro
                        try: self.out_queue.task_done()
                        except ValueError: pass # Se já foi feito ou a fila mudou

                    error_str_upper = str(e_send).upper()
                    if any(err_key in error_str_upper for err_key in ["LIVESESSION CLOSED", "LIVESESSION NOT CONNECTED", "DEADLINE EXCEEDED", "RST_STREAM", "UNAVAILABLE"]):
                        logger.info("Erro de envio indica sessão Gemini fechada/perdida. Sinalizando parada.")
                        self.stop_event.set()
                        break 
                    else:
                        # traceback.print_exc() # Já logado com exc_info=True
                        await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            logger.info("send_realtime cancelado.")
        except Exception as e:
            logger.error(f"Erro fatal em send_realtime: {e}", exc_info=True)
            self.stop_event.set()
        finally:
            logger.info("send_realtime finalizado.")

    async def listen_audio(self):
        if not pya:
            logger.error("PyAudio não inicializado. Tarefa listen_audio não pode iniciar.")
            return

        audio_stream = None
        try:
            logger.info("Configurando stream de áudio de entrada...")
            mic_info = pya.get_default_input_device_info()
            logger.info(f"Usando microfone: {mic_info['name']} (Taxa: {SEND_SAMPLE_RATE} Hz)")
            
            # pya.open é bloqueante
            audio_stream = await asyncio.to_thread(
                pya.open,
                format=FORMAT, channels=CHANNELS, rate=SEND_SAMPLE_RATE,
                input=True, input_device_index=mic_info["index"],
                frames_per_buffer=CHUNK_SIZE
            )
            logger.info("Escutando áudio do microfone...")

            while not self.stop_event.is_set():
                if self.thinking_event.is_set():
                    await asyncio.sleep(0.05)
                    continue

                if not audio_stream or not audio_stream.is_active(): # Checa se o stream está ok
                     logger.warning("Stream de áudio de entrada não está ativo. Encerrando listen_audio.")
                     self.stop_event.set() # Sinaliza para parar
                     break

                try:
                    # audio_stream.read é bloqueante
                    data = await asyncio.to_thread(
                        audio_stream.read, CHUNK_SIZE, exception_on_overflow=False # False para não levantar exceção em overflow
                    )
                    if self.out_queue:
                         try:
                             if self.out_queue.full():
                                 discarded = await self.out_queue.get()
                                 self.out_queue.task_done()
                                 # logger.debug("Fila de saída cheia, áudio descartado/atrasado.")
                             self.out_queue.put_nowait({"data": data, "mime_type": "audio/pcm"})
                         except asyncio.QueueFull:
                             logger.warning("Fila de saída cheia ao tentar enfileirar áudio (put_nowait).")
                         except Exception as q_e:
                              logger.error(f"Erro inesperado ao manipular out_queue em listen_audio: {q_e}", exc_info=True)
                
                except OSError as e_os:
                    if e_os.errno == -9988 or "Stream closed" in str(e_os) or "Input overflowed" in str(e_os): # Comum
                        logger.info(f"Stream de áudio fechado ou com overflow (OSError: {e_os}). Encerrando listen_audio.")
                    else: # Outros erros de OS
                        logger.error(f"Erro de OS ao ler do stream de áudio: {e_os}", exc_info=True)
                    self.stop_event.set()
                    break 
                except Exception as e_read:
                    logger.error(f"Erro durante a leitura do áudio em listen_audio: {e_read}", exc_info=True)
                    self.stop_event.set() 
                    break
        except asyncio.CancelledError:
            logger.info("listen_audio cancelado.")
        except Exception as e:
            logger.error(f"Erro crítico em listen_audio: {e}", exc_info=True)
            self.stop_event.set()
        finally:
            logger.info("Finalizando listen_audio...")
            if audio_stream:
                try:
                    if audio_stream.is_active(): # Só para se estiver ativo
                        await asyncio.to_thread(audio_stream.stop_stream)
                    await asyncio.to_thread(audio_stream.close)
                    logger.info("Stream de áudio de entrada fechado.")
                except Exception as e_close_stream:
                    logger.error(f"Erro ao fechar stream de áudio de entrada: {e_close_stream}", exc_info=True)
            logger.info("listen_audio concluído.")

    def _handle_save_known_face(self, person_name: str) -> str:
        logger.info(f"[DeepFace] Iniciando salvamento para: {person_name}")
        frame_to_process = None
        with self.frame_lock:
            if self.latest_bgr_frame is not None:
                frame_to_process = self.latest_bgr_frame.copy()

        if frame_to_process is None:
            logger.warning("[DeepFace] Erro: Nenhum frame disponível para salvar.")
            return "Não foi possível capturar a imagem para salvar o rosto."

        safe_person_name_dir = "".join(c if c.isalnum() or c in [' '] else '_' for c in person_name).strip().replace(" ", "_")
        if not safe_person_name_dir: safe_person_name_dir = "desconhecido"
        person_dir = os.path.join(DB_PATH, safe_person_name_dir)

        try:
            if not os.path.exists(person_dir):
                os.makedirs(person_dir)
                logger.info(f"[DeepFace] Diretório criado: {person_dir}")

            detected_faces = DeepFace.extract_faces(
                img_path=frame_to_process, # DeepFace espera BGR por padrão se for np.array
                detector_backend=DEEPFACE_DETECTOR_BACKEND,
                enforce_detection=True, 
                align=True,
                silent=True # Reduz output do DeepFace
            )

            if not detected_faces or not isinstance(detected_faces, list) or 'facial_area' not in detected_faces[0]:
                logger.info(f"[DeepFace] Nenhum rosto detectado para {person_name}.")
                return f"Não consegui detectar um rosto claro para {person_name}."

            face_data = detected_faces[0]['facial_area'] # x, y, w, h
            x, y, w, h = face_data['x'], face_data['y'], face_data['w'], face_data['h']
            
            # Recorta a imagem do rosto com uma margem (opcional, mas pode ajudar)
            margin = 10 
            y1, y2 = max(0, y - margin), min(frame_to_process.shape[0], y + h + margin)
            x1, x2 = max(0, x - margin), min(frame_to_process.shape[1], x + w + margin)
            face_image = frame_to_process[y1:y2, x1:x2]

            if face_image.size == 0:
                 logger.warning(f"[DeepFace] Erro ao recortar rosto para {person_name} (imagem vazia).")
                 return f"Erro ao processar o rosto de {person_name}."

            timestamp = int(time.time())
            safe_file_name_base = "".join(c if c.isalnum() else '_' for c in person_name).strip()
            if not safe_file_name_base: safe_file_name_base = "rosto"
            file_name = f"{safe_file_name_base.lower()}_{timestamp}.jpg"
            file_path = os.path.join(person_dir, file_name)

            save_success = cv2.imwrite(file_path, face_image)
            if not save_success:
                logger.error(f"[DeepFace] Erro ao salvar imagem em {file_path}")
                return f"Erro ao salvar a imagem do rosto de {person_name}."

            # Remove cache de representações para forçar recálculo na próxima chamada a DeepFace.find
            # O nome do arquivo pkl depende do modelo, detector e métrica.
            # DeepFace pode criar múltiplos arquivos .pkl se diferentes combinações são usadas.
            # A forma mais segura é limpar todos os .pkl no DB_PATH ou os específicos.
            # Exemplo: representations_vgg-face.pkl (DeepFace < 0.0.80)
            # Exemplo: df_model_VGG-Face_detector_opencv_distance_metric_cosine_normalization_base_align_True.pkl (DeepFace >= 0.0.80)
            # Para simplificar, vamos tentar remover um nome comum ou iterar.
            # A melhor abordagem é deixar o DeepFace gerenciar isso, mas se forçar, seja específico.
            # Por agora, vamos tentar um padrão comum.
            model_name_safe = DEEPFACE_MODEL_NAME.lower().replace('-', '_')
            # representations_pkl_path = os.path.join(DB_PATH, f"representations_{model_name_safe}.pkl") # Nome antigo
            # Nome mais provável com DeepFace >= 0.0.80
            representations_pkl_path = os.path.join(DB_PATH, f"df_model_{DEEPFACE_MODEL_NAME}_detector_{DEEPFACE_DETECTOR_BACKEND}_distance_metric_{DEEPFACE_DISTANCE_METRIC}_normalization_base_align_True.pkl")

            if os.path.exists(representations_pkl_path):
                try:
                    os.remove(representations_pkl_path)
                    logger.info(f"[DeepFace] Cache de representações '{representations_pkl_path}' removido para atualização.")
                except Exception as e_pkl:
                    logger.warning(f"[DeepFace] Aviso: Falha ao remover cache de representações '{representations_pkl_path}': {e_pkl}")
            else:
                # Tenta encontrar qualquer pkl no diretório se o nome exato falhar (mais arriscado)
                for f_name in os.listdir(DB_PATH):
                    if f_name.endswith(".pkl"):
                        try:
                            os.remove(os.path.join(DB_PATH, f_name))
                            logger.info(f"[DeepFace] Cache de representações genérico '{f_name}' removido.")
                        except Exception as e_pkl_gen:
                            logger.warning(f"[DeepFace] Aviso: Falha ao remover cache genérico '{f_name}': {e_pkl_gen}")


            logger.info(f"[DeepFace] Rosto de {person_name} salvo em {file_path}")
            return f"Rosto de {person_name} salvo com sucesso."

        except ValueError as ve: 
             logger.warning(f"[DeepFace] Nenhum rosto detectado (ValueError) para {person_name}: {ve}")
             return f"Não consegui detectar um rosto claro para salvar para {person_name}."
        except Exception as e:
            logger.error(f"[DeepFace] Erro inesperado ao salvar rosto para {person_name}: {e}", exc_info=True)
            return f"Ocorreu um erro ao tentar salvar o rosto de {person_name}."

    def _handle_identify_person_in_front(self) -> str:
        if pd is None:
            logger.error("[DeepFace] Erro: Biblioteca 'pandas' não está disponível. Identificação desabilitada.")
            return "Erro interno: uma dependência necessária para identificação de rostos não está instalada."

        logger.info("[DeepFace] Iniciando identificação de pessoa...")
        frame_to_process = None
        with self.frame_lock:
            if self.latest_bgr_frame is not None:
                frame_to_process = self.latest_bgr_frame.copy()

        if frame_to_process is None:
            logger.warning("[DeepFace] Erro: Nenhum frame disponível para identificar.")
            return "Não foi possível capturar a imagem para identificar."

        try:
            # DeepFace.find pode levar tempo, especialmente com DBs grandes
            dfs = DeepFace.find(
                img_path=frame_to_process, # BGR
                db_path=DB_PATH,
                model_name=DEEPFACE_MODEL_NAME,
                detector_backend=DEEPFACE_DETECTOR_BACKEND,
                distance_metric=DEEPFACE_DISTANCE_METRIC,
                enforce_detection=True, # Exige detecção clara no frame de entrada
                silent=True, # Reduz output
                align=True
            )

            # dfs é uma lista de DataFrames (geralmente uma se um rosto for detectado na img_path)
            if not dfs or not isinstance(dfs, list) or dfs[0].empty:
                logger.info("[DeepFace] Nenhuma correspondência encontrada ou rosto não detectado claramente na imagem de entrada.")
                return "Não consegui reconhecer ninguém ou não detectei um rosto claro."

            df = dfs[0] # Pega o primeiro DataFrame

            # A coluna de distância pode ter nomes como 'VGG-Face_cosine' ou apenas 'distance'
            # O DeepFace >= 0.0.80 padroniza melhor o nome da coluna de distância.
            # Ex: 'distance' ou 'model_metric' (ex: 'VGG-Face_cosine')
            # A coluna 'identity' contém o caminho para a imagem correspondente no DB.
            
            # A coluna de distância é nomeada dinamicamente em versões mais recentes do DeepFace
            # Ex: df_model_VGG-Face_detector_opencv_distance_metric_cosine_normalization_base_align_True.pkl
            # A coluna no DataFrame é geralmente 'distance' ou '<model_name>_<metric_name>'
            distance_col_name = 'distance' # Coluna padrão
            if distance_col_name not in df.columns:
                # Tenta encontrar uma coluna que contenha a métrica
                potential_cols = [col for col in df.columns if DEEPFACE_DISTANCE_METRIC in col.lower()]
                if potential_cols:
                    distance_col_name = potential_cols[0]
                else: # Se ainda não encontrar, loga erro
                    logger.error(f"[DeepFace] Erro: Coluna de distância ('distance' ou contendo '{DEEPFACE_DISTANCE_METRIC}') não encontrada. Colunas: {df.columns.tolist()}")
                    return "Erro ao processar resultado da identificação (coluna de distância)."


            # Ordena por distância (menor é melhor)
            df = df.sort_values(by=distance_col_name, ascending=True)
            best_match = df.iloc[0]

            best_match_identity_path = best_match['identity']
            person_name_from_db = os.path.basename(os.path.dirname(best_match_identity_path))
            distance_value = best_match[distance_col_name]

            logger.info(f"[DeepFace] Pessoa potencialmente identificada: {person_name_from_db} (Distância: {distance_value:.4f})")

            # Limiares de distância (ajustar experimentalmente!)
            # Estes são exemplos, os valores ótimos variam.
            thresholds = {
                'VGG-Face': {'cosine': 0.40, 'euclidean': 0.60, 'euclidean_l2': 0.86}, # Cosine: <0.40 é match
                'Facenet':  {'cosine': 0.40, 'euclidean': 10,   'euclidean_l2': 0.80}, # Valores de exemplo
                'Facenet512':{'cosine': 0.30, 'euclidean': 23.56,'euclidean_l2': 1.04},
                'ArcFace':  {'cosine': 0.68, 'euclidean': 4.15, 'euclidean_l2': 1.13},
                'Dlib':     {'cosine': 0.07, 'euclidean': 0.6,  'euclidean_l2': 0.4}, # Dlib é mais sensível
            }
            threshold = thresholds.get(DEEPFACE_MODEL_NAME, {}).get(DEEPFACE_DISTANCE_METRIC, 0.5) # Padrão genérico

            if distance_value <= threshold:
                # Calcula uma "confiança" simples (não é probabilidade estatística)
                confidence_percent = max(0, (1 - (distance_value / (threshold * 1.5))) * 100) # Heurística
                confidence_percent = min(99, int(confidence_percent)) # Limita a 99%
                return f"A pessoa na sua frente parece ser {person_name_from_db} (confiança: {confidence_percent}%)."
            else:
                logger.info(f"[DeepFace] Distância {distance_value:.4f} > limiar ({threshold}). Não reconhecido com confiança.")
                return "Detectei um rosto, mas não tenho certeza de quem é."

        except ValueError as ve: 
            logger.warning(f"[DeepFace] Erro (ValueError) ao identificar, provavelmente nenhum rosto detectado na imagem de entrada: {ve}")
            return "Não detectei um rosto claro para identificar."
        except Exception as e:
            logger.error(f"[DeepFace] Erro inesperado ao identificar: {e}", exc_info=True)
            return "Ocorreu um erro ao tentar identificar a pessoa."

    def _run_midas_inference(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        if not self.midas_model or not self.midas_transform:
            # logger.debug("[MiDaS] Modelo ou transformador não carregado.")
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
            logger.error(f"[MiDaS] Erro durante a inferência: {e}", exc_info=True)
            return None

    def _find_best_yolo_match(self, object_type_query: str, yolo_results_list: Optional[List[Any]]) -> Optional[Tuple[Dict[str, int], float, str]]:
        best_match_info = None
        highest_conf = -1.0
        
        # Normaliza a query do tipo de objeto
        normalized_query = object_type_query.lower().strip()
        target_yolo_class_names = YOLO_CLASS_MAP.get(normalized_query, [normalized_query]) # Lista de nomes YOLO
        # logger.debug(f"[YOLO Match] Procurando por tipo '{normalized_query}', classes YOLO alvo: {target_yolo_class_names}")

        if not yolo_results_list or not self.yolo_model:
             # logger.debug("[YOLO Match] Sem resultados YOLO ou modelo não carregado.")
             return None

        yolo_results_obj = yolo_results_list[0] # Pega o objeto Results
        if not yolo_results_obj or not hasattr(yolo_results_obj, 'boxes'):
            # logger.debug("[YOLO Match] Objeto Results inválido ou sem atributo 'boxes'.")
            return None

        for box in yolo_results_obj.boxes:
            if not (hasattr(box, 'cls') and hasattr(box, 'conf') and hasattr(box, 'xyxy')):
                # logger.debug("[YOLO Match] Caixa malformada encontrada.")
                continue 

            cls_id_tensor = box.cls
            if cls_id_tensor.nelement() == 0: continue 
            cls_id = int(cls_id_tensor[0])

            conf_tensor = box.conf
            if conf_tensor.nelement() == 0: continue
            conf = float(conf_tensor[0])

            if cls_id < len(self.yolo_model.names):
                detected_class_name = self.yolo_model.names[cls_id]
            else:
                # logger.debug(f"[YOLO Match] ID de classe inválido: {cls_id}")
                continue 

            if detected_class_name in target_yolo_class_names:
                if conf > highest_conf:
                    highest_conf = conf
                    coords_tensor = box.xyxy[0]
                    if coords_tensor.nelement() < 4: continue 
                    coords = list(map(int, coords_tensor))
                    bbox_dict = {'x1': coords[0], 'y1': coords[1], 'x2': coords[2], 'y2': coords[3]}
                    best_match_info = (bbox_dict, conf, detected_class_name) # Salva o nome da classe detectada
                    # logger.debug(f"[YOLO Match] Novo melhor match: {detected_class_name} ({conf:.2f})")
        
        return best_match_info

    def _estimate_direction(self, bbox: Dict[str, int], frame_width: int) -> str:
        box_center_x = (bbox['x1'] + bbox['x2']) / 2
        center_zone_third = frame_width / 3

        if box_center_x < center_zone_third:
            return "à sua esquerda"
        elif box_center_x > (frame_width - center_zone_third):
            return "à sua direita"
        else:
            return "à sua frente"

    def _check_if_on_surface(self, target_bbox: Dict[str, int], yolo_results_list: Optional[List[Any]]) -> bool:
        surface_classes_keys = ["mesa", "mesa de jantar", "bancada", "prateleira", "sofá", "cama"]
        surface_yolo_names = []
        for key in surface_classes_keys:
            surface_yolo_names.extend(YOLO_CLASS_MAP.get(key, []))
        surface_yolo_names = list(set(surface_yolo_names)) 

        if not surface_yolo_names: return False

        target_bottom_y = target_bbox['y2']
        target_center_x = (target_bbox['x1'] + target_bbox['x2']) / 2

        if not yolo_results_list or not self.yolo_model: return False
        
        yolo_results_obj = yolo_results_list[0]
        if not yolo_results_obj or not hasattr(yolo_results_obj, 'boxes'): return False

        for box in yolo_results_obj.boxes:
            if not (hasattr(box, 'cls') and hasattr(box, 'xyxy')): continue

            cls_id_tensor = box.cls
            if cls_id_tensor.nelement() == 0: continue
            cls_id = int(cls_id_tensor[0])

            if cls_id < len(self.yolo_model.names):
                class_name = self.yolo_model.names[cls_id]
            else: continue

            if class_name in surface_yolo_names:
                coords_tensor = box.xyxy[0]
                if coords_tensor.nelement() < 4: continue
                s_x1, s_y1, s_x2, s_y2 = map(int, coords_tensor)

                # Heurística:
                horizontally_aligned = s_x1 < target_center_x < s_x2
                # Objeto está "descansando" perto do topo da superfície (s_y1 é o topo da caixa da superfície)
                # A base do objeto (target_bottom_y) deve estar próxima e ligeiramente acima ou no mesmo nível do topo da superfície.
                y_tolerance_pixels = 30 
                vertically_aligned = (s_y1 - y_tolerance_pixels) < target_bottom_y < (s_y1 + y_tolerance_pixels * 1.5)
                
                # Adicional: A superfície deve ser razoavelmente larga em comparação com o objeto
                surface_width = s_x2 - s_x1
                target_width = target_bbox['x2'] - target_bbox['x1']
                reasonable_width = target_width < surface_width * 1.5 # Objeto não muito mais largo que a superfície

                if horizontally_aligned and vertically_aligned and reasonable_width:
                    # logger.debug(f"[Surface Check] Objeto em ({target_center_x},{target_bottom_y}) considerado sobre '{class_name}' em ({s_x1}-{s_x2}, {s_y1}-{s_y2})")
                    return True
        return False

    def _handle_find_object_and_estimate_distance(self, object_description: str, object_type: str) -> str:
        logger.info(f"[Localizar Objeto] Buscando por '{object_description}' (tipo YOLO: '{object_type}')...")
        frame_to_process = None
        yolo_results_for_frame = None # Esta será a List[Results]
        frame_height, frame_width = 0, 0

        with self.frame_lock:
            if self.latest_bgr_frame is not None:
                frame_to_process = self.latest_bgr_frame.copy()
                yolo_results_for_frame = self.latest_yolo_results # Pega a List[Results]
                if frame_to_process is not None: # Redundante se latest_bgr_frame não é None, mas seguro
                    frame_height, frame_width, _ = frame_to_process.shape
        
        if frame_to_process is None or frame_width == 0 or frame_height == 0:
             logger.warning("[Localizar Objeto] Erro: Nenhum frame válido disponível.")
             return f"{self.trckuser}, não estou enxergando nada no momento para localizar o {object_description}."

        if not yolo_results_for_frame: # Checa se a lista de resultados YOLO existe
            logger.warning("[Localizar Objeto] Erro: Nenhum resultado YOLO disponível para o frame atual.")
            return f"{self.trckuser}, não consegui processar a imagem a tempo para encontrar o {object_description}."

        # Usa object_type (que deve ser um nome de classe YOLO) para a busca inicial
        best_yolo_match = self._find_best_yolo_match(object_type, yolo_results_for_frame)

        # Fallback: Se não encontrou pelo object_type fornecido, tenta usar a descrição completa
        # Isso é menos preciso, pois _find_best_yolo_match espera um tipo de objeto.
        # Uma melhoria seria extrair um tipo de objeto da descrição usando NLP ou heurísticas.
        if not best_yolo_match:
            logger.info(f"[Localizar Objeto] Nenhum objeto do tipo '{object_type}' encontrado. Tentando usar a descrição '{object_description}' como fallback (pode ser impreciso).")
            # Tenta a última palavra da descrição como um tipo de objeto
            last_word_in_desc = object_description.split(" ")[-1].lower()
            if last_word_in_desc != object_type.lower(): # Evita repetir a mesma busca
                best_yolo_match = self._find_best_yolo_match(last_word_in_desc, yolo_results_for_frame)

            if not best_yolo_match:
                 logger.info(f"[Localizar Objeto] Objeto '{object_description}' não encontrado mesmo com fallback.")
                 return f"{self.trckuser}, não consegui encontrar um(a) {object_description} na imagem."

        target_bbox, confidence, detected_class_name = best_yolo_match
        logger.info(f"[Localizar Objeto] Melhor correspondência YOLO: Classe '{detected_class_name}', Conf: {confidence:.2f}, BBox: {target_bbox}")

        is_on_surface = self._check_if_on_surface(target_bbox, yolo_results_for_frame)
        surface_msg = "sobre uma superfície (como uma mesa ou bancada)" if is_on_surface else ""
        direction = self._estimate_direction(target_bbox, frame_width)

        distance_steps = -1
        if self.midas_model and frame_to_process is not None:
            # logger.debug("[Localizar Objeto] Executando MiDaS...")
            depth_map = self._run_midas_inference(frame_to_process)
            if depth_map is not None:
                try:
                    center_x = int((target_bbox['x1'] + target_bbox['x2']) / 2)
                    center_y = int((target_bbox['y1'] + target_bbox['y2']) / 2)
                    center_y = max(0, min(center_y, depth_map.shape[0] - 1))
                    center_x = max(0, min(center_x, depth_map.shape[1] - 1))
                    depth_value_at_center = depth_map[center_y, center_x]

                    # --- Conversão MiDaS (MUITO BRUTA - PRECISA CALIBRAR EXTENSIVAMENTE) ---
                    # MiDaS_small retorna profundidade relativa (maior valor = mais perto, menor = mais longe)
                    # Esta é uma tentativa de mapeamento muito simplificada e provavelmente imprecisa.
                    # Precisa de calibração com objetos a distâncias conhecidas.
                    if depth_value_at_center > 1e-6: 
                        # Exemplo de mapeamento (valores completamente arbitrários, ajustar!)
                        # Quanto MAIOR o depth_value_at_center (para MiDaS_small), MAIS PERTO.
                        if depth_value_at_center > 250:  # Muito perto
                            estimated_meters = np.random.uniform(0.3, 1.0) 
                        elif depth_value_at_center > 100: # Perto
                            estimated_meters = np.random.uniform(1.0, 3.0)
                        elif depth_value_at_center > 30:  # Médio
                            estimated_meters = np.random.uniform(3.0, 7.0)
                        else: # Longe
                            estimated_meters = np.random.uniform(7.0, 15.0)
                        
                        estimated_meters = max(0.3, min(estimated_meters, 20.0)) # Limita
                        distance_steps = max(1, round(estimated_meters / METERS_PER_STEP))
                        logger.info(f"[Localizar Objeto] Profundidade MiDaS no centro ({center_y},{center_x}): {depth_value_at_center:.4f}, Metros Estimados (heurístico): {estimated_meters:.2f}, Passos: {distance_steps}")
                    else:
                         logger.warning("[Localizar Objeto] Valor de profundidade MiDaS inválido ou muito baixo no centro do objeto.")
                except IndexError:
                    logger.error(f"[Localizar Objeto] Erro de índice ao acessar mapa de profundidade.")
                except Exception as e_depth:
                    logger.error(f"[Localizar Objeto] Erro ao processar profundidade MiDaS: {e_depth}", exc_info=True)
            else:
                logger.warning("[Localizar Objeto] MiDaS não retornou mapa de profundidade.")
        else:
            logger.info("[Localizar Objeto] MiDaS não disponível ou frame inválido. Não é possível estimar distância.")

        # Usa a descrição original do usuário para a resposta
        object_name_for_response = object_description 

        response_parts = [f"{self.trckuser}, o {object_name_for_response} está"]
        if surface_msg: response_parts.append(surface_msg)
        
        if distance_steps > 0:
            response_parts.append(f"a aproximadamente {distance_steps} passo{'s' if distance_steps > 1 else ''}")
        
        response_parts.append(direction) # Adiciona a direção

        # Constrói a frase
        if len(response_parts) > 1: # Se adicionou algo além de "Usuário, o obj está"
            if len(response_parts) == 2: # Só direção ou só distância/superfície
                 result_message = " ".join(response_parts) + "."
            else: # Combinação
                # Ex: "..., sobre a superfície, a X passos, à sua frente."
                # Junta os elementos do meio com vírgula, e o último com "e" ou diretamente.
                # Se tem distância E direção: "a X passos e à sua frente"
                # Se tem superfície E distância E direção: "sobre X, a Y passos e à sua frente"
                
                # Simplificado: junta com vírgulas, exceto o último elemento (direção)
                if len(response_parts) > 2: # Se tem mais do que [intro, direção]
                    # Junta todos os atributos (superfície, distância) com vírgula
                    attributes = ", ".join(response_parts[1:-1])
                    result_message = f"{attributes} {response_parts[-1]}." # Adiciona a direção no final
                    # Prepend o início: "Usuário, o obj está ..."
                    result_message = f"{response_parts[0]} {result_message}"

                else: # Só tem [intro, direção] ou [intro, um_atributo]
                    result_message = " ".join(response_parts) + "."
        else: # Caso muito raro
            result_message = f"{self.trckuser}, não consegui determinar a localização exata do {object_name_for_response}."
        
        logger.info(f"[Localizar Objeto] Resultado: {result_message}")
        return result_message

    async def receive_audio(self):
        logger.info("Receive_audio pronto para receber respostas do Gemini...")
        last_processed_response_part = None # Para o print no final

        try:
            if not self.session:
                logger.error("Sessão Gemini não estabelecida em receive_audio. Encerrando.")
                self.stop_event.set()
                return

            # Loop principal da tarefa de recebimento
            while not self.stop_event.is_set():
                if not self.session: # Checa se a sessão foi perdida
                    logger.warning("Sessão Gemini desconectada em receive_audio. Tentando reconectar ou parar.")
                    await asyncio.sleep(1)
                    if not self.session and not self.stop_event.is_set():
                        logger.info("Sessão ainda não disponível. Sinalizando parada para reconexão.")
                        self.stop_event.set() 
                    elif self.session:
                        logger.info("Sessão Gemini reconectada.")
                    break # Sai do while interno, o loop run tentará reconectar

                has_received_data_in_this_gemini_turn = False
                current_response_text_parts = [] # Para acumular texto fragmentado

                try:
                    # O timeout aqui é para o caso de o stream travar completamente sem fechar.
                    # Se o Gemini simplesmente não tiver nada a dizer, o `async for` não iterará.
                    async with asyncio.timeout(30.0): # Timeout mais longo para o stream de resposta
                        async for response_part in self.session.receive():
                            if self.stop_event.is_set(): break
                            
                            has_received_data_in_this_gemini_turn = True
                            last_processed_response_part = response_part # Guarda a última parte para o print final

                            # --- Processa Áudio ---
                            if response_part.data:
                                if self.audio_in_queue:
                                    try:
                                        self.audio_in_queue.put_nowait(response_part.data)
                                    except asyncio.QueueFull:
                                        logger.warning("Fila de áudio de entrada cheia. Áudio da IA descartado.")
                                continue # Processou áudio, vai para próxima parte

                            # --- Processa Nome Pendente (Fluxo save_known_face) ---
                            if self.awaiting_name_for_save_face:
                                user_provided_name = None
                                if response_part.text:
                                    user_provided_name = response_part.text.strip()
                                    logger.info(f"[Trackie] Recebido texto enquanto aguardava nome: '{user_provided_name}'")

                                if user_provided_name: # Se o usuário forneceu o nome
                                    logger.info(f"[Trackie] Processando nome '{user_provided_name}' para salvar rosto...")
                                    self.awaiting_name_for_save_face = False 
                                    original_function_name_pending = "save_known_face"

                                    self.thinking_event.set(); logger.info("Pensando...")
                                    
                                    voice_feedback_msg = f"{self.trckuser}, salvando rosto de {user_provided_name}, um momento..."
                                    if self.session:
                                        try:
                                            await self.session.send(input=voice_feedback_msg, end_of_turn=True)
                                        except Exception as e_feedback: logger.error(f"Erro ao enviar feedback (awaiting name): {e_feedback}")

                                    result_message = await asyncio.to_thread(self._handle_save_known_face, user_provided_name)
                                    
                                    logger.info(f"  [Trackie] Resultado da Função '{original_function_name_pending}': {result_message}")
                                    if self.session:
                                        try:
                                            await self.session.send(
                                                input=types.Content(role="tool", parts=[
                                                    types.Part.from_function_response(
                                                        name=original_function_name_pending,
                                                        response={"result": Value(string_value=result_message)}
                                                    )])
                                            ) # end_of_turn=False por padrão para FunctionResponse
                                        except Exception as e_send_fc_resp: logger.error(f"Erro ao enviar FunctionResponse (awaiting name): {e_send_fc_resp}")
                                    
                                    if self.thinking_event.is_set(): self.thinking_event.clear(); logger.info("Pensamento concluído.")
                                continue # Processou nome, vai para próxima parte

                            # --- Processa Texto da IA ---
                            if response_part.text:
                                current_time = time.time()
                                # Evita imprimir a *mesma string completa* repetidamente
                                if response_part.text == self.last_response_text and (current_time - self.last_response_time) < 2.0:
                                    logger.info(f"Descartando resposta de texto repetida: '{response_part.text}'")
                                else:
                                    print(f"{response_part.text}", end="") # Imprime partes do texto
                                    current_response_text_parts.append(response_part.text)
                                    # Atualiza last_response_text apenas quando o turno de texto parece completo
                                    # (difícil de determinar com streaming, então atualiza com a última parte)
                                    self.last_response_text = response_part.text # Ou concatenar e guardar
                                    self.last_response_time = current_time
                            
                            # --- Processa Chamada de Função ---
                            if getattr(response_part, "function_call", None):
                                # Se havia texto acumulado, imprime uma nova linha antes da chamada de função
                                if current_response_text_parts: 
                                    print()
                                    # self.last_response_text = "".join(current_response_text_parts) # Salva o texto completo
                                    current_response_text_parts = [] # Reseta para o próximo bloco de texto

                                fc = response_part.function_call
                                function_name = fc.name
                                args = {key: val for key, val in fc.args.items()} # args é um Struct, converter para dict
                                logger.info(f"\n[Gemini Function Call] Recebido: {function_name}, Args: {args}")

                                result_message = None 

                                if function_name == "save_known_face" and not args.get("person_name"):
                                    self.awaiting_name_for_save_face = True
                                    if self.thinking_event.is_set(): self.thinking_event.clear()
                                    logger.info("[Trackie] Nome não fornecido para save_known_face. Solicitando ao usuário.")
                                    if self.session:
                                        try:
                                            await self.session.send(input=f"{self.trckuser}, por favor, qual o nome da pessoa para salvar o rosto?", end_of_turn=True)
                                        except Exception as e_ask_name: logger.error(f"Erro ao pedir nome para save_face: {e_ask_name}")
                                    # Não executa a função local nem envia FC response agora, espera input do usuário
                                
                                else: # Outras funções ou save_known_face com nome
                                    self.thinking_event.set(); logger.info("Pensando...")
                                    
                                    voice_feedback_msg = f"{self.trckuser}, processando {function_name}, um momento..."
                                    # ... (mensagens de feedback específicas para cada função, como no original)
                                    if function_name == "save_known_face": voice_feedback_msg = f"{self.trckuser}, salvando rosto de {args.get('person_name', 'pessoa')}, um momento..."
                                    elif function_name == "identify_person_in_front": voice_feedback_msg = f"{self.trckuser}, identificando pessoa, um momento..."
                                    elif function_name == "find_object_and_estimate_distance": voice_feedback_msg = f"{self.trckuser}, localizando {args.get('object_description', 'objeto')}, um momento..."

                                    if self.session:
                                        try:
                                            await self.session.send(input=voice_feedback_msg, end_of_turn=True)
                                        except Exception as e_feedback: logger.error(f"Erro ao enviar feedback pré-função: {e_feedback}")

                                    vision_functions = ["save_known_face", "identify_person_in_front", "find_object_and_estimate_distance"]
                                    if self.video_mode != "camera" and function_name in vision_functions:
                                        result_message = "Desculpe, esta função só está disponível quando a câmera está ativa."
                                    else:
                                        logger.info(f"  [Trackie] Processando Função '{function_name}' em background...")
                                        try:
                                            if function_name == "save_known_face":
                                                person_name_arg = args.get("person_name")
                                                if person_name_arg: result_message = await asyncio.to_thread(self._handle_save_known_face, person_name_arg)
                                                else: result_message = "Erro: nome não fornecido para salvar rosto."; logger.error("LÓGICA: _handle_save_known_face chamado sem nome aqui.")
                                            
                                            elif function_name == "identify_person_in_front":
                                                if pd is None: result_message = "Erro interno: dependência 'pandas' faltando."
                                                else: result_message = await asyncio.to_thread(self._handle_identify_person_in_front)
                                            
                                            elif function_name == "find_object_and_estimate_distance":
                                                desc, obj_type = args.get("object_description"), args.get("object_type")
                                                if desc and obj_type:
                                                    if not self.midas_model: result_message = f"{self.trckuser}, desculpe, o módulo de estimativa de distância não está funcionando."
                                                    else: result_message = await asyncio.to_thread(self._handle_find_object_and_estimate_distance, desc, obj_type)
                                                else: result_message = "Descrição ou tipo do objeto não fornecido."; logger.error(f"Argumentos faltando para find_object: desc='{desc}', type='{obj_type}'")
                                            
                                            else: result_message = f"Função '{function_name}' desconhecida."; logger.warning(f"Função não mapeada: {function_name}")
                                        except Exception as e_handler:
                                             logger.error(f"Erro ao executar handler para '{function_name}': {e_handler}", exc_info=True)
                                             result_message = f"Ocorreu um erro interno ao processar a função {function_name}."
                                    
                                    if result_message is not None:
                                        logger.info(f"  [Trackie] Resultado da Função '{function_name}': {result_message}")
                                        if self.session:
                                            try:
                                                await self.session.send(input=types.Content(role="tool", parts=[
                                                        types.Part.from_function_response(
                                                            name=function_name,
                                                            response={"result": Value(string_value=result_message)}
                                                        )]))
                                            except Exception as e_send_fc_resp_main: logger.error(f"Erro ao enviar FunctionResponse (main): {e_send_fc_resp_main}")
                                    
                                    if self.thinking_event.is_set(): self.thinking_event.clear(); logger.info("Pensamento concluído.")
                        # Fim do `async for`
                        if current_response_text_parts: # Se o último `response_part` foi texto
                            print() # Garante nova linha
                            # self.last_response_text = "".join(current_response_text_parts) # Salva o texto completo
                            current_response_text_parts = []


                    # Fim do `async with asyncio.timeout`
                except asyncio.TimeoutError:
                    # logger.debug("Timeout (30s) esperando por partes da resposta do Gemini. Pode ser normal se não houver resposta.")
                    # Se houve dados neste turno, mas depois um timeout, o turno terminou.
                    if has_received_data_in_this_gemini_turn and last_processed_response_part and last_processed_response_part.text and not last_processed_response_part.text.endswith('\n'):
                        print() # Nova linha se o último texto não terminou com uma
                    continue # Volta para o início do `while not self.stop_event.is_set()`
                except Exception as e_inner_loop: # Erros dentro do processamento do stream
                    logger.error(f"Erro durante o recebimento/processamento de resposta: {e_inner_loop}", exc_info=True)
                    error_str_upper = str(e_inner_loop).upper()
                    if any(err_key in error_str_upper for err_key in ["LIVESESSION CLOSED", "LIVESESSION NOT CONNECTED", "DEADLINE EXCEEDED", "RST_STREAM", "UNAVAILABLE"]):
                        logger.info("Erro indica que a sessão Gemini foi fechada/perdida. Sinalizando parada.")
                        self.stop_event.set()
                        break # Sai do `while not self.stop_event.is_set()`
                    else: # Outros erros, tenta continuar após pausa
                        await asyncio.sleep(0.5)
                
                # Após o `async for` (seja por conclusão normal do stream ou timeout que não foi erro fatal)
                if not self.stop_event.is_set():
                    if has_received_data_in_this_gemini_turn:
                        # logger.debug("Fim do turno de resposta do Gemini.")
                        if last_processed_response_part and getattr(last_processed_response_part, 'text', None) and not last_processed_response_part.text.endswith('\n'):
                            print() 
                    else: # Nenhum dado recebido no stream (ex: Gemini não respondeu nada)
                        # logger.debug("Stream do turno atual terminou sem dados.")
                        await asyncio.sleep(0.05) # Pequena pausa antes de tentar receber de novo

            # Fim do `while not self.stop_event.is_set()`
            if self.stop_event.is_set():
                logger.info("Loop de recebimento de áudio/respostas interrompido pelo stop_event.")

        except asyncio.CancelledError:
            logger.info("receive_audio foi cancelado.")
        except Exception as e: # Erro crítico fora do loop principal (ex: na configuração inicial)
            logger.error(f"Erro crítico em receive_audio: {e}", exc_info=True)
            self.stop_event.set() 
        finally:
            logger.info("receive_audio finalizado.")
            self.awaiting_name_for_save_face = False
            if self.thinking_event.is_set(): self.thinking_event.clear()

    async def play_audio(self):
        if not pya:
            logger.error("PyAudio não inicializado. Tarefa play_audio não pode iniciar.")
            return

        stream = None
        output_rate = RECEIVE_SAMPLE_RATE 
        try:
            logger.info("Configurando stream de áudio de saída...")
            try:
                out_device_info = pya.get_default_output_device_info()
                # output_rate = int(out_device_info['defaultSampleRate']) # Usar taxa do dispositivo pode ser melhor
                logger.info(f"Usando dispositivo de saída: {out_device_info['name']} @ {output_rate} Hz")
            except Exception as e_dev_info:
                logger.warning(f"Não foi possível obter info do dispositivo de saída padrão ({e_dev_info}). Usando taxa padrão: {output_rate} Hz")

            stream = await asyncio.to_thread(
                pya.open, format=FORMAT, channels=CHANNELS, rate=output_rate, output=True
            )
            logger.info("Player de áudio pronto.")

            while not self.stop_event.is_set():
                if not self.audio_in_queue:
                    await asyncio.sleep(0.1)
                    continue
                
                bytestream = None
                try:
                    # Verifica se há nova entrada do usuário para interromper a fala da IA
                    if self.out_queue and not self.out_queue.empty() and stream.is_active():
                        logger.info("Nova entrada do usuário detectada, interrompendo áudio da IA e limpando fila de áudio.")
                        await asyncio.to_thread(stream.stop_stream) # Para o stream imediatamente
                        
                        # Limpa audio_in_queue para evitar reprodução de respostas antigas
                        if self.audio_in_queue:
                            while not self.audio_in_queue.empty():
                                try: self.audio_in_queue.get_nowait(); self.audio_in_queue.task_done()
                                except asyncio.QueueEmpty: break
                                except ValueError: pass # task_done em fila vazia
                        
                        await asyncio.sleep(0.1) # Pequena pausa
                        if not stream.is_active(): # Reinicia se parou
                           await asyncio.to_thread(stream.start_stream) 
                        continue # Volta para pegar novo áudio ou esperar

                    bytestream = await asyncio.wait_for(self.audio_in_queue.get(), timeout=0.5)

                    if bytestream is None: 
                        logger.info("Recebido sinal de encerramento (None) para play_audio.")
                        break 

                    if stream and stream.is_active():
                        await asyncio.to_thread(stream.write, bytestream)
                    else: # Stream não está ativo mas recebeu dados
                        logger.warning("Stream de áudio para playback não está ativo. Descartando áudio.")
                    
                    if self.audio_in_queue: self.audio_in_queue.task_done()

                except asyncio.TimeoutError: continue
                except asyncio.QueueEmpty: continue
                except OSError as e_os_play:
                    if "Stream closed" in str(e_os_play): logger.info("Stream de playback fechado (OSError).")
                    else: logger.error(f"Erro de OS ao reproduzir áudio: {e_os_play}", exc_info=True)
                    break 
                except Exception as e_inner:
                    logger.error(f"Erro ao reproduzir áudio (interno): {e_inner}", exc_info=True)
                    if "Stream closed" in str(e_inner): break 
                    # Considerar break para outros erros também

        except asyncio.CancelledError:
            logger.info("play_audio foi cancelado.")
        except Exception as e:
            logger.error(f"Erro crítico em play_audio: {e}", exc_info=True)
        finally:
            logger.info("Finalizando play_audio...")
            if stream:
                try:
                    if stream.is_active(): await asyncio.to_thread(stream.stop_stream)
                    await asyncio.to_thread(stream.close)
                    logger.info("Stream de áudio de saída fechado.")
                except Exception as e_close:
                    logger.error(f"Erro ao fechar stream de áudio de saída: {e_close}", exc_info=True)
            logger.info("play_audio concluído.")

    async def run(self):
        logger.info("Iniciando AudioLoop...")
        max_retries = 3
        retry_delay_base = 2.0

        attempt = 0
        while attempt <= max_retries and not self.stop_event.is_set():
            retry_delay = retry_delay_base * (2 ** attempt) 
            try:
                if attempt > 0:
                     logger.info(f"Tentativa de reconexão {attempt}/{max_retries} após {retry_delay:.1f}s...")
                     await asyncio.sleep(retry_delay)

                if self.session: # Limpa sessão anterior
                    try: await self.session.close()
                    except Exception: pass 
                self.session = None
                self.audio_in_queue = None 
                self.out_queue = None      
                self.awaiting_name_for_save_face = False 
                if self.thinking_event.is_set(): self.thinking_event.clear()
                
                if client is None: # Checagem crítica
                    logger.error("ERRO FATAL: Cliente Gemini não inicializado. Não é possível conectar.")
                    self.stop_event.set()
                    break

                logger.info(f"Tentando conectar ao Gemini (Tentativa {attempt+1})...")
                # Usar context manager para a sessão
                async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                    self.session = session
                    session_id_str = getattr(session, 'session_id', getattr(session, '_session_id', 'N/A'))
                    logger.info(f"Sessão Gemini LiveConnect estabelecida (ID: {session_id_str})")
                    attempt = 0 # Reseta tentativas em sucesso

                    self.audio_in_queue = asyncio.Queue(maxsize=200) # Fila para áudio da IA para tocar
                    self.out_queue = asyncio.Queue(maxsize=150) # Fila para dados (mic, video) para IA

                    async with asyncio.TaskGroup() as tg:
                        logger.info("Iniciando tarefas da sessão...")
                        tg.create_task(self.send_text(), name="send_text_task")
                        tg.create_task(self.send_realtime(), name="send_realtime_task")
                        if pya: tg.create_task(self.listen_audio(), name="listen_audio_task")

                        if self.video_mode == "camera":
                            tg.create_task(self.get_frames(), name="get_frames_task")
                        elif self.video_mode == "screen":
                            tg.create_task(self.get_screen(), name="get_screen_task")
                        
                        tg.create_task(self.receive_audio(), name="receive_audio_task")
                        if pya: tg.create_task(self.play_audio(), name="play_audio_task")
                        logger.info("Todas as tarefas da sessão iniciadas.")
                    # TaskGroup espera todas as tarefas terminarem

                    logger.info("TaskGroup da sessão finalizado.")
                    if not self.stop_event.is_set(): # Se terminou sem stop_event, sessão pode ter fechado
                         logger.warning("Sessão Gemini terminou inesperadamente ou TaskGroup concluído. Tentando reconectar...")
                         attempt += 1 
                    else: # Stop_event foi setado, sair do loop de conexão
                        logger.info("Stop event detectado após TaskGroup. Encerrando loop de conexão.")
                        break
            
            except asyncio.CancelledError:
                logger.info("Loop principal (run) cancelado.")
                self.stop_event.set()
                break
            except ExceptionGroup as eg: # Erro de uma ou mais tarefas no TaskGroup
                logger.error(f"Erro(s) no TaskGroup (Tentativa {attempt+1}):", exc_info=False) # exc_info=False para não duplicar tracebacks
                self.stop_event.set() 
                for i, exc in enumerate(eg.exceptions):
                    logger.error(f"  Erro {i+1} no TaskGroup: {type(exc).__name__} - {exc}", exc_info=True) # Log individual com traceback
                attempt += 1
                self.session = None 
            except Exception as e: # Erro na conexão inicial ou outro erro no loop run
                logger.error(f"Erro ao conectar ou erro inesperado no método run (Tentativa {attempt+1}): {e}", exc_info=True)
                
                error_str_upper = str(e).upper()
                is_connection_error = any(err_str in error_str_upper for err_str in [
                    "RST_STREAM", "UNAVAILABLE", "DEADLINE_EXCEEDED", "LIVESESSION CLOSED", 
                    "LIVESESSION NOT CONNECTED", "CONNECTIONCLOSEDERROR", "GOAWAY", 
                    "INTERNALERROR", "FAILED TO ESTABLISH CONNECTION", "AUTHENTICATION", "PERMISSION_DENIED"
                ])

                if is_connection_error: logger.info(f"Detectado erro relacionado à sessão ou conexão Gemini: {e}")
                else: logger.info("Erro não parece ser diretamente de conexão. Verifique o traceback.")
                
                attempt += 1
                self.session = None
                if attempt > max_retries:
                     logger.error("Máximo de tentativas de reconexão atingido. Encerrando.")
                     self.stop_event.set()
                     break 

        # Fim do Loop de Conexão
        if not self.stop_event.is_set() and attempt > max_retries:
             logger.info("Não foi possível restabelecer a conexão com Gemini. Encerrando.")
             self.stop_event.set()

        logger.info("Iniciando limpeza final em AudioLoop.run()...")
        self.stop_event.set() # Garante que todas as tarefas saibam que devem parar

        if self.session: # Fecha sessão se ainda existir (improvável se o loop terminou)
            try: await self.session.close(); logger.info("Sessão LiveConnect fechada na limpeza final.")
            except Exception as e_cs: logger.error(f"Erro ao fechar sessão na limpeza final: {e_cs}")
        self.session = None

        if self.audio_in_queue: # Sinaliza para play_audio parar
            try: self.audio_in_queue.put_nowait(None)
            except asyncio.QueueFull: logger.warning("Fila audio_in_queue cheia ao tentar colocar None na limpeza.")
            except Exception as e_q_put: logger.error(f"Erro ao colocar None na audio_in_queue (limpeza): {e_q_put}")

        if self.preview_window_active: # Fecha janelas OpenCV
            logger.info("Fechando janelas OpenCV na limpeza final...")
            try: cv2.destroyAllWindows()
            except Exception as e_cv_destroy: logger.warning(f"Erro ao fechar janelas OpenCV (limpeza): {e_cv_destroy}")
            self.preview_window_active = False

        if pya: # Termina PyAudio
            try: pya.terminate(); logger.info("Recursos de PyAudio liberados.")
            except Exception as e_pya: logger.error(f"Erro ao terminar PyAudio: {e_pya}")
        
        logger.info("Limpeza de AudioLoop.run() concluída.")


# --- Ponto de Entrada Principal ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trackie - Assistente IA visual e auditivo.")
    parser.add_argument(
        "--mode", type=str, default=DEFAULT_MODE, choices=["camera", "screen", "none"],
        help="Modo de operação para entrada de vídeo/imagem ('camera', 'screen', 'none')."
    )
    parser.add_argument(
        "--show_preview", action="store_true",
        help="Mostra janela com preview da câmera e detecções YOLO (apenas no modo 'camera')."
    )
    args = parser.parse_args()

    show_actual_preview = False
    if args.mode == "camera" and args.show_preview:
        show_actual_preview = True
        logger.info("Feedback visual da câmera (preview) ATIVADO.")
    elif args.mode != "camera" and args.show_preview:
        logger.info("Aviso: --show_preview só tem efeito com --mode camera. Ignorando.")
    # else: logger.info("Feedback visual da câmera (preview) DESATIVADO.") # Implicito

    if args.mode == "camera" and not os.path.exists(YOLO_MODEL_PATH):
        logger.error(f"ERRO CRÍTICO: Modelo YOLO '{YOLO_MODEL_PATH}' Não encontrado.")
        exit(1)

    if not pya: # PyAudio é essencial
         logger.error("ERRO CRÍTICO: PyAudio não pôde ser inicializado. Verifique a instalação e dependências (PortAudio). Encerrando.")
         exit(1)

    if client is None: # Cliente Gemini é essencial
        logger.error("ERRO CRÍTICO: Cliente Gemini não pôde ser inicializado (verifique API Key/conexão). Encerrando.")
        exit(1)

    # Verifica se o arquivo de prompt foi carregado (system_instruction_text deve existir)
    # A verificação `system_instruction_text == "Você é um assistente prestativo."` é frágil se o padrão mudar.
    # Melhor verificar se foi modificado do valor inicial padrão.
    default_prompt_check = "Você é Trackie, um assistente multimodal avançado. Seja conciso e direto ao ponto."
    if 'system_instruction_text' not in globals() or not system_instruction_text or system_instruction_text == default_prompt_check:
         logger.warning("AVISO: Usando prompt do sistema padrão. Verifique SYSTEM_INSTRUCTION_PATH se um prompt customizado era esperado.")

    main_loop_instance = None # Renomeado para evitar conflito com asyncio.run(main_loop.run())
    try:
        logger.info(f"Iniciando Trackie no modo: {args.mode}")
        main_loop_instance = AudioLoop(video_mode=args.mode, show_preview=show_actual_preview)
        asyncio.run(main_loop_instance.run())

    except KeyboardInterrupt:
        logger.info("\nInterrupção pelo teclado recebida (Ctrl+C). Encerrando...")
        if main_loop_instance:
            logger.info("Sinalizando parada para as tarefas...")
            main_loop_instance.stop_event.set()
            # asyncio.run já deve lidar com o cancelamento das tarefas ao sair
    except Exception as e_main:
        logger.critical(f"Erro inesperado e não tratado no bloco __main__: {e_main}", exc_info=True)
        if main_loop_instance:
            logger.info("Sinalizando parada devido a erro inesperado...")
            main_loop_instance.stop_event.set()
    finally:
        logger.info("Bloco __main__ finalizado.")
        # A limpeza principal deve ocorrer no `finally` do método `run` da classe AudioLoop.
        # PyAudio.terminate() é chamado lá.
        logger.info("Programa completamente finalizado.")
```
298,9s

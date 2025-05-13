
# 1. Importações e Dependências
# Bibliotecas Padrão (stdlib)
import os
import asyncio
import base64
import io
import json
import logging # Movido para cima para configurar antes

# --- Configuração de Logging ---
# Habilita logging DEBUG para acompanhar o fluxo de chamadas
logging.basicConfig(
    level=logging.DEBUG, # Alterado para DEBUG
    format="%(asctime)s %(levelname)s %(name)s %(message)s", # Formato ajustado
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
# Também habilita DEBUG para o cliente do GenAI
logging.getLogger("google.genai").setLevel(logging.DEBUG)

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
# Tenta importar pandas, necessário para DeepFace.find
try:
    import pandas as pd
except ImportError:
    logger.info("AVISO: Biblioteca 'pandas' não encontrada. 'identify_person_in_front' pode não funcionar.")
    logger.info("Instale com: pip install pandas")
    pd = None # Define como None se não encontrado

from google import genai
from google.genai import types
from google.protobuf.struct_pb2 import Value # Mantido para FunctionResponse
from ultralytics import YOLO
import numpy as np
from deepface import DeepFace
import torch
# import torchvision # torchvision não é usado diretamente, timm pode ser suficiente ou MiDaS o trará
import timm # timm é usado por MiDaS, verificar se é import explícito necessário ou dependência do MiDaS


# --- Constantes ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.0-flash-live-001" # Confirmar se este modelo suporta a API LiveConnect e Function Calling
DEFAULT_MODE = "camera"
BaseDir = "C:/Users/Pedro H/Downloads/TrackiePowerSHell/" # Considere usar caminhos relativos ou configuração

# --- Caminho para o arquivo de prompt ---
SYSTEM_INSTRUCTION_PATH = os.path.join(BaseDir,"UserSettings", "Prompt's", "trackiegem1.txt")

# YOLO
YOLO_MODEL_PATH = os.path.join(BaseDir,"WorkTools", "yolo11n.pt")
DANGER_CLASSES = {
    'faca':             ['knife'],
    'tesoura':          ['scissors'],
    'barbeador':        ['razor'],
    'serra':            ['saw'],
    'machado':          ['axe'],
    'machadinha':       ['hatchet'],
    'arma_de_fogo':     ['gun'],
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
    'fio_energizado':   ['live_wire'],
    'tomada_elétrica':  ['electric_outlet'],
    'bateria':          ['battery'],
    'vidro_quebrado':   ['broken_glass'],
    'estilhaço':        ['shard'],
    'agulha':           ['needle'],
    'seringa':         ['syringe'],
    'martelo':          ['hammer'],
    'chave_de_fenda':   ['wrench'], # 'wrench' é chave inglesa, 'screwdriver' é chave de fenda
    'furadeira':        ['drill'],
    'motosserra':       ['chainsaw'],
    'carro':            ['car'],
    'motocicleta':      ['motorcycle'],
    'bicicleta':        ['bicycle'],
    'caminhão':         ['truck'],
    'ônibus':           ['bus'],
    'urso':             ['bear'],
    'cobra':            ['snake'],
    'aranha':           ['spider'],
    'jacaré':           ['alligator'],
    'penhasco':         ['cliff'],
    'buraco':           ['hole'],
    'escada':           ['stairs'],
}
YOLO_CONFIDENCE_THRESHOLD = 0.40
YOLO_CLASS_MAP = {
    "pessoa":                     ["person"],
    "gato":                       ["cat"],
    "cachorro":                   ["dog"],
    "coelho":                     ["rabbit"],
    "urso":                       ["bear"],
    "elefante":                   ["elephant"],
    "zebra":                      ["zebra"],
    "girafa":                     ["giraffe"],
    "vaca":                       ["cow"],
    "cavalo":                     ["horse"],
    "ovelha":                     ["sheep"],
    "macaco":                     ["monkey"],
    "bicicleta":                  ["bicycle"],
    "moto":                       ["motorcycle"], # "motorcycle" já existe em DANGER_CLASSES
    "carro":                      ["car"],      # "car" já existe em DANGER_CLASSES
    "ônibus":                     ["bus"],      # "bus" já existe em DANGER_CLASSES
    "trem":                       ["train"],
    "caminhão":                   ["truck"],    # "truck" já existe em DANGER_CLASSES
    "avião":                      ["airplane"],
    "barco":                      ["boat"],
    "skate":                      ["skateboard"],
    "prancha de surf":            ["surfboard"],
    "tênis":                      ["tennis racket"], # Raquete de tênis
    "mesa de jantar":             ["dining table"],
    "mesa":                       ["table", "desk", "dining table"],
    "cadeira":                    ["chair"],
    "sofá":                       ["couch", "sofa"],
    "cama":                       ["bed"],
    "vaso de planta":             ["potted plant"],
    "banheiro":                   ["toilet"], # Vaso sanitário
    "televisão":                  ["tv", "tvmonitor"],
    "abajur":                     ["lamp"],
    "espelho":                    ["mirror"],
    "laptop":                     ["laptop"],
    "computador":                 ["computer", "desktop computer", "tv"],
    "teclado":                    ["keyboard"],
    "mouse":                      ["mouse"],
    "controle remoto":            ["remote"],
    "celular":                    ["cell phone"],
    "micro-ondas":                ["microwave"],
    "forno":                      ["oven"],
    "torradeira":                 ["toaster"],
    "geladeira":                  ["refrigerator"],
    "caixa de som":               ["speaker"],
    "câmera":                     ["camera"],
    "garrafa":                    ["bottle"],
    "copo":                       ["cup"],
    "taça de vinho":              ["wine glass"],
    "taça":                       ["wine glass", "cup"],
    "prato":                      ["plate", "dish"],
    "tigela":                     ["bowl"],
    "garfo":                      ["fork"],
    "faca":                       ["knife"],    # "knife" já existe em DANGER_CLASSES
    "colher":                     ["spoon"],
    "panela":                     ["pan", "pot"],
    "frigideira":                 ["skillet", "frying pan"],
    "martelo":                    ["hammer"],   # "hammer" já existe em DANGER_CLASSES
    "chave inglesa":              ["wrench"],   # "wrench" já existe em DANGER_CLASSES como 'chave_de_fenda' (erro no original, wrench é inglesa)
    "furadeira":                  ["drill"],    # "drill" já existe em DANGER_CLASSES
    "parafusadeira":              ["drill"],    # Mapeado para "drill"
    "serra":                      ["saw"],      # "saw" já existe em DANGER_CLASSES
    "roçadeira":                  ["brush cutter"],
    "alicate":                    ["pliers"],
    "chave de fenda":             ["screwdriver"], # Corrigido: screwdriver
    "lanterna":                   ["flashlight"],
    "fita métrica":               ["tape measure"],
    "mochila":                    ["backpack"],
    "bolsa":                      ["handbag", "purse", "bag"],
    "carteira":                   ["wallet"],
    "óculos":                     ["glasses", "eyeglasses"],
    "relógio":                    ["clock", "watch"],
    "chinelo":                    ["sandal", "flip-flop"],
    "sapato":                     ["shoe"],
    "sanduíche":                  ["sandwich"],
    "hambúrguer":                 ["hamburger"],
    "banana":                     ["banana"],
    "maçã":                       ["apple"],
    "laranja":                    ["orange"],
    "bolo":                       ["cake"],
    "rosquinha":                  ["donut"],
    "pizza":                      ["pizza"],
    "cachorro-quente":            ["hot dog"],
    "escova de dentes":           ["toothbrush"],
    "secador de cabelo":          ["hair drier", "hair dryer"],
    "cotonete":                   ["cotton swab"],
    "sacola plástica":            ["plastic bag"],
    "livro":                      ["book"],
    "vaso":                       ["vase"],
    "bola":                       ["sports ball", "ball"],
    "bexiga":                     ["balloon"],
    "pipa":                       ["kite"],
    "luva":                       ["glove"],
    "skis":                       ["skis"],
    "snowboard":                  ["snowboard"],
    "tesoura":                    ["scissors"], # "scissors" já existe em DANGER_CLASSES
}


# DeepFace
DB_PATH = os.path.join(BaseDir,"UserSettings", "known_faces")
DEEPFACE_MODEL_NAME = 'VGG-Face'
DEEPFACE_DETECTOR_BACKEND = 'opencv'
DEEPFACE_DISTANCE_METRIC = 'cosine'

# MiDaS
MIDAS_MODEL_TYPE = "MiDaS_small" # "MiDaS_small" é mais leve. Outras opções: "MiDaS", "DPT_Large", "DPT_Hybrid"
METERS_PER_STEP = 0.7 # Assumindo que um passo tem em média 0.7 metros

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

# ATENÇÃO: A chave hardcoded abaixo é um RISCO DE SEGURANÇA e NÃO DEVE SER USADA EM PRODUÇÃO.
# Priorize o uso de variáveis de ambiente ou arquivos .env seguros.
# Se API_KEY não foi carregada, o código tentará usar a chave hardcoded.
# Remova a chave hardcoded se possível.
HARDCODED_API_KEY = "AIzaSyCOZU2M9mrAsx8aC4tYptfoSEwaJ3IuDZM" # Chave de exemplo do código original

FINAL_API_KEY_TO_USE = API_KEY

if not FINAL_API_KEY_TO_USE:
    logger.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    logger.warning("!! AVISO CRÍTICO: Chave da API Gemini não encontrada nas variáveis de ambiente !!")
    logger.warning(f"!! ou .env. Tentando usar uma chave HARDCODED: {HARDCODED_API_KEY[:4]}...XXXX !!")
    logger.warning("!! ISTO NÃO É SEGURO PARA PRODUÇÃO. Configure GEMINI_API_KEY no ambiente.   !!")
    logger.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    FINAL_API_KEY_TO_USE = HARDCODED_API_KEY # Use a hardcoded como último recurso

client = None
if FINAL_API_KEY_TO_USE:
    try:
        client = genai.Client(
            api_key=FINAL_API_KEY_TO_USE,
            http_options=types.HttpOptions(api_version='v1alpha') # v1alpha para LiveConnect
        )
        logger.info("Cliente Gemini inicializado com sucesso.")
    except Exception as e_client:
        logger.error(f"ERRO CRÍTICO ao inicializar cliente Gemini: {e_client}")
        logger.info("Verifique a API Key, a conexão e se a API LiveConnect está habilitada para sua chave.")
        traceback.print_exc()
        # client permanece None, o que será tratado no if __name__ == "__main__"
else:
    logger.error("ERRO CRÍTICO: Nenhuma API Key do Gemini disponível (ambiente, .env ou hardcoded).")
    # client permanece None

# --- Declarações de Funções Individuais para Gemini ---
save_func_decl = types.FunctionDeclaration(
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
        # person_name não é obrigatório na chamada inicial, pois a IA pode pedir depois.
        # No entanto, para a execução da função em si, ele se torna necessário.
        # A lógica de pedir o nome se ausente é tratada no fluxo da conversa.
        # Se a IA sempre DEVE fornecer, então required=["person_name"]
    )
)

identify_func_decl = types.FunctionDeclaration(
    name="identify_person_in_front",
    description="Identifica a pessoa atualmente em foco pela camera usando o banco de dados de rostos conhecidos (localizado em DB_PATH/known_faces). Deve ser chamado somente quando o usuário solicitar explicitamente a identificação de uma pessoa ou rosto.",
    parameters=types.Schema(type=types.Type.OBJECT, properties={}) # Sem parâmetros de entrada do usuário para esta chamada
)

find_func_decl = types.FunctionDeclaration(
    name="find_object_and_estimate_distance",
    description="Localiza um objeto específico descrito pelo usuário na visão da câmera, usando detecção de objetos (YOLO) e estima sua distância em passos usando um modelo de profundidade (MiDaS). Informa também se o objeto está sobre uma superfície como uma mesa e sua direção relativa (frente, esquerda, direita) em relação ao campo de visão da câmera.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "object_description": types.Schema(
                type=types.Type.STRING,
                description="A descrição completa do objeto fornecida pelo usuário (ex: 'computador azul', 'chave de fenda preta', 'minha garrafa de água')."
            ),
            "object_type": types.Schema(
                type=types.Type.STRING,
                description="O tipo principal do objeto que a IA identificou a partir da descrição do usuário (ex: 'computador', 'chave de fenda', 'garrafa'). Usado para filtrar detecções YOLO. Deve ser um nome de classe YOLO conhecido ou um tipo genérico."
            )
        },
        required=["object_description", "object_type"]
    )
)

# --- Ferramentas Gemini (Function Calling) ---
# Cada função em seu próprio Tool
google_search_tool = types.Tool(google_search=types.GoogleSearch())
code_exec_tool = types.Tool(code_execution=types.ToolCodeExecution()) # Permite ao Gemini executar código Python
save_tool = types.Tool(function_declarations=[save_func_decl])
identify_tool = types.Tool(function_declarations=[identify_func_decl])
find_tool = types.Tool(function_declarations=[find_func_decl])

FTOOLS = [google_search_tool, code_exec_tool, save_tool, identify_tool, find_tool]
logger.debug(f"Ferramentas (FTOOLS) configuradas: {[type(tool).__name__ for tool in FTOOLS]}")
for tool in FTOOLS:
    if tool.function_declarations:
        for func_decl in tool.function_declarations:
            logger.debug(f"  - Função declarada: {func_decl.name}")


# --- Carregar Instrução do Sistema do Arquivo ---
system_instruction_text = "Você é Trackie, um assistente multimodal avançado e amigável. Seu objetivo principal é ajudar o usuário a interagir com o ambiente ao seu redor através de visão computacional e processamento de linguagem natural. Você pode ver o que a câmera vê, ouvir o usuário e responder por voz. Use as ferramentas disponíveis quando apropriado para atender aos pedidos do usuário. Seja proativo ao oferecer ajuda com base no que você 'vê', mas sempre confirme com o usuário. Se uma função requer um nome (como salvar um rosto) e o usuário não o forneceu, peça o nome antes de prosseguir. Para a função 'find_object_and_estimate_distance', o 'object_type' deve ser uma classe que o modelo YOLO possa reconhecer (ex: 'garrafa', 'livro', 'celular', 'faca', 'tesoura', 'computador', 'teclado', 'mouse')." # Prompt padrão robusto
try:
    if not os.path.exists(SYSTEM_INSTRUCTION_PATH):
         logger.warning(f"AVISO: Arquivo de instrução do sistema não encontrado em '{SYSTEM_INSTRUCTION_PATH}'. Usando prompt padrão robusto.")
    else:
        with open(SYSTEM_INSTRUCTION_PATH, 'r', encoding='utf-8') as f:
            system_instruction_text = f.read()
        logger.info(f"Instrução do sistema carregada de: {SYSTEM_INSTRUCTION_PATH}")
except Exception as e_prompt:
    logger.error(f"Erro ao ler o arquivo de instrução do sistema: {e_prompt}")
    logger.info("Usando um prompt padrão robusto.")
    traceback.print_exc()


# --- Configuração da Sessão LiveConnect Gemini ---
# A presença de 'tools' no LiveConnectConfig habilita a chamada de função.
# O modelo decide automaticamente quando usá-las (comportamento "auto").
CONFIG = types.LiveConnectConfig(
    temperature=0.1, # Um pouco de temperatura pode ajudar na naturalidade, mas 0 é bom para consistência
    response_modalities=["audio"], # Queremos que o Gemini responda com áudio
    speech_config=types.SpeechConfig(
        language_code="pt-BR",
        voice_config=types.VoiceConfig(
            # Escolha uma voz. "Orus" é uma opção. Verifique vozes disponíveis.
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Orus") # Exemplo, verificar nomes exatos
        )
    ),
    tools=FTOOLS, # Lista de ferramentas (incluindo as funções declaradas)
    system_instruction=types.Content(
        parts=[types.Part.from_text(text=system_instruction_text)],
        role="user" # Para LiveConnect, system_instruction é frequentemente o primeiro turno do "user"
    ),
    # Não há um parâmetro explícito `function_calling="auto"` no LiveConnectConfig do SDK google-genai.
    # A funcionalidade é ativada pela presença de `tools`.
)
logger.debug(f"LiveConnectConfig: Temperature={CONFIG.temperature}, ResponseModalities={CONFIG.response_modalities}, Tools presentes: {bool(CONFIG.tools)}")


# --- Inicialização do PyAudio ---
try:
    pya = pyaudio.PyAudio()
    logger.info("PyAudio inicializado.")
except Exception as e_pyaudio:
    logger.error(f"Erro CRÍTICO ao inicializar PyAudio: {e_pyaudio}. O áudio não funcionará.")
    traceback.print_exc()
    pya = None

# --- Classe Principal do Assistente ---
class AudioLoop:
    """
    Gerencia o loop principal do assistente multimodal.
    """
    def __init__(self, video_mode: str = DEFAULT_MODE, show_preview: bool = False):
        self.video_mode = video_mode
        self.show_preview = show_preview if video_mode == "camera" else False
        self.audio_in_queue: Optional[asyncio.Queue] = None # Fila para áudio recebido do Gemini
        self.out_queue: Optional[asyncio.Queue] = None      # Fila para dados (áudio/vídeo) a serem enviados ao Gemini

        self.thinking_event = asyncio.Event() # Sinaliza quando o Gemini está "pensando" (processando uma função)
        self.session: Optional[genai.live.AsyncLiveSession] = None
        self.yolo_model: Optional[YOLO] = None
        self.preview_window_active: bool = False
        self.stop_event = asyncio.Event() # Sinal para parar todas as tarefas
        self.frame_lock = threading.Lock() # Lock para acesso seguro ao último frame
        self.latest_bgr_frame: Optional[np.ndarray] = None
        self.latest_yolo_results: Optional[List[Any]] = None # Resultados YOLO para o latest_bgr_frame

        self.awaiting_name_for_save_face: bool = False # Estado para o fluxo de salvar rosto

        # Carregamento de Modelos
        if self.video_mode == "camera":
            try:
                self.yolo_model = YOLO(YOLO_MODEL_PATH)
                logger.info(f"Modelo YOLO '{YOLO_MODEL_PATH}' carregado.")
            except FileNotFoundError:
                 logger.error(f"ERRO: Modelo YOLO não encontrado em '{YOLO_MODEL_PATH}'. YOLO desabilitado.")
                 self.yolo_model = None
            except Exception as e_yolo_load:
                logger.error(f"Erro ao carregar o modelo YOLO: {e_yolo_load}. YOLO desabilitado.")
                traceback.print_exc()
                self.yolo_model = None

        if not os.path.exists(DB_PATH):
            try:
                os.makedirs(DB_PATH)
                logger.info(f"Diretório DeepFace DB criado em: {DB_PATH}")
            except Exception as e_db_create:
                logger.error(f"Erro ao criar diretório {DB_PATH}: {e_db_create}")

        try:
            logger.info("Pré-carregando modelos DeepFace (pode levar um momento)...")
            dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            # Uma análise leve para forçar o download/cache dos modelos DeepFace
            DeepFace.analyze(img_path=dummy_frame, actions=['emotion'], enforce_detection=False, silent=True)
            logger.info("Modelos DeepFace pré-carregados (ou download iniciado).")
        except Exception as e_deepface_preload:
            logger.warning(f"Aviso: Erro ao pré-carregar modelos DeepFace: {e_deepface_preload}.")
            # traceback.print_exc()

        self.midas_model = None
        self.midas_transform = None
        self.midas_device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            logger.info(f"Carregando modelo MiDaS ({MIDAS_MODEL_TYPE}) para o dispositivo: {self.midas_device}...")
            self.midas_model = torch.hub.load("intel-isl/MiDaS", MIDAS_MODEL_TYPE, trust_repo=True)
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            if MIDAS_MODEL_TYPE == "MiDaS_small":
                 self.midas_transform = midas_transforms.small_transform
            else: # Para "MiDaS" (não small) ou DPT, o transform pode ser diferente
                 self.midas_transform = midas_transforms.dpt_transform # Ajustar se usar outros modelos MiDaS
            self.midas_model.to(self.midas_device)
            self.midas_model.eval()
            logger.info("Modelo MiDaS carregado com sucesso.")
        except Exception as e_midas_load:
            logger.error(f"Erro ao carregar modelo MiDaS: {e_midas_load}. Estimativa de profundidade desabilitada.")
            # traceback.print_exc()
            self.midas_model = None
            self.midas_transform = None

    async def send_text(self):
        """Lê input de texto do usuário, trata comandos de debug e envia ao Gemini."""
        logger.info("Pronto para receber comandos de texto. Digite 'q' para sair.")
        logger.info("Comandos de debug: 'p <nome>' para salvar rosto (ex: p pedro).")
        while not self.stop_event.is_set():
            try:
                text_input = await asyncio.to_thread(input, "message > ")

                if text_input.lower() == "q":
                    self.stop_event.set()
                    logger.info("Sinal de parada ('q') recebido. Encerrando...")
                    break

                elif text_input.lower().startswith("p "):
                    name_to_save = text_input[2:].strip()
                    if not name_to_save:
                        logger.info("[DEBUG] Uso: p <nome_da_pessoa>")
                        continue
                    logger.info(f"[DEBUG] Comando 'p' recebido. Tentando salvar rosto como '{name_to_save}'...")
                    if self.video_mode == "camera":
                        try:
                            # self.thinking_event.set() # Opcional: pausar outros envios
                            logger.debug(f"  [DEBUG] Chamando _handle_save_known_face('{name_to_save}')...")
                            result = await asyncio.to_thread(self._handle_save_known_face, name_to_save)
                            logger.info(f"  [DEBUG] Resultado do salvamento direto: {result}")
                        except Exception as e_debug_save:
                            logger.error(f"  [DEBUG] Erro ao tentar salvar rosto diretamente: {e_debug_save}")
                            traceback.print_exc()
                        # finally:
                            # self.thinking_event.clear()
                    else:
                        logger.info("  [DEBUG] Salvar rosto (comando 'p') só funciona no modo câmera.")
                    continue

                if self.session:
                    logger.debug(f"Enviando texto para Gemini: '{text_input}'")
                    await self.session.send(input=text_input or ".", end_of_turn=True)
                else:
                    if not self.stop_event.is_set():
                        logger.warning("Sessão Gemini não está ativa. Não é possível enviar mensagem de texto.")
                        await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                logger.info("Tarefa send_text cancelada.")
                break
            except Exception as e_send_text:
                logger.error(f"Erro em send_text: {e_send_text}")
                error_str_upper = str(e_send_text).upper()
                if "LIVESESSION CLOSED" in error_str_upper or "LIVESESSION NOT CONNECTED" in error_str_upper:
                    logger.info("Erro em send_text indica sessão fechada. Sinalizando parada.")
                    self.stop_event.set()
                break
        logger.info("Tarefa send_text finalizada.")


    def _get_frame(self, cap: cv2.VideoCapture) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        # (Função _get_frame inalterada - omitida para brevidade, mas logs DEBUG podem ser adicionados se necessário)
        ret, frame = cap.read()
        latest_frame_copy = None
        current_yolo_results = None

        if ret:
            latest_frame_copy = frame.copy() # Copia para processamento e armazenamento

        yolo_alerts = []
        display_frame = None # Frame para mostrar no preview
        if ret and self.yolo_model:
            # Converte para RGB para YOLO e PIL
            frame_rgb = cv2.cvtColor(latest_frame_copy, cv2.COLOR_BGR2RGB)
            try:
                # verbose=False para menos output, conf para threshold de confiança
                results = self.yolo_model.predict(frame_rgb, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD)
                current_yolo_results = results # Armazena os resultados brutos

                if self.show_preview:
                    display_frame = latest_frame_copy.copy() # Copia para desenhar

                for result in results: # results é uma lista, geralmente com um item
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        class_name_yolo = self.yolo_model.names[cls_id]
                        conf = float(box.conf[0])

                        if display_frame is not None: # Desenha no frame de display
                            label = f"{class_name_yolo} {conf:.2f}"
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        # Verifica se o objeto detectado é perigoso
                        is_dangerous = any(class_name_yolo in danger_list for danger_list in DANGER_CLASSES.values())
                        if is_dangerous and conf >= YOLO_CONFIDENCE_THRESHOLD: # Usa o mesmo threshold
                            yolo_alerts.append(class_name_yolo)
            except Exception as e_yolo_infer:
                logger.error(f"Erro na inferência YOLO: {e_yolo_infer}")
                current_yolo_results = None # Limpa em caso de erro
        elif self.show_preview and ret: # Se não tem YOLO mas tem preview e frame
            display_frame = latest_frame_copy.copy()


        # Atualiza o frame mais recente e resultados YOLO de forma thread-safe
        with self.frame_lock:
            if ret:
                self.latest_bgr_frame = latest_frame_copy # Armazena o BGR original
                self.latest_yolo_results = current_yolo_results
            else: # Falha na leitura do frame
                self.latest_bgr_frame = None
                self.latest_yolo_results = None
                return None, [] # Retorna None se não conseguiu ler o frame

        # Mostra o preview se habilitado e o display_frame foi preparado
        if self.show_preview and display_frame is not None:
            try:
                cv2.imshow("Trackie YOLO Preview", display_frame)
                cv2.waitKey(1) # Essencial para a janela ser processada
                self.preview_window_active = True
            except cv2.error as e_cv_show: # Erros comuns de display (headless, etc)
                if "DISPLAY" in str(e_cv_show).upper() or "GTK" in str(e_cv_show).upper() or \
                   "QT" in str(e_cv_show).upper() or "COULD NOT CONNECT TO DISPLAY" in str(e_cv_show).upper() or \
                   "plugin \"xcb\"" in str(e_cv_show).lower(): # Adicionado xcb
                    logger.warning("--------------------------------------------------------------------")
                    logger.warning("AVISO: Não foi possível mostrar a janela de preview da câmera.")
                    logger.warning("Isso pode acontecer em ambientes headless ou sem servidor X.")
                    logger.warning("Desabilitando feedback visual para esta sessão.")
                    logger.warning("--------------------------------------------------------------------")
                    self.show_preview = False # Desabilita para não tentar de novo
                    self.preview_window_active = False
                else: # Outro erro OpenCV
                    logger.error(f"Erro inesperado no OpenCV ao tentar mostrar preview: {e_cv_show}")
            except Exception as e_gen_show: # Erro genérico
                logger.error(f"Erro geral ao tentar mostrar preview: {e_gen_show}")
                self.show_preview = False
                self.preview_window_active = False


        image_part_for_gemini = None
        if ret: # Se o frame foi lido com sucesso
            try:
                # Reusa frame_rgb se já convertido, senão converte latest_bgr_frame
                if 'frame_rgb' not in locals() or frame_rgb is None:
                     frame_rgb_for_pil = cv2.cvtColor(self.latest_bgr_frame, cv2.COLOR_BGR2RGB)
                else:
                     frame_rgb_for_pil = frame_rgb

                img = Image.fromarray(frame_rgb_for_pil)
                img.thumbnail([1024, 1024]) # Redimensiona mantendo proporção, max 1024x1024
                image_io = io.BytesIO()
                img.save(image_io, format="jpeg", quality=85) # Boa qualidade, tamanho razoável
                image_io.seek(0)
                image_part_for_gemini = {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(image_io.read()).decode('utf-8')
                }
            except Exception as e_img_convert:
                logger.error(f"Erro na conversão do frame para JPEG (Gemini): {e_img_convert}")

        return image_part_for_gemini, list(set(yolo_alerts)) # Remove duplicatas dos alertas


    async def get_frames(self):
        # (Função get_frames inalterada - omitida para brevidade, mas logs DEBUG podem ser adicionados se necessário)
        cap = None
        try:
            logger.info("Iniciando captura da câmera (get_frames)...")
            # cv2.VideoCapture(0) pode bloquear, então rodamos em thread separada
            cap = await asyncio.to_thread(cv2.VideoCapture, 0) # Câmera padrão
            
            # Tenta configurar FPS, mas nem todas as câmeras respeitam
            target_fps = 1 # Um frame por segundo é um bom começo para não sobrecarregar
            cap.set(cv2.CAP_PROP_FPS, target_fps)
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"FPS solicitado: {target_fps}, FPS real da câmera: {actual_fps if actual_fps > 0 else 'Não disponível/Padrão'}")

            # Define o intervalo de sleep com base no FPS (real ou alvo)
            # Garante um mínimo e máximo para o sleep para evitar busy-waiting ou delays muito longos
            sleep_interval = 1.0 / (actual_fps if actual_fps > 0 and actual_fps < target_fps * 5 else target_fps)
            sleep_interval = max(0.1, min(sleep_interval, 1.0)) # Limita entre 0.1s e 1.0s
            logger.info(f"Intervalo de captura de frame definido para: {sleep_interval:.2f}s")

            if not cap.isOpened():
                logger.error("Erro CRÍTICO: Não foi possível abrir a câmera.")
                with self.frame_lock: # Limpa o estado do frame
                    self.latest_bgr_frame = None
                    self.latest_yolo_results = None
                self.stop_event.set() # Sinaliza para parar tudo
                return

            while not self.stop_event.is_set():
                if not cap.isOpened(): # Verifica se a câmera foi desconectada
                    logger.warning("Câmera desconectada ou fechada inesperadamente durante o loop.")
                    self.stop_event.set()
                    break

                # Processamento do frame (leitura, YOLO, conversão) em thread separada
                image_part, yolo_alerts = await asyncio.to_thread(self._get_frame, cap)

                with self.frame_lock: # Verifica se a leitura foi bem-sucedida
                    frame_was_read_successfully = self.latest_bgr_frame is not None

                if not frame_was_read_successfully:
                     if not cap.isOpened(): # Se falhou e a câmera fechou
                         logger.info("Leitura do frame falhou e câmera está fechada. Encerrando get_frames.")
                         self.stop_event.set()
                         break
                     else: # Falha temporária
                         logger.warning("Aviso: Falha temporária na leitura do frame da câmera.")
                         await asyncio.sleep(0.5) # Espera um pouco antes de tentar de novo
                         continue

                # Envia frame para a fila de saída (para Gemini)
                if image_part is not None and self.out_queue:
                    try:
                        if self.out_queue.full():
                            discarded = await self.out_queue.get() # Descarta o mais antigo
                            self.out_queue.task_done()
                            logger.debug("Fila de saída (out_queue) cheia, descartando frame antigo.")
                        self.out_queue.put_nowait(image_part)
                    except asyncio.QueueFull:
                         logger.debug("Fila de saída (out_queue) ainda cheia ao tentar enfileirar frame (put_nowait).")
                    except Exception as q_e_put_frame:
                         logger.error(f"Erro inesperado ao enfileirar frame na out_queue: {q_e_put_frame}")

                # Envia alertas YOLO urgentes diretamente para o Gemini
                if yolo_alerts and self.session:
                    for alert_class_name in yolo_alerts:
                        try:
                            # Formata a mensagem de alerta
                            alert_msg = f"Usuário, ATENÇÃO IMEDIATA! Um objeto perigoso '{alert_class_name.upper()}' foi detectado próximo a você!"
                            # Envia como um turno completo para interromper e alertar
                            await self.session.send(input=alert_msg, end_of_turn=True)
                            logger.info(f"ALERTA URGENTE ENVIADO para Gemini: {alert_msg}")
                        except Exception as e_alert_send:
                            logger.error(f"Erro ao enviar alerta YOLO urgente: {e_alert_send}")
                            if "LiveSession closed" in str(e_alert_send) or "LiveSession not connected" in str(e_alert_send):
                                logger.warning("Sessão Gemini fechada ao tentar enviar alerta. Sinalizando parada.")
                                self.stop_event.set()
                                break # Sai do loop de alertas
                    if self.stop_event.is_set(): break # Sai do loop principal se a sessão fechou

                await asyncio.sleep(sleep_interval) # Aguarda antes do próximo ciclo

        except asyncio.CancelledError:
            logger.info("Tarefa get_frames cancelada.")
        except Exception as e_get_frames:
            logger.error(f"Erro crítico em get_frames: {e_get_frames}")
            traceback.print_exc()
            self.stop_event.set() # Para tudo em caso de erro crítico
        finally:
            logger.info("Finalizando get_frames...")
            if cap and cap.isOpened():
                cap.release()
                logger.info("Câmera liberada.")
            with self.frame_lock: # Limpa o estado final
                self.latest_bgr_frame = None
                self.latest_yolo_results = None
            if self.preview_window_active: # Garante que a janela feche
                try:
                    cv2.destroyWindow("Trackie YOLO Preview") # Tenta fechar a específica
                except: pass
                try:
                    cv2.destroyAllWindows() # Tenta fechar todas
                    logger.info("Janelas OpenCV fechadas.")
                except Exception as e_cv_destroy_all:
                    logger.warning(f"Aviso: erro ao tentar fechar janelas de preview no finally: {e_cv_destroy_all}")
            self.preview_window_active = False
            logger.info("Tarefa get_frames concluída.")


    def _get_screen(self) -> Optional[Dict[str, Any]]:
        # (Função _get_screen inalterada - omitida para brevidade, mas logs DEBUG podem ser adicionados se necessário)
        # Usa mss para captura de tela
        sct = mss.mss()
        monitor_number = 1 # Tenta o monitor 1 (geralmente o principal, se houver múltiplos)
        try:
            monitors = sct.monitors
            if not monitors:
                logger.error("Erro: Nenhum monitor detectado por mss.")
                return None

            # Seleciona o monitor: tenta o monitor 1, depois o 0 (todos), depois o segundo (índice 1)
            if len(monitors) > monitor_number: # Se existe monitor 1 (ou mais)
                 monitor_to_capture = monitors[monitor_number]
            elif len(monitors) == 2 and monitor_number == 1: # Comum: monitor 0 é 'all', monitor 1 é o primário
                 monitor_to_capture = monitors[1]
            elif monitors: # Fallback para o primeiro monitor disponível (pode ser o 'all monitors' ou o único)
                 monitor_to_capture = monitors[0]
                 if len(monitors) > 1 and monitors[0]['width'] > monitors[1]['width'] * 1.5 : # Se o monitor 0 for muito largo (provavelmente 'all')
                     monitor_to_capture = monitors[1] # Tenta pegar o segundo que é mais provável ser o primário
            else: # Redundante, já coberto por 'if not monitors'
                logger.error("Erro: Falha ao selecionar um monitor para captura.")
                return None
            
            logger.debug(f"Capturando tela do monitor: {monitor_to_capture}")

            sct_img = sct.grab(monitor_to_capture) # Captura a imagem do monitor

            # Cria imagem PIL a partir dos dados brutos (BGRA -> RGB)
            img = Image.frombytes('RGB', sct_img.size, sct_img.rgb, 'raw', 'BGR')
            # Não precisa converter para RGB explicitamente se mss já fornece .rgb
            # img = Image.frombytes('RGB', (sct_img.width, sct_img.height), sct_img.rgb)


            image_io = io.BytesIO()
            img.save(image_io, format="PNG") # PNG é melhor para screenshots (sem perdas)
            image_io.seek(0)

            return {
                "mime_type": "image/png",
                "data": base64.b64encode(image_io.read()).decode('utf-8')
            }
        except Exception as e_mss_capture:
            logger.error(f"Erro ao capturar tela com mss: {e_mss_capture}")
            # traceback.print_exc()
            return None


    async def get_screen(self):
        # (Função get_screen inalterada - omitida para brevidade, mas logs DEBUG podem ser adicionados se necessário)
        logger.info("Iniciando captura de tela (get_screen)...")
        try:
            while not self.stop_event.is_set():
                # Captura síncrona em outra thread
                frame_data = await asyncio.to_thread(self._get_screen)

                if frame_data is None:
                    logger.warning("Falha ao capturar frame da tela.")
                    await asyncio.sleep(1.0) # Espera antes de tentar novamente
                    continue

                if self.out_queue:
                    try:
                         if self.out_queue.full():
                             discarded = await self.out_queue.get()
                             self.out_queue.task_done()
                             logger.debug("Fila de saída (out_queue) cheia, descartando frame de tela antigo.")
                         self.out_queue.put_nowait(frame_data)
                    except asyncio.QueueFull:
                         logger.debug("Fila de saída (out_queue) ainda cheia ao tentar enfileirar frame de tela.")
                    except Exception as q_e_put_screen:
                         logger.error(f"Erro inesperado ao enfileirar frame de tela na out_queue: {q_e_put_screen}")

                await asyncio.sleep(1.0) # Intervalo entre capturas de tela
        except asyncio.CancelledError:
            logger.info("Tarefa get_screen cancelada.")
        except Exception as e_get_screen:
            logger.error(f"Erro crítico em get_screen: {e_get_screen}")
            traceback.print_exc()
            self.stop_event.set()
        finally:
            logger.info("Tarefa get_screen finalizada.")


    async def send_realtime(self):
        # (Função send_realtime inalterada - omitida para brevidade, mas logs DEBUG podem ser adicionados se necessário)
        logger.info("Tarefa send_realtime iniciada, pronta para enviar dados para Gemini...")
        try:
            while not self.stop_event.is_set():
                if self.thinking_event.is_set(): # Pausa se Gemini estiver processando função
                    await asyncio.sleep(0.05)
                    continue

                if not self.out_queue: # Fila pode não existir durante reconexão
                    await asyncio.sleep(0.1)
                    continue

                msg_to_send = None
                try:
                    # Espera por um item na fila com timeout
                    msg_to_send = await asyncio.wait_for(self.out_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue # Normal se a fila estiver vazia
                except asyncio.QueueEmpty: # Redundante com timeout, mas seguro
                    continue
                except Exception as q_get_e:
                    logger.error(f"Erro ao obter da out_queue em send_realtime: {q_get_e}")
                    await asyncio.sleep(0.1)
                    continue

                if not self.session: # Sessão pode ter caído
                    logger.warning("Sessão Gemini não está ativa (send_realtime). Descartando mensagem.")
                    if self.out_queue and msg_to_send: self.out_queue.task_done() # Marca como concluído
                    if not self.stop_event.is_set(): await asyncio.sleep(0.5)
                    continue

                try:
                    if isinstance(msg_to_send, dict) and "data" in msg_to_send and "mime_type" in msg_to_send:
                        # logger.debug(f"Enviando parte multimodal: {msg_to_send['mime_type']}")
                        await self.session.send(input=msg_to_send)
                    elif isinstance(msg_to_send, str): # Para alertas urgentes
                        logger.info(f"Enviando texto via send_realtime (alerta): {msg_to_send}")
                        await self.session.send(input=msg_to_send, end_of_turn=False) # Não finaliza turno da IA
                    else:
                        logger.warning(f"Tipo de mensagem desconhecido em send_realtime: {type(msg_to_send)}")

                    if self.out_queue: self.out_queue.task_done() # Confirma processamento do item

                except Exception as e_send_gemini:
                    logger.error(f"Erro ao enviar para Gemini em send_realtime: {e_send_gemini}")
                    if self.out_queue and msg_to_send: self.out_queue.task_done() # Garante task_done

                    error_str_upper = str(e_send_gemini).upper()
                    if any(err_indicator in error_str_upper for err_indicator in [
                        "LIVESESSION CLOSED", "LIVESESSION NOT CONNECTED", "DEADLINE EXCEEDED",
                        "RST_STREAM", "UNAVAILABLE", "CONNECTIONCLOSEDERROR", "GOAWAY"
                    ]):
                        logger.warning("Erro de envio indica sessão Gemini fechada/perdida. Sinalizando parada.")
                        self.stop_event.set()
                        break # Sai do loop while
                    else: # Outros erros podem ser temporários
                        # traceback.print_exc() # Log detalhado para erros inesperados
                        await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            logger.info("Tarefa send_realtime cancelada.")
        except Exception as e_send_realtime_outer:
            logger.error(f"Erro fatal em send_realtime: {e_send_realtime_outer}")
            traceback.print_exc()
            self.stop_event.set()
        finally:
            logger.info("Tarefa send_realtime finalizada.")


    async def listen_audio(self):
        # (Função listen_audio inalterada - omitida para brevidade, mas logs DEBUG podem ser adicionados se necessário)
        if not pya:
            logger.error("PyAudio não inicializado. Tarefa listen_audio não pode iniciar.")
            return

        audio_stream = None
        try:
            logger.info("Configurando stream de áudio de entrada (microfone)...")
            mic_info = pya.get_default_input_device_info()
            logger.info(f"Usando microfone: {mic_info['name']} (Taxa: {SEND_SAMPLE_RATE} Hz)")
            
            audio_stream = await asyncio.to_thread(
                pya.open,
                format=FORMAT, channels=CHANNELS, rate=SEND_SAMPLE_RATE,
                input=True, input_device_index=mic_info["index"],
                frames_per_buffer=CHUNK_SIZE
            )
            logger.info("Escutando áudio do microfone...")

            while not self.stop_event.is_set():
                if self.thinking_event.is_set(): # Pausa se Gemini estiver processando
                    await asyncio.sleep(0.05)
                    continue

                if not audio_stream or not audio_stream.is_active():
                     logger.warning("Stream de áudio de entrada não está ativo. Encerrando listen_audio.")
                     self.stop_event.set() # Sinaliza para parar se o stream morrer
                     break

                try:
                    # Leitura bloqueante do stream, executada em thread separada
                    audio_data = await asyncio.to_thread(
                        audio_stream.read, CHUNK_SIZE, exception_on_overflow=False
                    )
                    if self.out_queue:
                         try:
                             if self.out_queue.full():
                                 # Se a fila estiver cheia, áudio pode ser descartado ou atrasado.
                                 # Para áudio, é melhor descartar o mais antigo se for o caso,
                                 # mas aqui estamos adicionando novo, então o descarte seria de imagem/tela.
                                 # Se a fila está cheia de áudio, algo está lento no envio.
                                 logger.debug("Fila de saída (out_queue) cheia ao tentar enfileirar áudio.")
                                 # Poderia descartar um item da fila aqui se necessário:
                                 # await self.out_queue.get(); self.out_queue.task_done()
                             self.out_queue.put_nowait({"data": audio_data, "mime_type": "audio/pcm"})
                         except asyncio.QueueFull:
                             logger.debug("Fila de saída (out_queue) cheia (put_nowait áudio).")
                         except Exception as q_e_put_audio:
                              logger.error(f"Erro inesperado ao enfileirar áudio na out_queue: {q_e_put_audio}")

                except OSError as e_os_audio_read:
                    if e_os_audio_read.errno == -9988 or "Stream closed" in str(e_os_audio_read) or "Input overflowed" in str(e_os_audio_read):
                        logger.warning(f"Stream de áudio fechado ou com overflow (OSError: {e_os_audio_read}). Encerrando listen_audio.")
                        self.stop_event.set()
                        break
                    else:
                        logger.error(f"Erro de OS ao ler do stream de áudio: {e_os_audio_read}")
                        traceback.print_exc()
                        self.stop_event.set()
                        break
                except Exception as e_audio_read:
                    logger.error(f"Erro durante a leitura do áudio em listen_audio: {e_audio_read}")
                    traceback.print_exc()
                    self.stop_event.set()
                    break
        except asyncio.CancelledError:
            logger.info("Tarefa listen_audio cancelada.")
        except Exception as e_listen_audio_outer:
            logger.error(f"Erro crítico em listen_audio: {e_listen_audio_outer}")
            traceback.print_exc()
            self.stop_event.set()
        finally:
            logger.info("Finalizando listen_audio...")
            if audio_stream:
                try:
                    if audio_stream.is_active():
                        audio_stream.stop_stream()
                    audio_stream.close()
                    logger.info("Stream de áudio de entrada (microfone) fechado.")
                except Exception as e_close_mic_stream:
                    logger.error(f"Erro ao fechar stream de áudio de entrada: {e_close_mic_stream}")
            logger.info("Tarefa listen_audio concluída.")


    def _handle_save_known_face(self, person_name: str) -> str:
        """Handler para a função 'save_known_face'."""
        logger.debug(f"[HANDLER] Entrou em _handle_save_known_face com person_name={person_name!r}")
        start_time = time.time()

        if not person_name or not person_name.strip():
            logger.warning("[HANDLER][save_known_face] Nome da pessoa está vazio ou ausente.")
            return "Nome da pessoa não fornecido. Não posso salvar o rosto sem um nome."

        frame_to_process = None
        with self.frame_lock:
            if self.latest_bgr_frame is not None:
                frame_to_process = self.latest_bgr_frame.copy() # Copia para evitar problemas de concorrência

        if frame_to_process is None:
            logger.warning("[HANDLER][save_known_face] Nenhum frame de câmera disponível para processar.")
            return "Não foi possível capturar a imagem da câmera para salvar o rosto."

        # Sanitiza nome para diretório e arquivo
        safe_person_name_dir = "".join(c if c.isalnum() or c in [' '] else '_' for c in person_name).strip().replace(" ", "_")
        if not safe_person_name_dir: safe_person_name_dir = "rosto_desconhecido" # Fallback
        person_dir_path = os.path.join(DB_PATH, safe_person_name_dir)

        try:
            if not os.path.exists(person_dir_path):
                os.makedirs(person_dir_path)
                logger.info(f"[HANDLER][save_known_face] Diretório criado: {person_dir_path}")

            logger.debug(f"[HANDLER][save_known_face] Tentando extrair rostos do frame para '{person_name}'.")
            # DeepFace.extract_faces pode lançar ValueError se nenhum rosto for encontrado e enforce_detection=True
            detected_faces_data = DeepFace.extract_faces(
                img_path=frame_to_process,
                detector_backend=DEEPFACE_DETECTOR_BACKEND,
                enforce_detection=True, # Garante que um rosto seja detectado
                align=True,
                # silent=True # Reduz verbosidade do DeepFace
            )

            if not detected_faces_data or not isinstance(detected_faces_data, list) or 'facial_area' not in detected_faces_data[0]:
                logger.info(f"[HANDLER][save_known_face] Nenhum rosto detectado claramente para '{person_name}'.")
                return f"Não consegui detectar um rosto claro na imagem para salvar como {person_name}."

            # Pega o primeiro rosto detectado (geralmente o maior/mais central)
            face_meta = detected_faces_data[0]
            facial_area = face_meta['facial_area'] # x, y, w, h
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

            # Recorta a imagem do rosto com uma pequena margem para garantir que todo o rosto seja incluído
            margin = 20 # pixels de margem
            y1, y2 = max(0, y - margin), min(frame_to_process.shape[0], y + h + margin)
            x1, x2 = max(0, x - margin), min(frame_to_process.shape[1], x + w + margin)
            face_image_cropped = frame_to_process[y1:y2, x1:x2]

            if face_image_cropped.size == 0:
                 logger.warning(f"[HANDLER][save_known_face] Erro ao recortar rosto para '{person_name}' (imagem resultante vazia).")
                 return f"Erro ao processar a imagem do rosto de {person_name}."

            timestamp = int(time.time())
            safe_file_name_base = "".join(c if c.isalnum() else '_' for c in person_name).strip().lower()
            if not safe_file_name_base: safe_file_name_base = "rosto"
            file_name = f"{safe_file_name_base}_{timestamp}.jpg"
            full_file_path = os.path.join(person_dir_path, file_name)

            save_success = cv2.imwrite(full_file_path, face_image_cropped)
            if not save_success:
                logger.error(f"[HANDLER][save_known_face] Falha ao salvar imagem em '{full_file_path}' usando cv2.imwrite.")
                return f"Ocorreu um erro técnico ao tentar salvar a imagem do rosto de {person_name}."

            # Remove o arquivo .pkl de representações para forçar o DeepFace a reconstruí-lo na próxima chamada de `find`
            # Isso garante que o novo rosto seja incluído no reconhecimento.
            model_name_safe_for_pkl = DEEPFACE_MODEL_NAME.lower().replace('-', '_') # Ex: vgg_face
            representations_pkl_path = os.path.join(DB_PATH, f"representations_{model_name_safe_for_pkl}.pkl")
            if os.path.exists(representations_pkl_path):
                try:
                    os.remove(representations_pkl_path)
                    logger.info(f"[HANDLER][save_known_face] Cache de representações '{representations_pkl_path}' removido para atualização.")
                except Exception as e_pkl_remove:
                    logger.warning(f"[HANDLER][save_known_face] Aviso: Falha ao remover cache de representações '{representations_pkl_path}': {e_pkl_remove}")

            duration = time.time() - start_time
            logger.info(f"[HANDLER][save_known_face] Rosto de '{person_name}' salvo com sucesso em '{full_file_path}'. Duração: {duration:.2f}s")
            return f"Rosto de {person_name} salvo com sucesso!"

        except ValueError as ve: # Erro comum do DeepFace se enforce_detection=True e nenhum rosto for encontrado
             logger.warning(f"[HANDLER][save_known_face] Nenhum rosto detectado (ValueError) para '{person_name}': {ve}")
             return f"Não consegui detectar um rosto claro para salvar para {person_name}. Por favor, certifique-se de que o rosto está bem visível."
        except Exception as e_save:
            logger.error(f"[HANDLER][save_known_face] Erro inesperado ao salvar rosto para '{person_name}': {e_save}")
            traceback.print_exc()
            return f"Ocorreu um erro inesperado ao tentar salvar o rosto de {person_name}."


    def _handle_identify_person_in_front(self) -> str:
        """Handler para a função 'identify_person_in_front'."""
        logger.debug("[HANDLER] Entrou em _handle_identify_person_in_front")
        start_time = time.time()

        if pd is None: # Verifica se pandas está disponível
            logger.error("[HANDLER][identify_person] Biblioteca 'pandas' não disponível. Identificação desabilitada.")
            return "Desculpe, estou com um problema técnico (dependência 'pandas' ausente) e não posso realizar a identificação agora."

        frame_to_process = None
        with self.frame_lock:
            if self.latest_bgr_frame is not None:
                frame_to_process = self.latest_bgr_frame.copy()

        if frame_to_process is None:
            logger.warning("[HANDLER][identify_person] Nenhum frame de câmera disponível para identificar.")
            return "Não foi possível capturar a imagem da câmera para realizar a identificação."

        try:
            logger.debug("[HANDLER][identify_person] Tentando identificar pessoa no frame atual...")
            # DeepFace.find retorna uma lista de DataFrames, um para cada rosto detectado na img_path.
            # Se enforce_detection=True, ele primeiro garante que um rosto é detectado na img_path.
            dfs_results = DeepFace.find(
                img_path=frame_to_process,
                db_path=DB_PATH,
                model_name=DEEPFACE_MODEL_NAME,
                detector_backend=DEEPFACE_DETECTOR_BACKEND,
                distance_metric=DEEPFACE_DISTANCE_METRIC,
                enforce_detection=True, # Exige detecção clara de rosto na imagem de entrada
                align=True,
                # silent=True
            )

            if not dfs_results or not isinstance(dfs_results, list) or not isinstance(dfs_results[0], pd.DataFrame) or dfs_results[0].empty:
                logger.info("[HANDLER][identify_person] Nenhuma correspondência encontrada ou rosto não detectado claramente na imagem de entrada.")
                return "Não consegui reconhecer ninguém conhecido no banco de dados ou não detectei um rosto claro na imagem atual."

            # Processa o primeiro DataFrame de resultados (correspondente ao rosto mais proeminente na img_path)
            df_best_face = dfs_results[0]

            # A coluna de distância pode ter nomes como 'VGG-Face_cosine' ou apenas 'distance'
            # Tenta encontrar a coluna de distância correta
            distance_col_candidate = f"{DEEPFACE_MODEL_NAME}_{DEEPFACE_DISTANCE_METRIC}"
            if distance_col_candidate not in df_best_face.columns:
                if 'distance' in df_best_face.columns: # Fallback comum
                    distance_col_candidate = 'distance'
                else: # Tenta encontrar qualquer coluna que contenha a métrica
                    found_dist_col = None
                    for col_name in df_best_face.columns:
                        if DEEPFACE_DISTANCE_METRIC in col_name.lower():
                            found_dist_col = col_name
                            break
                    if not found_dist_col:
                        logger.error(f"[HANDLER][identify_person] Coluna de distância não encontrada no DataFrame. Colunas: {df_best_face.columns.tolist()}")
                        return "Erro ao processar os resultados da identificação (coluna de distância ausente)."
                    distance_col_candidate = found_dist_col
            
            logger.debug(f"[HANDLER][identify_person] Usando coluna de distância: {distance_col_candidate}")


            # Ordena por distância (menor é melhor) e pega o melhor match
            df_best_face = df_best_face.sort_values(by=distance_col_candidate, ascending=True)
            best_match_info = df_best_face.iloc[0]

            identity_path = best_match_info['identity']
            # O nome da pessoa é o nome do diretório pai do arquivo de imagem no DB_PATH
            person_name_identified = os.path.basename(os.path.dirname(identity_path))
            distance_value = best_match_info[distance_col_candidate]

            logger.info(f"[HANDLER][identify_person] Melhor correspondência: '{person_name_identified}' com distância: {distance_value:.4f}")

            # Limiares de distância (ajustar experimentalmente!)
            # Estes são exemplos e podem variar muito com o modelo e a qualidade das imagens.
            thresholds = {
                'VGG-Face': {'cosine': 0.40, 'euclidean': 0.60, 'euclidean_l2': 0.86}, # Cosine: <0.4 é bom match
                'Facenet': {'cosine': 0.40, 'euclidean': 0.90, 'euclidean_l2': 1.10},
                'Facenet512': {'cosine': 0.30, 'euclidean': 0.70, 'euclidean_l2': 0.95},
                'ArcFace': {'cosine': 0.68, 'euclidean': 1.13, 'euclidean_l2': 1.13}, # ArcFace cosine: <0.68
                'Dlib': {'cosine': 0.07, 'euclidean': 0.6, 'euclidean_l2': 0.6}, # Dlib cosine: <0.07
            }
            recognition_threshold = thresholds.get(DEEPFACE_MODEL_NAME, {}).get(DEEPFACE_DISTANCE_METRIC, 0.5) # Padrão 0.5 se não mapeado

            duration = time.time() - start_time
            if distance_value <= recognition_threshold:
                logger.info(f"[HANDLER][identify_person] Pessoa identificada como '{person_name_identified}'. Duração: {duration:.2f}s")
                return f"A pessoa na sua frente parece ser {person_name_identified}."
            else:
                logger.info(f"[HANDLER][identify_person] Distância {distance_value:.4f} > limiar ({recognition_threshold}). Não reconhecido com confiança. Duração: {duration:.2f}s")
                return "Detectei um rosto, mas não tenho certeza de quem é ou não corresponde a ninguém no banco de dados."

        except ValueError as ve: # Se enforce_detection=True e nenhum rosto for encontrado na imagem de entrada
            logger.warning(f"[HANDLER][identify_person] Nenhum rosto detectado na imagem de entrada (ValueError): {ve}")
            return "Não detectei um rosto claro na imagem atual para tentar a identificação."
        except Exception as e_identify:
            logger.error(f"[HANDLER][identify_person] Erro inesperado ao identificar pessoa: {e_identify}")
            traceback.print_exc()
            return "Ocorreu um erro inesperado ao tentar identificar a pessoa."


    def _run_midas_inference(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        # (Função _run_midas_inference inalterada - omitida para brevidade, mas logs DEBUG podem ser adicionados se necessário)
        if not self.midas_model or not self.midas_transform:
            logger.debug("[MiDaS Handler] Modelo MiDaS ou transformador não carregado. Não é possível inferir profundidade.")
            return None
        try:
            logger.debug(f"[MiDaS Handler] Iniciando inferência MiDaS no dispositivo {self.midas_device}.")
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            input_batch = self.midas_transform(img_rgb).to(self.midas_device)

            with torch.no_grad():
                prediction = self.midas_model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic", # "bicubic" geralmente dá resultados mais suaves que "bilinear"
                    align_corners=False,
                ).squeeze()
            depth_map_output = prediction.cpu().numpy()
            logger.debug("[MiDaS Handler] Inferência MiDaS concluída.")
            return depth_map_output
        except Exception as e_midas_infer:
            logger.error(f"[MiDaS Handler] Erro durante a inferência MiDaS: {e_midas_infer}")
            # traceback.print_exc()
            return None


    def _find_best_yolo_match(self, object_type_query: str, yolo_results_list: List[Any]) -> Optional[Tuple[Dict[str, int], float, str]]:
        # (Função _find_best_yolo_match inalterada - omitida para brevidade, mas logs DEBUG podem ser adicionados se necessário)
        best_match_found = None
        highest_confidence = -1.0
        
        # Mapeia o tipo de objeto consultado para nomes de classe YOLO
        # object_type_query é o que a IA passou (ex: "garrafa", "computador")
        target_yolo_class_names = YOLO_CLASS_MAP.get(object_type_query.lower(), [object_type_query.lower()])
        logger.debug(f"[YOLO Matcher] Procurando por classes YOLO: {target_yolo_class_names} (query: '{object_type_query}')")

        if not yolo_results_list or not self.yolo_model:
             logger.debug("[YOLO Matcher] Sem resultados YOLO ou modelo YOLO não carregado.")
             return None

        for yolo_result_item in yolo_results_list: # Geralmente uma lista com um resultado
            if hasattr(yolo_result_item, 'boxes') and yolo_result_item.boxes:
                for yolo_box in yolo_result_item.boxes:
                    if not (hasattr(yolo_box, 'cls') and hasattr(yolo_box, 'conf') and hasattr(yolo_box, 'xyxy')):
                        logger.debug("[YOLO Matcher] Caixa YOLO malformada encontrada, pulando.")
                        continue

                    class_id_tensor = yolo_box.cls
                    if class_id_tensor.nelement() == 0: continue
                    class_id = int(class_id_tensor[0])

                    confidence_tensor = yolo_box.conf
                    if confidence_tensor.nelement() == 0: continue
                    confidence = float(confidence_tensor[0])

                    if class_id < len(self.yolo_model.names):
                        detected_class_name = self.yolo_model.names[class_id]
                    else:
                        logger.debug(f"[YOLO Matcher] ID de classe YOLO inválido: {class_id}")
                        continue

                    if detected_class_name in target_yolo_class_names:
                        if confidence > highest_confidence:
                            highest_confidence = confidence
                            coords_tensor = yolo_box.xyxy[0]
                            if coords_tensor.nelement() < 4: continue
                            coords_list = list(map(int, coords_tensor))
                            bbox = {'x1': coords_list[0], 'y1': coords_list[1], 'x2': coords_list[2], 'y2': coords_list[3]}
                            best_match_found = (bbox, confidence, detected_class_name)
                            logger.debug(f"[YOLO Matcher] Novo melhor match: {detected_class_name} (Conf: {confidence:.2f})")
        
        if best_match_found:
            logger.debug(f"[YOLO Matcher] Melhor correspondência final para '{object_type_query}': {best_match_found[2]} com confiança {best_match_found[1]:.2f}")
        else:
            logger.debug(f"[YOLO Matcher] Nenhuma correspondência encontrada para '{object_type_query}' nas classes {target_yolo_class_names}")
        return best_match_found


    def _estimate_direction(self, bbox: Dict[str, int], frame_width: int) -> str:
        # (Função _estimate_direction inalterada - omitida para brevidade)
        box_center_x = (bbox['x1'] + bbox['x2']) / 2
        segment_width = frame_width / 3 # Divide o frame em 3 segmentos verticais

        if box_center_x < segment_width:
            return "à sua esquerda"
        elif box_center_x > (frame_width - segment_width): # ou box_center_x > 2 * segment_width
            return "à sua direita"
        else:
            return "à sua frente"


    def _check_if_on_surface(self, target_bbox: Dict[str, int], yolo_results_list: List[Any]) -> bool:
        # (Função _check_if_on_surface inalterada - omitida para brevidade, mas logs DEBUG podem ser adicionados se necessário)
        surface_class_keys_in_map = ["mesa", "mesa de jantar", "bancada", "prateleira", "sofá", "cama"] # Chaves do YOLO_CLASS_MAP
        surface_yolo_target_names = []
        for key in surface_class_keys_in_map:
            surface_yolo_target_names.extend(YOLO_CLASS_MAP.get(key, []))
        surface_yolo_target_names = list(set(surface_yolo_target_names)) # Nomes de classe YOLO reais

        if not surface_yolo_target_names:
            logger.debug("[Surface Check] Nenhuma classe de superfície definida no YOLO_CLASS_MAP.")
            return False
        logger.debug(f"[Surface Check] Procurando por superfícies: {surface_yolo_target_names}")

        target_bottom_y = target_bbox['y2']
        target_center_x = (target_bbox['x1'] + target_bbox['x2']) / 2
        target_height = target_bbox['y2'] - target_bbox['y1']


        if not yolo_results_list or not self.yolo_model:
            logger.debug("[Surface Check] Sem resultados YOLO ou modelo não carregado.")
            return False

        for yolo_result_item in yolo_results_list:
             if hasattr(yolo_result_item, 'boxes') and yolo_result_item.boxes:
                for surface_candidate_box in yolo_result_item.boxes:
                    if not (hasattr(surface_candidate_box, 'cls') and hasattr(surface_candidate_box, 'xyxy')):
                        continue

                    class_id_tensor = surface_candidate_box.cls
                    if class_id_tensor.nelement() == 0: continue
                    class_id = int(class_id_tensor[0])

                    if class_id < len(self.yolo_model.names):
                        detected_class_name = self.yolo_model.names[class_id]
                    else:
                        continue

                    if detected_class_name in surface_yolo_target_names:
                        coords_tensor = surface_candidate_box.xyxy[0]
                        if coords_tensor.nelement() < 4: continue
                        s_x1, s_y1, s_x2, s_y2 = map(int, coords_tensor) # Coordenadas da superfície

                        # Heurísticas para verificar se o objeto está SOBRE a superfície:
                        # 1. Alinhamento Horizontal: Centro X do objeto está dentro da largura da superfície.
                        is_horizontally_aligned = (s_x1 < target_center_x < s_x2)

                        # 2. Alinhamento Vertical: Base do objeto (target_bottom_y) está próxima ou
                        #    ligeiramente acima/abaixo do topo da superfície (s_y1).
                        #    Permite uma pequena sobreposição ou espaço.
                        vertical_tolerance_pixels = target_height * 0.3 # Tolera até 30% da altura do objeto
                        is_vertically_close_to_top = (s_y1 - vertical_tolerance_pixels) < target_bottom_y < (s_y1 + vertical_tolerance_pixels * 1.5)
                        
                        # 3. Objeto está acima da base da superfície (para evitar objetos "atrás" de superfícies altas)
                        is_above_surface_base = target_bottom_y < s_y2 + vertical_tolerance_pixels

                        # 4. (Opcional) Tamanho Relativo: Objeto não é significativamente maior que a superfície.
                        # surface_height = s_y2 - s_y1
                        # is_size_compatible = target_height < (surface_height * 1.5) if surface_height > 0 else True


                        if is_horizontally_aligned and is_vertically_close_to_top and is_above_surface_base: # and is_size_compatible:
                            logger.debug(f"[Surface Check] Objeto em ({target_center_x},{target_bottom_y}) considerado SOBRE '{detected_class_name}' em ({s_x1}-{s_x2}, {s_y1}-{s_y2})")
                            return True
        
        logger.debug("[Surface Check] Nenhuma superfície de apoio encontrada sob o objeto.")
        return False


    def _handle_find_object_and_estimate_distance(self, object_description: str, object_type: str) -> str:
        """Handler para a função 'find_object_and_estimate_distance'."""
        logger.debug(f"[HANDLER] Entrou em _handle_find_object_and_estimate_distance com object_description='{object_description}', object_type='{object_type}'")
        start_time = time.time()

        if not object_description or not object_type:
            logger.warning("[HANDLER][find_object] Descrição ou tipo do objeto ausente.")
            return "Por favor, forneça uma descrição e o tipo do objeto que você quer encontrar."

        frame_to_process_bgr = None
        yolo_results_for_this_frame = None
        frame_h, frame_w = 0, 0

        with self.frame_lock:
            if self.latest_bgr_frame is not None:
                frame_to_process_bgr = self.latest_bgr_frame.copy()
                yolo_results_for_this_frame = self.latest_yolo_results # Pega os resultados YOLO correspondentes
                if frame_to_process_bgr is not None: # Garante que a cópia foi bem sucedida
                    frame_h, frame_w, _ = frame_to_process_bgr.shape

        if frame_to_process_bgr is None or frame_w == 0 or frame_h == 0:
             logger.warning("[HANDLER][find_object] Nenhum frame de câmera válido disponível.")
             return f"Desculpe, não estou conseguindo ver nada no momento para localizar o {object_type}."

        if not yolo_results_for_this_frame:
            logger.warning("[HANDLER][find_object] Nenhum resultado YOLO disponível para o frame atual. Tente novamente em instantes.")
            return f"Não consegui processar a imagem a tempo para encontrar o {object_type}. Por favor, tente novamente."

        # Encontra a melhor correspondência YOLO para o object_type fornecido pela IA
        best_yolo_match_tuple = self._find_best_yolo_match(object_type, yolo_results_for_this_frame)

        # Fallback: Se não encontrou pelo object_type, tenta com a última palavra da descrição original
        if not best_yolo_match_tuple:
            logger.info(f"[HANDLER][find_object] Objeto do tipo '{object_type}' não encontrado. Tentando fallback com a descrição completa '{object_description}'...")
            # Tenta usar a descrição inteira ou partes dela se o object_type falhou.
            # Aqui, vamos tentar a última palavra como um tipo alternativo.
            last_word_in_desc = object_description.split(" ")[-1].lower()
            if last_word_in_desc != object_type.lower(): # Evita repetir a busca
                logger.info(f"[HANDLER][find_object] Fallback: Buscando por tipo derivado da descrição: '{last_word_in_desc}'")
                best_yolo_match_tuple = self._find_best_yolo_match(last_word_in_desc, yolo_results_for_this_frame)

            if not best_yolo_match_tuple:
                 logger.info(f"[HANDLER][find_object] Objeto '{object_description}' (tipo '{object_type}') não encontrado mesmo com fallback.")
                 return f"Desculpe, não consegui encontrar um(a) {object_description} na imagem."

        target_bbox_coords, confidence_score, detected_yolo_class = best_yolo_match_tuple
        logger.info(f"[HANDLER][find_object] Melhor correspondência YOLO: Classe '{detected_yolo_class}', Conf: {confidence_score:.2f}, BBox: {target_bbox_coords}")

        is_on_a_surface = self._check_if_on_surface(target_bbox_coords, yolo_results_for_this_frame)
        surface_info_msg = "sobre uma superfície (como uma mesa ou prateleira)" if is_on_a_surface else ""

        object_direction = self._estimate_direction(target_bbox_coords, frame_w)

        distance_in_steps = -1
        depth_map_from_midas = None
        if self.midas_model and self.midas_transform: # Verifica se MiDaS está funcional
            logger.debug("[HANDLER][find_object] Executando inferência MiDaS para estimativa de distância...")
            depth_map_from_midas = self._run_midas_inference(frame_to_process_bgr)
        else:
            logger.info("[HANDLER][find_object] Modelo MiDaS não disponível. Não é possível estimar distância.")

        if depth_map_from_midas is not None:
            try:
                # Pega o valor de profundidade no centro da caixa delimitadora do objeto
                obj_center_x = int((target_bbox_coords['x1'] + target_bbox_coords['x2']) / 2)
                obj_center_y = int((target_bbox_coords['y1'] + target_bbox_coords['y2']) / 2)
                
                # Garante que as coordenadas estão dentro dos limites do mapa de profundidade
                obj_center_y = max(0, min(obj_center_y, depth_map_from_midas.shape[0] - 1))
                obj_center_x = max(0, min(obj_center_x, depth_map_from_midas.shape[1] - 1))
                
                depth_value_at_center = depth_map_from_midas[obj_center_y, obj_center_x]

                # --- Conversão MiDaS para Metros (MUITO APROXIMADA - REQUER CALIBRAÇÃO) ---
                # MiDaS (especialmente _small) retorna profundidade inversa normalizada. Maior valor = mais perto.
                # A escala exata depende do modelo e da cena. Esta é uma heurística.
                if depth_value_at_center > 1e-5: # Evita divisão por zero ou valores muito pequenos
                    # Heurística de mapeamento (AJUSTAR COM BASE EM TESTES REAIS PARA SEU CENÁRIO)
                    # Exemplo: Se depth_value_at_center for alto (ex: > 250-300 para MiDaS_small), está perto.
                    # Se for baixo (ex: < 30-50), está longe.
                    # Esta é uma tentativa de conversão para metros, pode não ser linear.
                    if MIDAS_MODEL_TYPE == "MiDaS_small": # Valores típicos para MiDaS_small
                        if depth_value_at_center > 300:  estimated_dist_meters = np.random.uniform(0.3, 1.0)
                        elif depth_value_at_center > 200: estimated_dist_meters = np.random.uniform(1.0, 2.5)
                        elif depth_value_at_center > 100: estimated_dist_meters = np.random.uniform(2.5, 5.0)
                        elif depth_value_at_center > 50:  estimated_dist_meters = np.random.uniform(5.0, 10.0)
                        else: estimated_dist_meters = np.random.uniform(10.0, 20.0) # Longe
                    else: # Outros modelos MiDaS podem ter escalas diferentes
                        # Para modelos que dão profundidade métrica (ex: DPT_Large), a conversão é mais direta.
                        # Se for profundidade inversa, a lógica é similar a MiDaS_small mas com outros ranges.
                        # Assumindo uma escala similar a MiDaS_small por enquanto se não for ele.
                        if depth_value_at_center > 15:  estimated_dist_meters = np.random.uniform(0.3, 1.0) # Exemplo para DPT (valores menores são mais longe)
                        elif depth_value_at_center > 10: estimated_dist_meters = np.random.uniform(1.0, 2.5)
                        elif depth_value_at_center > 5: estimated_dist_meters = np.random.uniform(2.5, 5.0)
                        else: estimated_dist_meters = np.random.uniform(5.0, 15.0)


                    estimated_dist_meters = max(0.3, min(estimated_dist_meters, 25.0)) # Limita a um alcance razoável
                    distance_in_steps = max(1, round(estimated_dist_meters / METERS_PER_STEP)) # Pelo menos 1 passo
                    logger.info(f"[HANDLER][find_object] Profundidade MiDaS no centro ({obj_center_y},{obj_center_x}): {depth_value_at_center:.4f}. Dist. Estimada: {estimated_dist_meters:.2f}m. Passos: {distance_in_steps}")
                else:
                     logger.info("[HANDLER][find_object] Valor de profundidade MiDaS inválido ou muito baixo no centro do objeto.")
            except IndexError: # Coordenadas fora dos limites do mapa
                logger.warning(f"[HANDLER][find_object] Erro de índice ao acessar mapa de profundidade MiDaS. Coords: ({obj_center_y},{obj_center_x}), Mapa: {depth_map_from_midas.shape}.")
            except Exception as e_midas_process:
                logger.error(f"[HANDLER][find_object] Erro ao processar profundidade MiDaS: {e_midas_process}")
                distance_in_steps = -1 # Reseta se houve erro

        # --- Constrói a Mensagem de Resposta para o Usuário ---
        # Usa a descrição original do usuário para a resposta, pois é mais natural.
        object_name_for_user_response = object_description

        response_parts_list = [f"Encontrei o {object_name_for_user_response}"]
        if surface_info_msg:
            response_parts_list.append(surface_info_msg)

        if distance_in_steps > 0:
            response_parts_list.append(f"a aproximadamente {distance_in_steps} passo{'s' if distance_in_steps > 1 else ''}")

        response_parts_list.append(object_direction) # Adiciona a direção (sempre)

        # Junta as partes da resposta de forma gramaticalmente correta
        if len(response_parts_list) == 2: # Apenas "Encontrei o OBJ" e "DIREÇÃO"
            final_response_message = f"{response_parts_list[0]} {response_parts_list[1]}."
        elif len(response_parts_list) > 2:
            # Ex: "Encontrei o OBJ, sobre uma superfície, a X passos, à sua frente."
            # Junta os intermediários com vírgula, e o último com "e" ou diretamente.
            # "Encontrei o OBJ, [parte 2], [parte 3] ... e [última parte]."
            # Se a última parte é a direção, não precisa de "e".
            first_part = response_parts_list[0]
            last_part_direction = response_parts_list[-1]
            middle_parts = response_parts_list[1:-1]
            if middle_parts:
                final_response_message = f"{first_part}, {', '.join(middle_parts)}, {last_part_direction}."
            else: # Só tem o primeiro e a direção
                final_response_message = f"{first_part} {last_part_direction}."
        else: # Só o primeiro (não deveria acontecer se sempre adiciona direção)
            final_response_message = f"{response_parts_list[0]}."


        duration = time.time() - start_time
        logger.info(f"[HANDLER][find_object] Finalizado. Duração: {duration:.2f}s. Resposta: '{final_response_message}'")
        return final_response_message


    async def receive_audio(self):
        # (Função receive_audio inalterada - omitida para brevidade, mas logs DEBUG podem ser adicionados se necessário)
        logger.info("Tarefa receive_audio iniciada, aguardando respostas do Gemini...")
        try:
            if not self.session:
                logger.error("Sessão Gemini não estabelecida em receive_audio. Encerrando tarefa.")
                self.stop_event.set()
                return

            while not self.stop_event.is_set():
                if not self.session: # Verifica se a sessão foi perdida
                    logger.warning("Sessão Gemini desconectada em receive_audio. Aguardando reconexão ou parada.")
                    await asyncio.sleep(1)
                    if not self.session and not self.stop_event.is_set():
                        logger.warning("Sessão ainda não disponível. Sinalizando parada para o loop run tentar reconectar.")
                        self.stop_event.set()
                    elif self.session:
                        logger.info("Sessão Gemini reconectada (detectado em receive_audio).")
                    break # Sai do loop interno para o 'run' tentar reconectar ou parar

                try:
                    data_received_in_this_turn = False
                    # logger.debug("Aguardando próximo turno de resposta do Gemini...")

                    # Este loop itera sobre as partes da resposta do Gemini para um turno.
                    # Pode conter áudio, texto, ou chamadas de função.
                    async for response_chunk in self.session.receive():
                        data_received_in_this_turn = True
                        # logger.debug(f"Recebido chunk da resposta: {response_chunk}")


                        if self.stop_event.is_set():
                            logger.info("Sinal de parada recebido durante processamento de resposta em receive_audio.")
                            break # Sai do loop `async for`

                        # --- 1. Processa Áudio da Resposta ---
                        if response_chunk.data: # Contém bytes de áudio
                            if self.audio_in_queue:
                                try:
                                    self.audio_in_queue.put_nowait(response_chunk.data)
                                except asyncio.QueueFull:
                                    logger.warning("Fila de áudio de entrada (audio_in_queue) cheia. Áudio do Gemini descartado.")
                            # logger.debug("Chunk de áudio do Gemini enfileirado para playback.")
                            continue # Processou áudio, vai para o próximo chunk

                        # --- 2. Processa Nome Pendente (Fluxo save_known_face) ---
                        # Se a IA pediu um nome e o usuário respondeu (texto ou voz transcrita).
                        if self.awaiting_name_for_save_face:
                            user_provided_name_for_face = None
                            if response_chunk.text: # Gemini transcreveu fala do usuário ou usuário digitou
                                user_provided_name_for_face = response_chunk.text.strip()
                                logger.info(f"[Trackie][SaveFaceFlow] Recebido texto do usuário enquanto aguardava nome: '{user_provided_name_for_face}'")

                            if user_provided_name_for_face:
                                logger.info(f"[Trackie][SaveFaceFlow] Processando nome '{user_provided_name_for_face}' para salvar rosto...")
                                self.awaiting_name_for_save_face = False # Reseta a flag

                                original_function_to_call_after_name = "save_known_face"

                                logger.info("Pensando (após receber nome para salvar)...")
                                self.thinking_event.set() # Pausa envio de novos dados (áudio/vídeo)

                                # Feedback de voz ANTES de executar a função demorada
                                feedback_msg_before_save = f"Entendido, salvando o rosto de {user_provided_name_for_face} agora. Um momento..."
                                if self.session:
                                    try:
                                        await self.session.send(input=feedback_msg_before_save, end_of_turn=True)
                                        logger.debug(f"  [Feedback Enviado]: {feedback_msg_before_save}")
                                    except Exception as e_feedback_voice:
                                        logger.error(f"Erro ao enviar feedback de voz (save_face flow): {e_feedback_voice}")

                                # Executa a função síncrona em outra thread
                                function_execution_result = await asyncio.to_thread(self._handle_save_known_face, user_provided_name_for_face)

                                # Envia o resultado da função de volta para o Gemini
                                logger.info(f"  [Trackie][SaveFaceFlow] Resultado da Função '{original_function_to_call_after_name}': {function_execution_result}")
                                if self.session:
                                    try:
                                        function_response_content = types.Content(
                                            role="tool", # Importante: role="tool" para respostas de função
                                            parts=[types.Part.from_function_response(
                                                name=original_function_to_call_after_name,
                                                response={"result": Value(string_value=function_execution_result)} # Usa Value para struct_pb2
                                            )]
                                        )
                                        await self.session.send(input=function_response_content) # Não usa end_of_turn=True aqui
                                        logger.debug("  [Trackie][SaveFaceFlow] Resultado da função (após nome) enviado para Gemini.")
                                    except Exception as e_send_fc_resp:
                                        logger.error(f"Erro ao enviar FunctionResponse (save_face flow): {e_send_fc_resp}")
                                else:
                                    logger.warning("  [Trackie][SaveFaceFlow] Sessão inativa. Não foi possível enviar resultado da função.")

                                if self.thinking_event.is_set(): self.thinking_event.clear()
                                logger.info("Pensamento concluído (após receber nome para salvar).")
                                continue # Processamos este input, vamos para o próximo response_chunk

                        # --- 3. Processa Texto da IA (Resposta normal) ---
                        if response_chunk.text:
                            # Imprime texto recebido do Gemini (pode ser parcial se streamming)
                            # O logger.info já adiciona newline, então end="" não é estritamente necessário se cada chunk for logado.
                            logger.info(f"[Gemini Texto]: {response_chunk.text}")


                        # --- 4. Processa Chamada de Função Solicitada pelo Gemini ---
                        # O SDK google-genai usa response_chunk.candidates[0].content.parts[0].function_call
                        # Vamos verificar a estrutura exata do response_chunk para LiveConnect
                        # Geralmente, o response_chunk em si pode ter o atributo function_call
                        # ou ele está aninhado.
                        # No código original: getattr(response_part, "function_call", None)
                        # Vamos assumir que response_chunk pode ter function_call diretamente ou em candidates.
                        
                        # Tentativa 1: Acesso direto (como no código original)
                        fc_object = getattr(response_chunk, "function_call", None)
                        # Tentativa 2: Acesso via candidates (só se existir este atributo)
                        if not fc_object and hasattr(response_chunk, "candidates") and response_chunk.candidates:
                            candidate = response_chunk.candidates[0]
                            if candidate.content and candidate.content.parts:
                                part = candidate.content.parts[0]
                                if hasattr(part, "function_call"):
                                    fc_object = part.function_call
                                    logger.debug(f"Function call encontrada em response_chunk.candidates[0].content.parts[0].function_call")


                        if fc_object:
                            function_name_from_gemini = fc_object.name
                            function_args_from_gemini = {key: val for key, val in fc_object.args.items()} # Converte para dict Python
                            logger.info(f"[Gemini Function Call] Recebido: Nome='{function_name_from_gemini}', Args={function_args_from_gemini}")
                            # Log para verificar se o SDK chega a disparar o handler (Passo 2 do user)
                            logger.debug(f"Payload da Function Call recebida do Gemini: {fc_object}")


                            function_execution_result = None # Inicializa resultado da execução

                            # --- Caso Especial: save_known_face sem nome ---
                            # A IA deve pedir o nome ao usuário.
                            if function_name_from_gemini == "save_known_face" and not function_args_from_gemini.get("person_name"):
                                self.awaiting_name_for_save_face = True # Ativa a flag
                                if self.thinking_event.is_set(): self.thinking_event.clear() # Garante que não está "pensando" enquanto pergunta

                                logger.info("[Trackie] Nome não fornecido para save_known_face. Solicitando ao usuário via Gemini.")
                                # Pede o nome ao usuário via Gemini (voz)
                                if self.session:
                                    try:
                                        # Envia a pergunta e termina o turno da IA para que ela fale e espere resposta
                                        await self.session.send(input="Claro, posso salvar o rosto. Qual é o nome da pessoa?", end_of_turn=True)
                                        logger.debug("Pergunta sobre nome para save_face enviada ao Gemini.")
                                    except Exception as e_ask_name_fc:
                                        logger.error(f"Erro ao pedir nome para save_face via Gemini: {e_ask_name_fc}")
                                # Não executa a função local nem envia FC response agora; espera a resposta do usuário.

                            # --- Caso Normal: Outras funções ou save_known_face COM nome ---
                            else:
                                logger.info(f"Pensando (processando função {function_name_from_gemini})...")
                                self.thinking_event.set() # Pausa envio de novos dados

                                # Monta mensagem de feedback de voz ANTES de executar a função (que pode ser demorada)
                                voice_feedback_msg_before_func = f"Ok, processando seu pedido para {function_name_from_gemini}. Um momento..." # Padrão
                                if function_name_from_gemini == "save_known_face":
                                    person_name_fb = function_args_from_gemini.get('person_name', 'a pessoa')
                                    voice_feedback_msg_before_func = f"Entendido, salvando o rosto de {person_name_fb}. Só um instante..."
                                elif function_name_from_gemini == "identify_person_in_front":
                                    voice_feedback_msg_before_func = "Certo, estou tentando identificar a pessoa na sua frente..."
                                elif function_name_from_gemini == "find_object_and_estimate_distance":
                                    obj_desc_fb = function_args_from_gemini.get('object_description', 'o objeto')
                                    voice_feedback_msg_before_func = f"Ok, procurando por {obj_desc_fb} e estimando a distância..."

                                if self.session:
                                    try:
                                        await self.session.send(input=voice_feedback_msg_before_func, end_of_turn=True)
                                        logger.debug(f"  [Feedback de Voz Enviado]: {voice_feedback_msg_before_func}")
                                    except Exception as e_feedback_fc:
                                        logger.error(f"Erro ao enviar feedback de voz pré-função: {e_feedback_fc}")

                                # --- Executa a Função Local Correspondente ---
                                # Verifica se a função requer modo câmera e se está ativo
                                vision_dependent_functions = ["save_known_face", "identify_person_in_front", "find_object_and_estimate_distance"]
                                if self.video_mode != "camera" and function_name_from_gemini in vision_dependent_functions:
                                    logger.warning(f"[Function Call] Função '{function_name_from_gemini}' requer modo câmera, mas modo atual é '{self.video_mode}'.")
                                    function_execution_result = "Desculpe, esta função só está disponível quando a câmera está ativa e eu posso ver o ambiente."
                                else:
                                    logger.info(f"  [Trackie] Executando handler para '{function_name_from_gemini}' em background...")
                                    try:
                                        if function_name_from_gemini == "save_known_face":
                                            person_name_arg = function_args_from_gemini.get("person_name")
                                            if person_name_arg: # Nome foi fornecido diretamente pela IA
                                                function_execution_result = await asyncio.to_thread(self._handle_save_known_face, person_name_arg)
                                            else: # Deveria ter sido tratado pelo fluxo de pedir nome
                                                function_execution_result = "Erro interno: o nome não foi fornecido para salvar o rosto, e o fluxo de solicitação falhou."
                                                logger.error("ERRO LÓGICO: _handle_save_known_face chamado sem nome, fora do fluxo de solicitação.")
                                        
                                        elif function_name_from_gemini == "identify_person_in_front":
                                            function_execution_result = await asyncio.to_thread(self._handle_identify_person_in_front)
                                        
                                        elif function_name_from_gemini == "find_object_and_estimate_distance":
                                            desc_arg = function_args_from_gemini.get("object_description")
                                            obj_type_arg = function_args_from_gemini.get("object_type")
                                            if desc_arg and obj_type_arg:
                                                if not self.midas_model: # Verifica se MiDaS está funcional
                                                    function_execution_result = "Desculpe, o módulo de estimativa de distância não está funcionando no momento, então não posso dizer a que distância o objeto está, mas posso tentar localizá-lo."
                                                    # Poderia chamar uma versão simplificada que só localiza sem MiDaS
                                                function_execution_result = await asyncio.to_thread(
                                                    self._handle_find_object_and_estimate_distance, desc_arg, obj_type_arg
                                                )
                                            else:
                                                function_execution_result = "Para localizar um objeto, preciso da descrição e do tipo do objeto. Parece que faltou alguma informação."
                                                logger.error(f"Argumentos faltando para find_object_and_estimate_distance: desc='{desc_arg}', type='{obj_type_arg}'")
                                        else:
                                            function_execution_result = f"Função '{function_name_from_gemini}' é desconhecida ou não implementei um handler para ela."
                                            logger.warning(f"Recebida chamada para função não mapeada/desconhecida: {function_name_from_gemini}")
                                    
                                    except Exception as e_execute_handler:
                                         logger.error(f"Erro CRÍTICO ao executar handler para '{function_name_from_gemini}': {e_execute_handler}")
                                         traceback.print_exc()
                                         function_execution_result = f"Ocorreu um erro interno grave ao tentar processar a função {function_name_from_gemini}."

                            # --- Envia Resultado da Função de Volta para Gemini (se houver e não for o fluxo de pedir nome) ---
                            if function_execution_result is not None: # Só envia se um resultado foi gerado (não no caso de pedir nome)
                                logger.info(f"  [Trackie] Resultado da Função '{function_name_from_gemini}': {function_execution_result}")
                                if self.session:
                                    try:
                                        fc_response_content = types.Content(
                                            role="tool",
                                            parts=[types.Part.from_function_response(
                                                name=function_name_from_gemini,
                                                response={"result": Value(string_value=str(function_execution_result))} # Garante que é string
                                            )]
                                        )
                                        await self.session.send(input=fc_response_content)
                                        logger.debug(f"  [Trackie] Resultado da função '{function_name_from_gemini}' enviado para Gemini.")
                                    except Exception as e_send_fc_resp_main_flow:
                                        logger.error(f"Erro ao enviar FunctionResponse (fluxo principal) para '{function_name_from_gemini}': {e_send_fc_resp_main_flow}")
                                else:
                                    logger.warning(f"  [Trackie] Sessão inativa. Não foi possível enviar resultado da função '{function_name_from_gemini}'.")

                                # Libera o envio de dados após processar a função e enviar resposta
                                if self.thinking_event.is_set(): self.thinking_event.clear()
                                logger.info(f"Pensamento concluído (após função {function_name_from_gemini}).")
                            # Se function_execution_result é None (caso de pedir nome), thinking_event já foi/será tratado.

                    # --- Fim do processamento de um turno da IA ---
                    if self.stop_event.is_set(): break # Sai do loop `async for` se stop foi chamado

                    if data_received_in_this_turn:
                        # logger.debug("Fim do processamento do turno atual de resposta do Gemini.")
                        pass # Continua esperando o próximo turno/chunk
                    else:
                        # Se o loop `async for` terminar sem dados, pode indicar fim normal do stream do turno ou problema.
                        # logger.debug("Stream do turno atual do Gemini terminou sem dados (ou não houve dados).")
                        await asyncio.sleep(0.05) # Pequena pausa para não ocupar CPU em loop vazio


                except Exception as e_receive_inner_loop:
                    logger.error(f"Erro durante o recebimento/processamento de resposta do Gemini: {e_receive_inner_loop}")
                    error_str_upper = str(e_receive_inner_loop).upper()
                   # intercepta encerramento de sessão ou desconexão pelo texto da exceção
                    if any(err_indicator in error_str_upper for err_indicator in [
                        "LIVESESSION CLOSED", "LIVESESSION NOT CONNECTED"
                    ]):
                        logger.warning("Erro indica que a sessão Gemini foi fechada/perdida. Sinalizando parada.")
                        self.stop_event.set()
                        break  # sai do loop principal de receive_audio
                    else:
                        traceback.print_exc()
                        await asyncio.sleep(0.5)

            # Se o loop while terminar por causa do stop_event
            if self.stop_event.is_set():
                logger.info("Loop de receive_audio interrompido pelo stop_event.")

        except asyncio.CancelledError:
            logger.info("Tarefa receive_audio foi cancelada.")
        except Exception as e_receive_audio_outer:
            logger.error(f"Erro crítico em receive_audio (fora do loop principal): {e_receive_audio_outer}")
            traceback.print_exc()
            self.stop_event.set() # Garante que tudo pare
        finally:
            logger.info("Tarefa receive_audio finalizada.")
            self.awaiting_name_for_save_face = False # Limpa a flag em qualquer saída
            if self.thinking_event.is_set(): # Garante que thinking_event seja limpo
                self.thinking_event.clear()


    async def play_audio(self):
        # (Função play_audio inalterada - omitida para brevidade, mas logs DEBUG podem ser adicionados se necessário)
        if not pya:
            logger.error("PyAudio não inicializado. Tarefa play_audio não pode iniciar.")
            return

        audio_output_stream = None
        output_device_sample_rate = RECEIVE_SAMPLE_RATE # Taxa padrão do Gemini
        try:
            logger.info("Configurando stream de áudio de saída (playback)...")
            try:
                default_out_device_info = pya.get_default_output_device_info()
                logger.info(f"Usando dispositivo de saída: {default_out_device_info['name']} @ {output_device_sample_rate} Hz.")
            except Exception as e_out_dev_info:
                logger.warning(f"Não foi possível obter info do dispositivo de saída padrão ({e_out_dev_info}). Usando taxa padrão: {output_device_sample_rate} Hz.")

            audio_output_stream = await asyncio.to_thread(
                pya.open, format=FORMAT, channels=CHANNELS, rate=output_device_sample_rate, output=True
            )
            logger.info("Player de áudio (playback) pronto.")

            while not self.stop_event.is_set():
                if not self.audio_in_queue: # Fila pode não existir durante reconexão
                    await asyncio.sleep(0.1)
                    continue

                audio_bytes_to_play = None
                try:
                    # Espera por áudio da fila com timeout
                    audio_bytes_to_play = await asyncio.wait_for(self.audio_in_queue.get(), timeout=0.5)

                    if audio_bytes_to_play is None: # Sinal de parada da fila (enviado no 'run' cleanup)
                        logger.info("Recebido sinal de encerramento (None) na fila de playback. Encerrando play_audio.")
                        break # Sai do loop while

                    if audio_output_stream and audio_output_stream.is_active():
                        # Escreve no stream em outra thread (pode bloquear)
                        await asyncio.to_thread(audio_output_stream.write, audio_bytes_to_play)
                    else:
                        logger.warning("Stream de áudio para playback não está ativo. Descartando áudio do Gemini.")
                    
                    if self.audio_in_queue: self.audio_in_queue.task_done() # Marca o item como processado

                except asyncio.TimeoutError:
                    continue # Normal se não houver áudio por 0.5s
                except asyncio.QueueEmpty: # Redundante com timeout
                    continue
                except OSError as e_os_playback:
                    if "Stream closed" in str(e_os_playback): # Erro comum
                        logger.warning("Stream de playback fechado (OSError). Encerrando play_audio.")
                        break
                    else: # Outro erro de OS
                        logger.error(f"Erro de OS ao reproduzir áudio: {e_os_playback}")
                        traceback.print_exc()
                        break # Para em outros erros de OS também
                except Exception as e_playback_inner:
                    logger.error(f"Erro ao reproduzir áudio (interno): {e_playback_inner}")
                    if "Stream closed" in str(e_playback_inner):
                        logger.warning("Stream de playback fechado (detectado em Exception). Encerrando play_audio.")
                        break
                    # traceback.print_exc() # Log detalhado para outros erros
                    # Considerar se deve parar ou continuar em outros erros. Parar é mais seguro.
                    break

        except asyncio.CancelledError:
            logger.info("Tarefa play_audio cancelada.")
        except Exception as e_play_audio_outer:
            logger.error(f"Erro crítico em play_audio: {e_play_audio_outer}")
            traceback.print_exc()
            # Não seta stop_event aqui, deixa o 'run' gerenciar paradas globais
        finally:
            logger.info("Finalizando play_audio...")
            if audio_output_stream:
                try:
                    # Espera o buffer esvaziar antes de fechar (opcional, pode causar delay)
                    # await asyncio.to_thread(audio_output_stream.stop_stream) # Garante que todo o buffer seja tocado
                    if audio_output_stream.is_active():
                         audio_output_stream.stop_stream() # Para o stream imediatamente
                    audio_output_stream.close()
                    logger.info("Stream de áudio de saída (playback) fechado.")
                except Exception as e_close_playback_stream:
                    logger.error(f"Erro ao fechar stream de áudio de saída: {e_close_playback_stream}")
            logger.info("Tarefa play_audio concluída.")


    async def run(self):
        # (Função run inalterada - omitida para brevidade, mas logs DEBUG podem ser adicionados se necessário)
        logger.info("Iniciando AudioLoop.run()...")
        max_connection_retries = 3
        retry_delay_seconds_base = 2.0

        current_retry_attempt = 0
        while current_retry_attempt <= max_connection_retries and not self.stop_event.is_set():
            try:
                if current_retry_attempt > 0: # Se for uma tentativa de reconexão
                     actual_retry_delay = retry_delay_seconds_base * (2 ** (current_retry_attempt -1)) # Backoff exponencial
                     logger.info(f"Tentativa de reconexão {current_retry_attempt}/{max_connection_retries} ao Gemini após {actual_retry_delay:.1f}s...")
                     await asyncio.sleep(actual_retry_delay)

                # --- Limpa estado da sessão anterior (importante para reconexão) ---
                if self.session:
                    try: await self.session.close()
                    except Exception: pass # Ignora erros ao fechar sessão antiga
                self.session = None
                self.audio_in_queue = None # Será recriado
                self.out_queue = None      # Será recriado
                self.awaiting_name_for_save_face = False
                if self.thinking_event.is_set(): self.thinking_event.clear()
                # --- Fim da Limpeza ---

                if client is None: # Verificação crucial
                    logger.error("ERRO CRÍTICO: Cliente Gemini não inicializado. Não é possível conectar. Encerrando AudioLoop.")
                    self.stop_event.set() # Para todas as outras tarefas se o cliente falhou
                    break # Sai do loop de retries

                logger.info(f"Tentando conectar ao Gemini LiveConnect (Tentativa {current_retry_attempt + 1})...")
                # O `async with` garante que session.close() seja chamado na saída
                async with client.aio.live.connect(model=MODEL, config=CONFIG) as live_session:
                    self.session = live_session
                    session_id_for_log = 'N/A'
                    if hasattr(live_session, 'session_id'): session_id_for_log = live_session.session_id
                    elif hasattr(live_session, '_session_id'): session_id_for_log = live_session._session_id
                    logger.info(f"Sessão Gemini LiveConnect estabelecida (ID: {session_id_for_log}).")
                    current_retry_attempt = 0 # Reseta tentativas em caso de sucesso na conexão

                    # Recria filas para a nova sessão
                    self.audio_in_queue = asyncio.Queue()
                    self.out_queue = asyncio.Queue(maxsize=200) # Fila maior para dados de saída (áudio, vídeo)

                    # --- Inicia todas as tarefas da sessão usando TaskGroup ---
                    # TaskGroup garante que se uma tarefa falhar, as outras são canceladas.
                    async with asyncio.TaskGroup() as tg:
                        logger.info("Iniciando tarefas da sessão (send_text, send_realtime, etc.)...")
                        tg.create_task(self.send_text(), name="send_text_task")
                        tg.create_task(self.send_realtime(), name="send_realtime_task")
                        if pya: tg.create_task(self.listen_audio(), name="listen_audio_task")

                        if self.video_mode == "camera":
                            tg.create_task(self.get_frames(), name="get_frames_task")
                        elif self.video_mode == "screen":
                            tg.create_task(self.get_screen(), name="get_screen_task")

                        tg.create_task(self.receive_audio(), name="receive_audio_task")
                        if pya: tg.create_task(self.play_audio(), name="play_audio_task")
                        logger.info("Todas as tarefas da sessão foram iniciadas.")
                    # O bloco `async with tg:` espera todas as tarefas do grupo terminarem.
                    # Se sair daqui, ou o stop_event foi setado, ou uma tarefa falhou (gerando ExceptionGroup),
                    # ou a sessão Gemini fechou e as tarefas que dependem dela terminaram.

                    logger.info("TaskGroup da sessão Gemini finalizado.")
                    if not self.stop_event.is_set(): # Se o TaskGroup terminou, mas não por nossa causa
                         logger.warning("Sessão Gemini terminou inesperadamente ou todas as tarefas do TaskGroup concluídas sem stop_event. Tentando reconectar...")
                         current_retry_attempt += 1
                    else: # Se stop_event foi setado (ex: por 'q' ou erro crítico)
                        logger.info("Stop_event detectado após TaskGroup. Encerrando loop de conexão.")
                        break # Sai do loop `while current_retry_attempt <= max_connection_retries`

            except asyncio.CancelledError:
                logger.info("Loop principal (AudioLoop.run) foi cancelado.")
                self.stop_event.set() # Garante que o evento de parada seja definido para outras tarefas
                break
            except ExceptionGroup as eg: # Erro vindo do TaskGroup (uma ou mais tarefas falharam)
                logger.error(f"Erro(s) no TaskGroup da sessão (Tentativa {current_retry_attempt + 1}):")
                self.stop_event.set() # Para tudo se uma tarefa crítica falhar
                for i, exc_in_group in enumerate(eg.exceptions):
                    logger.error(f"  Erro {i+1} no grupo: {type(exc_in_group).__name__} - {exc_in_group}")
                    # traceback.print_exception(type(exc_in_group), exc_in_group, exc_in_group.__traceback__)
                current_retry_attempt += 1
                self.session = None # Garante que a sessão seja considerada inválida para a próxima tentativa
            except (errors.LiveSessionClosedError, errors.LiveSessionNotConnectedError, errors.DeadlineExceededError, errors.RetryError) as e_gemini_conn:
                logger.warning(f"Erro de conexão/sessão Gemini (Tentativa {current_retry_attempt + 1}): {type(e_gemini_conn).__name__} - {e_gemini_conn}")
                # traceback.print_exc()
                current_retry_attempt += 1
                self.session = None
            except Exception as e_run_outer:
                logger.error(f"Erro inesperado no método AudioLoop.run (Tentativa {current_retry_attempt + 1}): {type(e_run_outer).__name__} - {e_run_outer}")
                traceback.print_exc()
                current_retry_attempt += 1
                self.session = None
                if current_retry_attempt > max_connection_retries:
                     logger.error("Máximo de tentativas de reconexão atingido após erro genérico. Encerrando.")
                     self.stop_event.set()
                     break

        # --- Fim do Loop de Conexão ---
        if not self.stop_event.is_set() and current_retry_attempt > max_connection_retries:
             logger.critical("Não foi possível estabelecer ou restabelecer a conexão com Gemini após múltiplas tentativas. O programa será encerrado.")
             self.stop_event.set() # Garante que o evento de parada esteja definido

        # --- Limpeza Final ---
        logger.info("Iniciando limpeza final em AudioLoop.run() (após loop de conexão)...")
        self.stop_event.set() # Garante que todas as tarefas saibam que devem parar

        if self.session: # Fecha a sessão Gemini se ainda estiver ativa (improvável se saiu do loop)
            try:
                logger.info("Fechando sessão LiveConnect ativa (limpeza final)...")
                await self.session.close()
            except Exception as e_final_close_session:
                logger.error(f"Erro ao fechar sessão LiveConnect na limpeza final: {e_final_close_session}")
        self.session = None

        if self.audio_in_queue: # Sinaliza para a tarefa play_audio parar
            try: self.audio_in_queue.put_nowait(None)
            except asyncio.QueueFull: logger.debug("Fila audio_in_queue cheia ao tentar colocar None na limpeza.")
            except Exception as e_q_put_none: logger.error(f"Erro ao colocar None na audio_in_queue (limpeza): {e_q_put_none}")

        if self.preview_window_active: # Fecha janelas OpenCV
            logger.info("Fechando janelas OpenCV (limpeza final)...")
            try: cv2.destroyAllWindows()
            except Exception as e_cv_destroy_final: logger.warning(f"Erro ao fechar janelas OpenCV na limpeza final: {e_cv_destroy_final}")
            self.preview_window_active = False

        # PyAudio é terminado no finally do __main__ para garantir que seja o último.
        logger.info("Limpeza de AudioLoop.run() concluída.")


# --- Função de Teste Síncrono (Passo 5 das instruções) ---
def run_sync_function_call_test(test_ftools, test_model_name, test_system_instruction):
    """Executa um teste síncrono para verificar a chamada de função."""
    if not client:
        logger.error("[SYNC TEST] Cliente Gemini não inicializado. Teste cancelado.")
        return
    if not test_ftools:
        logger.error("[SYNC TEST] Lista de ferramentas (FTOOLS) vazia. Teste cancelado.")
        return

    logger.info("--- INICIANDO TESTE SÍNCRONO DE CHAMADA DE FUNÇÃO ---")
    try:
        # Usa o system_instruction no construtor do modelo, que é uma prática comum.
        # O role="user" para system_instruction é como o LiveConnectConfig faz.
        sync_model_instance = genai.GenerativeModel(
            model_name=test_model_name,
            system_instruction=types.Content(parts=[types.Part.from_text(test_system_instruction)], role="user"),
            tools=test_ftools, # Passa as ferramentas diretamente para o modelo
            # Configuração explícita de chamada de função (ToolConfig)
            # Tenta o modo AUTO. Se FunctionCallingConfig.Mode não existir, pode ser uma string "AUTO".
            # A simples presença de 'tools' pode ser suficiente para alguns SDKs/versões.
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfig.Mode.AUTO # ou .ANY para forçar
                    # mode="AUTO" # Alternativa se Mode.AUTO não for um enum
                )
            )
        )
        
        user_test_prompt = "Por favor, salve o rosto da pessoa na câmera com o nome 'Carlos Teste'."
        logger.info(f"[SYNC TEST] Enviando prompt: \"{user_test_prompt}\"")
        logger.debug(f"[SYNC TEST] Ferramentas fornecidas: {[decl.name for tool in test_ftools if tool.function_declarations for decl in tool.function_declarations]}")

        # `generate_content` é uma chamada síncrona (bloqueante)
        sync_response = sync_model_instance.generate_content(
            contents=[types.Content(parts=[types.Part.from_text(user_test_prompt)], role="user")]
        )

        logger.info("[SYNC TEST] Resposta recebida.")
        # logger.debug(f"[SYNC TEST] Resposta completa: {sync_response}") # Log muito verboso

        if sync_response.candidates and sync_response.candidates[0].content and sync_response.candidates[0].content.parts:
            response_part = sync_response.candidates[0].content.parts[0]
            if response_part.function_call:
                fc_details = response_part.function_call
                logger.info(f"  [SUCCESS] Teste síncrono resultou em Function Call:")
                logger.info(f"    Nome da Função: {fc_details.name}")
                logger.info(f"    Argumentos: {dict(fc_details.args)}")
            elif response_part.text:
                logger.info(f"  [TEXT] Teste síncrono resultou em Texto (esperava-se Function Call):")
                logger.info(f"    Texto: \"{response_part.text}\"")
            else:
                logger.warning(f"  [UNKNOWN PART] Teste síncrono retornou uma parte de resposta desconhecida: {response_part}")
        
        elif sync_response.candidates and sync_response.candidates[0].finish_reason:
             finish_reason_name = sync_response.candidates[0].finish_reason.name
             logger.warning(f"  [FINISH REASON] Teste síncrono: {finish_reason_name}")
             if finish_reason_name == "SAFETY":
                 logger.error("  [SAFETY BLOCK] Resposta do teste síncrono bloqueada por razões de segurança.")
             elif finish_reason_name == "RECITATION":
                 logger.warning("  [RECITATION BLOCK] Resposta do teste síncrono bloqueada por recitação.")
             # Verificar se há prompt_feedback para mais detalhes
             if sync_response.prompt_feedback and sync_response.prompt_feedback.block_reason:
                 logger.error(f"    Motivo do bloqueio do prompt: {sync_response.prompt_feedback.block_reason.name}")

        else:
            logger.error("  [NO VALID RESPONSE] Teste síncrono não retornou conteúdo ou candidatos na forma esperada.")
            logger.debug(f"Resposta completa do teste síncrono (sem conteúdo esperado): {sync_response}")


    except AttributeError as e_attr:
        logger.error(f"[SYNC TEST] Erro de atributo durante o teste: {e_attr}")
        logger.error("  Isso pode indicar uma incompatibilidade com a versão do SDK google-genai.")
        logger.error("  Verifique se 'FunctionCallingConfig.Mode.AUTO' ou a estrutura de 'tool_config' está correta.")
        traceback.print_exc()
    except Exception as e_sync_test:
        logger.error(f"[SYNC TEST] Erro CRÍTICO durante o teste síncrono: {e_sync_test}")
        traceback.print_exc()
    logger.info("--- TESTE SÍNCRONO DE CHAMADA DE FUNÇÃO CONCLUÍDO ---")


# --- Ponto de Entrada Principal ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trackie - Assistente IA visual e auditivo com Gemini.")
    parser.add_argument(
        "--mode", type=str, default=DEFAULT_MODE, choices=["camera", "screen", "none"],
        help="Modo de operação para entrada de vídeo/imagem ('camera', 'screen', 'none'). Padrão: camera."
    )
    parser.add_argument(
        "--show_preview", action="store_true",
        help="Mostra janela com preview da câmera e detecções YOLO (apenas no modo 'camera')."
    )
    parser.add_argument(
        "--run_sync_test", action="store_true",
        help="Executa um teste síncrono de chamada de função antes de iniciar o loop principal."
    )
    args = parser.parse_args()

    show_actual_preview_window = False
    if args.mode == "camera" and args.show_preview:
        show_actual_preview_window = True
        logger.info("Feedback visual da câmera (preview YOLO) ATIVADO.")
    elif args.mode != "camera" and args.show_preview:
        logger.info("Aviso: --show_preview só tem efeito com --mode camera. Preview será desativado.")
    else:
        logger.info("Feedback visual da câmera (preview YOLO) DESATIVADO.")

    # --- Verificações Críticas Antes de Iniciar ---
    if client is None: # Cliente Gemini falhou na inicialização
        logger.critical("Cliente Gemini não pôde ser inicializado. Verifique a API Key e logs anteriores. Encerrando.")
        exit(1)

    if not pya: # PyAudio falhou
         logger.critical("PyAudio não pôde ser inicializado. Verifique a instalação e dependências (PortAudio). Encerrando.")
         exit(1)

    if args.mode == "camera" and not os.path.exists(YOLO_MODEL_PATH):
            logger.critical(f"ERRO: Modelo YOLO '{YOLO_MODEL_PATH}' não encontrado, mas modo 'camera' está ativo.")
            logger.critical("Verifique o caminho ou baixe o modelo. Encerrando.")
            exit(1)

    if 'system_instruction_text' not in globals() or \
       not system_instruction_text or \
       system_instruction_text == "Você é um assistente prestativo.": # Verifica se o prompt padrão mínimo foi usado
         logger.warning("AVISO: Falha ao carregar a instrução do sistema do arquivo ou o arquivo não foi encontrado/está vazio.")
         logger.warning("O assistente usará um prompt padrão interno, o que pode afetar o comportamento.")
         # Não sair, mas alertar. O prompt padrão robusto definido acima será usado.

    # --- Executar Teste Síncrono (Opcional) ---
    if args.run_sync_test:
        if FTOOLS and MODEL and system_instruction_text:
            run_sync_function_call_test(FTOOLS, MODEL, system_instruction_text)
        else:
            logger.error("Não foi possível executar o teste síncrono: FTOOLS, MODEL ou system_instruction_text não estão definidos.")


    # --- Iniciar Loop Principal do Assistente ---
    main_assist_loop = None
    try:
        logger.info(f"Iniciando Trackie no modo: {args.mode.upper()}")
        main_assist_loop = AudioLoop(video_mode=args.mode, show_preview=show_actual_preview_window)
        asyncio.run(main_assist_loop.run())

    except KeyboardInterrupt:
        logger.info("\nInterrupção pelo teclado (Ctrl+C) recebida. Encerrando Trackie...")
        if main_assist_loop:
            logger.info("Sinalizando parada para todas as tarefas do assistente...")
            main_assist_loop.stop_event.set()
            # Um pequeno delay para permitir que as tarefas tentem limpar antes de sair abruptamente
            # O loop `run` já tem sua própria lógica de limpeza e espera por tarefas.
            # asyncio.run já lida com o cancelamento de tarefas restantes no TaskGroup.
    except Exception as e_main_unhandled:
        logger.critical(f"Erro CRÍTICO e não tratado no bloco __main__: {type(e_main_unhandled).__name__}: {e_main_unhandled}")
        traceback.print_exc()
        if main_assist_loop:
            logger.info("Sinalizando parada para as tarefas devido a erro crítico inesperado...")
            main_assist_loop.stop_event.set()
    finally:
        logger.info("Bloco __main__ finalizado. Iniciando limpeza de recursos globais (PyAudio)...")
        if pya: # Termina PyAudio globalmente, pois foi inicializado globalmente
            try:
                logger.info("Terminando PyAudio globalmente...")
                pya.terminate()
                logger.info("Recursos de PyAudio liberados globalmente.")
            except Exception as e_pya_terminate_final:
                logger.error(f"Erro ao terminar PyAudio na limpeza final do __main__: {e_pya_terminate_final}")
        
        # Garante que todas as janelas OpenCV sejam fechadas se algo deu muito errado
        # e o cleanup do AudioLoop não foi chamado ou falhou.
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        logger.info("Programa Trackie completamente finalizado.")


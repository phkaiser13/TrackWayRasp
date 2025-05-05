import os
import asyncio
import base64
import io
import traceback
import time
import json
import logging
from collections import deque
import argparse
import signal
import sys

# --- Bibliotecas de IA e Processamento ---
import cv2 # pip install opencv-python
import pyaudio # pip install pyaudio (pode precisar de portaudio dev lib)
import PIL.Image # pip install Pillow
import mss # pip install mss
import numpy as np # pip install numpy
import torch # pip install torch torchvision torchaudio (verificar versão CUDA se usar GPU)

# Tentar importar bibliotecas de IA e logar falhas se não encontradas
# Define as variáveis como None para permitir fallback gracioso
YOLO = None
try:
    from ultralytics import YOLO # pip install ultralytics
except ImportError:
    print("AVISO: Biblioteca 'ultralytics' (YOLO) não encontrada. YOLO será desativado.")
except Exception as e:
    print(f"AVISO: Erro ao importar ultralytics: {e}. YOLO será desativado.")

DPTImageProcessor, DPTForDepthEstimation = None, None
try:
    from transformers import DPTImageProcessor, DPTForDepthEstimation # pip install transformers
except ImportError:
     print("AVISO: Biblioteca 'transformers' não encontrada. MiDaS via transformers será desativado.")
except Exception as e:
     print(f"AVISO: Erro ao importar transformers: {e}. MiDaS será desativado.")


DeepFace = None
try:
    # DeepFace tem muitas dependências pesadas (tensorflow, etc.)
    from deepface import DeepFace # pip install deepface
except ImportError:
     print("AVISO: Biblioteca 'deepface' não encontrada. DeepFace será desativado.")
except Exception as e:
     print(f"AVISO: Erro ao importar deepface: {e}. DeepFace será desativado.")

# --- IMPORTANTE: Ajuste a importação do SAM2 para corresponder à sua biblioteca ---
# Exemplo: Se usar 'segment-anything' (SAM original):
# from segment_anything import sam_model_registry, SamPredictor
# Se for uma biblioteca 'sam2' específica, verifique os nomes corretos
sam_model_registry, SamPredictor = None, None
try:
    # TENTE SUBSTITUIR 'sam2_library' PELO NOME REAL DA BIBLIOTECA NO SEU 'pip list'
    # ou pelo nome do módulo python correto
    from sam2.build_sam import build_sam2 # <<< AJUSTE ESTA LINHA!
  #  from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator   # <<< E ESSA!
    # Pode ser que 'build_sam' também seja necessário se o registry não for direto
    # from sam2.build_sam import build_sam2 # Se necessário
except ImportError:
     print("AVISO: Biblioteca 'sam2_library' (ou similar para SAM2) não encontrada. SAM2 será desativado.")
     print("       Verifique o nome correto da biblioteca instalada e ajuste a importação.")
except Exception as e:
     print(f"AVISO: Erro ao importar biblioteca SAM2: {e}. SAM2 será desativado.")





vision = None
try:
    from google.cloud import vision # pip install google-cloud-vision
except ImportError:
     print("AVISO: Biblioteca 'google-cloud-vision' não encontrada. Cloud Vision será desativado.")
except Exception as e:
     print(f"AVISO: Erro ao importar google-cloud-vision: {e}. Cloud Vision será desativado.")

# --- Configurações Globais (Idealmente movidas para um arquivo de config ou args) ---
# Modelos e APIs
YOLO_MODEL_PATH = "/home/raspsenai/yolov8n.pt"
SAM2_CONFIG_PATH = "/home/raspsenai/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_MODEL_TYPE = "hiera_large"
SAM2_CHECKPOINT_PATH = "/home/raspsenai/sam2/checkpoints/sam2.1_hiera_large.pt"
MIDAS_MODEL_NAME = "Intel/dpt-swinv2-large-384"
MIDAS_WEIGHTS_PATH = "/home/raspsenai/MiDaS/weights/dpt_swin2_large_384.pt"
KNOWN_FACES_DIR = "/home/raspsenai/known_faces"
VISION_API_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/home/raspsenai/google-json-creds.json")
GEMINI_MODEL = "models/gemini-2.0-flash-live-001"
API_KEY = os.getenv("GEMINI_API_KEY")

# Outras Configurações
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000 # Taxa de envio para Gemini
RECEIVE_SAMPLE_RATE = 24000 # Taxa de recebimento de Gemini (pode variar)
CHUNK_SIZE = 1024 # Tamanho do chunk de áudio
VIDEO_FPS_TARGET = 3  # Frames por segundo alvo para processamento de IA (não captura)
CONTEXT_INJECTION_INTERVAL = 4 # Segundos entre injeções de contexto
DEFAULT_VIDEO_MODE = "camera" # Ou "screen", "none"
DEFAULT_USER_NAME = "Usuário"
SYSTEM_INSTRUCTION = """
Você é Trackie (pronuncia-se 'Tréqui'), uma IA assistente integrada a um dispositivo vestível, projetada para auxiliar pessoas com deficiência visual, especialmente em ambientes como oficinas, fábricas ou laboratórios. Seu objetivo principal é fornecer informações sobre o ambiente de forma clara, concisa e segura em português do Brasil.

**Instruções Fundamentais:**

1.  **Nome do Usuário:** Sempre comece suas respostas mencionando o nome do usuário (ex: "Pedro, ..."). Se o nome não foi fornecido, use "Usuário". (O nome será gerenciado pelo sistema externo). Por enquanto, use 'Usuário'.
2.  **Comunicação Clara:** Use frases curtas, vocabulário simples e direções inequívocas (esquerda, direita, frente, atrás). Evite jargões técnicos, a menos que essenciais para a segurança.
3.  **Estimativa de Distância:** Quando descrever a localização de objetos, sempre estime a distância em *passos* (considere 1 passo ≈ 0.7 metros). Você receberá informações de profundidade estimadas pelo sistema em metros (precedidas por 'm', ex: 'cadeira (3.1m)'). Use essas informações para calcular os passos e informe APENAS os passos. Exemplo: Se receber "cadeira (2.1m)", diga "Usuário, há uma cadeira a cerca de 3 passos à sua direita." (2.1 / 0.7 ≈ 3). Se receber "caixa (dist?)", diga "Usuário, vejo uma caixa, mas não consigo estimar a distância." Arredonde os passos para o inteiro mais próximo.
4.  **Prioridade à Segurança:** Identifique e avise IMEDIATAMENTE sobre perigos potenciais (fogo, obstáculos inesperados, produtos químicos derramados, fios desencapados, máquinas em operação próximas, etc.). Use um tom de alerta. Exemplo: "Usuário, CUIDADO! Fio desencapado no chão a 2 passos à sua frente!". Você receberá informações de detecção de objetos e cenas pelo sistema. Use seu raciocínio para inferir perigos. Objetos como 'knife', 'scissors', 'fire', 'smoke', 'machine' merecem atenção especial.
5.  **Descrição Ambiental:** Uma das suas principais funções, Deve ser feita somente quando o usuario mandar, forneça uma breve descrição do ambiente, focando em objetos principais, saídas, pessoas e textos relevantes (placas de aviso, sinalizações). Use as informações de contexto recebidas.
6.  **Leitura de Texto (OCR):** Quando detectar texto (via OCR fornecido pelo sistema como "Texto: '...'"), leia-o se parecer relevante (placas, etiquetas de aviso, nomes em crachás). Exemplo: "Usuário, uma placa na parede diz: 'PERIGO - Alta Voltagem'". Seja conciso.
7.  **Reconhecimento Facial:** Você receberá informações sobre pessoas identificadas ("Pessoas: Nome1 (2.5m), Pessoa Desconhecida (4.0m)"). Use essa informação para responder perguntas como "Quem está na minha frente?". Exemplo: "Usuário, a pessoa a uns 4 passos à sua frente é Ana Silva." (2.5 / 0.7 ≈ 4 passos). Se um rosto não for reconhecido, diga "Usuário, há uma pessoa não reconhecida a cerca de 6 passos à sua frente." (4.0 / 0.7 ≈ 6 passos).
8.  **Comandos Naturais:** Entenda e responda a comandos como: "Onde está a saída?", "O que é esse objeto perto da minha mão esquerda?", "Leia a placa na parede.", "Tem alguém vindo?".
9.  **INFORMAÇÕES DE CONTEXTO DO SISTEMA:** Você receberá periodicamente mensagens iniciadas com `[CONTEXTO_SISTEMA]`. Essas mensagens contêm dados processados da câmera e outros sensores (objetos detectados por YOLO com distância em metros, segmentação SAM2 [interpretar como descrição geral da cena], profundidade por MiDaS [usada para as distâncias], rostos por DeepFace com distância, texto por OCR). **É ESSENCIAL que você UTILIZE essas informações** para compor suas respostas, calcular distâncias (convertendo metros para passos), avaliar perigos e descrever o ambiente de forma precisa. Não leia a mensagem `[CONTEXTO_SISTEMA]` para o usuário, apenas use os dados dela para formular sua resposta em linguagem natural. Ignore itens com '(dist?)' se não for relevante mencioná-los sem distância. O sumário de 'Segmentos' dá uma ideia do que ocupa mais espaço na visão (ex: parede, chão, mesa).
10. **Tom de Voz:** Mantenha um tom de voz calmo, prestativo e seguro. Aumente a urgência apenas para avisos de perigo. Use a voz pré-definida se aplicável à API (a configuração é externa).
11. **Incerteza:** Se a visão estiver obstruída, a imagem ou o som estiverem ruins, informe ao usuário. Ex: "Usuário, não consigo ver claramente agora, a câmera parece obstruída." ou "Usuário, o som está muito ruidoso, pode repetir?". Se o contexto recebido for vazio ou limitado, diga algo como "Usuário, a visão do ambiente está limitada no momento."

**Exemplo de como usar o CONTEXTO_SISTEMA:**
Se receber: `[CONTEXTO_SISTEMA] Objetos: cadeira (2.1m), mesa (3.0m), faca (dist?). Pessoas: Ana Silva (3.5m), Pessoa Desconhecida (1.5m). Texto: 'SAIDA'. Segmentos: [Seg1(Area:15000,IoU:0.9), Seg2(Area:8000,IoU:0.8)] (2 máscaras total)` (Assume que segmentos são objetos não identificados ou partes grandes como parede/chão)
Sua descrição ou resposta poderia ser: "Usuário, à sua frente vejo uma Pessoa Desconhecida a cerca de 2 passos (1.5/0.7≈2). Mais adiante, a Ana Silva está a uns 5 passos (3.5/0.7≈5). Há também uma cadeira a 3 passos (2.1/0.7≈3) e uma mesa a cerca de 4 passos (3.0/0.7≈4). Vejo também o que parece ser uma faca por perto, mas não sei a distância exata. Uma placa escrita 'SAIDA' está visível."
"""
# Defina a instrução do sistema como uma variável separada



    
# --- Configuração de Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Desativar logs muito verbosos de bibliotecas de terceiros, se necessário
# logging.getLogger("ultralytics").setLevel(logging.WARNING)
# logging.getLogger("transformers").setLevel(logging.WARNING)
# logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
# logging.getLogger("deepface").setLevel(logging.WARNING)

logger = logging.getLogger("TrackieAssist") # Logger específico para a aplicação

# --- Validação Inicial Crítica ---
if not API_KEY:
    logger.critical("API Key do Gemini não configurada! Defina a variável de ambiente 'GEMINI_API_KEY'. Encerrando.")
    sys.exit(1)

# Tentar importar GenAI DEPOIS de checar a key
    # --- INICIALIZAÇÃO CORRETA DO CLIENTE GEMINI (Aqui, após ler os Args) ---
    global client, generation_config, tools # Declara que vamos *definir* estas globais aqui
    client = None # Reseta cliente para garantir que está limpo antes de tentar inicializar
    generation_config = None
    tools = []
    logger.info(f"--- Iniciando Cliente Google GenAI (Modelo: {GEMINI_MODEL}) ---") # Usa o GEMINI_MODEL atualizado pelos args

    try:
        # Tenta importar de novo para garantir (caso tenha falhado antes)
        # Usa aliases para evitar conflito se 'genai'/'glm' já existirem com outro significado
        import google.generativeai as genai_core
        from google.ai import generativelanguage as glm_core

        # Atribui aos nomes globais que o resto do código espera
        genai = genai_core
        glm = glm_core
        if not genai: raise ImportError("Falha ao reimportar google.generativeai")

        logger.info(f"SDK google-generativeai versão: {genai.__version__}")

        # 1. Configurar API Key (essencial antes de criar o modelo)
        genai.configure(api_key=API_KEY)
        logger.info("Gemini SDK configurado com API Key.")

        # 2. Preparar Configs (usando 'genai' diretamente)
        # Configuração de Geração
        generation_config = genai.GenerationConfig(
            candidate_count=1,
            temperature=0.2, # Exemplo de temperatura
            # max_output_tokens=4096, # Exemplo
            # stop_sequences=["\nHuman:", "\nUsuario:"] # Exemplo
        )
        # Ferramentas (Exemplo: Habilitar busca - REMOVA se não quiser)
        # tools = [genai.Tool(google_search_retrieval=genai.GoogleSearchRetrieval())]
        
                # (Dentro do try...except de inicialização do Gemini em main)
        # ... (após definir generation_config) ...

        # --- SUBSTITUIÇÃO: Definir Ferramentas usando 'genai' diretamente ---
        logger.info("Configurando ferramentas Gemini...")
        tools = [] # Começa com lista vazia por segurança
        try:
            # Verifica se o módulo 'genai' foi importado e está disponível
            if genai:
                # Tenta obter as classes necessárias diretamente do módulo 'genai'
                # Usar getattr é mais seguro caso alguma classe não exista em certas versões
                ToolClass = getattr(genai, 'Tool', None)
                CodeExecutionClass = getattr(genai, 'CodeExecution', None)
                GoogleSearchClass = getattr(genai, 'GoogleSearch', None)
                # FunctionDeclaration geralmente vem de glm, se precisar:
                # FunctionDeclarationClass = getattr(glm, 'FunctionDeclaration', None)

                # Verifica se conseguiu obter as classes necessárias
                if ToolClass and CodeExecutionClass and GoogleSearchClass:
                    # Monta a lista de ferramentas usando as classes encontradas
                    # Adicione ou remova ferramentas conforme sua necessidade
                    tools = [
                        ToolClass(code_execution=CodeExecutionClass()),
                        ToolClass(google_search=GoogleSearchClass()),
                        # Exemplo se fosse usar Function Calling:
                        # ToolClass(function_declarations=[
                        #     FunctionDeclarationClass(name='minha_funcao', description='...', parameters=...)
                        # ])
                    ]
                    logger.info("Ferramentas Gemini (CodeExecution, GoogleSearch) configuradas com sucesso.")
                else:
                    # Loga quais classes não foram encontradas
                    missing = [cls_name for cls_name, cls_obj in [('Tool', ToolClass), ('CodeExecution', CodeExecutionClass), ('GoogleSearch', GoogleSearchClass)] if cls_obj is None]
                    logger.warning(f"Não foi possível encontrar as classes de ferramentas ({', '.join(missing)}) no módulo 'genai'. Nenhuma ferramenta será usada.")
                    tools = [] # Garante que a lista está vazia
            else:
                logger.warning("Módulo 'genai' não disponível. Nenhuma ferramenta Gemini será usada.")
                tools = [] # Garante que a lista está vazia

        except AttributeError as e_tool_attr:
            logger.warning(f"Erro de atributo ao configurar ferramentas Gemini (API mudou?): {e_tool_attr}. Nenhuma ferramenta será usada.")
            tools = []
        except Exception as e_tool_other:
            logger.error(f"Erro inesperado ao configurar ferramentas Gemini: {e_tool_other}", exc_info=True)
            tools = []
        # --- FIM DA SUBSTITUIÇÃO ---

        # Log final sobre as ferramentas (já estava lá, mas confirma)
        logger.info(f"Configuração de Ferramentas: {'Sim (' + str(len(tools)) + ')' if tools else 'Não'}")

        # 3. Instanciar o Modelo Generativo (AQUI que o 'client' é criado!)
        # (O restante da inicialização do client continua igual)
        client = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=SYSTEM_INSTRUCTION,
            generation_config=generation_config,
            tools=tools if tools else None # Passa a lista 'tools' definida acima
            # safety_settings=[...]
        )
    
    
    
    # Deixe vazio se não usar ferramentas

        logger.info(f"Configuração de Geração: {generation_config}")
        logger.info(f"Configuração de Ferramentas: {'Sim' if tools else 'Não'}")

        # 3. Instanciar o Modelo Generativo (AQUI que o 'client' é criado!)
        client = genai.GenerativeModel(
            model_name=GEMINI_MODEL,         # Usa o modelo dos args/default
            system_instruction=SYSTEM_INSTRUCTION, # Passa a instrução do sistema global
            generation_config=generation_config, # Passa a config de geração
            tools=tools if tools else None       # Passa as ferramentas (ou None)
            # safety_settings=[...] # Adicione configurações de segurança se necessário
        )
        # ---> ESTA LINHA ACIMA É A QUE CRIA O OBJETO 'client' <---

        logger.info(f"SUCESSO: Cliente GenerativeModel para '{GEMINI_MODEL}' criado e atribuído à variável 'client'.")

    except ImportError as imp_err:
        logger.critical(f"FALHA: Biblioteca 'google.generativeai' não encontrada durante inicialização final: {imp_err}")
        client = None # Garante que client é None se import falhar
    except AttributeError as attr_err:
        logger.critical(f"FALHA: Atributo não encontrado no SDK Gemini (Versão incompatível? Chave inválida?): {attr_err}", exc_info=False)
        client = None # Garante que client é None
    except Exception as e_init:
        logger.critical(f"FALHA CRÍTICA ao inicializar cliente Gemini: {e_init}", exc_info=True)
        client = None # Garante que client é None

    # --- LOG DIAGNÓSTICO IMPORTANTE ---
    # Verifica se o 'client' foi realmente criado após o try/except
    if client:
        logger.info("VERIFICAÇÃO PÓS-INIT: Objeto 'client' Gemini FOI inicializado.")
    else:
        logger.error("VERIFICAÇÃO PÓS-INIT: Objeto 'client' Gemini NÃO FOI inicializado (continua None). A task Gemini Manager falhará.")
        # Considerar encerrar se Gemini for essencial:
        # logger.critical("Gemini é essencial e não inicializou. Encerrando.")
        # sys.exit(1)

    # --- FIM DO BLOCO DE INICIALIZAÇÃO DO GEMINI ---



# --- Inicialização PyAudio ---
pya = None
try:
    pya = pyaudio.PyAudio()
    logger.info(f"PyAudio inicializado (versão: {pyaudio.get_portaudio_version_text()})")
except Exception as e:
     logger.error(f"Falha ao inicializar PyAudio: {e}. Funcionalidades de áudio desativadas.")
     # Não sai, mas áudio não funcionará

# --- Placeholders para Modelos e Estados Globais ---
yolo_model = None
sam_predictor = None # Guarda o objeto predictor do SAM
midas_processor = None
midas_model = None
cloud_vision_client = None
# deepface_db_path é definido globalmente, usado diretamente

# --- Funções Auxiliares ---

def load_models(device: str):
    """Carrega os modelos de IA para o dispositivo especificado (CPU ou GPU)."""
    global yolo_model, sam_predictor, midas_processor, midas_model, cloud_vision_client
    logger.info(f"Iniciando carregamento de modelos para o dispositivo: {device}...")
    start_time = time.monotonic()

    # --- YOLO ---
    if YOLO and YOLO_MODEL_PATH:
        try:
            logger.info(f"Carregando modelo YOLO de: {YOLO_MODEL_PATH}...")
            yolo_model = YOLO(YOLO_MODEL_PATH)
            # Forçar modelo para o device correto (ultralytics geralmente faz isso automaticamente, mas pode ser explícito)
            # A chamada .to(device) é mais comum em torch puro, yolo() pode ter seu próprio arg
            # Exemplo de predição para forçar carregamento/mover para device
            _ = yolo_model(np.zeros((64, 64, 3), dtype=np.uint8), device=device, verbose=False)
            logger.info(f"Modelo YOLO '{YOLO_MODEL_PATH}' carregado e testado em {device}.")
        except FileNotFoundError:
            logger.error(f"Erro: Arquivo do modelo YOLO não encontrado em {YOLO_MODEL_PATH}")
            yolo_model = None
        except Exception as e:
            logger.error(f"Falha ao carregar ou testar modelo YOLO: {e}", exc_info=True)
            yolo_model = None
    else:
        logger.warning("YOLO desativado (biblioteca 'ultralytics' não importada ou YOLO_MODEL_PATH não definido).")


    # --- SAM2 (Segment Anything Model 2 ou similar) ---
    # Presume que sam_model_registry e SamPredictor foram importados corretamente
    if sam_model_registry and SamPredictor and SAM2_CHECKPOINT_PATH and SAM2_MODEL_TYPE:
        try:
            logger.info(f"Carregando modelo SAM2 tipo '{SAM2_MODEL_TYPE}' do checkpoint: {SAM2_CHECKPOINT_PATH} para {device}...")
            # Instancia o modelo SAM usando o registro de modelos
            sam_model_instance = sam_model_registry[SAM2_MODEL_TYPE](checkpoint=SAM2_CHECKPOINT_PATH)
            sam_model_instance.to(device=device)
            sam_model_instance.eval() # Modo de avaliação
            # Cria o objeto predictor, que será usado para inferência
            sam_predictor = SamPredictor(sam_model_instance)
            logger.info("Modelo SAM2 e Predictor carregados com sucesso.")
            # Teste rápido opcional (requer uma imagem dummy)
            # dummy_img_rgb = np.zeros((128, 128, 3), dtype=np.uint8)
            # sam_predictor.set_image(dummy_img_rgb)
            # logger.info("SAM2 Predictor testado com set_image.")
        except FileNotFoundError:
             logger.error(f"Erro: Checkpoint do SAM2 não encontrado em {SAM2_CHECKPOINT_PATH}")
             sam_predictor = None
        except KeyError:
             logger.error(f"Erro: Tipo de modelo SAM2 '{SAM2_MODEL_TYPE}' não reconhecido pelo 'sam_model_registry'. Verifique o nome e a biblioteca.")
             sam_predictor = None
        except ImportError: # Captura erro se SamPredictor/registry veio de uma lib não instalada
            logger.error(f"Erro de Importação ao tentar usar SAM2. A biblioteca está instalada corretamente?")
            sam_predictor = None
        except Exception as e:
            logger.error(f"Falha inesperada ao carregar ou testar modelo SAM2: {e}", exc_info=True)
            sam_predictor = None
    else:
        logger.warning("SAM2 desativado (biblioteca/classes não importadas, checkpoint ou tipo não definidos).")


    # --- MiDaS (usando Transformers) ---
    if DPTImageProcessor and DPTForDepthEstimation and MIDAS_MODEL_NAME:
        try:
            logger.info(f"Carregando MiDaS (DPT) processor e modelo: '{MIDAS_MODEL_NAME}' para {device}...")
            # Tenta carregar o modelo/processor usando o nome HuggingFace
            # Isso baixa os pesos automaticamente se não estiverem no cache local
            midas_processor = DPTImageProcessor.from_pretrained(MIDAS_MODEL_NAME)
            midas_model = DPTForDepthEstimation.from_pretrained(MIDAS_MODEL_NAME)
            midas_model.to(device)
            midas_model.eval() # Modo de avaliação é crucial
            logger.info(f"Modelo MiDaS (DPT '{MIDAS_MODEL_NAME}') carregado em {device}.")
            # Teste opcional com imagem dummy
            # dummy_input = midas_processor(images=np.zeros((64, 64, 3), dtype=np.uint8), return_tensors="pt").to(device)
            # with torch.no_grad():
            #    _ = midas_model(**dummy_input)
            # logger.info("Modelo MiDaS (DPT) testado.")
        except OSError as e:
             # OSError pode indicar modelo não encontrado online ou problema de permissão/disco no cache
             logger.error(f"Falha ao carregar MiDaS (DPT) '{MIDAS_MODEL_NAME}'. Verifique o nome, conexão com a internet ou o cache do HuggingFace (~/.cache/huggingface). Erro: {e}", exc_info=False)
             midas_processor, midas_model = None, None
        except Exception as e:
            logger.error(f"Falha inesperada ao carregar ou testar MiDaS (DPT): {e}", exc_info=True)
            midas_processor, midas_model = None, None
    else:
        logger.warning("MiDaS (DPT) desativado (biblioteca 'transformers' não importada ou MIDAS_MODEL_NAME não definido).")


    # --- DeepFace ---
    if DeepFace and KNOWN_FACES_DIR:
         try:
             logger.info("Verificando disponibilidade do DeepFace...")
             # Garante que o diretório DB exista antes de qualquer operação
             os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
             logger.info(f"Diretório para banco de dados de faces DeepFace: '{KNOWN_FACES_DIR}' (criado se não existia).")
             # Teste rápido opcional para forçar carregamento inicial de algum modelo (pode ser LENTO!)
             # try:
             #     logger.info("Tentando pré-carregar modelo base DeepFace (ex: VGG-Face)...")
             #     _ = DeepFace.build_model("VGG-Face")
             #     logger.info("Modelo base DeepFace pré-carregado.")
             # except Exception as build_e:
             #     logger.warning(f"Não foi possível pré-carregar modelo base DeepFace: {build_e}. Modelos serão carregados sob demanda.")
             logger.info("DeepFace parece estar disponível. Modelos serão carregados sob demanda nas análises.")
         except Exception as e:
             logger.error(f"Erro durante a verificação inicial do DeepFace (ex: criação do diretório '{KNOWN_FACES_DIR}'): {e}", exc_info=True)
             # DeepFace pode não funcionar corretamente se houve erro aqui. A task tentará usá-lo mesmo assim.
    else:
        logger.warning("DeepFace desativado (biblioteca 'deepface' não importada ou KNOWN_FACES_DIR não definido).")


    # --- Google Cloud Vision ---
    if vision and VISION_API_CREDENTIALS:
        try:
            logger.info(f"Carregando credenciais do Google Cloud Vision de: {VISION_API_CREDENTIALS}...")
            if not os.path.exists(VISION_API_CREDENTIALS):
                raise FileNotFoundError(f"Arquivo de credenciais não encontrado em: {VISION_API_CREDENTIALS}")

            # Usa as credenciais do arquivo JSON especificado
            cloud_vision_client = vision.ImageAnnotatorClient.from_service_account_json(
                VISION_API_CREDENTIALS
            )
            # Teste simples para verificar a conexão/autenticação (opcional, consome quota mínima)
            # logger.info("Testando autenticação do Google Cloud Vision...")
            # test_image = vision.Image(content=base64.b64decode("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")) # 1x1 pixel gif
            # response = cloud_vision_client.annotate_image({'image': test_image, 'features': [{'type_': vision.Feature.Type.LABEL_DETECTION}]})
            # if response.error.message: raise Exception(f"Erro na API Vision (teste): {response.error.message}")
            logger.info("Cliente Google Cloud Vision inicializado e autenticado com sucesso.")
        except FileNotFoundError as fnf_e:
            logger.error(str(fnf_e)) # Já tem a mensagem formatada
            cloud_vision_client = None
        except ImportError: # Se google.cloud.vision foi definido como None antes
             logger.error("Erro ao usar Google Cloud Vision: Biblioteca não foi importada corretamente.")
             cloud_vision_client = None
        except Exception as e:
            logger.error(f"Falha ao carregar ou testar Google Cloud Vision: {e}", exc_info=True)
            cloud_vision_client = None
    elif vision:
        logger.warning("Google Cloud Vision desativado: caminho das credenciais (GOOGLE_APPLICATION_CREDENTIALS ou var local) não definido ou inválido.")
    else:
        logger.warning("Google Cloud Vision desativado (biblioteca 'google-cloud-vision' não importada).")

    elapsed = time.monotonic() - start_time
    logger.info(f"Carregamento de modelos concluído em {elapsed:.2f} segundos.")


def preprocess_frame_for_midas_transformers(frame: np.ndarray, processor, device) -> torch.Tensor | None:
    """Converte frame OpenCV BGR para o formato esperado pelo MiDaS DPT via Transformers."""
    if processor is None or frame is None:
        logger.debug("Pré-processamento MiDaS pulado (processador ou frame nulo).")
        return None
    try:
        # Transformers DPTImageProcessor geralmente espera RGB PIL Image ou numpy array
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # return_tensors="pt" retorna tensores PyTorch
        inputs = processor(images=image_rgb, return_tensors="pt")
        # Retorna os 'pixel_values' prontos para o modelo, já movidos para o device correto
        return inputs['pixel_values'].to(device)
    except Exception as e:
        logger.error(f"Erro no pré-processamento MiDaS (Transformers): {e}", exc_info=False) # Log menos verboso
        return None


def get_depth_from_midas_output_transformers(output, original_frame_shape: tuple) -> np.ndarray | None:
    """Processa a saída do MiDaS (Transformers DPT) para obter um mapa de profundidade."""
    if output is None or not hasattr(output, 'predicted_depth'):
        logger.debug("Pós-processamento MiDaS pulado (output nulo ou sem 'predicted_depth').")
        return None
    try:
        # Processamento dentro de torch.no_grad() para economizar memória/tempo
        with torch.no_grad():
            # A saída do DPTForDepthEstimation está em `output.predicted_depth`
            # Precisa ser redimensionada para o tamanho da imagem original
            # A saída é geralmente [1, H_out, W_out], então usamos unsqueeze(1) se for [H_out, W_out]
            depth_tensor = output.predicted_depth
            if depth_tensor.ndim == 2:
                 depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0) # Adiciona Batch e Channel se necessário
            elif depth_tensor.ndim == 3:
                 depth_tensor = depth_tensor.unsqueeze(1) # Adiciona Channel

            # Redimensiona para o tamanho original (altura, largura)
            original_height, original_width = original_frame_shape[:2]
            prediction = torch.nn.functional.interpolate(
                depth_tensor,
                size=(original_height, original_width),
                mode="bicubic", # Bicubic costuma dar resultados mais suaves
                align_corners=False, # Importante para certos modos de interpolação
            )

            # Remove dimensões extras (Batch e Channel) e move para CPU como numpy array
            depth_map_DPT = prediction.squeeze().cpu().numpy()

        # logger.debug(f"Mapa de profundidade MiDaS gerado com shape: {depth_map_ DPT.shape}")
        # Valores DPT: Maiores indicam MAIS longe.
        return depth_map_DPT
    except Exception as e:
        logger.error(f"Erro no pós-processamento MiDaS (Transformers): {e}", exc_info=False)
        return None


def get_midas_distance_at_point_or_bbox(depth_map: np.ndarray | None, box: list | np.ndarray = None, point: tuple = None) -> float | None:
    """
    Obtém o valor de profundidade 'raw' do MiDaS DPT (MAIOR = LONGE)
    para o centro de uma bounding box ou um ponto específico.
    Retorna o valor 'raw' que depois precisa ser convertido para metros.
    """
    if depth_map is None:
        # logger.debug("Cálculo de distância MiDaS pulado (depth_map nulo).")
        return None

    h, w = depth_map.shape[:2] # Aceita mapas 2D ou 3D (pega só H e W)

    try:
        target_x, target_y = -1, -1

        if point:
            x, y = map(int, point)
            if 0 <= y < h and 0 <= x < w:
                target_x, target_y = x, y
            else:
                logger.warning(f"Ponto ({x},{y}) fora dos limites do mapa de profundidade ({w}x{h}).")
                return None # Ponto fora da imagem

        elif box is not None and len(box) == 4:
            # Converte box para int e garante limites
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2) # Usa w-1 e h-1 como índices máximos

            # Verifica se a bbox é válida após o clipping
            if x1 >= x2 or y1 >= y2:
                logger.warning(f"Bounding box inválida após clipping: [{x1},{y1},{x2},{y2}]")
                return None

            # Calcula o centro da bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            target_x, target_y = center_x, center_y
            # logger.debug(f"Centro da bbox {box} -> ({target_x}, {target_y})")

        else:
            # Se nem ponto nem box forem fornecidos, retorna None (evita pegar centro da imagem por padrão)
            # logger.debug("Nenhum ponto ou bbox fornecido para buscar profundidade.")
            # Alternativa: Poderia retornar profundidade do centro da imagem, mas é menos útil
            # center_y, center_x = h // 2, w // 2
            # target_x, target_y = center_x, center_y
            return None

        # Se temos um alvo válido, pega o valor de profundidade
        if target_x != -1 and target_y != -1:
             midas_raw_value = float(depth_map[target_y, target_x])
             # logger.debug(f"Valor raw MiDaS em ({target_x},{target_y}): {midas_raw_value:.4f}")
             return midas_raw_value
        else:
             # logger.warning(f"Não foi possível determinar ponto alvo no mapa {w}x{h} para box={box}, point={point}")
             return None

    except IndexError:
         # Este erro pode ocorrer se target_x/target_y forem calculados incorretamente
         logger.error(f"Índice fora dos limites ao acessar depth_map[{target_y}, {target_x}] (shape: {h}x{w}).", exc_info=False)
         return None
    except Exception as e:
         logger.error(f"Erro inesperado ao obter valor raw MiDaS: {e}", exc_info=True)
         return None

# --- Conversão MiDaS -> Metros ---
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!! MUITO IMPORTANTE: ESTA FUNÇÃO PRECISA SER CALIBRADA EXPERIMENTALMENTE !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Modelos MiDaS/DPT retornam profundidade relativa (geralmente maior = mais longe).
# Esta função tenta mapear esse valor 'raw' para metros, mas a relação DEPENDE MUITO
# do modelo específico, da câmera, da iluminação e da cena.
# A implementação abaixo é um CHUTE / PLACEHOLDER e provavelmente estará INCORRETA.
MIN_DIST_M = 0.2 # Distância mínima em metros a ser considerada (ex: 20cm)
MAX_DIST_M = 25.0 # Distância máxima em metros a ser considerada (ex: 25m)

def midas_depth_to_meters(midas_raw_value: float | None, depth_map: np.ndarray | None = None) -> float | None:
    """
    TENTA converter o valor 'raw' do mapa de profundidade MiDaS DPT para metros.
    !!! REQUER CALIBRAÇÃO URGENTE PARA FUNCIONAR CORRETAMENTE !!!
    Retorna distância estimada em metros ou None.
    """
    if midas_raw_value is None or np.isnan(midas_raw_value) or np.isinf(midas_raw_value):
        # logger.debug(f"Conversão para metros pulada (valor raw inválido: {midas_raw_value}).")
        return None

    # --- ESTRATÉGIA DE CALIBRAÇÃO (Exemplo Placeholder - Linear Invertido Simples) ---
    # Esta é uma das *muitas* formas possíveis, e os valores são CHUTADOS.
    # Você precisa medir distâncias REAIS e ver quais valores RAW o MiDaS retorna
    # para esses pontos na sua configuração e então ajustar a fórmula.

    # Valores RAW esperados (MAIOR = LONGE para DPT). Chutes:
    RAW_VAL_AT_MIN_DIST = 2.0  # Qual valor raw o MiDaS dá para ~MIN_DIST_M (0.2m)? CHUTE!
    RAW_VAL_AT_MAX_DIST = 15.0 # Qual valor raw o MiDaS dá para ~MAX_DIST_M (25m)? CHUTE!

    # 1. Normalização Simples (Mapeia raw para 0-1)
    #    Evita divisão por zero se RAW_VAL_AT_MAX_DIST == RAW_VAL_AT_MIN_DIST
    if RAW_VAL_AT_MAX_DIST <= RAW_VAL_AT_MIN_DIST:
        # Se os valores de calibração são inválidos ou iguais, retorna uma média? Ou None?
        # logger.warning("Valores de calibração MiDaS inválidos (max <= min).")
        # Talvez retornar a distância mínima?
        return MIN_DIST_M # Ou None

    normalized_depth = (midas_raw_value - RAW_VAL_AT_MIN_DIST) / (RAW_VAL_AT_MAX_DIST - RAW_VAL_AT_MIN_DIST)

    # 2. Clipping da Normalização (Garante que fique entre 0 e 1)
    normalized_depth = np.clip(normalized_depth, 0.0, 1.0)

    # 3. Mapeamento Linear para Metros
    #    normalized_depth = 0   => meters = MIN_DIST_M
    #    normalized_depth = 1   => meters = MAX_DIST_M
    estimated_meters = MIN_DIST_M + normalized_depth * (MAX_DIST_M - MIN_DIST_M)

    # 4. Retorna o valor final, clipando novamente só por segurança
    final_meters = float(np.clip(estimated_meters, MIN_DIST_M, MAX_DIST_M))
    # logger.debug(f"Raw MiDaS {midas_raw_value:.2f} -> Norm {normalized_depth:.2f} -> Est Meters {final_meters:.2f}")

    # Alternativa: Relação Inversa (se MAIOR fosse PERTO, como em MiDaS v2.1)
    # scale = 500.0 # Fator de escala (CALIBRAR)
    # shift = 10.0  # Offset (CALIBRAR)
    # estimated_meters = scale / (midas_raw_value + shift)

    # Alternativa: Relação de Potência (Flexível, mas precisa calibrar K e EXP)
    # k_scale = 0.5   # CALIBRAR
    # exp_factor = 1.1 # CALIBRAR (positivo se raw maior = longe)
    # estimated_meters = k_scale * (midas_raw_value ** exp_factor)


    # LOG IMPORTANTE: Avisa que a calibração é necessária se ainda não foi feita
    # Poderia ter uma flag global `IS_CALIBRATED = False`
    # if not IS_CALIBRATED:
    #     logger.warning("Função midas_depth_to_meters está usando valores PLACEHOLDER! CALIBRAÇÃO é necessária.")

    return final_meters


# --- Classe Principal AudioLoop ---

# --- Classe Principal AudioLoop ---

class AudioLoop:
    # ADICIONADO: gemini_client, genai_module, glm_module aos parâmetros
    def __init__(self, video_mode=DEFAULT_VIDEO_MODE, user_name=DEFAULT_USER_NAME, device='cpu', gemini_client=None, genai_module=None, glm_module=None):
        """Inicializa o sistema do assistente."""
        self.video_mode = video_mode
        self.user_name = user_name
        self.device = device
        # ADICIONADO: Armazena as referências passadas como atributos da instância
        self.gemini_client = gemini_client
        self.genai = genai_module
        self.glm = glm_module
        # --- FIM DAS LINHAS ADICIONADAS/MODIFICADAS ---

        logger.info(f"Inicializando AudioLoop: Modo Vídeo='{self.video_mode}', User='{self.user_name}', Device='{self.device}'")
        # Log de verificação
        if self.gemini_client:
            logger.info(f"AudioLoop recebeu instância do cliente Gemini.")
        else:
            logger.warning("AudioLoop NÃO recebeu instância do cliente Gemini.")

        # --- Async Queues ---
        # (O restante do __init__ continua igual)

        # --- Async Queues ---
        # Essas filas são a espinha dorsal da comunicação entre as tasks assíncronas

        # Filas para Comunicação com Gemini (bidirecional)
        self.gemini_audio_in_queue = asyncio.Queue(maxsize=50)   # Áudio Recebido do Gemini -> Para tocar
        self.gemini_text_in_queue = asyncio.Queue()      # Texto Recebido do Gemini -> Para logar/exibir
        self.mic_audio_out_queue = asyncio.Queue(maxsize=20) # Áudio Capturado do Mic -> Para enviar ao Gemini
        self.video_frame_out_queue = asyncio.Queue(maxsize=5) # Frames de Vídeo (Encoded) -> Para enviar ao Gemini
        self.context_injection_queue = asyncio.Queue(maxsize=10) # String de Contexto Formatada -> Para enviar ao Gemini
        self.text_command_out_queue = asyncio.Queue(maxsize=5) # Comandos de Texto do Usuário -> Para enviar ao Gemini


        # Filas para distribuição de frames para Processadores de IA (frames BGR crus)
        self.processor_frame_queues = {
            "yolo": asyncio.Queue(maxsize=3),
            "sam2": asyncio.Queue(maxsize=3),
            "midas": asyncio.Queue(maxsize=3),
            "deepface": asyncio.Queue(maxsize=3),
            "cloud_vision": asyncio.Queue(maxsize=3) # OCR pode ser menos frequente
        }
        self._active_processors = {k: False for k in self.processor_frame_queues} # Flags para saber quem está ativo

        # Filas para coletar Resultados dos Processadores de IA
        self.results_queues = {
            "yolo": asyncio.Queue(),
            "sam2": asyncio.Queue(),
            "midas": asyncio.Queue(),
            "deepface": asyncio.Queue(),
            "cloud_vision": asyncio.Queue()
        }

        # --- Estado da Aplicação ---
        self.chat_session = None # Armazena a sessão de chat ativa com Gemini (se start_chat for usado)
        self.audio_input_stream = None # Stream PyAudio para captura de microfone
        self.audio_output_stream = None # Stream PyAudio para playback de áudio Gemini
        self.last_context_update_time = time.monotonic() # Controle de frequência de envio de contexto
        self.last_processed_frame_time = time.monotonic() # Controle de FPS do processamento de IA
        self.frame_drop_counters = {k: 0 for k in self.processor_frame_queues} # Contadores de frames descartados por fila cheia
        self.shutdown_event = asyncio.Event() # Evento para sinalizar encerramento para as tasks

        # Dicionário para guardar referências das tasks asyncio criadas
        self.tasks = {}

        # Registra os processadores que foram carregados com sucesso
        self._register_active_processors()

    def _register_active_processors(self):
        """Verifica quais modelos foram carregados e marca as tasks como ativas."""
        if yolo_model: self._active_processors["yolo"] = True
        if sam_predictor: self._active_processors["sam2"] = True
        if midas_model: self._active_processors["midas"] = True
        if DeepFace: self._active_processors["deepface"] = True # Assume ativo se lib existe e dir está ok
        if cloud_vision_client: self._active_processors["cloud_vision"] = True
        active_list = [name for name, active in self._active_processors.items() if active]
        logger.info(f"Processadores de IA ativos: {active_list if active_list else 'Nenhum'}")

    async def _encode_frame_for_gemini(self, frame: np.ndarray) -> dict | None:
        """Converte frame numpy BGR para formato blob (bytes JPEG) para Gemini API v1.5+."""
        if frame is None:
            return None
        try:
            # Codificar para JPEG em memória (formato eficiente)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85] # Qualidade JPEG (0-100)
            result, encoded_jpeg = cv2.imencode(".jpg", frame, encode_param)

            if not result:
                logger.warning("Falha ao codificar frame para JPEG.")
                return None

            # API mais nova espera Blob: { mime_type, data: bytes }
            mime_type = "image/jpeg"
            image_bytes = encoded_jpeg.tobytes()

            # Verifica tamanho (Gemini pode ter limites)
            # logger.debug(f"Frame JPEG encoded: {len(image_bytes) / 1024:.1f} KB")
            if len(image_bytes) > 4 * 1024 * 1024: # Exemplo de limite (4MB) - VERIFICAR DOCS!
                 logger.warning(f"Frame JPEG ({len(image_bytes) / 1024:.1f} KB) excede limite, descartando.")
                 # Poderia tentar reduzir qualidade ou tamanho aqui
                 return None

            # Retorna no formato esperado pela API (verificar documentação se mudou)
            # O 'data' é diretamente os bytes da imagem.
            return {"mime_type": mime_type, "data": image_bytes}

        except cv2.error as cv_err:
             logger.error(f"Erro OpenCV ao codificar frame para Gemini: {cv_err}")
             return None
        except Exception as e:
            logger.error(f"Erro inesperado ao codificar frame para Gemini: {e}", exc_info=True)
            return None

    async def _queue_put_robust(self, queue: asyncio.Queue, item, queue_name: str, drop_oldest_if_full: bool = True):
        """ Tenta colocar item na fila, opcionalmente descartando o mais antigo se cheia. """
        try:
            if drop_oldest_if_full and queue.full():
                try:
                    old_item = queue.get_nowait()
                    queue.task_done() # Marca o antigo como "feito" para não bloquear join()s
                    logger.warning(f"Fila '{queue_name}' estava cheia. Item mais antigo descartado.")
                except asyncio.QueueEmpty:
                    pass # Fila esvaziou entre a checagem e o get, ignora
            await queue.put(item)
            return True
        except asyncio.QueueFull:
            # Só ocorre se drop_oldest_if_full=False e a fila encher
            logger.error(f"Fila '{queue_name}' cheia e não foi possível descartar item. Item atual perdido.")
            return False
        except Exception as e:
            logger.error(f"Erro inesperado ao colocar item na fila '{queue_name}': {e}")
            return False


    # --------------------------------------------------------------------------
    # Tasks de Processamento de Frames de IA
    # --------------------------------------------------------------------------

    async def process_yolo(self):
        """Task para processar frames com YOLO."""
        if not self._active_processors["yolo"]:
             logger.warning("Task YOLO não iniciada (modelo não carregado/ativo).")
             return
        logger.info("YOLO Processor Task iniciada.")
        queue = self.processor_frame_queues["yolo"]
        results_queue = self.results_queues["yolo"]

        while not self.shutdown_event.is_set():
            try:
                # Espera por um frame na fila específica do YOLO
                frame_data = await queue.get()
                timestamp = frame_data['timestamp']
                frame_bgr = frame_data['frame'] # Frame BGR Numpy

                # Confirma se o modelo ainda está carregado (poderia ter falhado depois)
                if yolo_model is None:
                    logger.warning("YOLO modelo tornou-se None, pulando frame.")
                    queue.task_done()
                    await asyncio.sleep(5) # Espera antes de tentar de novo
                    continue

                # Executa a inferência YOLO. `to_thread` pode ser usado se for pesado em CPU
                # mas YOLO com GPU costuma liberar o event loop rapidamente. Testar.
                # results = await asyncio.to_thread(yolo_model, frame_bgr, device=self.device, verbose=False)
                yolo_start_time = time.monotonic()
                results = yolo_model(frame_bgr, device=self.device, verbose=False, conf=0.3) # Conf thresh na chamada
                yolo_elapsed = time.monotonic() - yolo_start_time
                # logger.debug(f"YOLO inference took {yolo_elapsed:.3f}s")


                detections = []
                # Processa os resultados (estrutura pode variar ligeiramente entre versões do ultralytics)
                if results and isinstance(results, list) and len(results) > 0:
                    res = results[0] # Primeiro resultado geralmente contém as detecções
                    if res.boxes: # Verifica se há bounding boxes detectadas
                        boxes = res.boxes.xyxy.cpu().numpy() # Formato [x1, y1, x2, y2]
                        confs = res.boxes.conf.cpu().numpy() # Confianças
                        clss = res.boxes.cls.cpu().numpy() # Classes (índices)
                        names = res.names # Dicionário {index: label_str}

                        for i in range(len(boxes)):
                            detections.append({
                                "label": names[int(clss[i])],
                                "confidence": float(confs[i]),
                                "box": boxes[i].tolist() # Converte para lista python
                            })

                if detections:
                    # logger.debug(f"YOLO Detectou: {[(d['label'], f\"{d['confidence']:.2f}\") for d in detections]}")
                    # Envia resultado para fila de resultados
                    await self._queue_put_robust(results_queue,
                                                 {"timestamp": timestamp, "detections": detections},
                                                 "YOLO Results")
                # else: logger.debug("YOLO não detectou objetos neste frame.")


                # Sinaliza que o item foi processado na fila de entrada
                queue.task_done()

                # Pequeno sleep para ceder controle, mesmo que a task seja rápida
                # await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                logger.info("YOLO Processor Task cancelada.")
                break # Sai do loop while
            except Exception as e:
                logger.error(f"Erro crítico no YOLO Processor: {e}", exc_info=True)
                # Tentar consumir o item da fila para não bloquear em caso de erro persistente
                if not queue.empty():
                     try: queue.get_nowait(); queue.task_done()
                     except asyncio.QueueEmpty: pass
                await asyncio.sleep(2) # Espera antes de tentar próximo frame
        logger.info("YOLO Processor Task finalizada.")

    async def process_sam2(self):
        """Task para processar frames com SAM2 (Segment Anything). **USA SIMULAÇÃO!**"""
        if not self._active_processors["sam2"]:
            logger.warning("Task SAM2 não iniciada (predictor não carregado/ativo).")
            return
        logger.info("SAM2 Processor Task iniciada (usando placeholder/simulação).")
        queue = self.processor_frame_queues["sam2"]
        results_queue = self.results_queues["sam2"]

        while not self.shutdown_event.is_set():
            try:
                frame_data = await queue.get()
                timestamp = frame_data['timestamp']
                frame_bgr = frame_data['frame']

                if sam_predictor is None: # Checa novamente
                     logger.warning("SAM2 predictor tornou-se None, pulando frame.")
                     queue.task_done()
                     await asyncio.sleep(5)
                     continue

                sam_start_time = time.monotonic()
                # SAM geralmente requer frame RGB
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # --- Lógica de Inferência SAM2 ---
                # A implementação REAL dependeria do que se quer:
                # 1. Segmentar TUDO: Usaria `SamAutomaticMaskGenerator(sam_predictor.model).generate(frame_rgb)` (Pode ser LENTO!)
                # 2. Segmentar com Prompts: `sam_predictor.set_image(frame_rgb)` e depois `sam_predictor.predict(point_coords=..., point_labels=..., box=..., multimask_output=...)`
                #    Os prompts (pontos, caixas) poderiam vir do YOLO ou de outra fonte.

                # --- PLACEHOLDER ATUAL: Simular resultado de segmentação ---
                # Função simulada que retorna uma estrutura parecida com a real
                def segment_simulation():
                    # Simula algumas máscaras com dados básicos
                    height, width = frame_rgb.shape[:2]
                    masks_data = []
                    num_masks = np.random.randint(2, 6) # Simula de 2 a 5 máscaras
                    for i in range(num_masks):
                         # Gera uma bbox aleatória como base para a "máscara"
                         x1 = np.random.randint(0, width // 2)
                         y1 = np.random.randint(0, height // 2)
                         x2 = np.random.randint(x1 + width // 10, width -1)
                         y2 = np.random.randint(y1 + height // 10, height -1)
                         area = (x2 - x1) * (y2 - y1)
                         # Dados que `SamAutomaticMaskGenerator` costuma retornar:
                         masks_data.append({
                             # "segmentation": [[x1,y1, x2,y1, x2,y2, x1,y2]], # Contorno (poderia ser mais complexo)
                             "area": area,
                             "predicted_iou": np.random.uniform(0.75, 0.98), # IOU simulado
                             "stability_score": np.random.uniform(0.8, 0.99), # Estabilidade simulada
                             "bbox": [x1, y1, x2, y2], # Bbox da máscara
                             # "point_coords": [[x1+(x2-x1)//2, y1+(y2-y1)//2]], # Ponto central
                         })
                    # logger.debug(f"SAM2 (simulado) gerou {len(masks_data)} máscaras.")
                    time.sleep(0.05 + np.random.rand() * 0.1) # Simula tempo de processamento
                    return masks_data

                # Roda a simulação em thread separada para não bloquear (mesmo sendo simulação)
                masks_info = await asyncio.to_thread(segment_simulation)
                sam_elapsed = time.monotonic() - sam_start_time
                # logger.debug(f"SAM2 (sim) inference took {sam_elapsed:.3f}s")

                if masks_info:
                     # Formatar resultado para a fila (sumário simplificado)
                     processed_segments = []
                     # Ordena por área simulada (maior primeiro)
                     masks_info.sort(key=lambda x: x.get('area', 0), reverse=True)
                     for i, mask in enumerate(masks_info[:5]): # Pega as Top 5 maiores
                         area = mask.get('area', 0)
                         iou = mask.get('predicted_iou', 0.0)
                         # Simplifica a info - talvez só área? Ou só IOU? Ou uma label genérica?
                         # A IA final vai receber este resumo. Como tornar útil?
                         # Ex: "ObjGrande(iou=0.95)"
                         size_label = "Desconhecido"
                         if area > (frame_rgb.shape[0]*frame_rgb.shape[1]*0.3): size_label="Enorme"
                         elif area > (frame_rgb.shape[0]*frame_rgb.shape[1]*0.1): size_label="Grande"
                         elif area > (frame_rgb.shape[0]*frame_rgb.shape[1]*0.02): size_label="Medio"
                         else: size_label="Pequeno"

                         # Usar 'Seg' (Segmento) ou 'Obj' (Objeto)? 'Seg' é mais neutro.
                         processed_segments.append(f"Seg{i+1}[{size_label},IoU:{iou:.2f}]")

                     if processed_segments:
                         await self._queue_put_robust(results_queue, {
                             "timestamp": timestamp,
                             "segmentation_summary": ", ".join(processed_segments), # Envia sumário textual
                             "raw_masks_count": len(masks_info) # Envia contagem total
                             # "masks_data": masks_info # Poderia enviar dados completos (mas seria GRANDE)
                         }, "SAM2 Results")

                queue.task_done()
                # SAM (real) pode ser pesado, dar uma pausa? Ou deixar o scheduler decidir?
                # await asyncio.sleep(0.05) # Pequena pausa opcional

            except asyncio.CancelledError:
                logger.info("SAM2 Processor Task cancelada.")
                break
            except Exception as e:
                logger.error(f"Erro crítico no SAM2 Processor: {e}", exc_info=True)
                if not queue.empty():
                    try: queue.get_nowait(); queue.task_done()
                    except asyncio.QueueEmpty: pass
                await asyncio.sleep(2)
        logger.info("SAM2 Processor Task finalizada.")

    async def process_midas(self):
        """Task para processar frames com MiDaS (DPT) e gerar mapa de profundidade."""
        if not self._active_processors["midas"]:
             logger.warning("Task MiDaS não iniciada (modelo/processador não carregado/ativo).")
             return
        logger.info("MiDaS Processor Task iniciada.")
        queue = self.processor_frame_queues["midas"]
        results_queue = self.results_queues["midas"]

        while not self.shutdown_event.is_set():
            try:
                frame_data = await queue.get()
                timestamp = frame_data['timestamp']
                frame_bgr = frame_data['frame'] # Frame BGR Numpy

                if midas_model is None or midas_processor is None:
                    logger.warning("MiDaS modelo/proc tornou-se None, pulando frame.")
                    queue.task_done()
                    await asyncio.sleep(5)
                    continue

                midas_start_time = time.monotonic()
                # Função de inferência MiDaS (pode ser pesada, especialmente em CPU)
                # Roda em thread separada para não bloquear o loop de eventos asyncio
                def midas_inference_thread():
                    inputs = preprocess_frame_for_midas_transformers(frame_bgr, midas_processor, self.device)
                    if inputs is None: return None

                    with torch.no_grad(): # Essencial para inferência
                        outputs = midas_model(inputs) # Executa o modelo DPT

                    # Processa a saída para obter o mapa de profundidade numpy
                    depth_map = get_depth_from_midas_output_transformers(outputs, frame_bgr.shape)
                    return depth_map

                depth_map_result = await asyncio.to_thread(midas_inference_thread)
                midas_elapsed = time.monotonic() - midas_start_time
                # logger.debug(f"MiDaS inference took {midas_elapsed:.3f}s")


                if depth_map_result is not None:
                    # logger.debug(f"MiDaS gerou mapa de profundidade com shape {depth_map_result.shape}")
                    await self._queue_put_robust(results_queue,
                                                 {"timestamp": timestamp, "depth_map": depth_map_result},
                                                 "MiDaS Results")
                # else: logger.warning("Falha ao gerar mapa de profundidade MiDaS para este frame.")

                queue.task_done()
                # await asyncio.sleep(0.02) # Pausa opcional

            except asyncio.CancelledError:
                logger.info("MiDaS Processor Task cancelada.")
                break
            except Exception as e:
                logger.error(f"Erro crítico no MiDaS Processor: {e}", exc_info=True)
                if not queue.empty():
                    try: queue.get_nowait(); queue.task_done()
                    except asyncio.QueueEmpty: pass
                await asyncio.sleep(2)
        logger.info("MiDaS Processor Task finalizada.")


    async def process_deepface(self):
        """Task para detectar e reconhecer faces com DeepFace."""
        if not self._active_processors["deepface"]:
             logger.warning("Task DeepFace não iniciada (biblioteca não carregada ou dir não definido).")
             return
        logger.info("DeepFace Processor Task iniciada.")
        queue = self.processor_frame_queues["deepface"]
        results_queue = self.results_queues["deepface"]

        # Configurações DeepFace (podem vir de args/config)
        model_name = "VGG-Face" # Outros: "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"
        distance_metric = "cosine" # Ou "euclidean", "euclidean_l2"
        # Backend de detecção: 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe', 'yolov8', 'yunet', 'centerface'
        detector_backend = 'retinaface' # 'retinaface' costuma ser bom, 'mediapipe' é leve, 'yolov8' se YOLO já estiver carregado?
        similarity_threshold = 0.60 # Limiar para considerar um match (1.0 - distance > threshold). Ajustar experimentalmente!

        while not self.shutdown_event.is_set():
            try:
                frame_data = await queue.get()
                timestamp = frame_data['timestamp']
                frame_bgr = frame_data['frame']

                # Função de reconhecimento DeepFace (é BLOQUEANTE - CPU/GPU intensive)
                # Deve rodar em thread separada
                deepface_start_time = time.monotonic()
                def find_faces_thread():
                    try:
                        # DeepFace `find` pode usar BGR ou RGB dependendo do backend.
                        # Passar BGR costuma ser seguro, a lib deve converter se necessário.
                        # db_path usa KNOWN_FACES_DIR global.
                        # enforce_detection=False: retorna mesmo se não achar *nenhum* rosto
                        # align=True: alinha os rostos antes de extrair features (recomendado)
                        # silent=True: suprime logs internos do DeepFace
                        dfs_results = DeepFace.find(img_path=frame_bgr.copy(), # Passa cópia para segurança
                                             db_path=KNOWN_FACES_DIR,
                                             model_name=model_name,
                                             distance_metric=distance_metric,
                                             detector_backend=detector_backend,
                                             enforce_detection=False, # NÃO falha se não achar rosto
                                             align=True,
                                             silent=True)

                        recognized_list = []
                        # dfs_results é uma LISTA de DataFrames. Cada DF corresponde a um rosto detectado na img_path.
                        # Cada linha no DF é um match potencial da DB.
                        if dfs_results and isinstance(dfs_results, list):
                            for df in dfs_results:
                                if not df.empty:
                                    # Ordena matches para este rosto pela distância (menor é melhor)
                                    df = df.sort_values(by="distance", ascending=True)

                                    # Pega o MELHOR match (primeira linha após ordenar)
                                    best_match = df.iloc[0]
                                    identity_path = best_match['identity']
                                    distance = best_match['distance']
                                    similarity = max(0.0, 1.0 - distance) # Similaridade (0 a 1)

                                    # Extrai a BBox do rosto detectado (source_x/y/w/h)
                                    source_x = best_match.get('source_x', 0)
                                    source_y = best_match.get('source_y', 0)
                                    source_w = best_match.get('source_w', 1) # Evita w/h=0
                                    source_h = best_match.get('source_h', 1)
                                    box_xyxy = [source_x, source_y, source_x + source_w, source_y + source_h]

                                    face_info = {"box_xyxy": box_xyxy, "similarity": float(similarity)}

                                    # Verifica se o melhor match supera o limiar de similaridade
                                    if similarity >= similarity_threshold:
                                        # Tenta extrair nome da pasta do arquivo de identidade
                                        try:
                                            # Assume estrutura .../db_path/NomePessoa/img.jpg
                                            # Pega o nome do diretório pai do arquivo de identidade
                                            name = os.path.basename(os.path.dirname(identity_path))
                                            # Evita nomes genéricos que o DeepFace possa usar internamente
                                            if "representations_" in name or not name:
                                                name = "Desconhecido"
                                        except Exception:
                                            name = "Desconhecido"
                                        face_info["name"] = name
                                        # logger.debug(f"Face reconhecida: {name} (Sim: {similarity:.2f})")
                                    else:
                                        # Match abaixo do limiar é considerado 'Desconhecido'
                                        face_info["name"] = "Desconhecido"
                                        # logger.debug(f"Face detectada mas não reconhecida (Sim: {similarity:.2f} < {similarity_threshold})")

                                    recognized_list.append(face_info)
                                # else: logger.debug("DataFrame vazio encontrado em resultados DeepFace.") # Rosto detectado, mas nenhum match na DB?

                        # logger.debug(f"DeepFace thread encontrou {len(recognized_list)} rostos (reconhecidos ou não).")
                        return recognized_list

                    except ValueError as ve:
                        # ValueError específico que DeepFace levanta se não detectar NENHUM rosto
                        if "Face Detector could not find any face" in str(ve):
                             # logger.debug("DeepFace não detectou nenhum rosto no frame.")
                             return [] # Retorna lista vazia é normal neste caso
                        else:
                            # Outro ValueError inesperado
                            logger.error(f"Erro de Valor inesperado no DeepFace.find: {ve}", exc_info=False)
                            return []
                    except Exception as find_err:
                         # Captura qualquer outra exceção durante o DeepFace.find
                         logger.error(f"Erro crítico dentro da thread DeepFace.find: {find_err}", exc_info=True)
                         return []

                # Executa a função bloqueante em uma thread separada
                recognized_faces_result = await asyncio.to_thread(find_faces_thread)
                deepface_elapsed = time.monotonic() - deepface_start_time
                # logger.debug(f"DeepFace find+process took {deepface_elapsed:.3f}s")

                # Se encontrou algum rosto (reconhecido ou não), envia para fila de resultados
                if recognized_faces_result:
                    await self._queue_put_robust(results_queue,
                                                 {"timestamp": timestamp, "faces": recognized_faces_result},
                                                 "DeepFace Results")

                queue.task_done()
                # DeepFace pode ser lento, dar pausa maior?
                # await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                logger.info("DeepFace Processor Task cancelada.")
                break
            except Exception as e:
                logger.error(f"Erro crítico no DeepFace Processor: {e}", exc_info=True)
                if not queue.empty():
                     try: queue.get_nowait(); queue.task_done()
                     except asyncio.QueueEmpty: pass
                await asyncio.sleep(2)
        logger.info("DeepFace Processor Task finalizada.")


    async def process_cloud_vision(self):
        """Task para fazer OCR usando Google Cloud Vision API."""
        if not self._active_processors["cloud_vision"]:
             logger.warning("Task Cloud Vision não iniciada (cliente não carregado/ativo).")
             return
        logger.info("Cloud Vision Processor Task iniciada.")
        queue = self.processor_frame_queues["cloud_vision"]
        results_queue = self.results_queues["cloud_vision"]
        # Reimporta aqui para garantir que `vision` esteja disponível se o import inicial falhou
        try:
            from google.cloud import vision as gcp_vision
        except ImportError:
             logger.error("Falha ao reimportar google.cloud.vision dentro da task. Encerrando task.")
             return

        while not self.shutdown_event.is_set():
            try:
                frame_data = await queue.get()
                timestamp = frame_data['timestamp']
                frame_bgr = frame_data['frame']

                if cloud_vision_client is None: # Verifica se o cliente global ainda existe
                    logger.warning("Google Vision cliente tornou-se None, pulando frame.")
                    queue.task_done()
                    await asyncio.sleep(10) # Espera bastante antes de tentar de novo
                    continue

                ocr_start_time = time.monotonic()
                # Prepara imagem para a API (bytes JPEG)
                success, encoded_image = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90]) # Qualidade um pouco maior para OCR
                if not success:
                     logger.warning("Falha ao codificar frame para JPEG (Cloud Vision).")
                     queue.task_done()
                     continue

                image_content = encoded_image.tobytes()
                # Cria o objeto Image da API
                gcp_image = gcp_vision.Image(content=image_content)

                # Função para chamar a API Cloud Vision (BLOQUEANTE devido I/O de rede)
                # Roda em thread separada
                def detect_text_thread():
                     try:
                         # logger.debug("Chamando Google Cloud Vision API (Text Detection)...")
                         response = cloud_vision_client.text_detection(image=gcp_image)

                         # Verifica erros na resposta da API
                         if response.error.message:
                              # Loga o erro mas não necesariamente para a task
                              logger.error(f"Erro da API Google Cloud Vision: {response.error.message}")
                              return None # Retorna None para indicar falha

                         # Extrai o texto completo detectado (geralmente o primeiro annotation)
                         if response.text_annotations:
                             full_text = response.text_annotations[0].description
                             # logger.debug(f"Cloud Vision OCR detectou texto: '{full_text[:80]}...'")
                             return full_text.strip() # Retorna texto limpo
                         else:
                             # logger.debug("Cloud Vision não detectou texto neste frame.")
                             return "" # Retorna string vazia se nada foi detectado

                     except Exception as api_err:
                          logger.error(f"Erro durante chamada à API Cloud Vision (Text Detection): {api_err}", exc_info=False) # Não loga traceback completo por padrão
                          return None # Indica falha

                # Executa a chamada da API em uma thread
                detected_text_result = await asyncio.to_thread(detect_text_thread)
                ocr_elapsed = time.monotonic() - ocr_start_time
                # logger.debug(f"Google Vision OCR call took {ocr_elapsed:.3f}s")


                # Se a API retornou texto (não None e não vazio)
                if detected_text_result:
                     await self._queue_put_robust(results_queue,
                                                 {"timestamp": timestamp, "ocr_text": detected_text_result},
                                                 "CloudVision Results")

                # Se retornou None, indica que houve erro na API
                #elif detected_text_result is None:
                     # logger.warning("Chamada OCR falhou ou API retornou erro.")

                queue.task_done()
                # API tem custo e limites, ESPAÇAR as chamadas é CRUCIAL
                await asyncio.sleep(2.0) # <<< Intervalo de segurança - AJUSTAR CONFORME USO/CUSTO

            except asyncio.CancelledError:
                logger.info("Cloud Vision Processor Task cancelada.")
                break
            except Exception as e:
                logger.error(f"Erro crítico no Cloud Vision Processor: {e}", exc_info=True)
                if not queue.empty():
                    try: queue.get_nowait(); queue.task_done()
                    except asyncio.QueueEmpty: pass
                await asyncio.sleep(5) # Espera mais em caso de erro
        logger.info("Cloud Vision Processor Task finalizada.")

    # --------------------------------------------------------------------------
    # Task de Síntese de Contexto
    # --------------------------------------------------------------------------

    async def context_synthesizer(self):
        """
        Coleta periodicamente os resultados mais recentes de cada processador de IA,
        formata uma string de contexto consolidada e a envia para Gemini.
        """
        logger.info("Context Synthesizer Task iniciada.")
        # Guarda o último resultado válido recebido de cada processador
        last_valid_results = {k: None for k in self.results_queues}
        # Guarda o último mapa de profundidade válido para cálculos de distância
        latest_depth_map = None

        # Função helper para consumir TUDO de uma fila e retornar APENAS o último item
        async def get_latest_item_from_queue(q_name: str) -> dict | None:
            queue = self.results_queues[q_name]
            last_item = None
            try:
                # Esvazia a fila pegando todos os itens disponíveis sem bloquear
                while True:
                    item = queue.get_nowait()
                    last_item = item # Guarda o último item lido
                    queue.task_done() # Marca como processado
            except asyncio.QueueEmpty:
                # Fila vazia, retorna o último item que conseguiu pegar (ou None se estava vazia)
                # if last_item: logger.debug(f"Got latest result from {q_name}")
                return last_item
            except Exception as e_get:
                logger.error(f"Erro ao esvaziar fila '{q_name}': {e_get}")
                return last_item # Retorna o que conseguiu até o erro

        while not self.shutdown_event.is_set():
            try:
                synth_start_time = time.monotonic()

                # 1. Atualizar últimos resultados válidos e o mapa de profundidade
                data_changed_since_last_synthesis = False
                for name in self.results_queues.keys():
                    # Pega o item MAIS RECENTE da fila de resultados (descarta antigos)
                    latest_item = await get_latest_item_from_queue(name)
                    if latest_item:
                        last_valid_results[name] = latest_item
                        data_changed_since_last_synthesis = True # Marca que algo novo chegou
                        # Se for resultado do MiDaS, atualiza o mapa de profundidade mais recente
                        if name == "midas" and "depth_map" in latest_item:
                            latest_depth_map = latest_item["depth_map"]
                            # logger.debug("Mapa de profundidade MiDaS atualizado para síntese.")

                # Verifica se é hora de injetar contexto OU se dados importantes mudaram
                # (Poderia ter lógica mais complexa aqui, ex: enviar só se perigo detectado)
                current_time = time.monotonic()
                should_send_context = (current_time - self.last_context_update_time >= CONTEXT_INJECTION_INTERVAL)

                # Só monta e envia o contexto se for a hora OU se dados relevantes mudaram?
                # Por simplicidade, vamos enviar na frequência definida se houver *algo* nos dados recentes.
                if should_send_context and any(last_valid_results.values()):
                    context_parts = [] # Lista para guardar as strings de cada tipo de dado
                    # logger.debug("Montando string de contexto...")

                    # --- Processa Objetos YOLO + Distância MiDaS ---
                    yolo_res = last_valid_results["yolo"]
                    # logger.debug(f"YOLO data for context: {'Yes' if yolo_res else 'No'}, Depth map available: {'Yes' if latest_depth_map is not None else 'No'}")
                    if yolo_res and "detections" in yolo_res:
                         objects_str_parts = []
                         # Ordena por confiança (descendente) e pega os N primeiros
                         sorted_detections = sorted(yolo_res["detections"], key=lambda x: x['confidence'], reverse=True)
                         for det in sorted_detections[:7]: # Limita a 7 objetos mais confiantes
                             label = det['label']
                             conf = det['confidence']
                             box = det['box']
                             dist_str = "(dist?)" # Placeholder se não conseguir calcular distância

                             # Tenta calcular a distância usando o mapa de profundidade mais recente
                             if latest_depth_map is not None:
                                 midas_raw = get_midas_distance_at_point_or_bbox(latest_depth_map, box=box)
                                 dist_meters = midas_depth_to_meters(midas_raw, latest_depth_map) # Tenta converter para metros
                                 if dist_meters is not None:
                                      # Formato para IA: 'objeto (X.Ym)'
                                      dist_str = f"({dist_meters:.1f}m)"
                                      # logger.debug(f"  -> YOLO: {label} at {dist_str}")
                                 # else: logger.debug(f"  -> YOLO: {label} - Failed distance (raw: {midas_raw})")


                             # Monta a string para este objeto: "label(dist)" ou "label(dist?)"
                             objects_str_parts.append(f"{label}{dist_str}")

                         if objects_str_parts:
                             context_parts.append(f"Objetos: {', '.join(objects_str_parts)}")
                         # else: logger.debug("Nenhuma detecção YOLO encontrada ou processada para contexto.")
                    # else: logger.debug("Dados YOLO não disponíveis para contexto.")


                    # --- Processa Pessoas DeepFace + Distância MiDaS ---
                    face_res = last_valid_results["deepface"]
                    # logger.debug(f"DeepFace data for context: {'Yes' if face_res else 'No'}")
                    if face_res and "faces" in face_res:
                         people_str_parts = []
                         # Ordenar rostos pelo tamanho da BBox (maior primeiro) para dar prioridade aos mais próximos/maiores?
                         def bbox_area(f):
                             box = f.get('box_xyxy')
                             return (box[2]-box[0]) * (box[3]-box[1]) if box and len(box)==4 and box[2]>box[0] and box[3]>box[1] else 0

                         sorted_faces = sorted(face_res["faces"], key=bbox_area, reverse=True)

                         for face in sorted_faces[:5]: # Limita a 5 rostos (os maiores na imagem)
                             name = face.get('name', 'Desconhecido')
                             box = face.get('box_xyxy')
                             dist_str = "(dist?)"

                             if latest_depth_map is not None and box:
                                 midas_raw = get_midas_distance_at_point_or_bbox(latest_depth_map, box=box)
                                 dist_meters = midas_depth_to_meters(midas_raw, latest_depth_map)
                                 if dist_meters is not None:
                                     dist_str = f"({dist_meters:.1f}m)"
                                     # logger.debug(f"  -> Face: {name} at {dist_str}")
                                 # else: logger.debug(f"  -> Face: {name} - Failed distance (raw: {midas_raw})")

                             people_str_parts.append(f"{name}{dist_str}")

                         if people_str_parts:
                             context_parts.append(f"Pessoas: {', '.join(people_str_parts)}")
                         # else: logger.debug("Nenhum rosto DeepFace encontrado ou processado para contexto.")
                    # else: logger.debug("Dados DeepFace não disponíveis para contexto.")


                    # --- Processa Texto Cloud Vision ---
                    cv_res = last_valid_results["cloud_vision"]
                    # logger.debug(f"Cloud Vision data for context: {'Yes' if cv_res else 'No'}")
                    if cv_res and "ocr_text" in cv_res and cv_res["ocr_text"]:
                         # Limpa e formata o texto OCR
                         ocr_text = cv_res["ocr_text"].replace('\n', ' ').strip()
                         ocr_text = ' '.join(ocr_text.split()) # Remove espaços múltiplos
                         if ocr_text:
                             # Envia texto truncado se for muito longo
                             max_ocr_len = 200
                             truncated_text = ocr_text[:max_ocr_len] + ('...' if len(ocr_text) > max_ocr_len else '')
                             context_parts.append(f"Texto: '{truncated_text}'")
                             # logger.debug(f"  -> OCR: '{truncated_text}'")
                         # else: logger.debug("Texto OCR vazio após limpeza.")
                    # else: logger.debug("Dados OCR não disponíveis ou vazios.")


                    # --- Processa Sumário SAM2 (Segmentação Simulada) ---
                    sam_res = last_valid_results["sam2"]
                    # logger.debug(f"SAM2 data for context: {'Yes' if sam_res else 'No'}")
                    if sam_res and "segmentation_summary" in sam_res:
                        summary = sam_res["segmentation_summary"]
                        count = sam_res.get("raw_masks_count", "?")
                        # Formato: "Segmentos: [Seg1[Grande,IoU:0.95], Seg2[Medio,IoU:0.88]] (Total: 5)"
                        context_parts.append(f"Segmentos: [{summary}] (Total: {count})")
                        # logger.debug(f"  -> SAM2 Summary: [{summary}] (Total: {count})")
                    # else: logger.debug("Dados SAM2 não disponíveis.")


                    # --- Monta string final e envia para Gemini ---
                    if context_parts: # Só envia se alguma informação foi adicionada
                        context_str = "[CONTEXTO_SISTEMA] " + ". ".join(context_parts) + "."
                        logger.info(f"Injetando contexto: {context_str}") # Log importante
                        # Envia para a fila de saída de contexto
                        await self._queue_put_robust(self.context_injection_queue, context_str, "Context Injection")
                        self.last_context_update_time = current_time # Atualiza o tempo do último envio
                    # else: logger.debug("Nenhuma parte de contexto gerada, nada a enviar.")

                # else:
                    # if not should_send_context: logger.debug("Ainda não é hora de enviar contexto.")
                    # if not any(last_valid_results.values()): logger.debug("Nenhum dado válido recente para contexto.")

                # Espera um intervalo antes da próxima tentativa de síntese
                # O tempo de espera deve ser menor que CONTEXT_INJECTION_INTERVAL
                # para permitir checagens mais frequentes dos dados, mas não tão baixo
                # a ponto de causar busy-waiting.
                synth_elapsed = time.monotonic() - synth_start_time
                wait_time = max(0.1, 0.75 - synth_elapsed) # Espera ~0.75s no total por ciclo
                await asyncio.sleep(wait_time)

            except asyncio.CancelledError:
                 logger.info("Context Synthesizer Task cancelada.")
                 break
            except Exception as e:
                 logger.error(f"Erro crítico no Context Synthesizer: {e}", exc_info=True)
                 await asyncio.sleep(1) # Espera antes de tentar de novo
        logger.info("Context Synthesizer Task finalizada.")


    # --------------------------------------------------------------------------
    # Tasks de Captura e Distribuição (Vídeo, Áudio)
    # --------------------------------------------------------------------------

    async def capture_video(self):
        """
        Captura frames de vídeo (câmera ou tela) e distribui cópias para:
        1. A fila de envio para Gemini (codificado).
        2. As filas dos processadores de IA ativos (frame cru BGR, com throttling).
        """
        logger.info(f"Iniciando task de captura de vídeo (Modo: {self.video_mode}, Target FPS Proc: {VIDEO_FPS_TARGET})")
        cap = None # VideoCapture object
        sct = None # Screen Capture object
        monitor = None # Monitor selecionado para screen capture

        # --- Configuração Inicial da Fonte de Vídeo ---
        if self.video_mode == "camera":
            try:
                camera_index = 0 # Tentar câmera 0 primeiro
                logger.info(f"Tentando abrir câmera index {camera_index}...")
                # Usar asyncio.to_thread para chamadas bloqueantes do OpenCV
                cap = await asyncio.to_thread(cv2.VideoCapture, camera_index)
                if not cap or not cap.isOpened():
                    logger.warning(f"Falha ao abrir câmera {camera_index}. Tentando index 1...")
                    camera_index = 1
                    if cap: await asyncio.to_thread(cap.release) # Libera anterior se falhou parcialmente
                    cap = await asyncio.to_thread(cv2.VideoCapture, camera_index)

                if not cap or not cap.isOpened():
                    logger.error("Erro fatal ao abrir câmera! Tentou index 0 e 1. Verifique conexões/drivers.")
                    self.shutdown_event.set() # Sinaliza para encerrar outras tasks
                    return # Encerra a task de captura

                # Opcional: Configurar resolução e FPS desejados (se a câmera suportar)
                # await asyncio.to_thread(cap.set, cv2.CAP_PROP_FRAME_WIDTH, 640)
                # await asyncio.to_thread(cap.set, cv2.CAP_PROP_FRAME_HEIGHT, 480)
                # await asyncio.to_thread(cap.set, cv2.CAP_PROP_FPS, 30) # Pode não ter efeito

                # Logar configurações reais da câmera
                actual_width = int(await asyncio.to_thread(cap.get, cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(await asyncio.to_thread(cap.get, cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = await asyncio.to_thread(cap.get, cv2.CAP_PROP_FPS)
                logger.info(f"Câmera {camera_index} aberta com sucesso: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS (nominal)")

            except cv2.error as cv_err:
                 logger.error(f"Erro OpenCV ao abrir câmera: {cv_err}. Encerrando task de captura.", exc_info=True)
                 self.shutdown_event.set()
                 if cap: await asyncio.to_thread(cap.release)
                 return
            except Exception as e:
                 logger.error(f"Erro inesperado ao abrir câmera: {e}. Encerrando task de captura.", exc_info=True)
                 self.shutdown_event.set()
                 if cap: await asyncio.to_thread(cap.release)
                 return

        elif self.video_mode == "screen":
             try:
                 logger.info("Iniciando captura de tela com MSS...")
                 sct = mss.mss()
                 # Monitor 0 é a área total de todos os monitores
                 # Monitor 1 é geralmente o monitor primário. Testar qual funciona melhor.
                 monitor_index = 1
                 if len(sct.monitors) > monitor_index:
                      monitor = sct.monitors[monitor_index]
                      logger.info(f"Captura de tela configurada para monitor {monitor_index}: {monitor['width']}x{monitor['height']} at ({monitor['left']},{monitor['top']})")
                 else:
                     logger.error(f"Monitor {monitor_index} não encontrado! Monitores disponíveis: {sct.monitors}")
                     sct = None # Desativa captura
                     self.shutdown_event.set()
                     return
             except Exception as e:
                 logger.error(f"Erro ao iniciar captura de tela com MSS: {e}. Encerrando task.", exc_info=True)
                 sct = None
                 self.shutdown_event.set()
                 return
        else:
             logger.info("Modo de vídeo 'none'. Task de captura de vídeo não fará nada útil, encerrando.")
             return # Encerra a task se modo for 'none'

        # --- Loop Principal de Captura e Distribuição ---
        frame_capture_counter = 0
        frame_processed_counter = 0
        capture_start_time = time.monotonic()

        while not self.shutdown_event.is_set():
            try:
                frame_bgr = None
                grab_time_start = time.monotonic()

                # 1. Capturar Frame (Câmera ou Tela) - Usar to_thread para operações bloqueantes
                if cap:
                    ret, frame = await asyncio.to_thread(cap.read) # cap.read() é bloqueante
                    if ret:
                        frame_bgr = frame
                    else:
                        logger.warning("Câmera retornou 'ret=False'. Fim do vídeo ou erro.")
                        await asyncio.sleep(0.5) # Espera antes de tentar ler de novo
                        continue # Pula para próxima iteração
                elif sct and monitor:
                    # sct.grab() é relativamente rápido, mas ainda pode bloquear um pouco. to_thread é mais seguro.
                    img_raw = await asyncio.to_thread(sct.grab, monitor)
                    # Converte BGRA (MSS) para BGR (OpenCV)
                    frame_bgr = cv2.cvtColor(np.array(img_raw), cv2.COLOR_BGRA2BGR)

                grab_time_elapsed = time.monotonic() - grab_time_start

                if frame_bgr is not None:
                    frame_capture_counter += 1
                    timestamp = time.time() # Timestamp da captura

                    # Log FPS de captura a cada N frames
                    if frame_capture_counter % 100 == 0:
                         elapsed_total = time.monotonic() - capture_start_time
                         avg_fps = frame_capture_counter / elapsed_total if elapsed_total > 0 else 0
                         logger.info(f"Captura: {frame_capture_counter} frames @ {avg_fps:.1f} FPS (Proc: {frame_processed_counter}) Grab time: {grab_time_elapsed*1000:.1f}ms")


                    # 2. Enviar frame codificado para Gemini (sem throttling aqui, Gemini pode querer mais frames)
                    #    A codificação acontece em _encode_frame_for_gemini
                    gemini_frame_blob = self._encode_frame_for_gemini(frame_bgr.copy()) # Envia cópia
                    if gemini_frame_blob:
                        await self._queue_put_robust(self.video_frame_out_queue,
                                                     gemini_frame_blob,
                                                     "Gemini Video",
                                                     drop_oldest_if_full=True)


                    # 3. Distribuir frame cru (BGR) para processadores de IA (COM THROTTLING)
                    current_time = time.monotonic()
                    if current_time - self.last_processed_frame_time >= (1.0 / VIDEO_FPS_TARGET):
                        self.last_processed_frame_time = current_time
                        frame_processed_counter += 1
                        # Cria o payload UMA VEZ
                        frame_payload = {"timestamp": timestamp, "frame": frame_bgr.copy()} # Nova cópia

                        # Envia para as filas de TODOS os processadores ATIVOS
                        put_tasks = []
                        for name, is_active in self._active_processors.items():
                            if is_active:
                                queue = self.processor_frame_queues[name]
                                put_tasks.append(
                                    self._queue_put_robust(queue, frame_payload, f"{name.upper()} Frame Input", drop_oldest_if_full=True)
                                )

                        if put_tasks:
                            await asyncio.gather(*put_tasks) # Envia em paralelo (embora put seja rápido)
                        # logger.debug(f"Frame distribuído para {len(put_tasks)} processadores ativos.")
                    # else: logger.debug("Throttling: Pulando distribuição para processadores IA.")


                    # Pequeno sleep para evitar 100% CPU se a captura for muito rápida
                    # O tempo de grab já age como um sleep natural, mas adiciona um mínimo.
                    process_time_elapsed = time.monotonic() - grab_time_start
                    # Tenta manter um ciclo total próximo ao target FPS de processamento, mas não menor que 1ms
                    desired_cycle_time = 1.0 / (VIDEO_FPS_TARGET * 1.5) # Um pouco mais rápido que o target
                    sleep_duration = max(0.001, desired_cycle_time - process_time_elapsed)
                    await asyncio.sleep(sleep_duration)

                else:
                     logger.warning("Frame capturado resultou em None.")
                     await asyncio.sleep(0.1) # Espera se a captura falhou


            except asyncio.CancelledError:
                logger.info("Capture Video Task cancelada.")
                break
            except Exception as e:
                logger.error(f"Erro crítico na task Capture Video: {e}", exc_info=True)
                await asyncio.sleep(1) # Espera antes de tentar de novo

        # --- Limpeza Final ---
        logger.info("Encerrando captura de vídeo...")
        if cap:
            logger.info("Liberando recurso da câmera...")
            await asyncio.to_thread(cap.release)
            logger.info("Câmera liberada.")
        if sct:
            # MSS não tem um método close explícito, o objeto pode ser apenas deletado pelo GC
            logger.info("Recurso de captura de tela MSS será liberado.")
            sct = None

        logger.info("Capture Video Task finalizada.")

    async def listen_audio(self):
        """Lê áudio do microfone, o coloca na fila mic_audio_out_queue para ser enviado a Gemini."""
        if not pya:
             logger.warning("Task Listen Audio não iniciada - PyAudio indisponível.")
             return
        logger.info("Listen Audio Task iniciada.")

        stream = None
        try:
            # Identifica dispositivo de entrada padrão
            default_device_info = await asyncio.to_thread(pya.get_default_input_device_info)
            device_index = default_device_info["index"]
            device_name = default_device_info["name"]
            native_rate = int(default_device_info["defaultSampleRate"])
            logger.info(f"Microfone Padrão: '{device_name}' (Index: {device_index}, Rate Nativa: {native_rate} Hz)")
            # Idealmente, usaríamos a taxa nativa, mas Gemini espera 16kHz.
            # PyAudio pode fazer resampling, mas verificar se a qualidade fica boa.

            # Abre o stream de áudio
            logger.info(f"Abrindo stream de áudio do microfone (Rate: {SEND_SAMPLE_RATE} Hz)...")
            stream = await asyncio.to_thread(
                pya.open,
                format=FORMAT, # paInt16
                channels=CHANNELS, # Mono
                rate=SEND_SAMPLE_RATE, # Taxa de envio para Gemini
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK_SIZE, # Tamanho do buffer
            )
            self.audio_input_stream = stream # Guarda referência para fechar depois
            logger.info(f"Stream de áudio do microfone aberta com sucesso.")

            # Loop de leitura
            while not self.shutdown_event.is_set() and stream.is_active():
                 try:
                     # Lê dados do stream (bloqueante, por isso to_thread)
                     audio_chunk = await asyncio.to_thread(stream.read, CHUNK_SIZE, exception_on_overflow=False)

                     # Coloca o chunk de áudio na fila de saída para Gemini
                     # API v1.5+ espera bytes diretamente no 'data' do blob de áudio.
                     await self._queue_put_robust(self.mic_audio_out_queue,
                                                 {"mime_type": f"audio/l16;rate={SEND_SAMPLE_RATE}", "data": audio_chunk},
                                                 "Mic Audio Out",
                                                 drop_oldest_if_full=True)
                     # logger.debug(f"Chunk de áudio do Mic ({len(audio_chunk)} bytes) enfileirado.")

                 except IOError as e:
                      logger.error(f"Erro de I/O no stream de áudio do Microfone: {e}. Tentando continuar.", exc_info=False)
                      await asyncio.sleep(0.5) # Espera um pouco após erro de IO
                      # Tentar recriar stream aqui poderia ser uma opção avançada
                 except asyncio.CancelledError:
                     # Captura cancelamento dentro do loop de leitura também
                     logger.info("Cancelamento recebido durante leitura do Mic.")
                     break # Sai do loop de leitura


        except asyncio.CancelledError:
             logger.info("Listen Audio Task cancelada (durante inicialização ou loop externo).")
        except OSError as e_os:
             logger.error(f"Erro de Sistema ao configurar microfone (dispositivo indisponível?): {e_os}. Listen Audio Task encerrada.", exc_info=False)
             self.shutdown_event.set() # Sinaliza erro para outras tasks
        except Exception as e:
             logger.error(f"Erro crítico na Listen Audio Task: {e}. Encerrando.", exc_info=True)
             self.shutdown_event.set() # Sinaliza erro grave
        finally:
             # --- Limpeza do Stream ---
             logger.info("Limpando recursos da Listen Audio Task...")
             if stream and stream.is_active():
                  logger.info("Fechando stream de áudio do microfone...")
                  try:
                      # Operações de stream são bloqueantes
                      await asyncio.to_thread(stream.stop_stream)
                      await asyncio.to_thread(stream.close)
                      logger.info("Stream de áudio do microfone fechada.")
                  except Exception as e_close:
                       logger.error(f"Erro ao fechar stream de áudio do Mic: {e_close}")
             self.audio_input_stream = None
             logger.info("Listen Audio Task finalizada.")


    async def play_audio(self):
        """Lê chunks de áudio da fila gemini_audio_in_queue e os toca no dispositivo de saída padrão."""
        if not pya:
             logger.warning("Task Play Audio não iniciada - PyAudio indisponível.")
             return
        logger.info("Play Audio Task iniciada.")
        stream = None
        queue = self.gemini_audio_in_queue

        try:
             # Abre stream de saída
             # Usamos RECEIVE_SAMPLE_RATE, pois é a taxa que Gemini (supostamente) envia
             logger.info(f"Abrindo stream de áudio de saída (Playback Rate: {RECEIVE_SAMPLE_RATE} Hz)...")
             stream = await asyncio.to_thread(
                 pya.open,
                 format=FORMAT, # paInt16
                 channels=CHANNELS, # Mono (verificar se Gemini envia estéreo)
                 rate=RECEIVE_SAMPLE_RATE, # Taxa que Gemini envia
                 output=True,
                 # output_device_index=... # Poderia especificar dispositivo de saída
                 frames_per_buffer=CHUNK_SIZE, # Tamanho do buffer de saída
             )
             self.audio_output_stream = stream
             logger.info("Stream de áudio de saída (playback) aberta com sucesso.")

             # Loop de consumo da fila e escrita no stream
             while not self.shutdown_event.is_set():
                 try:
                     # Espera por um chunk de áudio na fila de entrada do Gemini
                     audio_chunk = await queue.get()

                     if audio_chunk:
                         # Escreve o chunk no stream de áudio (bloqueante)
                         # logger.debug(f"Tocando chunk de áudio ({len(audio_chunk)} bytes)...")
                         await asyncio.to_thread(stream.write, audio_chunk)
                     else:
                         # logger.debug("Recebido chunk de áudio None, ignorando.")
                         pass

                     # Sinaliza que o item foi processado/tocado
                     queue.task_done()

                 except asyncio.CancelledError:
                      logger.info("Cancelamento recebido durante espera/playback de áudio.")
                      # Antes de sair, tentar tocar o resto da fila? Opcional.
                      logger.info("Tentando tocar áudio restante na fila antes de sair...")
                      try:
                          while True:
                              remaining_chunk = queue.get_nowait()
                              if remaining_chunk: await asyncio.to_thread(stream.write, remaining_chunk)
                              queue.task_done()
                      except asyncio.QueueEmpty:
                          logger.info("Fila de áudio esvaziada.")
                      break # Sai do loop principal

                 except IOError as e:
                      logger.error(f"Erro de I/O ao escrever no stream de áudio Playback: {e}. Tentando continuar.", exc_info=False)
                      await asyncio.sleep(0.1) # Pausa curta
                      # Considerar recriar o stream aqui em caso de erro grave?
                 except Exception as e_play:
                     logger.error(f"Erro inesperado durante playback de áudio: {e_play}", exc_info=True)
                     queue.task_done() # Marca task done mesmo com erro para não travar
                     await asyncio.sleep(0.5)


        except asyncio.CancelledError:
             logger.info("Play Audio Task cancelada (durante inicialização ou loop externo).")
        except OSError as e_os:
             logger.error(f"Erro de Sistema ao configurar saída de áudio: {e_os}. Play Audio Task encerrada.", exc_info=False)
             self.shutdown_event.set()
        except Exception as e:
             logger.error(f"Erro crítico na Play Audio Task: {e}. Encerrando.", exc_info=True)
             self.shutdown_event.set()
        finally:
            # --- Limpeza do Stream de Saída ---
             logger.info("Limpando recursos da Play Audio Task...")
             if stream and stream.is_active():
                  logger.info("Fechando stream de áudio de saída (playback)...")
                  try:
                      # Espera a escrita do último chunk terminar antes de fechar
                      await asyncio.to_thread(stream.stop_stream)
                      await asyncio.to_thread(stream.close)
                      logger.info("Stream de áudio de saída (playback) fechada.")
                  except Exception as e_close:
                       logger.error(f"Erro ao fechar stream de áudio Playback: {e_close}")
             self.audio_output_stream = None
             logger.info("Play Audio Task finalizada.")


    async def send_text_command(self):
         """
         Lê input de texto do usuário no terminal de forma assíncrona
         e o coloca na fila text_command_out_queue para envio a Gemini.
         Também lida com comandos locais como 'q' e 'regface'.
         """
         logger.info("Send Text Command Task iniciada (digite comandos ou 'q' para sair).")
         queue = self.text_command_out_queue

         while not self.shutdown_event.is_set():
             try:
                 # Usa run_in_executor para rodar input() bloqueante em outra thread
                 loop = asyncio.get_running_loop()
                 prompt = f"{self.user_name}> "
                 text_input = await loop.run_in_executor(None, input, prompt) # 'input' é bloqueante
                 text_input = text_input.strip() # Remove espaços extras

                 if not text_input: # Ignora input vazio
                     continue

                 # --- Processamento de Comandos Locais ---
                 if text_input.lower() == "q":
                     logger.info("Comando 'q' recebido no terminal. Solicitando encerramento global...")
                     self.shutdown_event.set() # Sinaliza para todas as tasks encerrarem
                     # A task de envio para Gemini precisa pegar shutdown_event também
                     # Forçar um último put para garantir que a send task veja o shutdown? Opcional.
                     # await queue.put(None) # Sinalizar fim? Ou deixar shutdown_event bastar?
                     break # Sai do loop da send_text_command

                 elif text_input.lower().startswith("regface"):
                      parts = text_input.split(maxsplit=1)
                      if len(parts) == 2 and parts[1]:
                           person_name = parts[1].strip()
                           logger.warning(f"--- REGISTRO DE ROSTO (Placeholder) ---")
                           logger.warning(f"Solicitado registro para: '{person_name}'")
                           logger.warning(f"Funcionalidade AINDA NÃO IMPLEMENTADA.")
                           logger.warning(f"Para implementar:")
                           logger.warning(f"  1. Capturar frame de vídeo recente (ex: da fila de processamento).")
                           logger.warning(f"  2. Usar DeepFace.extract_faces() para detectar e alinhar rosto.")
                           logger.warning(f"  3. Salvar imagem do rosto em '{KNOWN_FACES_DIR}/{person_name}/face_01.jpg'.")
                           logger.warning(f"  4. Opcional: Remover representação pré-calculada ({KNOWN_FACES_DIR}/representations_*.pkl) para forçar recálculo.")
                           print(f"Sistema: [AVISO] Registro de rosto para '{person_name}' ainda não implementado.", file=sys.stderr)
                      else:
                           print("Uso: regface <NomeDaPessoa>", file=sys.stderr)
                           logger.warning("Comando 'regface' usado sem nome.")

                 # --- Enviar Texto como Comando para Gemini ---
                 else:
                     # logger.debug(f"Enviando comando de texto para Gemini: '{text_input}'")
                     # Coloca o texto na fila para a task send_to_gemini pegar
                     await self._queue_put_robust(queue, text_input, "Text Command Out", drop_oldest_if_full=False)

             except asyncio.CancelledError:
                  logger.info("Send Text Command Task cancelada.")
                  break # Sai do loop
             except RuntimeError as e:
                 # Handle caso rode em ambiente sem terminal interativo
                 if "Cannot run input()" in str(e) or "Event loop is closed" in str(e):
                     logger.warning("Terminal não interativo ou loop fechado. Encerrando task de comando de texto.")
                     break # Sai da task se não pode ler input
                 else:
                     logger.error(f"Erro inesperado de Runtime na Send Text Command Task: {e}", exc_info=True)
                     break # Sai em caso de erro desconhecido
             except Exception as e:
                  logger.error(f"Erro inesperado na Send Text Command Task: {e}", exc_info=True)
                  await asyncio.sleep(1) # Espera antes de tentar ler input de novo
         logger.info("Send Text Command Task finalizada.")

    # --------------------------------------------------------------------------
    # Task Principal de Comunicação com Gemini API (v1.5+ Chat)
    # --------------------------------------------------------------------------

    async def gemini_communication_manager(self):
        """
        Gerencia a sessão de chat multimodal com Gemini 1.5+ (`start_chat`).
        Usa o cliente Gemini e módulos armazenados em 'self'.
        """
        # Verifica se os componentes necessários foram passados para a instância
        if not self.gemini_client or not self.genai or not self.glm:
             logger.error("Task Gemini Manager não iniciada - Instância do cliente Gemini (self.gemini_client) ou módulos (self.genai/self.glm) não disponíveis na classe AudioLoop.")
             self.shutdown_event.set()
             return
        logger.info("Gemini Communication Manager Task iniciada (usando self.gemini_client).")

        # Cria aliases locais para os módulos/cliente para facilitar a leitura
        genai_local = self.genai
        glm_local = self.glm
        local_client_instance = self.gemini_client

        chat = None # Armazena a sessão de chat ativa *desta task*
        history = [] # Histórico para esta sessão de chat

        # --- Loop Principal de Gerenciamento da Sessão de Chat ---
        while not self.shutdown_event.is_set():
            try:
                # 1. Iniciar ou Reiniciar a Sessão de Chat se necessário
                if chat is None:
                     logger.info("Iniciando nova sessão de chat multimodal com Gemini...")
                     # --- CORRIGIDO: Usa 'local_client_instance' (que veio de self.gemini_client) ---
                     chat = local_client_instance.start_chat(history=history)
                     # -----------------------------------------------------------------------------
                     logger.info("Sessão de chat multimodal iniciada com sucesso.")

                # 2. Esperar por qualquer tipo de Input para Enviar
                tasks_waiting = {}
                tasks_waiting["mic"] = asyncio.create_task(self.mic_audio_out_queue.get())
                tasks_waiting["video"] = asyncio.create_task(self.video_frame_out_queue.get())
                tasks_waiting["context"] = asyncio.create_task(self.context_injection_queue.get())
                tasks_waiting["text_cmd"] = asyncio.create_task(self.text_command_out_queue.get())
                tasks_waiting["shutdown"] = asyncio.create_task(self.shutdown_event.wait())

                done, pending = await asyncio.wait(
                    tasks_waiting.values(), return_when=asyncio.FIRST_COMPLETED
                )

                if tasks_waiting["shutdown"] in done:
                    logger.info("Shutdown detectado no Gemini Manager. Encerrando...")
                    for task in pending: task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    break

                # 3. Processar Inputs que Chegaram e Montar Partes
                content_parts = []
                queues_to_mark_done = [] # Guarda tuplas (queue, data) para marcar done depois

                for task in done:
                    task_name = [name for name, t in tasks_waiting.items() if t == task][0]
                    queue = getattr(self, f"{task_name}_out_queue", None) # Ex: self.mic_audio_out_queue

                    if task.cancelled():
                         if queue: queues_to_mark_done.append((queue, None)) # Marca done mesmo se cancelado
                         continue
                    if task.exception():
                         ex = task.exception()
                         logger.error(f"Erro ao obter dados da fila '{task_name}': {ex}")
                         if queue: queues_to_mark_done.append((queue, None)) # Marca done mesmo com erro
                         continue

                    data = task.result()
                    if queue: queues_to_mark_done.append((queue, data)) # Adiciona à lista para marcar done

                    # Montar o conteúdo usando glm_local
                    if task_name == "mic" and isinstance(data, dict) and "data" in data:
                         # --- CORRIGIDO: Usa glm_local ---
                         content_parts.append(glm_local.Part(inline_data=glm_local.Blob(mime_type=data.get('mime_type', 'audio/l16;rate=16000'), data=data['data'])))
                    elif task_name == "video" and isinstance(data, dict) and "data" in data:
                         # --- CORRIGIDO: Usa glm_local ---
                         content_parts.append(glm_local.Part(inline_data=glm_local.Blob(mime_type=data.get('mime_type', 'image/jpeg'), data=data['data'])))
                    elif task_name == "context" and isinstance(data, str) and data:
                         # --- CORRIGIDO: Usa glm_local ---
                         content_parts.append(glm_local.Part(text=data))
                    elif task_name == "text_cmd" and isinstance(data, str) and data:
                         # --- CORRIGIDO: Usa glm_local ---
                         content_parts.append(glm_local.Part(text=data))

                # Marca as tarefas das filas como concluídas APÓS processar os dados
                for q, item_data in queues_to_mark_done:
                    try:
                        q.task_done()
                    except ValueError: # Se task_done() for chamado mais vezes que put()
                        logger.warning(f"Tentativa de chamar task_done() em fila já vazia ou desbalanceada: {q}")
                    except Exception as e_done:
                        logger.error(f"Erro ao chamar task_done() na fila: {e_done}")


                # Cancela tasks pendentes
                for task in pending:
                    if not task.done(): task.cancel()
                if pending: await asyncio.gather(*pending, return_exceptions=True)


                # 4. Enviar Mensagem Agrupada para Gemini (se houver partes)
                if content_parts and chat:
                    send_start_time = time.monotonic()
                    try:
                        response_stream = await chat.send_message_async(content_parts, stream=True)
                        send_elapsed = time.monotonic() - send_start_time
                        # logger.debug(f"Gemini send_message_async took {send_elapsed:.3f}s")

                        # 5. Processar Resposta (Stream)
                        async for chunk in response_stream:
                            if self.shutdown_event.is_set(): break

                            # Processar Texto
                            if hasattr(chunk, 'text') and chunk.text:
                                await self._queue_put_robust(self.gemini_text_in_queue, chunk.text, "Gemini Text In", drop_oldest_if_full=False)

                            # Processar Áudio (se houver)
                            if hasattr(chunk, 'parts'):
                                for part in chunk.parts:
                                     if hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith('audio/'):
                                         await self._queue_put_robust(self.gemini_audio_in_queue, part.inline_data.data, "Gemini Audio In", drop_oldest_if_full=True)
                                         break # Assume um audio por chunk

                            # Processar Function Calls (se houver)
                            # if hasattr(chunk, 'function_calls')...

                        # Fim do stream

                    # --- CORRIGIDO: Usa genai_local.types para exceções ---
                    except genai_local.types.BlockedPromptException as block_ex:
                         logger.error(f"PROMPT BLOQUEADO pela API Gemini: {block_ex}")
                         # chat = None # Opcional: Forçar reinício do chat
                    except genai_local.types.StopCandidateException as stop_ex:
                         logger.warning(f"Geração Gemini interrompida (Stop): {stop_ex}")
                    # ----------------------------------------------------
                    except Exception as e_send_recv:
                        logger.error(f"Erro durante envio/recebimento Gemini: {e_send_recv}", exc_info=True)
                        chat = None # Força reinício
                        history.clear()
                        await asyncio.sleep(5)
                else:
                    # Se não há partes para enviar, espera um pouco para não ocupar CPU
                    await asyncio.sleep(0.05)

            except asyncio.CancelledError:
                logger.info("Gemini Communication Manager Task cancelada.")
                break
            except Exception as e:
                logger.error(f"Erro crítico inesperado no loop Gemini Communication Manager: {e}", exc_info=True)
                chat = None # Tenta reiniciar
                history.clear()
                await asyncio.sleep(5)

        # --- Limpeza Final ---
        logger.info("Encerrando sessão de chat com Gemini (se ativa)...")
        chat = None # Libera referência
        logger.info("Gemini Communication Manager Task finalizada.")

        # --- Limpeza Final ---
        logger.info("Encerrando sessão de chat com Gemini (se ativa)...")
        # Não há um método 'close' explícito para o chat, apenas deixar de usar deve bastar
        self.chat_session = None
        chat = None
        logger.info("Gemini Communication Manager Task finalizada.")

    # --------------------------------------------------------------------------
    # Loop Principal de Execução e Gerenciamento de Tasks
    # --------------------------------------------------------------------------
    def _cancel_all_tasks(self):
        """ Cancela todas as tasks asyncio gerenciadas. """
        logger.debug(f"Iniciando cancelamento de {len(self.tasks)} tasks...")
        cancelled_count = 0
        for name, task in self.tasks.items():
            if task and not task.done():
                task.cancel()
                # logger.debug(f"Task '{name}' cancelada.")
                cancelled_count += 1
        logger.debug(f"{cancelled_count} tasks ativas foram solicitadas a cancelar.")
        # Não espera aqui, o TaskGroup ou loop principal deve esperar.

    # Função _run_main_loop corrigida:
    async def _run_main_loop(self):
        """
        Ponto central que cria e gerencia o ciclo de vida das tasks principais
        usando um TaskGroup para garantir que todas terminem juntas ou que erros
        sejam propagados corretamente.
        """
        logger.info("Iniciando loop principal de gerenciamento de tasks...")
        # self.shutdown_event.clear() # Garante que o evento de shutdown comece limpo

        try:
            # Cria um TaskGroup para gerenciar as tasks de forma estruturada
            async with asyncio.TaskGroup() as tg:
                logger.info("Criando tasks principais dentro do TaskGroup...")

                # --- Comunicação Gemini & Usuário ---
                self.tasks['gemini_manager'] = tg.create_task(self.gemini_communication_manager(), name="GeminiMgr")
                self.tasks['play_audio'] = tg.create_task(self.play_audio(), name="PlayAudio")
                if pya: # Só cria task de mic se PyAudio funcionou
                    self.tasks['listen_audio'] = tg.create_task(self.listen_audio(), name="ListenAudio")
                # Task de comando de texto (pode falhar em ambientes não interativos)
                self.tasks['send_text'] = tg.create_task(self.send_text_command(), name="SendTextCmd")


                # --- Captura e Processamento de Vídeo ---
                if self.video_mode != "none":
                    self.tasks['capture_video'] = tg.create_task(self.capture_video(), name="CaptureVideo")
                    # Cria tasks de processamento de IA somente se o vídeo estiver ativo
                    if self._active_processors["yolo"]: self.tasks['yolo'] = tg.create_task(self.process_yolo(), name="ProcessYOLO")
                    if self._active_processors["sam2"]: self.tasks['sam2'] = tg.create_task(self.process_sam2(), name="ProcessSAM2")
                    if self._active_processors["midas"]: self.tasks['midas'] = tg.create_task(self.process_midas(), name="ProcessMiDaS")
                    if self._active_processors["deepface"]: self.tasks['deepface'] = tg.create_task(self.process_deepface(), name="ProcessDeepFace")
                    if self._active_processors["cloud_vision"]: self.tasks['cloud_vision'] = tg.create_task(self.process_cloud_vision(), name="ProcessCloudVision")

                    # Task de síntese (depende dos processadores estarem ativos implicitamente)
                    if any(self._active_processors.values()):
                         self.tasks['context_synthesizer'] = tg.create_task(self.context_synthesizer(), name="ContextSynth")
                    else:
                        logger.warning("Nenhum processador de IA ativo, Context Synthesizer não será iniciado.")
                else:
                    logger.info("Modo de vídeo 'none', tasks de captura e processamento de vídeo não iniciadas.")

                active_task_count = len(self.tasks)
                logger.info(f"--- {active_task_count} Tasks Criadas e Rodando ---")
                # O TaskGroup espera aqui até que todas as tasks terminem ou uma lance exceção não tratada

            # Se TaskGroup terminar sem exceções (raro para loop contínuo sem shutdown)
            logger.info("TaskGroup finalizado sem exceções não tratadas diretamente pelo except*.")

        # --- Tratamento de Exceções do TaskGroup com except* ---
        # Captura grupos que contêm APENAS CancelledError (shutdown limpo)
        except* asyncio.CancelledError as eg_cancel:
            logger.info(f"TaskGroup cancelado (Shutdown solicitado?). Grupo contém {len(eg_cancel.exceptions)} CancelledError(s).")
            # Não precisa sinalizar shutdown_event aqui, já deve ter sido setado para causar isso.

        # Captura grupos que contêm QUALQUER outra exceção (incluindo misturas com CancelledError)
        except* Exception as eg_error:
            logger.error(f"Erro(s) no TaskGroup! O grupo contém {len(eg_error.exceptions)} exceção(ões).")

            # Separa os CancelledErrors dos erros reais dentro do grupo
            cancels, other_errors = eg_error.split(asyncio.CancelledError)

            if other_errors:  # Se houve erros além de cancelamentos
                logger.error(f"---> Erros Reais Encontrados ({len(other_errors.exceptions)}):")
                for i, exc in enumerate(other_errors.exceptions):
                    # Loga o traceback completo para os erros reais
                    logger.error(f"     Erro Real {i+1}: {type(exc).__name__}: {exc}", exc_info=exc)
                logger.warning("Erro(s) detectado(s) no TaskGroup. Sinalizando shutdown para garantir encerramento...")
                self.shutdown_event.set()  # Sinaliza shutdown por causa dos erros
            else:
                # Se chegou aqui, o grupo SÓ tinha CancelledErrors, mas foi pego por except* Exception
                # (pode acontecer dependendo de como a exceção é levantada internamente)
                logger.info("TaskGroup continha apenas CancelledErrors (mas capturado por except* Exception). Tratando como shutdown normal.")

            # Opcional: Logar os cancelamentos separados se existirem (pode ser verboso)
            # if cancels:
            #    logger.info(f"  -> O grupo também continha {len(cancels.exceptions)} CancelledError(s).")

        finally:
            logger.info("Saindo do loop principal (_run_main_loop)...")
            # Garante que o evento de shutdown esteja setado ao sair do loop principal
            if not self.shutdown_event.is_set():
                 logger.warning("Shutdown event não estava setado no finally do _run_main_loop, setando agora.")
                 self.shutdown_event.set()
            # Código opcional para esperar tasks (mantido comentado)
            # logger.debug("Aguardando finalização de todas as tasks...")
            # await asyncio.gather(...)
            # logger.debug("Tasks finalizadas.")

        logger.info("--- Loop principal (_run_main_loop) Concluído ---")
# --- Fim da função corrigida ---


# --- Ponto de Entrada Principal ---

async def main():
    """Função principal assíncrona: parseia args, carrega modelos, inicia o loop."""

    # --- DECLARAÇÃO GLOBAL MOVIDA PARA O INÍCIO DA FUNÇÃO ---
    # Declara que vamos modificar essas variáveis globais dentro desta função.
    # Isso deve vir ANTES de qualquer uso delas (leitura ou escrita) aqui.
    global YOLO_MODEL_PATH, SAM2_MODEL_TYPE, SAM2_CHECKPOINT_PATH, MIDAS_MODEL_NAME, KNOWN_FACES_DIR
    # --- FIM DA CORREÇÃO ---

    parser = argparse.ArgumentParser(description="Trackie - Assistente Visual com IA e Gemini")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Dispositivo para rodar modelos de IA ('auto', 'cpu', 'cuda'). Default: auto")
    parser.add_argument("--video-mode", type=str, default=DEFAULT_VIDEO_MODE, choices=["camera", "screen", "none"],
                        help="Fonte do vídeo ('camera', 'screen', 'none'). Default: camera")
    parser.add_argument("--user-name", type=str, default=DEFAULT_USER_NAME,
                        help="Nome do usuário para interação com Gemini. Default: Usuário")

    # Agora o uso como default é válido, pois o 'global' já foi declarado antes
    parser.add_argument("--yolo-model", type=str, default=YOLO_MODEL_PATH,
                        help=f"Caminho para o modelo YOLO (ex: yolov8n.pt). Default: '{YOLO_MODEL_PATH}'")
    parser.add_argument("--sam-model", type=str, default=SAM2_MODEL_TYPE,
                        help=f"Tipo do modelo SAM2 (ex: vit_h, hiera_large). Default: '{SAM2_MODEL_TYPE}'")
    parser.add_argument("--sam-ckpt", type=str, default=SAM2_CHECKPOINT_PATH,
                        help=f"Caminho para o checkpoint SAM2 (.pth/.pt). Default: '{SAM2_CHECKPOINT_PATH}'")
    parser.add_argument("--midas-model", type=str, default=MIDAS_MODEL_NAME,
                        help=f"Nome do modelo MiDaS/DPT no HuggingFace (ex: Intel/dpt-large). Default: '{MIDAS_MODEL_NAME}'")
    parser.add_argument("--faces-db", type=str, default=KNOWN_FACES_DIR,
                        help=f"Diretório do banco de dados de faces conhecidas. Default: '{KNOWN_FACES_DIR}'")
    # Add mais args se necessário (taxa de FPS, etc.)

    args = parser.parse_args()

    # Atualiza configurações globais baseadas nos args (importante para load_models)
    # A declaração 'global' no início permite estas atribuições às variáveis globais.
    YOLO_MODEL_PATH = args.yolo_model
    SAM2_MODEL_TYPE = args.sam_model
    SAM2_CHECKPOINT_PATH = args.sam_ckpt
    MIDAS_MODEL_NAME = args.midas_model
    KNOWN_FACES_DIR = args.faces_db

    logger.info(f"Configuração Final - YOLO Model Path: {YOLO_MODEL_PATH}")
    logger.info(f"Configuração Final - SAM2 Model Type: {SAM2_MODEL_TYPE}")
    logger.info(f"Configuração Final - SAM2 Checkpoint Path: {SAM2_CHECKPOINT_PATH}")
    logger.info(f"Configuração Final - MiDaS Model Name: {MIDAS_MODEL_NAME}")
    logger.info(f"Configuração Final - Known Faces Dir: {KNOWN_FACES_DIR}")


    # Determina o dispositivo de IA
    selected_device = args.device
    if selected_device == 'auto':
        if torch.cuda.is_available():
             selected_device = 'cuda'
             logger.info("Dispositivo CUDA detectado e selecionado.")
        else:
             selected_device = 'cpu'
             logger.info("CUDA não disponível, selecionando CPU.")
    elif selected_device == 'cuda' and not torch.cuda.is_available():
        logger.warning("Dispositivo CUDA solicitado, mas não disponível! Usando CPU.")
        selected_device = 'cpu'
    else:
        logger.info(f"Dispositivo selecionado manualmente: {selected_device}")


    # --- Carregar Modelos ANTES de iniciar o loop ---
    # Esta chamada é bloqueante e pode demorar
    # A função load_models usa as variáveis globais que foram atualizadas acima.
    load_models(selected_device)


    # --- Criar e Iniciar Instância Principal ---
    try:
        # ---> SUBSTITUÍDO: Passa 'client', 'genai', e 'glm' definidos DENTRO de main <---
        audio_loop_instance = AudioLoop(
            video_mode=args.video_mode,
            user_name=args.user_name,
            device=selected_device,
            gemini_client=client,   # Passa a instância do cliente criada em main
            genai_module=genai,     # Passa o módulo genai importado em main
            glm_module=glm          # Passa o módulo glm importado em main
        )
        # ---> FIM DA SUBSTITUIÇÃO <---
    except Exception as loop_init_err:
         logger.critical(f"Falha CRÍTICA ao inicializar AudioLoop: {loop_init_err}", exc_info=True)
         sys.exit(1) # Não pode continuar sem a instância principal

    # --- Capturar Sinais de Encerramento (Ctrl+C) ---
    loop = asyncio.get_running_loop()
    stop_signals = (signal.SIGINT, signal.SIGTERM)
    for sig in stop_signals:
         loop.add_signal_handler(
              sig,
              lambda s=sig: asyncio.create_task(shutdown_signal_handler(s, audio_loop.shutdown_event))
         )

    try:
        # O método _run_main_loop agora é executado após a configuração correta
        await audio_loop._run_main_loop()

    except Exception as main_err:
        logger.critical(f"Erro fatal não capturado no loop principal: {main_err}", exc_info=True)
        # Tenta sinalizar shutdown mesmo em caso de erro fatal
        audio_loop.shutdown_event.set()
    finally:
        logger.info("Programa principal (main) finalizando...")
        # --- Limpeza Global Final ---
        if pya:
             logger.info("Terminando PyAudio globalmente...")
             try:
                # Não precisa mais fechar streams individuais aqui, pois as tasks fazem isso no finally
                pya.terminate()
                logger.info("PyAudio terminado.")
             except Exception as e_term:
                 logger.error(f"Erro ao terminar PyAudio: {e_term}")

        logger.info("--- Aplicação Encerrada ---")

# Restante do código permanece igual... (incluindo if __name__ == "__main__":)


async def shutdown_signal_handler(signal_received, shutdown_event: asyncio.Event):
    """ Handler para sinais SIGINT/SIGTERM. """
    logger.warning(f"Recebido sinal de encerramento: {signal_received.name}. Iniciando shutdown...")
    if not shutdown_event.is_set():
        shutdown_event.set() # Sinaliza para todas as tasks que usam o evento
    # Não fazer sys.exit aqui, deixar o loop principal terminar graciosamente


if __name__ == "__main__":
    # Definir política de eventos para Windows se causar problemas (raro hoje em dia)
    # if sys.platform == 'win32':
    #    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Rodar a função main assíncrona
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Isso pode acontecer se o Ctrl+C for pressionado ANTES do loop asyncio ser totalmente estabelecido
        logger.info("KeyboardInterrupt capturado no nível mais externo. Encerrando.")
    except Exception as e_top:
        logger.critical(f"Exceção não tratada no nível raiz: {e_top}", exc_info=True)
        # Tenta garantir que PyAudio seja finalizado se possível
        if pya:
            try: pya.terminate()
            except: pass
    finally:
        logging.shutdown() # Garante que todos os logs foram escritos

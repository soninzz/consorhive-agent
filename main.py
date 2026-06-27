import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types
from supabase import create_client, Client

load_dotenv()

# --- INICIALIZAÇÃO ---
genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

app = FastAPI(title="AXIS Cérebro - Agente de Prospecção")

# --- MODELO DE ENTRADA (FastAPI) ---
class ChatRequest(BaseModel):
    lead_id: str
    lead_name: str
    ultima_mensagem: str

# --- SCHEMA NATIVO DO GEMINI (Evita o erro 'additionalProperties') ---
ESQUEMA_RESPOSTA_GEMINI = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "pensamento_interno": types.Schema(
            type=types.Type.STRING, 
            description="Seu raciocínio lógico antes de responder."
        ),
        "intencao_cliente": types.Schema(
            type=types.Type.STRING, 
            description="Classifique a mensagem do usuário como: 'QUALIFICACAO', 'SPAM_OUTRO' ou 'DUVIDA_SUPORTE'."
        ),
        "mensagem_para_cliente": types.Schema(
            type=types.Type.STRING, 
            description="A mensagem para o WhatsApp. DEVE SER 100% HUMANA, curta e direta. Se for SPAM, deixe vazia."
        ),
        "dados_extraidos": types.Schema(
            type=types.Type.OBJECT, 
            description="Dados de qualificação preenchidos até o momento."
        ),
        "acionar_handoff": types.Schema(
            type=types.Type.BOOLEAN, 
            description="True se o lead estiver qualificado OU se pedir explicitamente para falar com um humano."
        )
    },
    required=["pensamento_interno", "intencao_cliente", "mensagem_para_cliente", "dados_extraidos", "acionar_handoff"]
)

# --- PROMPT DO SISTEMA ---
SYSTEM_PROMPT = """Você é a Bianca, consultora de prospecção da TechXAP.
Sua missão é qualificar o lead de forma leve, dinâmica e extremamente humana pelo WhatsApp.

### DIRETRIZES DE TOM E COMPORTAMENTO:
1. APRESENTAÇÃO: Identifique-se como Bianca da TechXAP APENAS na primeira mensagem da conversa. Nas mensagens seguintes, nunca repita seu nome ou saudação (nada de "Olá!", "Tudo bem?" de novo). Vá direto ao ponto.
2. TOM DE WHATSAPP: Escreva como uma pessoa real. Use frases curtas, direta ao ponto, sem formalidades excessivas (evite "Prezado", "Gostaria de saber", "Por gentileza"). Pode usar letra minúscula casual e um emoji pontual aqui e ali.
3. ESTILO DE CONVERSA: Faça o cliente sentir que está conversando com alguém que quer ajudar, não com um robô de telemarketing.
4. UMA PERGUNTA POR VEZ: Nunca faça duas perguntas na mesma mensagem. Espere o cliente responder para avançar.

### ROTEIRO DE QUALIFICAÇÃO DINÂMICA (Pule etapas se o cliente já tiver respondido antes):
- Q1 (Perfil): Descobrir se é o dono, gestor ou diretor da operação.
- Q2 (Tamanho): Entender o tamanho da equipe atual.
- Q3 (Ferramenta): Saber qual ferramenta ou processo usam hoje para vender.
- Q4 (Gargalo): Qual o maior problema ou dificuldade que enfrentam hoje.
- Q5 (Impacto): Há quanto tempo sofrem com isso e o que estão perdendo.
- Q6 (Expectativa): O que esperam alcançar de resultado ideal.
- Q7 (Budget/Agenda): Validar se têm interesse em ver uma demonstração com um especialista e agendar.

### REGRA DE OURO DA CONTEXTUALIZAÇÃO:
Analise o HISTÓRICO da conversa antes de responder. Se o cliente voluntariamente já entregou a resposta de uma pergunta futura, NÃO faça essa pergunta. Valide o que ele disse de forma empática e passe para o próximo passo natural do roteiro.
"""

# --- ROTA 1: PROCESSAMENTO PRINCIPAL DE CHAT ---
@app.post("/api/v1/prospeccao/chat")
async def process_prospect_message(req: ChatRequest):
    try:
        # 1. BUSCA O LEAD E VERIFICA AS TRAVAS DE SEGURANÇA
        lead_query = supabase.table("leads_qualificacao").select("*").eq("lead_id", req.lead_id).execute()
        
        if lead_query.data:
            lead_atual = lead_query.data[0]
            
            # TRAVA 1: O humano assumiu a conversa?
            if lead_atual.get("status_bot") == "pausado_humano":
                print(f"⚠️ Bot ignorou mensagem de {req.lead_id} pois status está 'pausado_humano'.")
                return {"mensagem_para_cliente": "", "acionar_handoff": False, "status": "ignorado_humano_assumiu"}
                
            # TRAVA 2: Já existe um processamento em andamento? (Prevenção de loop)
            if lead_atual.get("processando") == True:
                return {"mensagem_para_cliente": "aguarde", "status": "processando"}

        # Bloqueia o lead (Ativa a Trava 2)
        if lead_query.data:
            supabase.table("leads_qualificacao").update({"processando": True}).eq("lead_id", req.lead_id).execute()
            historico = lead_query.data[0].get("historico_conversa", [])
        else:
            # Lead totalmente novo
            lead_data = {
                "lead_id": req.lead_id, 
                "nome": req.lead_name, 
                "historico_conversa": [], 
                "processando": True,
                "status_bot": "ativo"
            }
            supabase.table("leads_qualificacao").insert(lead_data).execute()
            historico = []

        # Adiciona a nova mensagem do usuário ao histórico
        historico.append({"role": "user", "content": req.ultima_mensagem})

        # 2. JANELA DE CONTEXTO LIMITADA (Últimos 10 eventos)
        historico_recente = historico[-10:]
        historico_formatado = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in historico_recente])

        # 3. CHAMA O GEMINI COM ANÁLISE DE INTENÇÃO (Mapeado com o Schema Nativo)
        prompt_completo = f"{SYSTEM_PROMPT}\n\nHISTÓRICO:\n{historico_formatado}"
        
        response_ia = genai_client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt_completo,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", 
                response_schema=ESQUEMA_RESPOSTA_GEMINI, # <--- Mudança aqui
                temperature=0.3
            ),
        )
        
        resultado = json.loads(response_ia.text)
        
        # 4. ATUALIZAÇÃO E CONTROLE DE ESTADO
        nova_mensagem = resultado.get("mensagem_para_cliente", "")
        if nova_mensagem: 
            historico.append({"role": "assistant", "content": nova_mensagem})

        # Se a IA decidiu que precisa de humano, pausa o robô no banco.
        novo_status_bot = "pausado_humano" if resultado["acionar_handoff"] else "ativo"
        status_lead = "qualificado" if resultado["acionar_handoff"] else "qualificando"

        update_data = {
            "historico_conversa": historico,
            "status": status_lead,
            "status_bot": novo_status_bot,
            "processando": False # Libera a trava
        }
        
        supabase.table("leads_qualificacao").update(update_data).eq("lead_id", req.lead_id).execute()
        
        return {
            "mensagem_para_cliente": nova_mensagem,
            "acionar_handoff": resultado["acionar_handoff"],
            "intencao_detectada": resultado.get("intencao_cliente")
        }

    except Exception as e:
        if 'req' in locals():
            supabase.table("leads_qualificacao").update({"processando": False}).eq("lead_id", req.lead_id).execute()
        raise HTTPException(status_code=500, detail=str(e))


# --- ROTA 2: INTERCEPÇÃO HUMANA ---
@app.post("/api/v1/prospeccao/intercepcao-humana")
async def verificar_intercepcao_humana(req: ChatRequest):
    try:
        lead_query = supabase.table("leads_qualificacao").select("historico_conversa").eq("lead_id", req.lead_id).execute()
        
        if not lead_query.data:
            return {"status": "ignore"}
            
        historico = lead_query.data[0].get("historico_conversa", [])
        
        ultima_msg_bot = ""
        for msg in reversed(historico):
            if msg.get("role") == "assistant":
                ultima_msg_bot = msg.get("content", "")
                break
        
        if req.ultima_mensagem.strip() != ultima_msg_bot.strip():
            supabase.table("leads_qualificacao").update({"status_bot": "pausado_humano"}).eq("lead_id", req.lead_id).execute()
            print(f"🛑 [INTERCEPÇÃO] Humano assumiu a conversa com {req.lead_id}. Bot desativado.")
            return {"status": "bot_pausado", "humano_assumiu": True}
            
        return {"status": "mensagem_do_proprio_bot", "humano_assumiu": False}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

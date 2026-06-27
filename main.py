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

app = FastAPI(title="ConsorHive — Bianca Agente")

# --- MODELO DE ENTRADA ---
class ChatRequest(BaseModel):
    lead_id: str
    lead_name: str
    ultima_mensagem: str

# --- SCHEMA DE RESPOSTA DO GEMINI ---
ESQUEMA_RESPOSTA_GEMINI = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "pensamento_interno": types.Schema(
            type=types.Type.STRING,
            description="Seu raciocínio antes de responder. Analise o histórico, o que já foi coletado e qual é o próximo passo lógico."
        ),
        "intencao_cliente": types.Schema(
            type=types.Type.STRING,
            description="Classifique: 'QUALIFICACAO' (interesse real), 'SPAM_OUTRO' (fora do escopo, grosseria), 'OUTRO_CONSULTOR' (já tem consultor Ademicon), 'OPTOUT' (não quer contato)."
        ),
        "mensagem_para_cliente": types.Schema(
            type=types.Type.STRING,
            description="Mensagem para enviar no WhatsApp. Curta, humana, direta. Máximo 3 linhas. Se for SPAM_OUTRO ou OPTOUT, deixe vazia."
        ),
        "dados_extraidos": types.Schema(
            type=types.Type.OBJECT,
            description="Preencha os campos que o lead revelou: project_type (imovel/veiculo/outros/servico), timing_months (int), ticket_brl (int), installment_capacity_brl (int), decision_maker (sozinho/dividido), product_maturity (nunca/ja_olhou/ja_tem_cota), other_consultant_client (bool)."
        ),
        "score": types.Schema(
            type=types.Type.INTEGER,
            description="Score de 0 a 100 indicando quão qualificado está o lead. 0-40=frio, 41-60=morno, 61-80=quente, 81-100=quente prioritário."
        ),
        "acionar_handoff": types.Schema(
            type=types.Type.BOOLEAN,
            description="True se: score >= 61 E tem project_type + timing_months + ticket_brl preenchidos, OU se ticket_brl > 1000000, OU se lead pediu explicitamente para falar com humano."
        )
    },
    required=["pensamento_interno", "intencao_cliente", "mensagem_para_cliente", "dados_extraidos", "score", "acionar_handoff"]
)

# --- PROMPT BASE (sempre presente, independente da KB) ---
SYSTEM_PROMPT_BASE = """Você é a Bianca, assistente virtual de estratégia patrimonial do consultor Maikon Festinalli (Ademicon).

### QUEM VOCÊ É:
- Assistente virtual — se perguntada, confirme que é uma assistente virtual, NÃO finja ser humana
- Você representa o Maikon, mas não é o Maikon
- Trabalha com consórcio Ademicon: imóvel, veículo, serviços e outros bens

### REGRAS ABSOLUTAS (nunca viole):
1. NUNCA prometa contemplação rápida, taxa exclusiva, vaga limitada ou retorno garantido
2. NUNCA mencione prestamista durante a prospecção
3. NUNCA opere lead que já é cliente de outro consultor Ademicon — encerre educadamente
4. NUNCA feche negócio — só o Maikon fecha
5. NUNCA faça mais de uma pergunta por mensagem
6. Se o lead pedir para parar ("pare", "não quero", "sai", "descadastra") — confirme opt-out educadamente e pare

### TOM:
- WhatsApp real: frases curtas, direto ao ponto, sem formalidades
- Consultivo, não vendedor de esquina
- Um emoji pontual no máximo por mensagem
- Primeira mensagem: se apresente como Bianca, assistente do Maikon. Nas demais, vá direto ao ponto

### ROTEIRO DE QUALIFICAÇÃO (7 dimensões — uma pergunta por vez, pule se já respondido):
- D1: Qual o projeto? (imóvel, veículo, outros)
- D2: Qual o prazo que pensa em concretizar? (meses)
- D3: Qual o valor aproximado do bem? (ticket em R$)
- D4: Quanto consegue pagar por mês de parcela?
- D5: Decisão é sozinho ou dividido com alguém?
- D6: Já conhece consórcio? Já teve cota antes?
- D7: É cliente de outro consultor Ademicon? (crítico — se sim, encerre)

### REGRA DE CONTEXTUALIZAÇÃO:
Analise o histórico completo. Se o lead já respondeu uma dimensão espontaneamente, não pergunte de novo. Valide e avance.

### HANDOFF:
Quando tiver project_type + timing_months + ticket_brl preenchidos E score >= 61, acione handoff=true. O Maikon assumirá a conversa.
"""

# --- FUNÇÃO: Busca KB ativa do Supabase ---
def buscar_kb_ativa() -> str:
    """Busca a Base de Conhecimento ativa do portal. Fallback para string vazia se não encontrar."""
    try:
        result = supabase.table("base_conhecimento").select("conteudo").eq("ativa", True).limit(1).execute()
        if result.data:
            return result.data[0]["conteudo"]
    except Exception as e:
        print(f"⚠️ Erro ao buscar KB: {e}")
    return ""

# --- FUNÇÃO: Monta o system prompt completo ---
def montar_system_prompt() -> str:
    kb = buscar_kb_ativa()
    if kb:
        return f"{SYSTEM_PROMPT_BASE}\n\n---\n### BASE DE CONHECIMENTO DO MAIKON:\n{kb}"
    return SYSTEM_PROMPT_BASE

# --- FUNÇÃO: Calcula temperatura pelo score ---
def score_para_temperatura(score: int) -> str:
    if score >= 61:
        return "quente"
    elif score >= 41:
        return "morno"
    return "frio"

# --- ROTA 1: PROCESSAMENTO PRINCIPAL DE CHAT ---
@app.post("/api/v1/prospeccao/chat")
async def process_prospect_message(req: ChatRequest):
    try:
        # 1. BUSCA O LEAD E VERIFICA TRAVAS DE SEGURANÇA
        lead_query = supabase.table("leads_qualificacao").select("*").eq("lead_id", req.lead_id).execute()

        if lead_query.data:
            lead_atual = lead_query.data[0]

            # TRAVA 1: Humano assumiu a conversa?
            if lead_atual.get("status_bot") in ("pausado_humano", "pausado"):
                print(f"⚠️ Bot ignorou {req.lead_id} — humano está conduzindo.")
                return {"mensagem_para_cliente": "", "acionar_handoff": False, "status": "ignorado_humano_assumiu"}

            # TRAVA 2: Processamento em andamento? (anti-loop)
            if lead_atual.get("processando") == True:
                return {"mensagem_para_cliente": "aguarde", "status": "processando"}

        # Bloqueia o lead (ativa trava 2)
        if lead_query.data:
            supabase.table("leads_qualificacao").update({"processando": True}).eq("lead_id", req.lead_id).execute()
            historico = lead_query.data[0].get("historico_conversa", [])
            qualification_atual = lead_query.data[0].get("dados_extraidos") or {}
        else:
            # Lead novo
            supabase.table("leads_qualificacao").insert({
                "lead_id": req.lead_id,
                "nome": req.lead_name,
                "historico_conversa": [],
                "processando": True,
                "status_bot": "ativo",
                "status": "qualificando",
                "score": 0,
                "temperatura": "frio"
            }).execute()
            historico = []
            qualification_atual = {}

        # Adiciona mensagem do usuário ao histórico
        historico.append({"role": "user", "content": req.ultima_mensagem})

        # 2. JANELA DE CONTEXTO (últimas 20 mensagens)
        historico_recente = historico[-20:]
        historico_formatado = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}" for msg in historico_recente
        ])

        # 3. MONTA PROMPT COM KB ATUAL + QUALIFICAÇÃO JÁ COLETADA
        system_prompt = montar_system_prompt()

        qualificacao_str = ""
        if qualification_atual:
            qualificacao_str = f"\n\n### DADOS JÁ COLETADOS DESTE LEAD:\n{json.dumps(qualification_atual, ensure_ascii=False, indent=2)}"

        prompt_completo = f"{system_prompt}{qualificacao_str}\n\n### HISTÓRICO DA CONVERSA:\n{historico_formatado}"

        # 4. CHAMA O GEMINI
        response_ia = genai_client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt_completo,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ESQUEMA_RESPOSTA_GEMINI,
                temperature=0.3
            ),
        )

        resultado = json.loads(response_ia.text)

        # 5. PROCESSA RESULTADO
        nova_mensagem = resultado.get("mensagem_para_cliente", "")
        intencao = resultado.get("intencao_cliente", "QUALIFICACAO")
        score = min(100, max(0, resultado.get("score", 0)))
        acionar_handoff = resultado.get("acionar_handoff", False)
        dados_extraidos = resultado.get("dados_extraidos", {})

        # Merge dos dados extraídos (não sobrescreve campos já preenchidos com None)
        qualification_atualizada = {**qualification_atual}
        for k, v in dados_extraidos.items():
            if v is not None and v != "" and v != qualification_atual.get(k):
                qualification_atualizada[k] = v

        # Detecta outro consultor — encerra imediatamente
        if intencao == "OUTRO_CONSULTOR" or qualification_atualizada.get("other_consultant_client") == True:
            acionar_handoff = False
            novo_status_bot = "descartado"
            status_lead = "descartado"
            etapa_funil = "PERDIDO"
        elif intencao == "OPTOUT":
            # Registra opt-out
            try:
                supabase.table("opt_outs").insert({
                    "telefone": req.lead_id,
                    "motivo": "lead_pediu",
                    "origem": "mensagem_lead"
                }).execute()
            except Exception:
                pass
            novo_status_bot = "descartado"
            status_lead = "descartado"
            etapa_funil = "PERDIDO"
        elif acionar_handoff:
            novo_status_bot = "pausado_humano"
            status_lead = "qualificado"
            etapa_funil = "QUALIFICANDO"
        else:
            novo_status_bot = "ativo"
            status_lead = "qualificando"
            etapa_funil = "EM_CONVERSA"

        # Adiciona resposta da Bianca ao histórico
        if nova_mensagem:
            historico.append({"role": "assistant", "content": nova_mensagem})

        temperatura = score_para_temperatura(score)

        # 6. SALVA NO BANCO
        update_data = {
            "historico_conversa": historico,
            "status": status_lead,
            "status_bot": novo_status_bot,
            "processando": False,
            "score": score,
            "temperatura": temperatura,
            "etapa_funil": etapa_funil,
            # Campos individuais de qualificação para o dashboard
            "cargo_decisor": qualification_atualizada.get("decision_maker"),
            "dor_principal": qualification_atualizada.get("project_type"),
            "resultado_esperado": qualification_atualizada.get("timing_months"),
            "budget_status": str(qualification_atualizada.get("ticket_brl", "")) if qualification_atualizada.get("ticket_brl") else None,
        }

        supabase.table("leads_qualificacao").update(update_data).eq("lead_id", req.lead_id).execute()

        print(f"✅ [{req.lead_id}] Score={score} Temp={temperatura} Handoff={acionar_handoff} Intenção={intencao}")

        return {
            "mensagem_para_cliente": nova_mensagem,
            "acionar_handoff": acionar_handoff,
            "intencao_detectada": intencao,
            "score": score,
            "temperatura": temperatura
        }

    except Exception as e:
        # Libera trava em caso de erro
        try:
            supabase.table("leads_qualificacao").update({"processando": False}).eq("lead_id", req.lead_id).execute()
        except Exception:
            pass
        print(f"❌ Erro ao processar {req.lead_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- ROTA 2: INTERCEPÇÃO HUMANA ---
@app.post("/api/v1/prospeccao/intercepcao-humana")
async def verificar_intercepcao_humana(req: ChatRequest):
    """
    Detecta se o consultor respondeu diretamente pelo WhatsApp (fora do portal).
    Se sim, pausa o bot automaticamente.
    """
    try:
        lead_query = supabase.table("leads_qualificacao").select("historico_conversa").eq("lead_id", req.lead_id).execute()

        if not lead_query.data:
            return {"status": "ignore"}

        historico = lead_query.data[0].get("historico_conversa", [])

        # Busca a última mensagem enviada pelo bot
        ultima_msg_bot = ""
        for msg in reversed(historico):
            if msg.get("role") == "assistant":
                ultima_msg_bot = msg.get("content", "")
                break

        # Se a mensagem outbound não é do bot, foi o humano respondendo pelo celular
        if req.ultima_mensagem.strip() != ultima_msg_bot.strip():
            supabase.table("leads_qualificacao").update({
                "status_bot": "pausado_humano",
                "responsavel_humano": "Maikon"
            }).eq("lead_id", req.lead_id).execute()
            print(f"🛑 [INTERCEPÇÃO] Maikon assumiu conversa com {req.lead_id} pelo celular.")
            return {"status": "bot_pausado", "humano_assumiu": True}

        return {"status": "mensagem_do_proprio_bot", "humano_assumiu": False}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- ROTA 3: HEALTH CHECK ---
@app.get("/health")
async def health():
    kb = buscar_kb_ativa()
    return {
        "status": "ok",
        "kb_ativa": bool(kb),
        "kb_chars": len(kb)
    }

import requests
import re


def fetch_all_core_data(api_base_url="http://localhost:8000/api"):
    """
    Preia datele brute din toate endpointurile principale pentru analiză completă.
    Returnează un dict cu toate tabelele relevante.
    """
    endpoints = {
        "dealerships": "dealerships",
        "vehicles": "vehicles",
        "service_items": "service-items",
        "sale_orders": "sale-orders",
        "car_sale_items": "car-sale-items",
        "employees": "employees",
        "customers": "customers"
    }
    all_data = {}
    for key, endpoint in endpoints.items():
        try:
            resp = requests.get(f"{api_base_url}/{endpoint}", timeout=60)
            resp.raise_for_status()
            all_data[key] = resp.json()
        except Exception as e:
            print(f"[WARN] Could not fetch {key}: {e}")
            all_data[key] = []
    return all_data

def summarize_for_openai(all_data: dict) -> dict:
    """
    Sumarizează local datele brute pentru a trimite doar statistici relevante la OpenAI.
    Returnează un dict cu sumaruri pentru fiecare tabel.
    """
    summary = {}
    # Exemplu sumarizare: număr entități, top 5, statistici simple
    for key, rows in all_data.items():
        summary[key] = {
            "count": len(rows),
            "sample": rows[:5] if isinstance(rows, list) else rows
        }
    # Poți adăuga aici calcule suplimentare: medii, topuri, outlieri etc.
    return summary
#!/usr/bin/env python3
import os, sys, json, datetime as dt, re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
from chromadb.config import Settings

# ---------- CONFIG ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "sales")

MODEL_RECOMMEND = os.getenv("OPENAI_MODEL_REC", "gpt-4o-mini")
MODEL_MESSAGE  = os.getenv("OPENAI_MODEL_MSG", "gpt-4o-mini")


ACTION_VERBS = [
    "increase","grow","improve","boost","raise","recover","reduce","decrease",
    "crește","cresc","îmbunătățește","reduce","recuperează"
]


def make_concrete_fallback(user_text: str) -> str:
    """
    Construiește o sugestie CONCRETĂ adaptată la structura reală a bazei de date (dealership, regiuni, vânzări, servicii, etc).
    """
    txt = (user_text or "").lower()

    # Domeniu: vânzări mașini, servicii, clienți, regiuni
    if any(k in txt for k in ["golf", "tiguan", "passat", "model", "vehicle", "mașină", "masina"]):
        domain = "vehicle sales"
    elif any(k in txt for k in ["serviciu", "service", "warranty", "casco", "sunroof", "oil"]):
        domain = "service sales"
    elif any(k in txt for k in ["client", "customer"]):
        domain = "customer engagement"
    else:
        domain = "dealership sales"

    # Regiune: dacă apare un nume de regiune din baza de date, îl folosim
    regions = ["NW", "NE", "Cluj-Napoca", "Iasi"]
    region_phrase = "in all regions"
    for reg in regions:
        if reg.lower() in txt:
            region_phrase = f"in region {reg}"
            break

    # metrică & orizont
    target_pct = "by 10%"
    horizon = "in the next 30 days"

    return f"Increase {domain} {region_phrase} {target_pct} {horizon}."

def validate_goal_with_ai(goal: str, horizon: int = None) -> dict:
    # Nu mai extragem automat perioada și regiunea cu regex, doar AI decide ce lipsește
    horizon = None
    region = None
    """
    Returnează dict cu chei: is_valid (bool), why (str), suggested (str – mereu concret).
    Promptul este adaptat la structura reală a bazei de date (dealership, regiuni, vânzări, servicii, etc).
    """
    system = (
        "You are a validator for business objectives for a car dealership network. "
        "Determine if the input is a concrete business goal suitable as LLM context. "
        "Accept any natural language goal that contains: "
        "- an action verb (accept any synonym or similar word: increase, improve, grow, boost, raise, expand, enhance, maximize, accelerate, develop, advance, strengthen, promote, escalate, elevate, reduce, decrease, cut, minimize, recover, fix, solve, address, optimize, stimulate, drive, push, achieve, reach, deliver, accomplish, etc.) "
        "- what to change (accept any business target: vehicle sales, service sales, casco, casco sales, insurance, customer engagement, revenue, profit, margin, leads, contracts, appointments, test drives, upsell, cross-sell, retention, satisfaction, etc.) "
        "- a scope (accept any region, city, dealership, employee, team, group, or 'all regions', 'all', 'entire', 'everywhere', 'national', 'local', 'area', 'zone', 'territory', etc., in any natural language form) "
        "- a time window/period (accept any natural language expression for period, e.g. 'in the last 30 days', 'in September', 'look at the last 60 days', 'previous 3 months', 'for the last quarter', 'in Q3', 'recently', 'this year', 'since January', 'in 2025', 'for the summer', etc.) "
        "If a time window/period is missing from the goal, but a period is provided separately (e.g. as a parameter or in the UI), consider the goal valid. "
        "If the input is not concrete, return STRICT JSON with keys: is_valid (true/false), why (short, enumerate ONLY what is missing: e.g. 'Missing time window, region'), suggested (empty string). Do NOT mention details that are already present. No extra text. "
        "Be very permissive with natural language, and accept any reasonable synonym or similar word for each element. All four elements must be present, but allow for creative or indirect phrasing."
    )
    # Trimite goal și perioada ca JSON structurat către LLM
    if horizon:
        user = json.dumps({"goal": goal.strip(), "period_days": horizon}, ensure_ascii=False)
    else:
        user = goal.strip()
    suggested = ""
    is_valid = False
    why = "Input is not a business goal."
    try:
        raw = _openai_chat(MODEL_RECOMMEND, system, user, temperature=0)
        s, e = raw.find("{"), raw.rfind("}") + 1
        data = json.loads(raw[s:e])
        is_valid = bool(data.get("is_valid", False))
        why = str(data.get("why", "")).strip() or why
        suggested = ""  # Nu mai sugerăm obiectiv complet
    except Exception:
        pass
    return {"is_valid": is_valid, "why": why, "suggested": suggested, "horizon": horizon, "region": region}



# ---------- KPI ----------
class KPIService:
    def compute(self, rows: List[Dict[str, Any]], since_days: int):
        if not rows:
            return pd.DataFrame(), []

        df = pd.DataFrame(rows)

    # Accepts both ISO string ("2025-07-15") and timestamp (fallback)

        # Folosește 'order_date' dacă 'date' nu există
        date_col = None
        for c in df.columns:
            if c.lower() in ("date", "order_date", "created_at", "data", "timestamp"):
                date_col = c
                break
        if not date_col:
            raise ValueError(f"No date column found in input! Columns: {list(df.columns)}")
        try:
            df[date_col] = pd.to_datetime(df[date_col]).dt.date
        except Exception:
            df[date_col] = pd.to_datetime(df[date_col], unit="s").dt.date

        # redenumește pentru restul codului
        if date_col != "date":
            df["date"] = df[date_col]

    # time windows
        end = dt.date.today()
        start = end - dt.timedelta(days=since_days)
        prev_start = start - dt.timedelta(days=since_days)
        prev_end = start - dt.timedelta(days=1)

        curr = df[(df["date"] >= start) & (df["date"] <= end)].copy()
        prev = df[(df["date"] >= prev_start) & (df["date"] <= prev_end)].copy()

        if curr.empty:
            return pd.DataFrame(), []

    # stable aggregation on current window (no apply)
        g_curr = (
            curr.groupby(["dealer_id", "region", "tier"], as_index=False)
                .agg(
                    leads=("leads", "sum"),
                    deals=("deals", "sum"),
                    revenue=("revenue", "sum"),
                    points=("points", "sum"),
                )
        )
        g_curr["conversion"] = np.where(g_curr["leads"] > 0, g_curr["deals"] / g_curr["leads"], 0.0)

    # previous window (for trend)
        g_prev = (
            prev.groupby(["dealer_id"], as_index=False)
                .agg(
                    leads_prev=("leads", "sum"),
                    deals_prev=("deals", "sum"),
                    revenue_prev=("revenue", "sum"),
                    points_prev=("points", "sum"),
                )
        )
        g_prev["conversion_prev"] = np.where(
            g_prev["leads_prev"] > 0, g_prev["deals_prev"] / g_prev["leads_prev"], 0.0
        )

    # merge & fill
        kpis = g_curr.merge(g_prev, on="dealer_id", how="left")
        for col in ["leads_prev", "deals_prev", "revenue_prev", "points_prev", "conversion_prev"]:
            if col not in kpis:
                kpis[col] = 0.0
        kpis[["leads_prev","deals_prev","revenue_prev","points_prev","conversion_prev"]] = \
            kpis[["leads_prev","deals_prev","revenue_prev","points_prev","conversion_prev"]].fillna(0.0)

    # trends
        def trend(curr_val, prev_val):
            if prev_val == 0:
                return 0.0 if curr_val == 0 else 1.0
            return (curr_val - prev_val) / abs(prev_val)

        kpis["trend_revenue_30d"] = kpis.apply(lambda r: trend(r["revenue"], r["revenue_prev"]), axis=1)
        kpis["trend_conversion_30d"] = kpis.apply(lambda r: trend(r["conversion"], r["conversion_prev"]), axis=1)

    # simple risk 0..1
        conv_benchmark = max(0.05, float(kpis["conversion"].mean()) if len(kpis) else 0.08)
        risk = (
            ((conv_benchmark - kpis["conversion"]).clip(lower=0) / max(conv_benchmark, 1e-6))
            + (-kpis["trend_revenue_30d"]).clip(lower=0)
        ).clip(lower=0)
        kpis["risk_of_underperform"] = (risk / (risk.max() or 1)).fillna(0)

        preview_cols = [
            "dealer_id","region","tier","leads","deals","revenue","points","conversion",
            "trend_revenue_30d","trend_conversion_30d","risk_of_underperform"
        ]

        preview = kpis[preview_cols].sort_values("risk_of_underperform", ascending=False).head(15)
        return kpis, preview.to_dict(orient="records")

# ---------- LLM helpers ----------
def _openai_chat(model: str, system_msg: str, user_msg: str, temperature: float = 0.2) -> str:
    # Assistant API v2: persistent thread, agentic reasoning
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    import time
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "assistants=v2",
        "Content-Type": "application/json"
    }
    # 1. Creează asistentul (sau folosește unul existent)
    assistant_payload = {
        "instructions": system_msg,
        "model": model
    }
    a_resp = requests.post("https://api.openai.com/v1/assistants", headers=headers, json=assistant_payload, timeout=60)
    a_resp.raise_for_status()
    assistant_id = a_resp.json()["id"]
    # 2. Creează thread
    t_resp = requests.post("https://api.openai.com/v1/threads", headers=headers, timeout=60)
    t_resp.raise_for_status()
    thread_id = t_resp.json()["id"]
    # 3. Trimite mesajul user
    msg_payload = {"role": "user", "content": user_msg}
    m_resp = requests.post(f"https://api.openai.com/v1/threads/{thread_id}/messages", headers=headers, json=msg_payload, timeout=60)
    m_resp.raise_for_status()
    # 4. Rulează asistentul pe thread
    run_payload = {"assistant_id": assistant_id}
    run_resp = requests.post(f"https://api.openai.com/v1/threads/{thread_id}/runs", headers=headers, json=run_payload, timeout=60)
    run_resp.raise_for_status()
    run_id = run_resp.json()["id"]
    # 5. Polling pentru finalizare
    for _ in range(60):
        run_status = requests.get(f"https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}", headers=headers, timeout=60).json()
        if run_status["status"] == "completed":
            break
        time.sleep(2)
    # 6. Obține răspunsul final
    messages = requests.get(f"https://api.openai.com/v1/threads/{thread_id}/messages", headers=headers, timeout=60).json()
    for msg in messages["data"]:
        if msg["role"] == "assistant":
            # Extrage doar textul principal
            return msg["content"][0]["text"]["value"]
    raise RuntimeError("No assistant response received.")

@dataclass
class ManagerPrompt:
    goal: str
    horizon_days: int = 30
    constraints: List[str] = None
    filters: Dict[str, Any] = None

def generate_recommendations(kpis_preview: List[Dict[str,Any]], prompt: ManagerPrompt, ai_summary: str = None):
    system = (
        "You are an AI Sales Coach for FRONT-LINE sales advisors. "
        "You receive the following summary of business data and issues found in the data: "
        f"{ai_summary}\n"
        "Return a STRICT JSON array with 2-4 items. Each item must have keys: "
        "id, title, explanation. "
        "IMPORTANT: Recommendations must be ONLY SIMPLE, CONCRETE, and MEASURABLE ACTIONS that sales advisors (employees) can do directly and whose achievement can be evaluated (e.g., 'increase sales by 25%', 'contact 10 new leads per week', 'follow up with all test drive customers within 48 hours'). "
        "Recommendations MUST be based ONLY on the specific problems, statistics, and issues found in the summary above. Do NOT suggest generic actions. "
        "DO NOT suggest marketing campaigns, SEO, pricing strategy, management tasks, or anything that requires manager approval or company-level changes. "
        "Do NOT include general advice or analysis unless it results in a specific, trackable action (e.g., 'collect and review 20 customer feedback forms this month and implement at least one suggestion'). "
        "No prose, only JSON array."
    )
    user = json.dumps({
        "goal": prompt.goal,
        "horizon_days": prompt.horizon_days,
        "constraints": prompt.constraints or [],
        "filters": prompt.filters or {},
        "kpis_preview": kpis_preview[:10]
    }, ensure_ascii=False)
    raw = _openai_chat(MODEL_RECOMMEND, system, user, 0.2)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        s, e = raw.find("["), raw.rfind("]")+1
        data = json.loads(raw[s:e])
    # normalize & score
    for i, r in enumerate(data, 1):
        if "id" not in r: r["id"] = i
        r["impact"] = float(max(0, min(1, r.get("impact", 0.6))))
        r["effort"] = float(max(0, min(1, r.get("effort", 0.4))))
        r["feasibility"] = float(max(0, min(1, r.get("feasibility", 0.7))))
    data.sort(key=lambda x: -(x["impact"] * x["feasibility"] / (0.2 + x["effort"])))
    return data[:4]

def compose_message(
    goal: str,
    selected: List[Dict[str,Any]],
    style: str = "professional",
    kpis_preview: Optional[List[Dict[str,Any]]] = None,
    performance_level: str = None
) -> str:
    # 1) Identified issues (strictly from data) – top 3 at risk, but skip UNKNOWN or empty
    issues_lines: List[str] = []
    if kpis_preview:
        top = sorted(kpis_preview, key=lambda x: x.get("risk_of_underperform", 0), reverse=True)[:3]
        for r in top:
            dealer_id = r.get('dealer_id','').strip()
            region = r.get('region','').strip()
            tier = r.get('tier','').strip()
            # Skip if dealer_id is UNKNOWN or empty
            if not dealer_id or dealer_id.upper() == 'UNKNOWN':
                continue
            # Optionally, you can add a short description if you want, or just skip
            issues_lines.append(f"- Issue detected for {dealer_id} ({region}, {tier})")
    # If no real issues, do not add any line
    #if not issues_lines:
    #    issues_lines = ["(No issues identified from data for the selected period.)"]

    # 2) Allowed actions – only what advisors can do (from selected recommendations)
    allowed_actions = [
        {"title": s.get("title","").strip(), "explanation": s.get("explanation","").strip()}
        for s in (selected or []) if s.get("title")
    ]

    style_instructions = {
        "motivational": "Use an inspirational, energetic, and encouraging tone. Make the team feel empowered and excited to act.",
        "professional": "Use a formal, direct, and efficient tone. Focus on clarity, responsibility, and execution. Avoid exclamation marks and emotional language.",
        "friendly": "Use a warm, approachable, and supportive tone. Make the message feel personal and positive, as if talking to colleagues you appreciate.",
    }
    style = (locals().get('style') or 'motivational').lower()
    style_note = style_instructions.get(style, style_instructions['motivational'])
    perf_note = ""
    if performance_level:
        if performance_level == "high":
            perf_note = (
                " This message is for salespeople with HIGH sales results."
                " Use a celebratory and appreciative tone."
                " Explicitly praise their achievements and encourage them to share best practices with others."
                " The message should sound like a recognition and a push to maintain or exceed their current performance."
                " Example: 'Fantastic work! Your results are outstanding. Keep inspiring the team!'"
            )
        elif performance_level == "medium":
            perf_note = (
                " This message is for salespeople with MEDIUM/average sales results."
                " Use a supportive and motivating tone."
                " Acknowledge their effort, but encourage them to take specific steps to reach the next level."
                " The message should include practical tips and a call to action for improvement."
                " Example: 'Good job so far! With a few focused actions, you can achieve even more.'"
            )
        elif performance_level == "low":
            perf_note = (
                " This message is for salespeople with LOW/below-expectation sales results."
                " Use a constructive, direct, but empathetic tone."
                " Clearly state that improvement is needed, offer simple and concrete steps, and express confidence that they can recover."
                " The message should be honest about the situation, but also encouraging."
                " Example: 'Your results are currently below expectations. Focus on these actions to get back on track—we believe in your potential.'"
            )
    system = (
        "Write a SHORT, WELL-STRUCTURED ANNOUNCEMENT for sales advisors, IN ENGLISH.\n"
        "STRICT rules:\n"
        f"- The target audience is the sales team (front-line). {style_note}{perf_note}\n"
        "- Start with a brief, relevant introduction (1-2 sentences) that sets the context and tone, but do NOT use standard email formulas like 'Dear team' or 'Best regards'.\n"
        "- After the intro, present ALL actions from 'allowed_actions' as a clear, easy-to-read list (use bullet points or numbers). Reformulate each action in a clear, natural way, but keep all essential details (such as dealership names, models, numbers, etc.). Do NOT omit or merge any action.\n"
        "- Do NOT mention statistics, numbers, or generalities from the summary.\n"
        "- The message must focus ONLY on concrete, simple actions that each employee can do directly (e.g.: follow up with leads, ask for feedback, improve product presentation, respond faster to customer requests, etc.).\n"
        "- DO NOT include any actions or suggestions that require manager approval, company-level changes, or management analysis.\n"
        "- DO NOT mention marketing, SEO, pricing, promotional campaigns, or anything outside the direct control of a sales advisor.\n"
        "- The message should encourage teamwork, initiative, and customer focus.\n"
        "- End with a short, style-appropriate closing phrase (motivational, professional, or friendly, depending on 'style'). Optionally, you may close with a collective address such as 'Dear Team' or similar, to give the message a warm and unified feel.\n"
        "- Return only plain text, no 'Subject', no automatic signature."
    )

    user = json.dumps({
        "goal": goal,
        "style": style,
        "specific_issues": issues_lines,   # facts from KPI
        "allowed_actions": allowed_actions # actions for advisors
    }, ensure_ascii=False)

    return _openai_chat(MODEL_MESSAGE, system, user, 0.2).strip()

def ai_classify_employee_performance(all_data: dict, business_goal: str = None, period_days: int = 30) -> dict:
    """
    Trimite toate datele brute la Assistant API și cere să clasifice angajații în underperformers, average, overperformers.
    Returnează dict cu chei: underperformers, average, overperformers (fiecare listă de dict cu nume, id, dealership, motiv).
    """
    import json
    today_str = __import__('datetime').date.today().strftime('%B %d, %Y')
    system = (
        f"Today is {today_str}. "
        "You are an expert business analyst for a car dealership group. "
        "You receive ALL raw data tables (sale_orders, car_sale_items, service_sale_items, dealerships, employees, customers, vehicles, service_items). "
        f"{'The manager\'s business objective is: \'%s\'. ' % business_goal if business_goal else ''}"
        f"Analyze the most recent {period_days} days of data. "
        "Your task: Based strictly on the data, classify all employees (salespersons) into three groups: underperformers (below target), average, and overperformers (above target). "
        "For each employee, provide: id, name, dealership, region, and a short reason (e.g. 'sales 30% below avg', 'top 10% revenue', etc). "
        "If possible, estimate the target from the data (e.g. average or median sales). Do NOT use any local statistics, only what you deduce from the data. "
        "Return STRICT JSON with three keys: underperformers, average, overperformers. Each is a list of employee objects as described. No extra text, no explanations."
    )
    user = json.dumps({"all_data": all_data}, ensure_ascii=False)
    try:
        result = _openai_chat(MODEL_MESSAGE, system, user, 0.2)
        # Extrage JSON robust (code block sau direct)
        import re
        match = re.search(r"```json\\s*([\\s\\S]+?)```", result)
        if match:
            json_str = match.group(1).strip()
            return json.loads(json_str)
        # fallback: caută primul { ... }
        s, e = result.find("{"), result.rfind("}") + 1
        if s != -1 and e > s:
            return json.loads(result[s:e])
        raise ValueError("Could not extract JSON from response")
    except Exception as e:
        return {"error": str(e)}

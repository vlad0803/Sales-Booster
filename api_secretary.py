from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import datetime as dt
import requests
from fastapi import Query, Request
import openai
import os
from ai_secretary_cli import ai_classify_employee_performance
from ai_secretary_cli import (
    fetch_all_core_data,
    validate_goal_with_ai,
    make_concrete_fallback,
    summarize_for_openai,
    KPIService,
    generate_recommendations,
    compose_message,
    ManagerPrompt
)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Endpoint to get grade for a specific employee using OpenAI completion (separate from chatbot) ===
@app.get("/employee_grade_openai")
def employee_grade_openai(employee_id: int = Query(...)):
    """
    Returns the grade (bronze/silver/gold) for a given employee_id using OpenAI completion, based on sales and diversity.
    """
    all_data = fetch_all_core_data()
    sale_orders = [o for o in all_data.get('sale_orders', []) if o.get('salesperson_id') == employee_id]
    sales_count = len(sale_orders)
    car_items = [c for c in all_data.get('car_sale_items', []) if c.get('order_id') in {o.get('order_id') for o in sale_orders}]
    total_models = len(car_items)
    service_items = [s for s in all_data.get('service_sale_items', []) if s.get('order_id') in {o.get('order_id') for o in sale_orders}]
    total_services = len(service_items)
    auto_sales = total_models
    service_sales = total_services
    prompt = f"""
You are an expert business analyst. Based on the following employee's performance, assign a grade: bronze, silver, or gold.
Rules:
- Bronze: default, for low or new performers.
- Silver: assign if the employee has at least 5 car sales OR at least 3 service sales (either condition is enough).
- Gold: assign if the employee has at least 10 car sales OR at least 5 service sales (either condition is enough).
If the employee does not meet the exact threshold for silver or gold, do not assign that grade. Be strict and do not round up. Return only the grade (bronze, silver, or gold), no explanation.

Examples:
- 9 car sales, 0 service sales → silver
- 10 car sales, 0 service sales → gold
- 5 car sales, 3 service sales → silver
- 10 car sales, 5 service sales → gold
- 5 car sales, 2 service sales → silver

Employee stats:
Car sales: {auto_sales}
Service sales: {service_sales}
"""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert business analyst. Based on the following employee's performance, assign a grade: bronze, silver, or gold. Rules: Bronze: default, for low or new performers. Silver: assign if the employee has at least 5 car sales OR at least 3 service sales (either condition is enough). Gold: assign if the employee has at least 10 car sales OR at least 5 service sales (either condition is enough). If the employee does not meet the exact threshold for silver or gold, do not assign that grade. Be strict and do not round up. Return only the grade (bronze, silver, or gold), no explanation. Examples: 9 car sales, 0 service sales → silver; 10 car sales, 0 service sales → gold; 5 car sales, 3 service sales → silver; 10 car sales, 5 service sales → gold; 5 car sales, 2 service sales → silver."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10,
        temperature=0
    )
    grade = response.choices[0].message.content.strip().lower()
    return {
        "employee_id": employee_id,
        "grade": grade,
        "sales_count": sales_count,
        "auto_sales": auto_sales,
        "service_sales": service_sales
    }


class GoalRequest(BaseModel):
    goal: str
    horizon: Optional[int] = None


class RecommendationRequest(BaseModel):
    goal: str
    horizon: Optional[int] = None
    region: Optional[str] = None
    selected: Optional[List[int]] = None
    style: Optional[str] = "friendly"
    performance_level: Optional[str] = None


class ReviewRequest(BaseModel):
    manager_id: int
    salesperson_id: int
    review_text: str


class GroupReviewRequest(BaseModel):
    group: str  # 'low', 'medium', 'high', 'all'
    review_text: str
    manager_id: Optional[int] = 1


@app.post("/send_group_review")
def send_group_review(req: GroupReviewRequest):
    """
    Primește grupa ('low', 'medium', 'high', 'all') și mesajul, trimite review la toți angajații din grupă.
    """
    all_data = fetch_all_core_data()
    # Clasifică angajații pe grupe
    performance = ai_classify_employee_performance(all_data)
    group_map = {
        "low": "underperformers",
        "medium": "average",
        "high": "overperformers"
    }
    groups = []
    if req.group == "all":
        groups = ["underperformers", "average", "overperformers"]
    elif req.group in group_map:
        groups = [group_map[req.group]]
    else:
        return {"success": False, "msg": "Invalid group."}
    # Trimite review la fiecare angajat din grupă
    results = []
    for group in groups:
        for emp in performance.get(group, []):
            salesperson_id = emp.get('id') or emp.get('employee_id') or emp.get('dealer_id')
            name = emp.get('name') or emp.get('full_name') or salesperson_id
            url = "http://localhost:8000/api/reviews"
            payload = {
                "manager_id": req.manager_id or 1,
                "salesperson_id": salesperson_id,
                "review_text": req.review_text
            }
            try:
                response = requests.post(url, json=payload)
                if response.status_code in (200, 201):
                    results.append({"salesperson_id": salesperson_id, "name": name, "success": True})
                else:
                    results.append({"salesperson_id": salesperson_id, "name": name, "success": False, "error": response.text})
            except Exception as e:
                results.append({"salesperson_id": salesperson_id, "name": name, "success": False, "error": str(e)})
    return {"success": True, "results": results, "msg": f"Sent to {len(results)} employees."}


@app.post("/recommendations")
def recommendations(req: RecommendationRequest):
    all_data = fetch_all_core_data()
    sales_rows = []
    for item in all_data.get("car_sale_items", []):
        sales_rows.append({
            "dealer_id": item.get("dealership_id", item.get("dealer_id", "UNKNOWN")),
            "date": item.get("date", item.get("order_date", dt.date.today().isoformat())),
            "region": item.get("region", "-"),
            "tier": item.get("tier", "-"),
            "leads": float(item.get("leads", 1)),
            "deals": float(item.get("deals", 1)),
            "revenue": float(item.get("revenue", item.get("amount", 0))),
            "points": float(item.get("points", 0)),
            "type": "car"
        })
    for item in all_data.get("service_items", []):
        sales_rows.append({
            "dealer_id": item.get("dealership_id", item.get("dealer_id", "UNKNOWN")),
            "date": item.get("date", item.get("order_date", dt.date.today().isoformat())),
            "region": item.get("region", "-"),
            "tier": item.get("tier", "-"),
            "leads": float(item.get("leads", 1)),
            "deals": float(item.get("deals", 1)),
            "revenue": float(item.get("revenue", item.get("amount", 0))),
            "points": float(item.get("points", 0)),
            "type": "service"
        })
    for item in all_data.get("service_sale_item", []):
        sales_rows.append({
            "dealer_id": item.get("dealership_id", item.get("dealer_id", "UNKNOWN")),
            "date": item.get("date", item.get("order_date", dt.date.today().isoformat())),
            "region": item.get("region", "-"),
            "tier": item.get("tier", "-"),
            "leads": float(item.get("leads", 1)),
            "deals": float(item.get("deals", 1)),
            "revenue": float(item.get("revenue", item.get("amount", 0))),
            "points": float(item.get("points", 0)),
            "type": "service_sale"
        })
    kpi_service = KPIService()
    horizon = req.horizon if req.horizon else 30
    _, kpis_preview = kpi_service.compute(sales_rows, since_days=horizon)
    # Obține sumarul AI pentru context
    import json
    today_str = dt.date.today().strftime('%B %d, %Y')
    system_summary = (
        f"Today is {today_str}. "
        "You are a data analyst. You receive ALL raw data tables (sale_orders, car_sale_items, service_sale_items, dealerships, employees, customers, vehicles, service_items) from a car dealership group. "
        f"The manager's business objective is: '{req.goal}'. "
        "When generating your summary, always deduce the analysis period from the business objective (e.g., 'in the next 60 days', 'last 14 days', etc.). If no period is mentioned, always analyze the most recent 30 days of data. "
        "The analysis period must always end today (the current date). For example, if the period is 'last 55 days', analyze from (today minus 55 days) up to today. Never use a future date. "
        "Write a concise, well-structured summary, split into short sections or paragraphs with clear headings or line breaks for readability. "
        "Focus your analysis, statistics, patterns, problems, and possible causes on what is most relevant for the given objective. "
        "If the objective mentions a specific product, service, or region, prioritize those in your summary and filter the findings accordingly. "
        "If the objective is general, provide a global summary. "
        "Include a section for dealerships with the lowest sales or negative trends, and another for concrete issues or opportunities visible in the data. "
        "Explicitly mention which product or service types (vehicle models, service items, etc.) have the lowest sales and in which dealership(s) these issues are most pronounced. "
        "Do NOT generalize, do NOT invent, do NOT add recommendations or generic business advice. "
        "Do NOT repeat information. Be clear, factual, and as brief as possible, but explain the main findings so a manager can understand the situation. "
        "Return only this structured summary, using headings or line breaks for clarity."
    )
    user_summary = json.dumps({"all_data": all_data}, ensure_ascii=False)
    from ai_secretary_cli import _openai_chat, MODEL_MESSAGE
    try:
        ai_summary = _openai_chat(MODEL_MESSAGE, system_summary, user_summary, 0.2)
    except Exception as e:
        ai_summary = f"[ERROR] LLM analysis failed: {e}"
    # Prompt pentru recomandări strict pe baza problemelor/statisticilor din sumar
    from ai_secretary_cli import generate_recommendations
    prompt = ManagerPrompt(goal=req.goal, horizon_days=horizon, filters={"region": req.region} if req.region else None)
    recommendations = generate_recommendations(kpis_preview, prompt, ai_summary=ai_summary)
    return {"recommendations": recommendations, "kpis_preview": kpis_preview, "ai_summary": ai_summary}


@app.post("/compose_message")
def compose_team_message(req: RecommendationRequest):
    all_data = fetch_all_core_data()
    sales_rows = []
    for item in all_data.get("car_sale_items", []):
        sales_rows.append({
            "dealer_id": item.get("dealership_id", item.get("dealer_id", "UNKNOWN")),
            "date": item.get("date", item.get("order_date", dt.date.today().isoformat())),
            "region": item.get("region", "-"),
            "tier": item.get("tier", "-"),
            "leads": float(item.get("leads", 1)),
            "deals": float(item.get("deals", 1)),
            "revenue": float(item.get("revenue", item.get("amount", 0))),
            "points": float(item.get("points", 0)),
            "type": "car"
        })
    for item in all_data.get("service_items", []):
        sales_rows.append({
            "dealer_id": item.get("dealership_id", item.get("dealer_id", "UNKNOWN")),
            "date": item.get("date", item.get("order_date", dt.date.today().isoformat())),
            "region": item.get("region", "-"),
            "tier": item.get("tier", "-"),
            "leads": float(item.get("leads", 1)),
            "deals": float(item.get("deals", 1)),
            "revenue": float(item.get("revenue", item.get("amount", 0))),
            "points": float(item.get("points", 0)),
            "type": "service"
        })
    for item in all_data.get("service_sale_item", []):
        sales_rows.append({
            "dealer_id": item.get("dealership_id", item.get("dealer_id", "UNKNOWN")),
            "date": item.get("date", item.get("order_date", dt.date.today().isoformat())),
            "region": item.get("region", "-"),
            "tier": item.get("tier", "-"),
            "leads": float(item.get("leads", 1)),
            "deals": float(item.get("deals", 1)),
            "revenue": float(item.get("revenue", item.get("amount", 0))),
            "points": float(item.get("points", 0)),
            "type": "service_sale"
        })
    kpi_service = KPIService()
    _, kpis_preview = kpi_service.compute(sales_rows, since_days=req.horizon)
    prompt = ManagerPrompt(goal=req.goal, horizon_days=req.horizon, filters={"region": req.region} if req.region else None)
    recommendations = generate_recommendations(kpis_preview, prompt)
    selected = [recommendations[i] for i in (req.selected or [0]) if i < len(recommendations)]
    message = compose_message(req.goal, selected, req.style, kpis_preview=kpis_preview, performance_level=req.performance_level)
    return {"message": message}


@app.post("/send_review")
def send_review(req: ReviewRequest):
    url = "http://localhost:8000/api/reviews"
    payload = {
        "manager_id": req.manager_id,
        "salesperson_id": req.salesperson_id,
        "review_text": req.review_text
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code in (200, 201):
            return {"success": True, "msg": f"Review created for salesperson_id={req.salesperson_id}."}
        else:
            return {"success": False, "msg": f"Failed to create review: {response.text}"}
    except Exception as e:
        return {"success": False, "msg": f"Error sending review: {e}"}


@app.post("/data_summary")
async def data_summary(request: Request):
    body = await request.json()
    goal = body.get("goal", "")
    all_data = fetch_all_core_data()
    import json
    # Prompt adaptat: summary-ul trebuie să fie focusat pe obiectivul dat
    today_str = dt.date.today().strftime('%B %d, %Y')
    system = (
        f"Today is {today_str}. "
        "You are a data analyst. You receive ALL raw data tables (sale_orders, car_sale_items, service_sale_items, dealerships, employees, customers, vehicles, service_items) from a car dealership group. "
        f"The manager's business objective is: '{goal}'. "
        "When generating your summary, always deduce the analysis period from the business objective (e.g., 'in the next 60 days', 'last 14 days', etc.). If no period is mentioned, always analyze the most recent 30 days of data. "
        "The analysis period must always end today (the current date). For example, if the period is 'last 55 days', analyze from (today minus 55 days) up to today. Never use a future date. "
        "Write a concise, well-structured summary, split into short sections or paragraphs with clear headings or line breaks for readability. "
        "Focus your analysis, statistics, patterns, problems, and possible causes on what is most relevant for the given objective. "
        "If the objective mentions a specific product, service, or region, prioritize those in your summary and filter the findings accordingly. "
        "If the objective is general, provide a global summary. "
        "Include a section for dealerships with the lowest sales or negative trends, and another for concrete issues or opportunities visible in the data. "
        "Explicitly mention which product or service types (vehicle models, service items, etc.) have the lowest sales and in which dealership(s) these issues are most pronounced. "
        "Do NOT generalize, do NOT invent, do NOT add recommendations or generic business advice. "
        "Do NOT repeat information. Be clear, factual, and as brief as possible, but explain the main findings so a manager can understand the situation. "
        "Return only this structured summary, using headings or line breaks for clarity."
    )
    user = json.dumps({"all_data": all_data}, ensure_ascii=False)
    from ai_secretary_cli import _openai_chat, MODEL_MESSAGE
    try:
        report = _openai_chat(MODEL_MESSAGE, system, user, 0.2)
    except Exception as e:
        report = f"[ERROR] LLM analysis failed: {e}"
    return {"ai_summary": report}


@app.get("/")
def root():
    return {"status": "API is running"}

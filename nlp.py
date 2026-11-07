"""NLP module powered by the Gemini API for natural language strategy parsing and portfolio commentary."""
import json
import os
try:
    import google.generativeai as genai
except ImportError:
    genai = None


def get_gemini_client(api_key=None):
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key and hasattr(importlib.import_module("streamlit"), "secrets"):
        try:
            import streamlit as st
            key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        except Exception:
            pass
    if key and genai:
        genai.configure(api_key=key)
        return True
    return False


def parse_question(prompt_text, api_key=None):
    """Parses a natural language investment prompt into backtest parameters using Gemini."""
    if not get_gemini_client(api_key):
        raise ValueError("Gemini API key not found. Please add GEMINI_API_KEY to your Streamlit secrets.")

    system_instruction = (
        "You are a quantitative backtesting assistant. Parse the user's investment strategy prompt "
        "and return a JSON object with a 'widgets' dictionary containing matching parameters: "
        "universe ('large_cap', 'tech', 'nyse', 'custom'), rule ('prior_year_winner', 'prior_year_loser', 'mom_6m', 'mom_12_1', 'custom_basket'), "
        "top_n (int 1-10), start_year (int 2005-2025), initial (int), annual_contrib (int), stop_pct (int 0-50). "
        "Also include 'explanation' (string describing what was applied) and 'unsupported' (string if anything was ignored)."
    )

    model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=system_instruction)
    response = model.generate_content(prompt_text)
    
    # Clean and parse JSON response
    text = response.text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
        
    return json.loads(text.strip())


def generate_ai_commentary(metrics_dict, strategy_name, api_key=None):
    """Generates an institutional-grade AI portfolio review using Gemini."""
    if not get_gemini_client(api_key):
        return "AI Commentary unavailable (Gemini API key missing)."

    prompt = (
        f"Analyze this backtested investment strategy '{strategy_name}' with the following metrics:\n"
        f"- Final Portfolio Value: ${metrics_dict.get('Final value', 0):,.2f}\n"
        f"- CAGR: {metrics_dict.get('CAGR', 0)*100:.2f}%\n"
        f"- Sharpe Ratio: {metrics_dict.get('Sharpe (rf=0)', 0):.2f}\n"
        f"- Max Drawdown: {metrics_dict.get('Max drawdown', 0)*100:.2f}%\n\n"
        "Provide a concise, highly professional 3-sentence quantitative critique highlighting risk-adjusted performance, "
        "drawdown resilience, and behavioral considerations for an investor."
    )

    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text.strip()

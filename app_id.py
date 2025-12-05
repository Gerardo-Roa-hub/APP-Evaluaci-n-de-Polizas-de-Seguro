# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 21:17:12 2025

@author: Dell
"""

import gettext
import google.generativeai as genai
import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.graph_objects as go
import os # Importar os
from dotenv import load_dotenv

#from PIL import Image
# --- Configuración de Idioma ---
# Directorio donde se almacenan las traducciones compiladas (.mo)
LOCALE_DIR = 'locales'

# Idiomas soportados
LANGUAGES = ['en', 'es']

# Inicializar estado de sesión
if 'language' not in st.session_state:
    st.session_state.language = 'en' # Inglés por defecto

# Selector de idioma (ej. en la sidebar)
selected_language = st.sidebar.selectbox(
    'Language/Idioma',
    LANGUAGES,
    format_func=lambda lang: 'English' if lang == 'en' else 'Español', # Mostrar nombres amigables
    index=LANGUAGES.index(st.session_state.language)
)

# Actualizar idioma y volver a ejecutar si cambia
if selected_language != st.session_state.language:
    st.session_state.language = selected_language
    st.experimental_rerun()

# --- Cargar Traducción con gettext ---
try:
    lang = gettext.translation('base', localedir=LOCALE_DIR, languages=[st.session_state.language])
    lang.install()
    _ = lang.gettext # La función mágica de traducción
except FileNotFoundError:
    # Si no se encuentra el archivo .mo, usar gettext nulo (no traduce)
    _ = gettext.gettext
    st.sidebar.warning(_("Language file not found for '{}'. Using default.".format(st.session_state.language)))


# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="🚀 LaunchPad Finance Analyzer :: [Roadvisors](https://roadvisors.com.mx)",
    page_icon="💡",
    layout="wide"
)

# --- Load Environment Variables & Configure AI ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_available = False
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        gemini_available = True
    except Exception as e:
        st.sidebar.error(f"⚠️ Gemini Connection Error: {e}")
else:
    st.sidebar.warning("🔑 Gemini API Key not found. AI Analysis Disabled.")

# --- Custom CSS Injection ---
custom_css = """
<style>
    /* ... (El CSS anterior se mantiene igual - omitido por brevedad) ... */
    /* General Body - Font is set in config.toml */
    body {
        font-family: 'Poppins', sans-serif;
        color: #0A2540; /* textColor from config */
    }

    /* Main Headers */
    h1, h2, h3 {
        font-weight: 700; /* Bolder headers */
        color: #0052CC; /* A slightly deeper blue for headers */
    }
    h1 {
        font-size: 2.8em;
        margin-bottom: 0.5em;
    }
    h2 {
        font-size: 2.0em;
        border-bottom: 2px solid #00A9FF; /* Primary color underline */
        padding-bottom: 0.3em;
        margin-top: 1.5em;
        margin-bottom: 1em;
    }
    h3 {
        font-size: 1.6em;
        color: #0A2540; /* Darker text color for subheaders */
        margin-bottom: 0.8em;
    }

    /* Main Calculation Button Style */
    .stButton>button {
        border: none;
        border-radius: 12px; /* Rounded corners */
        padding: 12px 28px;
        font-size: 1.1em;
        font-weight: 600;
        color: white;
        background: linear-gradient(90deg, #00A9FF, #007bff); /* Blue Gradient */
        box-shadow: 0 4px 15px rgba(0, 169, 255, 0.3); /* Soft shadow */
        transition: all 0.3s ease;
        display: block;
        margin: 20px auto; /* Center button */
        width: fit-content;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #007bff, #0056b3); /* Darker gradient on hover */
        box-shadow: 0 6px 20px rgba(0, 123, 255, 0.4);
        transform: translateY(-2px); /* Slight lift effect */
    }

    /* Input Field Labels (Year/Income/Expense) */
    .stNumberInput label, .stSelectbox label {
        font-weight: 500;
        color: #0A2540;
        font-size: 1.0em; /* Slightly larger labels */
    }
    /* Custom Styling for Input Table Headers */
    .input-table-header {
        font-weight: 600;
        font-size: 1.1em;
        color: #0052CC;
        text-align: center;
        padding-bottom: 10px;
    }

    /* KPI Table Styling */
    .kpi-table {
        width: 95%;
        max-width: 800px; /* Limit width */
        margin: 2em auto; /* Center table */
        border-collapse: separate; /* Allows for border-radius */
        border-spacing: 0;
        font-family: 'Poppins', sans-serif;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08); /* Subtle shadow */
        border-radius: 15px; /* Rounded corners for the table */
        overflow: hidden; /* Clip content to rounded corners */
    }
    .kpi-table th, .kpi-table td {
        padding: 16px 22px;
        border-bottom: 1px solid #E0E7FF; /* Light blue separator */
        text-align: left;
    }
    .kpi-table th {
        background-color: #00A9FF; /* Primary color header */
        color: white;
        font-weight: 600;
        font-size: 1.1em;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-top-left-radius: 15px; /* Rounded top corners */
        border-top-right-radius: 15px;
    }
    .kpi-table tr:last-child td {
        border-bottom: none; /* Remove border for last row */
    }
    .kpi-table tr:nth-child(even) td { /* Zebra striping - subtle */
       background-color: #F7FBFF; /* Very light blue */
    }
    .kpi-table td:nth-child(1) { /* KPI Name column */
        font-weight: 500;
        color: #0A2540;
    }
    .kpi-table td:nth-child(2) { /* Result column */
        font-weight: 700; /* Bold result */
        text-align: right;
        font-size: 1.15em; /* Slightly larger result */
        font-family: 'Courier New', Courier, monospace; /* Monospace for numbers */
    }

    /* Separator lines */
    hr {
        border-top: 1px solid #E0E7FF; /* Light blue separator */
    }

    /* Success/Warning/Error Messages */
    .stAlert {
        border-radius: 8px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# --- App Title & Intro ---
st.title("🚀 LaunchPad Finance Analyzer ::  [Roadvisors](https://roadvisors.com.mx)")
st.markdown("##### Turn Your Brilliant Idea into a **Financially Sound** Business!")
st.markdown("Hey Future Mogul! 👋 Got a game-changing idea? Let's crunch the numbers and see if it's got the financial firepower to succeed. Use this tool to quickly evaluate your project's potential.")
st.markdown("---")


# --- Functions (Keeping calculation logic, removing debug prints) ---
# ... (Las funciones calcular_vpn, calcular_irr, etc., se mantienen igual que en la versión anterior) ...
def calculate_vpn(discount_rate, cash_flows):
    """Calculates Net Present Value (NPV) using numpy_financial convention."""
    try:
        if cash_flows:
            return npf.npv(discount_rate, cash_flows)
        else:
            return 0.0
    except Exception as e:
        st.error(f"Error calculating NPV: {e}")
        return None

def calculate_irr(cash_flows):
    """Calculates Internal Rate of Return (IRR)."""
    try:
        if len(cash_flows) > 1 and cash_flows[0] < 0 and any(cf > 0 for cf in cash_flows[1:]):
            irr = npf.irr(cash_flows)
            return irr if not np.isnan(irr) and np.isfinite(irr) else None
        else:
            return None
    except Exception as e: # Catch broader exceptions including potential ValueErrors from npf.irr
        # st.error(f"Error calculating IRR: {e}") # Option to show error in UI
        print(f"DEBUG IRR Error: {e}") # Keep debug print for terminal for now
        return None


def calculate_payback_period(cash_flows):
    """Calculates Simple Payback Period."""
    if not cash_flows or cash_flows[0] >= 0:
        return None
    initial_investment_abs = abs(cash_flows[0])
    cumulative_flow = 0
    for i, flow in enumerate(cash_flows[1:], start=1):
        prev_cumulative_flow = cumulative_flow
        cumulative_flow += flow
        if cumulative_flow >= initial_investment_abs:
            if flow <= 0:
                return float('inf') if flow < 0 else i # Recovered end of year if flow=0, never if flow<0
            fraction_needed = initial_investment_abs - prev_cumulative_flow
            payback = (i - 1) + (fraction_needed / flow)
            return payback
    return float('inf') # Not recovered


def calculate_bcr(discount_rate, initial_investment, annual_income, annual_expenses):
    """Calculates Benefit-Cost Ratio (BCR). PV(Benefits) / PV(Costs)."""
    try:
        num_years = len(annual_income)
        if num_years == 0: return None

        # PV of Benefits (Incomes)
        pv_benefits = sum(annual_income[i] / ((1 + discount_rate) ** (i + 1)) for i in range(num_years))

        # PV of Costs (Initial Investment + PV of Expenses)
        pv_expenses = sum(abs(annual_expenses[i]) / ((1 + discount_rate) ** (i + 1)) for i in range(num_years))
        pv_total_costs_abs = abs(initial_investment) + pv_expenses

        if pv_total_costs_abs == 0:
            return float('inf') if pv_benefits > 0 else 0.0
        else:
            return pv_benefits / pv_total_costs_abs
    except Exception as e:
        st.error(f"Error calculating BCR: {e}")
        return None


def calculate_simple_roi(cash_flows):
    """Calculates Simple Return on Investment (Net Profit / Investment).""" # Descripción actualizada
    if not cash_flows or cash_flows[0] >= 0:
        return None
    initial_investment_abs = abs(cash_flows[0])
    if initial_investment_abs == 0: # Evitar división por cero
        return float('inf') if sum(cash_flows[1:]) > 0 else 0.0 # O puedes devolver None o 0.0

    total_future_inflows = sum(cash_flows[1:]) # Suma de los flujos de caja futuros (años 1 en adelante)
    net_profit = total_future_inflows - initial_investment_abs # Calcular el beneficio neto

    return net_profit / initial_investment_abs # Dividir el beneficio NETO por la inversión

# --- User Input Section ---
st.header("1. 💡 Your Project's Core Numbers")

col_inv, col_params = st.columns([1, 2])

with col_inv:
    st.subheader("Initial Spark 💰")
    initial_investment = st.number_input(
        "Investment Needed (€/$)",
        step=100.0,
        format="%.2f",
        value=-5935.0, # Example value
        help="Enter the total upfront cost (as a negative number)."
    )
    if initial_investment > 0:
        st.error("🚨 Whoops! Investment must be negative (it's cash outflow).")
        st.session_state.valid_investment = False
    elif initial_investment == 0:
        st.warning("⚠️ Zero investment? Some metrics (like IRR, PP, ROI) won't work.")
        st.session_state.valid_investment = True
    else:
        st.markdown(f"✅ Investment: **{initial_investment:,.2f}**")
        st.session_state.valid_investment = True

with col_params:
    st.subheader("The Ground Rules ⚖️")
    col_tasa, col_anos = st.columns(2)
    with col_tasa:
        discount_rate_str = st.selectbox(
            "Your Hurdle Rate (%)",
            options=[f"{i}%" for i in range(1, 51)],
            index=14, # Default 15%
            help="Minimum expected return (like WACC or your target profit rate)."
        )
        discount_rate = float(discount_rate_str.replace('%', '')) / 100.0
    with col_anos:
        project_years = st.selectbox(
            "Project Lifespan (Years)",
            options=[3, 5, 7, 10, 15, 20],
            index=1, # Default 5 years
            help="How many years will this project generate cash flow?"
        )

st.markdown("---")
st.header("2. 💸 The Cash Flow Forecast")
st.caption("Estimate the money coming IN (+) and going OUT (-) each year.")

# Dynamic Input for Cash Flows
annual_income_list = []
annual_expense_list = []
annual_net_flow_list = []

# Styling Headers for the Input Table
cols_header = st.columns(3)
cols_header[0].markdown("<p class='input-table-header'>Year</p>", unsafe_allow_html=True)
cols_header[1].markdown("<p class='input-table-header'>Income (+)</p>", unsafe_allow_html=True)
cols_header[2].markdown("<p class='input-table-header'>Expenses (-)</p>", unsafe_allow_html=True)

# Default values for 5 years matching the previous example
default_income = {1: 1967.0, 2: 1967.0, 3: 1967.0, 4: 1967.0, 5: 5096.0}
default_expense = {i: 0.0 for i in range(1, 6)}

for i in range(1, project_years + 1):
    row_cols = st.columns(3)
    with row_cols[0]:
        st.markdown(f"<div style='text-align: center; padding-top: 10px;'>Year {i}</div>", unsafe_allow_html=True)
    with row_cols[1]:
        # Use default if project_years is 5, else 0
        default_val_inc = default_income.get(i, 0.0) if project_years == 5 else 0.0
        income = st.number_input(
            f"Income_{i}", min_value=0.0, step=100.0, format="%.2f",
            label_visibility="collapsed", key=f"income_{i}", value=default_val_inc
        )
        annual_income_list.append(income if income is not None else 0.0)
    with row_cols[2]:
         # Use default if project_years is 5, else 0
        default_val_exp = default_expense.get(i, 0.0) if project_years == 5 else 0.0
        expense = st.number_input(
            f"Expense_{i}", max_value=0.0, step=100.0, format="%.2f",
            label_visibility="collapsed", key=f"expense_{i}", value=default_val_exp
        )
        annual_expense_list.append(expense if expense is not None else 0.0)

    # Calculate net flow for the year
    annual_net_flow_list.append(annual_income_list[-1] + annual_expense_list[-1])

# --- Prepare Final Cash Flow List ---
final_cash_flows = []
if 'valid_investment' in st.session_state and st.session_state.valid_investment:
    try:
        initial_investment_float = float(initial_investment)
        net_flows_float = [float(f) for f in annual_net_flow_list]
        final_cash_flows = [initial_investment_float] + net_flows_float
    except ValueError:
        st.error("Invalid number format in cash flows. Please check inputs.")
        final_cash_flows = [] # Reset if error
else:
    # Warning already shown in the investment section
    pass

# --- Calculation Trigger ---
st.markdown("---")
calculate_button = st.button("✨ Analyze My Project's Future! ✨")
st.markdown("---")

# --- Results Display ---
if calculate_button and final_cash_flows:
    st.header("3. 📊 Your Financial Dashboard :: [Roadvisors](https://roadvisors.com.mx)")


    # --- Calculate All KPIs ---
    # ... (Cálculos de KPIs como antes) ...
    npv_result = calculate_vpn(discount_rate, final_cash_flows)
    irr_result = calculate_irr(final_cash_flows)
    pp_result = calculate_payback_period(final_cash_flows)
    bcr_result = calculate_bcr(discount_rate, initial_investment_float, annual_income_list, annual_expense_list)
    roi_result = calculate_simple_roi(final_cash_flows)


    # --- Format KPIs for Display ---
    # ... (Formateo de KPIs como antes) ...
    npv_str = f"{npv_result:,.2f} €/$" if npv_result is not None else "🤔 Error/NA"
    irr_str = f"{irr_result:.2%}" if irr_result is not None else "📉 N/A or Complex"
    if pp_result is None: pp_str = "N/A (No Inv.)"
    elif pp_result == float('inf'): pp_str = f"⏳ > {project_years} yrs"
    elif isinstance(pp_result, (int, float)): pp_str = f"{pp_result:.2f} yrs"
    else: pp_str = "🤔 Error"
    bcr_str = f"{bcr_result:.3f}" if bcr_result is not None else "🤔 Error/NA"
    roi_str = f"{roi_result:.2%}" if roi_result is not None else "N/A (No Inv.)"

    kpi_data = {
        "⭐ Net Present Value (NPV)": (npv_result, npv_str),
        "🔥 Internal Rate of Return (IRR)": (irr_result, irr_str),
        "⏱️ Payback Period (PP)": (pp_result, pp_str),
        "⚖️ Benefit/Cost Ratio (BCR)": (bcr_result, bcr_str),
        "📈 Simple ROI": (roi_result, roi_str)
    }

    # --- Display KPIs in Styled Table ---
    # ... (Código HTML y lógica de colores de la tabla KPI como antes) ...
    st.subheader("Key Performance Indicators (KPIs)")
    st.caption("Is your project hitting the mark? Green is generally good!")

    html_table = """<table class="kpi-table">
        <thead><tr><th>Metric</th><th>Result</th></tr></thead>
        <tbody>
    """
    # --- Color Logic ---
    # VPN: Green > 0, Red < 0, Orange = 0
    vpn_color = "orange" if npv_result is not None and abs(npv_result) < 1e-6 else ("green" if (npv_result is not None and npv_result > 0) else ("red" if npv_result is not None else "grey"))
    # IRR: Green > Hurdle, Red < Hurdle
    irr_color = "green" if (irr_result is not None and irr_result > discount_rate) else ("red" if (irr_result is not None) else "grey")
    # PP: Green <= Half Life, Orange <= Full Life, Red > Full Life
    pp_color = "red" if pp_result == float('inf') else ("green" if (pp_result is not None and pp_result <= project_years / 2) else ("orange" if (pp_result is not None and pp_result <= project_years) else "grey"))
    # BCR: Green > 1, Red < 1, Orange = 1
    bcr_color = "orange" if bcr_result is not None and abs(bcr_result - 1.0) < 1e-6 else ("green" if (bcr_result is not None and bcr_result > 1.0) else ("red" if bcr_result is not None else "grey"))
    # ROI: Green > 0, Red < 0
    roi_color = "green" if (roi_result is not None and roi_result > 1e-6) else ("red" if (roi_result is not None and roi_result < -1e-6) else "grey")

    kpi_colors = {
        "⭐ Net Present Value (NPV)": vpn_color,
        "🔥 Internal Rate of Return (IRR)": irr_color,
        "⏱️ Payback Period (PP)": pp_color,
        "⚖️ Benefit/Cost Ratio (BCR)": bcr_color,
        "📈 Simple ROI": roi_color
    }

    for kpi_name, (kpi_value_raw, kpi_value_str) in kpi_data.items():
        color = kpi_colors.get(kpi_name, "grey")
        html_table += f"<tr><td>{kpi_name}</td><td><span style='color:{color};'>{kpi_value_str}</span></td></tr>"

    html_table += "</tbody></table>"
    st.markdown(html_table, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Cash Flow Chart ---
    # ... (Código del gráfico Plotly como antes) ...
    st.subheader("Visualizing Your Cash Flow Journey")
    years_axis = list(range(project_years + 1))
    cumulative_flows = np.cumsum(final_cash_flows).tolist()

    fig = go.Figure()
    # Bars for Net Annual Flow
    fig.add_trace(go.Bar(
        x=years_axis, y=final_cash_flows, name='Net Annual Flow',
        marker=dict(
            color=[('red' if x < 0 else ('#28a745' if x > 0 else 'grey')) for x in final_cash_flows], # Theme colors
            line=dict(width=0)
        ),
        hovertemplate='Year %{x}: Net Flow <b>%{y:,.2f}</b><extra></extra>'
    ))
    # Line for Cumulative Flow
    fig.add_trace(go.Scatter(
        x=years_axis, y=cumulative_flows, name='Cumulative Cash',
        mode='lines+markers',
        line=dict(color='#00A9FF', width=3), # Primary theme color
        marker=dict(size=8, symbol='circle', color='#FFFFFF', line=dict(width=2, color='#00A9FF')), # White markers with blue border
        hovertemplate='Year %{x}: Cumulative <b>%{y:,.2f}</b><extra></extra>'
    ))
    # Reference Line at Zero
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")

    fig.update_layout(
        title=f'Cash Flow Forecast & Payback ({project_years} Years)',
        xaxis_title='Project Year', yaxis_title='Amount (€/$)',
        xaxis=dict(tickmode='linear', dtick=1, showgrid=False, zeroline=False),
        yaxis=dict(gridcolor='#E0E7FF', zeroline=False), # Light blue grid
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.7)'),
        plot_bgcolor='rgba(0,0,0,0)', # Transparent plot background
        paper_bgcolor='rgba(0,0,0,0)', # Transparent paper background
        font=dict(family="Poppins, sans-serif", color="#0A2540"), # Theme font and color
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)


    # --- AI Analysis Section ---
    # ... (Sección de análisis IA como antes) ...
    st.markdown("---")
    st.subheader("🧠 Powered Insights (Sumary)")
    st.caption("Let me analyze these numbers and give you the executive summary.")

    if gemini_available and gemini_model:
        prompt = f"""
        Analyze the financial viability of a startup project for young entrepreneurs based on these inputs and calculated KPIs. The project spans {project_years} years with a hurdle rate (discount rate) of {discount_rate:.1%}.

        **Project Data:**
        *   Initial Investment: {initial_investment_float:,.2f}
        *   Annual Net Cash Flows (Year 1 to {project_years}): {[f"{cf:,.2f}" for cf in final_cash_flows[1:]]}
        *   Full Cash Flow Series (Year 0 to {project_years}): {[f"{cf:,.2f}" for cf in final_cash_flows]}

        **Calculated KPIs:**
        *   Net Present Value (NPV): {npv_str}
        *   Internal Rate of Return (IRR): {irr_str}
        *   Payback Period (PP): {pp_str}
        *   Benefit/Cost Ratio (BCR): {bcr_str}
        *   Simple ROI: {roi_str}

        **Your Task:** Provide a concise, encouraging, and actionable analysis for a young entrepreneur. Use markdown formatting.
        1.  **The Bottom Line (NPV):** Does the project create value ({npv_str}) compared to the {discount_rate:.1%} hurdle? Is it a 'Go' or 'No-Go' based purely on this?
        2.  **Profit Engine (IRR):** What's the project's *actual* rate of return ({irr_str})? How does it stack up against the hurdle rate? Explain what 'N/A or Complex' might mean (e.g., unusual cash flows).
        3.  **Cash Back Time (PP):** How quickly does the project pay back the initial investment ({pp_str})? Is this fast enough for a startup? What if it's never paid back?
        4.  **Bang for Buck (BCR):** For every dollar invested (in present value terms), how much benefit does the project generate ({bcr_str})? ( > 1 is good!).
        5.  **Simple Return (ROI):** What's the overall raw return ({roi_str}) on the initial cash invested (ignoring time value)?
        6.  **Founder's Takeaway:** Summarize the key financial signals. Is this project looking promising? What are the biggest financial strengths or weaknesses revealed? What should the entrepreneur consider next (e.g., refine estimates, seek funding, pivot)? Frame this as advice for their startup journey.

        Be positive but realistic. Focus on clarity and actionable insights for someone potentially new to these financial terms.
        """
        try:
            with st.spinner("⏳ AI is analyzing... Give it a moment!"):
                response = gemini_model.generate_content(prompt)
                st.markdown(response.text)
                st.success("✅ Roadvisors : Analysis Complete!")
        except Exception as e:
            st.error(f"🤖 Oops! AI Analysis failed: {e}")
            st.info("Check your inputs or try again later. You can still interpret the KPIs manually!")
    else:
        st.warning("AI Analysis is disabled. Please configure your Gemini API Key if you want AI insights.")


elif calculate_button:
    st.warning("Please ensure you have entered a valid negative investment and the cash flow table is populated.")

# --- Footer ---
st.markdown("---")
st.caption("© LaunchPad Finance Analyzer v0.6 | Path Fix | Fueling Young Entrepreneurship") # Updated version/note
st.caption("Made By [Roadvisors](https://roadvisors.com.mx)")
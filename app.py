"""
UAE Promo Pulse - Main Dashboard
================================
Streamlit dashboard with Executive and Manager views.

Features:
- 5+ Sidebar Filters: Date Range, City, Channel, Category, Brand, Fulfillment
- Toggle between Executive and Manager views
- 15 KPIs with full visualization
- What-If Simulation with constraint checking
- Download buttons for cleaned data and issues log
- Auto-generated recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os

from data_generator import generate_all_data
from cleaner import DataCleaner
from simulator import KPICalculator, PromoSimulator, generate_recommendation

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="UAE Promo Pulse",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# LOGO AND STYLES
# =============================================================================
def render_logo():
    """Render SVG logo for UAE Promo Pulse"""
    logo_html = """
    <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 1rem;">
        <svg width="420" height="100" viewBox="0 0 420 100" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#1E3A5F;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#2E5A8F;stop-opacity:1" />
                </linearGradient>
                <linearGradient id="pulseGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:#00D4AA;stop-opacity:1" />
                    <stop offset="50%" style="stop-color:#00E5BB;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#00F5CC;stop-opacity:1" />
                </linearGradient>
                <linearGradient id="uaeGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" style="stop-color:#00A86B;stop-opacity:1" />
                    <stop offset="33%" style="stop-color:#FFFFFF;stop-opacity:1" />
                    <stop offset="66%" style="stop-color:#000000;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#CE1126;stop-opacity:1" />
                </linearGradient>
                <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
                    <feDropShadow dx="2" dy="2" stdDeviation="3" flood-opacity="0.3"/>
                </filter>
            </defs>
            <rect x="5" y="10" width="410" height="80" rx="15" ry="15" fill="url(#bgGrad)" filter="url(#shadow)"/>
            <rect x="5" y="10" width="8" height="80" rx="4" ry="0" fill="url(#uaeGrad)"/>
            <g transform="translate(25, 25)">
                <circle cx="25" cy="25" r="22" fill="rgba(255,255,255,0.1)" stroke="url(#pulseGrad)" stroke-width="2"/>
                <path d="M15 20 L18 20 L22 35 L35 35 L38 23 L20 23" fill="none" stroke="url(#pulseGrad)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                <circle cx="24" cy="42" r="3" fill="url(#pulseGrad)"/>
                <circle cx="34" cy="42" r="3" fill="url(#pulseGrad)"/>
                <path d="M42 25 Q47 15, 52 25 Q57 35, 62 25" fill="none" stroke="url(#pulseGrad)" stroke-width="2" stroke-linecap="round"/>
            </g>
            <text x="95" y="45" font-family="Arial Black, sans-serif" font-size="22" font-weight="900" fill="#FFFFFF">UAE</text>
            <text x="150" y="45" font-family="Arial Black, sans-serif" font-size="22" font-weight="900" fill="url(#pulseGrad)">PROMO</text>
            <text x="260" y="45" font-family="Arial Black, sans-serif" font-size="22" font-weight="900" fill="#FFFFFF">PULSE</text>
            <text x="95" y="70" font-family="Arial, sans-serif" font-size="11" fill="rgba(255,255,255,0.8)" letter-spacing="1">RETAIL ANALYTICS &amp; PROMOTION SIMULATOR</text>
            <path d="M355 35 L365 35 L370 25 L375 45 L380 30 L385 40 L390 35 L400 35" 
                  fill="none" stroke="url(#pulseGrad)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <animate attributeName="opacity" values="0.5;1;0.5" dur="2s" repeatCount="indefinite"/>
            </path>
        </svg>
    </div>
    """
    st.markdown(logo_html, unsafe_allow_html=True)


def render_sidebar_logo():
    """Render smaller logo for sidebar"""
    sidebar_logo = """
    <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 1rem; padding: 10px;">
        <svg width="200" height="55" viewBox="0 0 200 55" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="sbBgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#1E3A5F;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#2E5A8F;stop-opacity:1" />
                </linearGradient>
                <linearGradient id="sbPulseGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:#00D4AA;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#00F5CC;stop-opacity:1" />
                </linearGradient>
            </defs>
            <rect x="2" y="2" width="196" height="50" rx="10" fill="url(#sbBgGrad)"/>
            <text x="15" y="25" font-family="Arial Black, sans-serif" font-size="11" font-weight="900" fill="#FFFFFF">UAE</text>
            <text x="45" y="25" font-family="Arial Black, sans-serif" font-size="11" font-weight="900" fill="url(#sbPulseGrad)">PROMO PULSE</text>
            <text x="15" y="40" font-family="Arial, sans-serif" font-size="8" fill="rgba(255,255,255,0.7)">Retail Analytics Dashboard</text>
            <path d="M150 20 L160 20 L165 12 L170 28 L175 18 L180 24 L185 20 L195 20" 
                  fill="none" stroke="url(#sbPulseGrad)" stroke-width="1.5" stroke-linecap="round">
                <animate attributeName="opacity" values="0.5;1;0.5" dur="2s" repeatCount="indefinite"/>
            </path>
        </svg>
    </div>
    """
    st.sidebar.markdown(sidebar_logo, unsafe_allow_html=True)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 1.1rem; color: #666; text-align: center; 
        margin-bottom: 1.5rem; margin-top: -0.5rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #1E3A5F 0%, #2E5A8F 100%);
        padding: 1rem; border-radius: 12px; color: white; text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1); height: 100%;
    }
    .kpi-value {font-size: 1.6rem; font-weight: bold; color: #00D4AA;}
    .kpi-label {font-size: 0.8rem; color: rgba(255,255,255,0.8); margin-top: 0.2rem;}
    .kpi-delta {font-size: 0.75rem; margin-top: 0.2rem;}
    .kpi-delta-positive {color: #00D4AA;}
    .kpi-delta-negative {color: #f5576c;}
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.2rem; border-radius: 10px; color: white;
        margin: 0.5rem 0 1rem 0; font-size: 0.9rem; line-height: 1.6;
    }
    .insight-success {background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);}
    .insight-warning {background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);}
    .insight-title {font-weight: bold; font-size: 1rem; margin-bottom: 0.5rem;}
    .insight-detail {
        font-size: 0.85rem; margin-top: 0.5rem; padding-top: 0.5rem; 
        border-top: 1px solid rgba(255,255,255,0.3);
    }
    .section-header {
        background: linear-gradient(90deg, #1E3A5F 0%, transparent 100%);
        padding: 0.7rem 1rem; border-radius: 8px; color: white;
        font-size: 1.1rem; font-weight: 600; margin: 1.2rem 0 0.8rem 0;
    }
    .view-badge {
        display: inline-block; padding: 0.3rem 0.8rem; border-radius: 20px;
        font-size: 0.75rem; font-weight: bold; margin-bottom: 0.5rem;
    }
    .view-executive {background: linear-gradient(90deg, #667eea, #764ba2); color: white;}
    .view-manager {background: linear-gradient(90deg, #11998e, #38ef7d); color: white;}
    .constraint-card {
        background: #fff3cd; border-left: 4px solid #ffc107;
        padding: 0.8rem; border-radius: 5px; margin: 0.3rem 0;
    }
    .constraint-card-error {
        background: #f8d7da; border-left: 4px solid #dc3545;
    }
    .stApp > header {background-color: transparent;}
    .block-container {padding-top: 1rem;}
    div[data-testid="stMetricValue"] {font-size: 1.5rem;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def kpi_card(label, value, delta=None, delta_type="positive"):
    """Render a styled KPI card"""
    delta_html = ""
    if delta is not None:
        delta_class = f"kpi-delta-{delta_type}"
        delta_icon = "↑" if delta_type == "positive" else "↓"
        delta_html = f'<div class="kpi-delta {delta_class}">{delta_icon} {delta}</div>'
    
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def detailed_insight(title, main_text, details, actions, itype="info"):
    """Render detailed insight box with actions"""
    css = f"insight-box insight-{itype}" if itype != "info" else "insight-box"
    icon = {"info": "💡", "success": "✅", "warning": "⚠️"}.get(itype, "💡")
    details_html = "".join([f"<br>• {d}" for d in details]) if details else ""
    actions_html = ""
    if actions:
        actions_html = f"<div class='insight-detail'><strong>🎯 Actions:</strong>{''.join([f'<br>→ {a}' for a in actions])}</div>"
    st.markdown(f'<div class="{css}"><div class="insight-title">{icon} {title}</div>{main_text}{details_html}{actions_html}</div>', unsafe_allow_html=True)


def section_header(title, icon="📊"):
    """Render section header"""
    st.markdown(f'<div class="section-header">{icon} {title}</div>', unsafe_allow_html=True)


def view_badge(view):
    """Render view badge"""
    css = "view-executive" if view == "Executive" else "view-manager"
    emoji = "👔" if view == "Executive" else "🔧"
    st.markdown(f'<span class="view-badge {css}">{emoji} {view} View</span>', unsafe_allow_html=True)


def validate_data(products_df, stores_df, sales_df, inventory_df):
    """Validate uploaded data has required columns"""
    errors = []
    if products_df is not None:
        required = ['product_id', 'category', 'base_price_aed', 'unit_cost_aed']
        missing = [c for c in required if c not in products_df.columns]
        if missing:
            errors.append(f"Products missing columns: {missing}")
    if stores_df is not None:
        required = ['store_id', 'city', 'channel']
        missing = [c for c in required if c not in stores_df.columns]
        if missing:
            errors.append(f"Stores missing columns: {missing}")
    if sales_df is not None:
        required = ['order_id', 'order_time', 'product_id', 'store_id', 'qty', 'selling_price_aed', 'payment_status']
        missing = [c for c in required if c not in sales_df.columns]
        if missing:
            errors.append(f"Sales missing columns: {missing}")
    if inventory_df is not None:
        required = ['snapshot_date', 'product_id', 'store_id', 'stock_on_hand']
        missing = [c for c in required if c not in inventory_df.columns]
        if missing:
            errors.append(f"Inventory missing columns: {missing}")
    return errors


def add_defaults(df, dtype):
    """Add default columns if missing"""
    if dtype == 'products':
        if 'brand' not in df.columns:
            df['brand'] = 'Unknown'
        if 'tax_rate' not in df.columns:
            df['tax_rate'] = 0.05
        if 'launch_flag' not in df.columns:
            df['launch_flag'] = 'Regular'
    elif dtype == 'stores':
        if 'fulfillment_type' not in df.columns:
            df['fulfillment_type'] = 'Own'
    elif dtype == 'sales':
        if 'discount_pct' not in df.columns:
            df['discount_pct'] = 0
        if 'return_flag' not in df.columns:
            df['return_flag'] = 0
    elif dtype == 'inventory':
        if 'reorder_point' not in df.columns:
            df['reorder_point'] = 20
        if 'lead_time_days' not in df.columns:
            df['lead_time_days'] = 7
    return df


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
for key in ['data_loaded', 'data', 'raw_data', 'raw_data_generated', 'cleaning_stats', 'upload_mode']:
    if key not in st.session_state:
        st.session_state[key] = None if key in ['data', 'raw_data', 'cleaning_stats'] else False


# =============================================================================
# SIDEBAR
# =============================================================================
render_sidebar_logo()

st.sidebar.markdown("## 🎛️ Dashboard Controls")

# View Toggle
view = st.sidebar.radio(
    "📊 Select View",
    ["Executive (CEO/CFO)", "Manager (Ops)"],
    help="Executive: Financial KPIs | Manager: Operational KPIs"
)
view_type = "Executive" if "Executive" in view else "Manager"

st.sidebar.markdown("---")

# Data Source
st.sidebar.markdown("### 📁 Data Source")
data_source = st.sidebar.radio("Choose:", ["Generate Sample Data", "Upload Your Own Data"])

if data_source == "Generate Sample Data":
    st.session_state.upload_mode = False
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🎲 Generate", use_container_width=True):
            with st.spinner("Generating synthetic data..."):
                st.session_state.raw_data = generate_all_data('data/raw')
                st.session_state.raw_data_generated = True
                st.session_state.data_loaded = False
            st.rerun()
    
    with col2:
        if st.button("🧹 Clean", use_container_width=True, 
                    disabled=not st.session_state.raw_data_generated):
            if st.session_state.raw_data:
                with st.spinner("Cleaning data..."):
                    orig = {k: len(v) for k, v in st.session_state.raw_data.items() if isinstance(v, pd.DataFrame)}
                    cleaner = DataCleaner()
                    cleaned = cleaner.clean_all(
                        st.session_state.raw_data['products'],
                        st.session_state.raw_data['stores'],
                        st.session_state.raw_data['sales'],
                        st.session_state.raw_data['inventory'],
                        'data/cleaned'
                    )
                    st.session_state.data = cleaned
                    cc = {k: len(v) for k, v in cleaned.items() if isinstance(v, pd.DataFrame) and k != 'issues'}
                    st.session_state.cleaning_stats = {
                        'original': orig,
                        'cleaned': cc,
                        'removed': {k: orig.get(k, 0) - cc.get(k, 0) for k in cc},
                        'total_issues': len(cleaned['issues']),
                        'issues_summary': cleaned['issues']['issue_type'].value_counts().to_dict() if len(cleaned['issues']) > 0 else {}
                    }
                    st.session_state.data_loaded = True
                st.rerun()
    
    if st.session_state.raw_data_generated and not st.session_state.data_loaded:
        st.sidebar.success("✅ Raw data generated!")
    if st.session_state.data_loaded:
        st.sidebar.success("✅ Data cleaned & ready!")

else:
    st.session_state.upload_mode = True
    st.sidebar.markdown("#### Upload CSV Files")
    
    products_file = st.sidebar.file_uploader("📦 Products", type=['csv'], key='products_upload')
    stores_file = st.sidebar.file_uploader("🏪 Stores", type=['csv'], key='stores_upload')
    sales_file = st.sidebar.file_uploader("💰 Sales", type=['csv'], key='sales_upload')
    inventory_file = st.sidebar.file_uploader("📊 Inventory", type=['csv'], key='inventory_upload')
    
    if st.sidebar.button("📤 Load & Clean Data", use_container_width=True):
        if all([products_file, stores_file, sales_file, inventory_file]):
            try:
                products_df = pd.read_csv(products_file)
                stores_df = pd.read_csv(stores_file)
                sales_df = pd.read_csv(sales_file)
                inventory_df = pd.read_csv(inventory_file)
                
                errors = validate_data(products_df, stores_df, sales_df, inventory_df)
                if errors:
                    for e in errors:
                        st.sidebar.error(e)
                else:
                    products_df = add_defaults(products_df, 'products')
                    stores_df = add_defaults(stores_df, 'stores')
                    sales_df = add_defaults(sales_df, 'sales')
                    inventory_df = add_defaults(inventory_df, 'inventory')
                    
                    orig = {
                        'products': len(products_df),
                        'stores': len(stores_df),
                        'sales': len(sales_df),
                        'inventory': len(inventory_df)
                    }
                    
                    with st.spinner("Processing uploaded data..."):
                        cleaner = DataCleaner()
                        cleaned = cleaner.clean_all(products_df, stores_df, sales_df, inventory_df, 'data/cleaned')
                        st.session_state.data = cleaned
                        cc = {k: len(v) for k, v in cleaned.items() if isinstance(v, pd.DataFrame) and k != 'issues'}
                        st.session_state.cleaning_stats = {
                            'original': orig,
                            'cleaned': cc,
                            'removed': {k: orig.get(k, 0) - cc.get(k, 0) for k in cc},
                            'total_issues': len(cleaned['issues']),
                            'issues_summary': cleaned['issues']['issue_type'].value_counts().to_dict() if len(cleaned['issues']) > 0 else {}
                        }
                        st.session_state.data_loaded = True
                        st.session_state.raw_data_generated = False
                    
                    st.sidebar.success("✅ Data uploaded and cleaned!")
                    st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
        else:
            st.sidebar.warning("⚠️ Please upload all 4 CSV files")

# =============================================================================
# FILTERS (shown when data is loaded)
# =============================================================================
st.sidebar.markdown("---")
filters = {}

if st.session_state.data_loaded and st.session_state.data:
    st.sidebar.markdown("### 🔍 Filters")
    data = st.session_state.data
    
    # 1. Date Range Filter
    if 'sales' in data:
        sales_dates = pd.to_datetime(data['sales']['order_time'], errors='coerce')
        min_date = sales_dates.min().date() if not sales_dates.isna().all() else datetime.now().date() - timedelta(days=120)
        max_date = sales_dates.max().date() if not sales_dates.isna().all() else datetime.now().date()
        
        date_range = st.sidebar.date_input(
            "📅 Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="date_filter"
        )
        if len(date_range) == 2:
            filters['date_from'] = date_range[0]
            filters['date_to'] = date_range[1]
    
    # 2. City Filter
    if 'stores' in data:
        cities = ['All'] + sorted(data['stores']['city'].unique().tolist())
        filters['city'] = st.sidebar.selectbox("🏙️ City", cities)
    
    # 3. Channel Filter
    if 'stores' in data:
        channels = ['All'] + sorted(data['stores']['channel'].unique().tolist())
        filters['channel'] = st.sidebar.selectbox("📱 Channel", channels)
    
    # 4. Category Filter
    if 'products' in data:
        categories = ['All'] + sorted(data['products']['category'].unique().tolist())
        filters['category'] = st.sidebar.selectbox("📦 Category", categories)
    
    # 5. Brand Filter
    if 'products' in data and 'brand' in data['products'].columns:
        brands = ['All'] + sorted(data['products']['brand'].unique().tolist())
        filters['brand'] = st.sidebar.selectbox("🏷️ Brand", brands)
    
    # 6. Fulfillment Filter
    if 'stores' in data and 'fulfillment_type' in data['stores'].columns:
        fulfillments = ['All'] + sorted(data['stores']['fulfillment_type'].unique().tolist())
        filters['fulfillment'] = st.sidebar.selectbox("🚚 Fulfillment", fulfillments)

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎮 Simulation Parameters")

sim_params = {
    'discount_pct': st.sidebar.slider("💸 Discount %", 0, 50, 15, help="Discount percentage to simulate"),
    'promo_budget': st.sidebar.number_input("💰 Promo Budget (AED)", 10000, 500000, 50000, step=5000),
    'margin_floor': st.sidebar.slider("📉 Margin Floor %", 5, 30, 15, help="Minimum acceptable margin"),
    'simulation_days': st.sidebar.selectbox("📆 Simulation Days", [7, 14], index=1)
}

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; font-size: 0.75rem; color: #888;">
    UAE Promo Pulse v2.0<br>
    Retail Analytics Dashboard
</div>
""", unsafe_allow_html=True)


# =============================================================================
# MAIN CONTENT
# =============================================================================
render_logo()
st.markdown('<p class="main-header">Advanced Retail Analytics & Promotion Simulator for UAE Market</p>', unsafe_allow_html=True)

# Show view badge
view_badge(view_type)

if not st.session_state.data_loaded:
    # Welcome screen
    st.markdown("---")
    
    if st.session_state.upload_mode:
        st.info("👈 Upload your CSV files in the sidebar to begin analysis")
        
        with st.expander("📋 Required Data Format", expanded=True):
            st.markdown("""
            **Products CSV:** `product_id`, `category`, `base_price_aed`, `unit_cost_aed`, (optional: `brand`)
            
            **Stores CSV:** `store_id`, `city`, `channel`, (optional: `fulfillment_type`)
            
            **Sales CSV:** `order_id`, `order_time`, `product_id`, `store_id`, `qty`, `selling_price_aed`, `payment_status`, (optional: `discount_pct`, `return_flag`)
            
            **Inventory CSV:** `snapshot_date`, `product_id`, `store_id`, `stock_on_hand`
            """)
            
            st.markdown("#### 📥 Download Templates")
            cols = st.columns(4)
            with cols[0]:
                st.download_button(
                    "📦 Products",
                    pd.DataFrame({
                        'product_id': ['PROD_0001'],
                        'category': ['Electronics'],
                        'brand': ['Samsung'],
                        'base_price_aed': [1500],
                        'unit_cost_aed': [900]
                    }).to_csv(index=False),
                    "products_template.csv",
                    use_container_width=True
                )
            with cols[1]:
                st.download_button(
                    "🏪 Stores",
                    pd.DataFrame({
                        'store_id': ['STORE_01'],
                        'city': ['Dubai'],
                        'channel': ['App'],
                        'fulfillment_type': ['Own']
                    }).to_csv(index=False),
                    "stores_template.csv",
                    use_container_width=True
                )
            with cols[2]:
                st.download_button(
                    "💰 Sales",
                    pd.DataFrame({
                        'order_id': ['ORD_000001'],
                        'order_time': ['2024-01-15 10:30:00'],
                        'product_id': ['PROD_0001'],
                        'store_id': ['STORE_01'],
                        'qty': [1],
                        'selling_price_aed': [1350],
                        'discount_pct': [10],
                        'payment_status': ['Paid'],
                        'return_flag': [0]
                    }).to_csv(index=False),
                    "sales_template.csv",
                    use_container_width=True
                )
            with cols[3]:
                st.download_button(
                    "📊 Inventory",
                    pd.DataFrame({
                        'snapshot_date': ['2024-01-15'],
                        'product_id': ['PROD_0001'],
                        'store_id': ['STORE_01'],
                        'stock_on_hand': [50]
                    }).to_csv(index=False),
                    "inventory_template.csv",
                    use_container_width=True
                )
    
    elif st.session_state.raw_data_generated:
        st.success("✅ Raw data generated! Click 'Clean' in the sidebar to process.")
        
        if st.session_state.raw_data:
            meta = st.session_state.raw_data.get('metadata', {})
            if meta:
                st.markdown("#### 📊 Generated Data Summary")
                cols = st.columns(4)
                cols[0].metric("Products", f"{meta.get('products_count', 0):,}")
                cols[1].metric("Stores", f"{meta.get('stores_count', 0):,}")
                cols[2].metric("Sales", f"{meta.get('sales_count', 0):,}")
                cols[3].metric("Inventory", f"{meta.get('inventory_records', 0):,}")
                
                st.markdown("#### 🔧 Injected Data Issues (for cleaning practice)")
                dirty = meta.get('dirty_data_injected', {})
                if dirty:
                    issue_df = pd.DataFrame([
                        {"Issue Type": k.replace('_', ' ').title(), "Count": v}
                        for k, v in dirty.items()
                    ])
                    st.dataframe(issue_df, hide_index=True, use_container_width=True)
    
    else:
        st.info("👈 Generate sample data or upload your own CSV files to start")
        
        # Feature cards
        st.markdown("---")
        st.markdown("### 🚀 Dashboard Features")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1E3A5F 0%, #2E5A8F 100%); 
                        padding: 1.5rem; border-radius: 10px; color: white; min-height: 200px;">
                <h4 style="color: #00D4AA;">📊 15+ KPIs</h4>
                <ul style="font-size: 0.85rem; padding-left: 1.2rem;">
                    <li>Gross & Net Revenue</li>
                    <li>Margin Analysis</li>
                    <li>Return & Failure Rates</li>
                    <li>Stockout Risk Metrics</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1E3A5F 0%, #2E5A8F 100%); 
                        padding: 1.5rem; border-radius: 10px; color: white; min-height: 200px;">
                <h4 style="color: #00D4AA;">🎯 What-If Simulation</h4>
                <ul style="font-size: 0.85rem; padding-left: 1.2rem;">
                    <li>Demand Uplift Modeling</li>
                    <li>Constraint Checking</li>
                    <li>Budget Optimization</li>
                    <li>Risk Assessment</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1E3A5F 0%, #2E5A8F 100%); 
                        padding: 1.5rem; border-radius: 10px; color: white; min-height: 200px;">
                <h4 style="color: #00D4AA;">🧹 Data Quality</h4>
                <ul style="font-size: 0.85rem; padding-left: 1.2rem;">
                    <li>7+ Validation Rules</li>
                    <li>Issue Logging</li>
                    <li>Auto-Cleaning</li>
                    <li>Quality Reports</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

else:
    # ==========================================================================
    # DASHBOARD WITH DATA LOADED
    # ==========================================================================
    data = st.session_state.data
    
    # Initialize calculators
    kpi_calc = KPICalculator(
        data['sales'], data['products'], data['stores'], data['inventory']
    )
    simulator = PromoSimulator(
        data['sales'], data['products'], data['stores'], data['inventory']
    )
    
    # Apply filters
    filtered_df = kpi_calc.filter_data(
        city=filters.get('city'),
        channel=filters.get('channel'),
        category=filters.get('category'),
        brand=filters.get('brand'),
        fulfillment=filters.get('fulfillment'),
        date_from=filters.get('date_from'),
        date_to=filters.get('date_to')
    )
    
    # Compute KPIs
    kpis = kpi_calc.compute_kpis(filtered_df)
    daily = kpi_calc.compute_daily(filtered_df)
    
    # Run simulation
    sim_results = simulator.run_simulation(
        sim_params['discount_pct'],
        sim_params['promo_budget'],
        sim_params['margin_floor'],
        sim_params['simulation_days'],
        city=filters.get('city'),
        channel=filters.get('channel'),
        category=filters.get('category')
    )
    
    # ==========================================================================
    # DATA QUALITY REPORT
    # ==========================================================================
    if st.session_state.cleaning_stats:
        with st.expander("📊 Data Quality Report", expanded=False):
            stats = st.session_state.cleaning_stats
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Original Records", f"{sum(stats['original'].values()):,}")
            col2.metric("Records Removed", f"{sum(stats['removed'].values()):,}", delta=f"-{sum(stats['removed'].values())}", delta_color="inverse")
            col3.metric("Cleaned Records", f"{sum(stats['cleaned'].values()):,}")
            col4.metric("Issues Logged", f"{stats['total_issues']:,}")
            
            if stats['issues_summary']:
                st.markdown("#### Issue Types Distribution")
                issues_df = pd.DataFrame([
                    {"Issue Type": k, "Count": v}
                    for k, v in stats['issues_summary'].items()
                ]).sort_values('Count', ascending=False)
                
                fig = px.bar(
                    issues_df, x='Issue Type', y='Count',
                    color='Count', color_continuous_scale='Reds'
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================================================
    # EXECUTIVE VIEW
    # ==========================================================================
    if view_type == "Executive":
        
        # KPI Cards Row 1
        section_header("Financial KPIs", "💰")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            kpi_card("Net Revenue", f"AED {kpis['net_revenue']:,.0f}")
        with col2:
            kpi_card("Gross Margin", f"{kpis['gross_margin_pct']:.1f}%",
                    "Healthy" if kpis['gross_margin_pct'] >= 15 else "Low",
                    "positive" if kpis['gross_margin_pct'] >= 15 else "negative")
        with col3:
            kpi_card("COGS", f"AED {kpis['cogs']:,.0f}")
        with col4:
            kpi_card("Avg Discount", f"{kpis['avg_discount_pct']:.1f}%")
        with col5:
            profit = sim_results['results']['profit_proxy'] if sim_results['results'] else 0
            kpi_card("Profit Proxy (Sim)", f"AED {profit:,.0f}",
                    "Profitable" if profit > 0 else "Loss",
                    "positive" if profit > 0 else "negative")
        
        # KPI Cards Row 2
        st.markdown("")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            kpi_card("Gross Revenue", f"AED {kpis['gross_revenue']:,.0f}")
        with col2:
            kpi_card("Refund Amount", f"AED {kpis['refund_amount']:,.0f}")
        with col3:
            kpi_card("Total Orders", f"{kpis['total_orders']:,}")
        with col4:
            kpi_card("AOV", f"AED {kpis['aov']:,.0f}")
        with col5:
            budget_util = sim_results['results']['budget_utilization'] if sim_results['results'] else 0
            kpi_card("Budget Utilization", f"{budget_util:.1f}%")
        
        st.markdown("---")
        
        # Chart Row 1: Revenue Trend & Revenue by City
        col1, col2 = st.columns(2)
        
        with col1:
            section_header("Net Revenue Trend", "📈")
            if len(daily) > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=daily['date'], y=daily['revenue'],
                    mode='lines+markers',
                    name='Daily Revenue',
                    line=dict(color='#1E3A5F', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(30,58,95,0.1)'
                ))
                
                # Add 7-day moving average
                if len(daily) >= 7:
                    daily['ma7'] = daily['revenue'].rolling(7, min_periods=1).mean()
                    fig.add_trace(go.Scatter(
                        x=daily['date'], y=daily['ma7'],
                        mode='lines',
                        name='7-Day MA',
                        line=dict(color='#00D4AA', width=3, dash='dot')
                    ))
                
                fig.update_layout(
                    height=350,
                    legend=dict(orientation="h", y=-0.15),
                    plot_bgcolor='rgba(0,0,0,0)',
                    yaxis_title="Revenue (AED)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Insight
                if len(daily) >= 7:
                    trend = daily['revenue'].iloc[-7:].mean() - daily['revenue'].iloc[:7].mean()
                    detailed_insight(
                        "Revenue Trend Analysis",
                        f"Revenue is <strong>{'increasing' if trend > 0 else 'decreasing'}</strong> with AED {abs(trend):,.0f} change in weekly average",
                        [f"Total days analyzed: {len(daily)}", f"Peak day revenue: AED {daily['revenue'].max():,.0f}"],
                        ["Scale marketing on high-performing days" if trend > 0 else "Investigate decline causes"],
                        "success" if trend > 0 else "warning"
                    )
            else:
                st.info("No data for trend analysis")
        
        with col2:
            section_header("Revenue by City/Channel", "🏙️")
            city_breakdown = kpi_calc.compute_breakdown(filtered_df, 'city')
            channel_breakdown = kpi_calc.compute_breakdown(filtered_df, 'channel')
            
            tab1, tab2 = st.tabs(["By City", "By Channel"])
            
            with tab1:
                if len(city_breakdown) > 0:
                    fig = px.pie(
                        city_breakdown, values='revenue', names='city',
                        hole=0.4,
                        color_discrete_sequence=['#1E3A5F', '#00D4AA', '#667eea', '#f5576c']
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                if len(channel_breakdown) > 0:
                    fig = px.bar(
                        channel_breakdown, x='channel', y='revenue',
                        color='margin_pct',
                        color_continuous_scale='Viridis',
                        text=channel_breakdown['revenue'].apply(lambda x: f'{x/1000:.0f}K')
                    )
                    fig.update_layout(height=300)
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
        
        # Chart Row 2: Margin by Category & Scenario Impact
        col1, col2 = st.columns(2)
        
        with col1:
            section_header("Margin % by Category", "📊")
            cat_breakdown = kpi_calc.compute_breakdown(filtered_df, 'category')
            
            if len(cat_breakdown) > 0:
                fig = px.bar(
                    cat_breakdown.sort_values('margin_pct', ascending=True),
                    x='margin_pct', y='category',
                    orientation='h',
                    color='margin_pct',
                    color_continuous_scale=[[0, '#f5576c'], [0.5, '#FFD700'], [1, '#00D4AA']]
                )
                fig.add_vline(x=sim_params['margin_floor'], line_dash="dash", line_color="red",
                             annotation_text=f"Floor: {sim_params['margin_floor']}%")
                fig.update_layout(height=350, showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
                
                # Insight
                low_margin = cat_breakdown[cat_breakdown['margin_pct'] < sim_params['margin_floor']]
                if len(low_margin) > 0:
                    detailed_insight(
                        "Margin Alert",
                        f"<strong>{len(low_margin)}</strong> categories below {sim_params['margin_floor']}% margin floor",
                        [f"At risk: {', '.join(low_margin['category'].tolist())}"],
                        ["Review pricing", "Negotiate supplier costs", "Reduce promotions"],
                        "warning"
                    )
        
        with col2:
            section_header("Scenario Impact (Profit vs Discount)", "🎯")
            
            # Generate scenario matrix
            discounts = [5, 10, 15, 20, 25, 30]
            scenario_data = []
            
            for d in discounts:
                res = simulator.run_simulation(
                    d, sim_params['promo_budget'], sim_params['margin_floor'],
                    sim_params['simulation_days'],
                    filters.get('city'), filters.get('channel'), filters.get('category')
                )
                if res['results']:
                    scenario_data.append({
                        'Discount %': d,
                        'Profit Proxy': res['results']['profit_proxy'],
                        'Margin %': res['results']['sim_margin_pct'],
                        'Viable': res['results']['sim_margin_pct'] >= sim_params['margin_floor']
                    })
            
            if scenario_data:
                scenario_df = pd.DataFrame(scenario_data)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=scenario_df['Discount %'],
                    y=scenario_df['Profit Proxy'],
                    marker_color=['#00D4AA' if v else '#f5576c' for v in scenario_df['Viable']],
                    text=scenario_df['Profit Proxy'].apply(lambda x: f'{x/1000:.0f}K'),
                    textposition='outside'
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(
                    height=350,
                    xaxis_title="Discount %",
                    yaxis_title="Profit Proxy (AED)",
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Find optimal
                viable = scenario_df[scenario_df['Viable']]
                if len(viable) > 0:
                    optimal = viable.loc[viable['Profit Proxy'].idxmax()]
                    detailed_insight(
                        "Optimal Scenario",
                        f"Best discount: <strong>{optimal['Discount %']}%</strong> with profit <strong>AED {optimal['Profit Proxy']:,.0f}</strong>",
                        [f"Maintains margin above {sim_params['margin_floor']}%", "Green bars = viable scenarios"],
                        ["Apply recommended discount", "Monitor actual vs projected"],
                        "success"
                    )
        
        # Recommendation Box
        section_header("Auto-Generated Recommendations", "💡")
        recommendations = generate_recommendation(kpis, sim_results['results'], sim_results['violations'])
        
        rec_cols = st.columns(2)
        for i, rec in enumerate(recommendations):
            with rec_cols[i % 2]:
                if rec.startswith("✅"):
                    st.success(rec)
                elif rec.startswith("⚠️") or rec.startswith("💡"):
                    st.warning(rec)
                elif rec.startswith("🚫"):
                    st.error(rec)
                else:
                    st.info(rec)
    
    # ==========================================================================
    # MANAGER VIEW
    # ==========================================================================
    else:  # Manager View
        
        # Operational KPI Cards
        section_header("Operational KPIs", "🔧")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        stockout_risk = sim_results['results']['stockout_risk_pct'] if sim_results['results'] else 0
        high_risk_skus = sim_results['results']['high_risk_skus'] if sim_results['results'] else 0
        
        with col1:
            kpi_card("Stockout Risk", f"{stockout_risk:.1f}%",
                    "High Risk" if stockout_risk > 30 else "Acceptable",
                    "negative" if stockout_risk > 30 else "positive")
        with col2:
            kpi_card("High Risk SKUs", f"{high_risk_skus:,}")
        with col3:
            kpi_card("Return Rate", f"{kpis['return_rate']:.1f}%",
                    "High" if kpis['return_rate'] > 10 else "Normal",
                    "negative" if kpis['return_rate'] > 10 else "positive")
        with col4:
            kpi_card("Payment Failure", f"{kpis['payment_failure_rate']:.1f}%",
                    "High" if kpis['payment_failure_rate'] > 5 else "Normal",
                    "negative" if kpis['payment_failure_rate'] > 5 else "positive")
        with col5:
            kpi_card("Total Units", f"{kpis['total_units']:,}")
        
        st.markdown("---")
        
        # Chart Row 1: Stockout by City/Channel & Top Risk Items
        col1, col2 = st.columns(2)
        
        with col1:
            section_header("Stockout Risk by City/Channel", "📍")
            
            if sim_results.get('simulation_detail') is not None:
                sim_detail = sim_results['simulation_detail']
                
                # Merge city/channel info
                sim_detail = sim_detail.merge(
                    data['stores'][['store_id', 'city', 'channel']].drop_duplicates(),
                    on='store_id', how='left', suffixes=('', '_store')
                )
                
                tab1, tab2 = st.tabs(["By City", "By Channel"])
                
                with tab1:
                    city_risk = sim_detail.groupby('city').agg({
                        'stockout_risk': ['sum', 'count'],
                        'excess_demand': 'sum'
                    }).reset_index()
                    city_risk.columns = ['City', 'At Risk', 'Total', 'Excess Demand']
                    city_risk['Risk %'] = (city_risk['At Risk'] / city_risk['Total'] * 100).round(1)
                    
                    fig = px.bar(
                        city_risk, x='City', y='Risk %',
                        color='Risk %',
                        color_continuous_scale=[[0, '#00D4AA'], [0.5, '#FFD700'], [1, '#f5576c']],
                        text='Risk %'
                    )
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Threshold")
                    fig.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    channel_risk = sim_detail.groupby('channel').agg({
                        'stockout_risk': ['sum', 'count'],
                        'excess_demand': 'sum'
                    }).reset_index()
                    channel_risk.columns = ['Channel', 'At Risk', 'Total', 'Excess Demand']
                    channel_risk['Risk %'] = (channel_risk['At Risk'] / channel_risk['Total'] * 100).round(1)
                    
                    fig = px.bar(
                        channel_risk, x='Channel', y='Risk %',
                        color='Risk %',
                        color_continuous_scale=[[0, '#00D4AA'], [0.5, '#FFD700'], [1, '#f5576c']],
                        text='Risk %'
                    )
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig.add_hline(y=30, line_dash="dash", line_color="red")
                    fig.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            section_header("Top 10 Stockout Risk Items", "🔴")
            
            if sim_results.get('top_risk_items') is not None:
                risk_df = sim_results['top_risk_items'].copy()
                
                # Style the dataframe
                def color_risk(val):
                    if val > 150:
                        return 'background-color: #f8d7da'
                    elif val > 100:
                        return 'background-color: #fff3cd'
                    return ''
                
                st.dataframe(
                    risk_df.style.applymap(color_risk, subset=['Risk %']),
                    use_container_width=True,
                    height=350
                )
                
                if len(risk_df) > 0:
                    detailed_insight(
                        "Stockout Alert",
                        f"<strong>{len(risk_df[risk_df['Risk %'] > 100])}</strong> product-stores with demand exceeding stock",
                        [f"Highest risk: {risk_df.iloc[0]['Product']} at {risk_df.iloc[0]['Store']}"],
                        ["Expedite replenishment", "Consider inventory reallocation"],
                        "warning"
                    )
        
        # Chart Row 2: Inventory Distribution & Demand vs Stock
        col1, col2 = st.columns(2)
        
        with col1:
            section_header("Inventory Distribution", "📦")
            
            # Get latest inventory
            inv = data['inventory'].copy()
            inv['snapshot_date'] = pd.to_datetime(inv['snapshot_date'])
            latest_inv = inv[inv['snapshot_date'] == inv['snapshot_date'].max()]
            
            fig = px.histogram(
                latest_inv, x='stock_on_hand',
                nbins=30,
                color_discrete_sequence=['#1E3A5F']
            )
            fig.add_vline(x=latest_inv['stock_on_hand'].median(), line_dash="dash", 
                         line_color="#00D4AA", annotation_text="Median")
            fig.update_layout(
                height=300,
                xaxis_title="Stock on Hand",
                yaxis_title="Count",
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Stats
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Avg Stock", f"{latest_inv['stock_on_hand'].mean():.0f}")
            col_b.metric("Median Stock", f"{latest_inv['stock_on_hand'].median():.0f}")
            col_c.metric("Zero Stock", f"{(latest_inv['stock_on_hand'] == 0).sum()}")
        
        with col2:
            section_header("Demand vs Stock Scatter", "📊")
            
            if sim_results.get('simulation_detail') is not None:
                sim_detail = sim_results['simulation_detail']
                
                # Sample for performance
                sample = sim_detail.sample(min(500, len(sim_detail)))
                
                fig = px.scatter(
                    sample,
                    x='stock_on_hand',
                    y='sim_demand',
                    color='stockout_risk',
                    color_discrete_map={0: '#00D4AA', 1: '#f5576c'},
                    opacity=0.6,
                    labels={'stockout_risk': 'At Risk'}
                )
                # Add diagonal line (demand = stock)
                max_val = max(sample['stock_on_hand'].max(), sample['sim_demand'].max())
                fig.add_trace(go.Scatter(
                    x=[0, max_val], y=[0, max_val],
                    mode='lines', name='Demand = Stock',
                    line=dict(color='gray', dash='dash')
                ))
                fig.update_layout(
                    height=300,
                    xaxis_title="Stock on Hand",
                    yaxis_title="Simulated Demand",
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Chart Row 3: Issues Pareto & Constraint Violations
        col1, col2 = st.columns(2)
        
        with col1:
            section_header("Issues Pareto (from Cleaning)", "⚠️")
            
            if len(data['issues']) > 0:
                issue_counts = data['issues']['issue_type'].value_counts().reset_index()
                issue_counts.columns = ['Issue Type', 'Count']
                issue_counts['Cumulative %'] = (issue_counts['Count'].cumsum() / issue_counts['Count'].sum() * 100)
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(
                    go.Bar(x=issue_counts['Issue Type'], y=issue_counts['Count'],
                          name='Count', marker_color='#1E3A5F'),
                    secondary_y=False
                )
                fig.add_trace(
                    go.Scatter(x=issue_counts['Issue Type'], y=issue_counts['Cumulative %'],
                              name='Cumulative %', line=dict(color='#00D4AA', width=3)),
                    secondary_y=True
                )
                fig.add_hline(y=80, line_dash="dash", line_color="#f5576c", secondary_y=True)
                fig.update_layout(
                    height=350,
                    legend=dict(orientation="h", y=-0.25),
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No issues logged during cleaning")
        
        with col2:
            section_header("Constraint Violations", "🚫")
            
            violations = sim_results.get('violations', [])
            
            if violations:
                for v in violations:
                    severity_class = "constraint-card-error" if v['severity'] == 'HIGH' else "constraint-card"
                    icon = "🚫" if v['severity'] == 'HIGH' else "⚠️"
                    st.markdown(f"""
                    <div class="{severity_class}">
                        <strong>{icon} {v['constraint']}</strong><br>
                        {v['message']}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show top violators if available
                if sim_results.get('constraint_violators'):
                    with st.expander("📋 View Top Constraint Violators"):
                        for cv in sim_results['constraint_violators']:
                            st.markdown(f"**{cv['constraint']}**")
                            violators_df = pd.DataFrame(cv['top_violators'])
                            st.dataframe(violators_df, use_container_width=True, height=150)
            else:
                st.success("✅ All constraints satisfied!")
                st.markdown("""
                - ✅ Margin above floor
                - ✅ Budget not exceeded
                - ✅ Stock levels adequate
                """)
        
        # Drill-down section
        st.markdown("---")
        section_header("Drill-Down Analysis", "🔍")
        
        drill_col1, drill_col2 = st.columns([1, 2])
        
        with drill_col1:
            st.markdown("#### Select Filters")
            drill_city = st.selectbox("City", ['All'] + data['stores']['city'].unique().tolist(), key='drill_city')
            drill_cat = st.selectbox("Category", ['All'] + data['products']['category'].unique().tolist(), key='drill_cat')
        
        with drill_col2:
            st.markdown("#### Filtered Risk Table")
            
            if sim_results.get('simulation_detail') is not None:
                drill_df = sim_results['simulation_detail'].copy()
                
                # Merge for filtering
                drill_df = drill_df.merge(
                    data['stores'][['store_id', 'city']].drop_duplicates(),
                    on='store_id', how='left', suffixes=('', '_s')
                )
                
                if drill_city != 'All':
                    drill_df = drill_df[drill_df['city'] == drill_city]
                if drill_cat != 'All':
                    drill_df = drill_df[drill_df['category'] == drill_cat]
                
                if len(drill_df) > 0:
                    drill_summary = drill_df.groupby(['product_id', 'store_id']).agg({
                        'stock_on_hand': 'first',
                        'sim_demand': 'first',
                        'stockout_risk': 'first',
                        'excess_demand': 'first'
                    }).reset_index().nlargest(10, 'excess_demand')
                    
                    st.dataframe(drill_summary, use_container_width=True, height=250)
                else:
                    st.info("No data for selected filters")
    
    # ==========================================================================
    # DOWNLOAD SECTION
    # ==========================================================================
    st.markdown("---")
    section_header("Export Data", "📥")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.download_button(
            "📄 Cleaned Sales",
            data['sales'].to_csv(index=False),
            "cleaned_sales.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        if len(data['issues']) > 0:
            st.download_button(
                "📄 Issues Log",
                data['issues'].to_csv(index=False),
                "issues.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.button("📄 Issues Log", disabled=True, use_container_width=True)
    
    with col3:
        if sim_results['results']:
            sim_export = pd.DataFrame([sim_results['results']])
            st.download_button(
                "📄 Simulation Results",
                sim_export.to_csv(index=False),
                "simulation_results.csv",
                "text/csv",
                use_container_width=True
            )
    
    with col4:
        if sim_results['top_risk_items'] is not None:
            st.download_button(
                "📄 Risk Items",
                sim_results['top_risk_items'].to_csv(index=False),
                "risk_items.csv",
                "text/csv",
                use_container_width=True
            )


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: center; align-items: center; padding: 1rem;">
    <svg width="200" height="40" viewBox="0 0 200 40" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="footerGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" style="stop-color:#00D4AA;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#00F5CC;stop-opacity:1" />
            </linearGradient>
        </defs>
        <text x="10" y="25" font-family="Arial, sans-serif" font-size="12" fill="#888">UAE</text>
        <text x="40" y="25" font-family="Arial, sans-serif" font-size="12" fill="url(#footerGrad)">PROMO PULSE</text>
        <text x="130" y="25" font-family="Arial, sans-serif" font-size="10" fill="#888">v2.0</text>
        <path d="M160 20 L170 20 L173 14 L176 26 L179 18 L182 22 L185 20 L195 20" 
              fill="none" stroke="url(#footerGrad)" stroke-width="1.5" stroke-linecap="round"/>
    </svg>
</div>
""", unsafe_allow_html=True)

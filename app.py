import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import base64

from data_generator import generate_all_data
from cleaner import DataCleaner
from simulator import KPICalculator, PromoSimulator, generate_recommendation

# Page Config
st.set_page_config(page_title="UAE Promo Pulse", page_icon="🛒", layout="wide")

# ============================================================================
# LOGO AND STYLES
# ============================================================================
def render_logo():
    """Render SVG logo for UAE Promo Pulse"""
    logo_html = """
    <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 1rem;">
        <svg width="400" height="100" viewBox="0 0 400 100" xmlns="http://www.w3.org/2000/svg">
            <!-- Background Shape -->
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
            
            <!-- Main Container -->
            <rect x="5" y="10" width="390" height="80" rx="15" ry="15" fill="url(#bgGrad)" filter="url(#shadow)"/>
            
            <!-- UAE Flag Stripe -->
            <rect x="5" y="10" width="8" height="80" rx="4" ry="0" fill="url(#uaeGrad)"/>
            
            <!-- Shopping Cart Icon -->
            <g transform="translate(25, 25)">
                <circle cx="25" cy="25" r="22" fill="rgba(255,255,255,0.1)" stroke="url(#pulseGrad)" stroke-width="2"/>
                <path d="M15 20 L18 20 L22 35 L35 35 L38 23 L20 23" fill="none" stroke="url(#pulseGrad)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                <circle cx="24" cy="42" r="3" fill="url(#pulseGrad)"/>
                <circle cx="34" cy="42" r="3" fill="url(#pulseGrad)"/>
                <!-- Pulse waves -->
                <path d="M42 25 Q47 15, 52 25 Q57 35, 62 25" fill="none" stroke="url(#pulseGrad)" stroke-width="2" stroke-linecap="round"/>
            </g>
            
            <!-- Text: UAE -->
            <text x="95" y="45" font-family="Arial Black, sans-serif" font-size="22" font-weight="900" fill="#FFFFFF">UAE</text>
            
            <!-- Text: PROMO -->
            <text x="145" y="45" font-family="Arial Black, sans-serif" font-size="22" font-weight="900" fill="url(#pulseGrad)">PROMO</text>
            
            <!-- Text: PULSE -->
            <text x="250" y="45" font-family="Arial Black, sans-serif" font-size="22" font-weight="900" fill="#FFFFFF">PULSE</text>
            
            <!-- Tagline -->
            <text x="95" y="70" font-family="Arial, sans-serif" font-size="11" fill="rgba(255,255,255,0.8)" letter-spacing="1">RETAIL ANALYTICS &amp; PROMOTION SIMULATOR</text>
            
            <!-- Decorative pulse line -->
            <path d="M340 35 L350 35 L355 25 L360 45 L365 30 L370 40 L375 35 L385 35" 
                  fill="none" stroke="url(#pulseGrad)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <animate attributeName="opacity" values="0.5;1;0.5" dur="2s" repeatCount="indefinite"/>
            </path>
            
            <!-- Version badge -->
            <rect x="340" y="55" width="45" height="18" rx="9" fill="url(#pulseGrad)"/>
            <text x="362" y="68" font-family="Arial, sans-serif" font-size="10" font-weight="bold" fill="#1E3A5F" text-anchor="middle">v2.0</text>
        </svg>
    </div>
    """
    st.markdown(logo_html, unsafe_allow_html=True)


def render_sidebar_logo():
    """Render smaller logo for sidebar"""
    sidebar_logo = """
    <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 1rem; padding: 10px;">
        <svg width="200" height="60" viewBox="0 0 200 60" xmlns="http://www.w3.org/2000/svg">
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
            <rect x="2" y="5" width="196" height="50" rx="10" fill="url(#sbBgGrad)"/>
            <circle cx="35" cy="30" r="18" fill="rgba(255,255,255,0.1)" stroke="url(#sbPulseGrad)" stroke-width="2"/>
            <text x="60" y="28" font-family="Arial Black, sans-serif" font-size="12" font-weight="900" fill="#FFFFFF">UAE</text>
            <text x="90" y="28" font-family="Arial Black, sans-serif" font-size="12" font-weight="900" fill="url(#sbPulseGrad)">PROMO</text>
            <text x="60" y="43" font-family="Arial Black, sans-serif" font-size="12" font-weight="900" fill="#FFFFFF">PULSE</text>
            <text x="105" y="43" font-family="Arial, sans-serif" font-size="8" fill="rgba(255,255,255,0.7)">v2.0</text>
            <!-- Cart icon -->
            <path d="M28 22 L30 22 L33 32 L42 32 L44 25 L31 25" fill="none" stroke="url(#sbPulseGrad)" stroke-width="1.5" stroke-linecap="round"/>
            <circle cx="34" cy="36" r="2" fill="url(#sbPulseGrad)"/>
            <circle cx="40" cy="36" r="2" fill="url(#sbPulseGrad)"/>
            <!-- Pulse -->
            <path d="M160 25 L165 25 L168 18 L172 32 L176 22 L180 28 L185 25 L190 25" 
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
    .main-header {font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 1.5rem; margin-top: -0.5rem;}
    .insight-box {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem 1.2rem; border-radius: 10px; color: white; margin: 0.5rem 0 1.5rem 0; font-size: 0.9rem; line-height: 1.6;}
    .insight-success {background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);}
    .insight-warning {background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);}
    .insight-title {font-weight: bold; font-size: 1rem; margin-bottom: 0.5rem;}
    .insight-detail {font-size: 0.85rem; margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid rgba(255,255,255,0.3);}
    .stApp > header {background-color: transparent;}
    .block-container {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

# Session State
for k in ['data_loaded', 'data', 'raw_data', 'raw_data_generated', 'cleaning_stats', 'upload_mode']:
    if k not in st.session_state:
        st.session_state[k] = None if k in ['data', 'raw_data', 'cleaning_stats'] else False

def detailed_insight(title, main_text, details, actions, itype="info"):
    css = f"insight-box insight-{itype}" if itype != "info" else "insight-box"
    icon = {"info": "💡", "success": "✅", "warning": "⚠️"}.get(itype, "💡")
    details_html = "".join([f"<br>• {d}" for d in details]) if details else ""
    actions_html = f"<div class='insight-detail'><strong>🎯 Actions:</strong>{''.join([f'<br>→ {a}' for a in actions])}</div>" if actions else ""
    st.markdown(f'<div class="{css}"><div class="insight-title">{icon} {title}</div>{main_text}{details_html}{actions_html}</div>', unsafe_allow_html=True)

def validate_data(products_df, stores_df, sales_df, inventory_df):
    errors = []
    if products_df is not None:
        missing = [c for c in ['product_id', 'category', 'base_price_aed', 'unit_cost_aed'] if c not in products_df.columns]
        if missing: errors.append(f"Products missing: {missing}")
    if stores_df is not None:
        missing = [c for c in ['store_id', 'city', 'channel'] if c not in stores_df.columns]
        if missing: errors.append(f"Stores missing: {missing}")
    if sales_df is not None:
        missing = [c for c in ['order_id', 'order_time', 'product_id', 'store_id', 'qty', 'selling_price_aed', 'payment_status'] if c not in sales_df.columns]
        if missing: errors.append(f"Sales missing: {missing}")
    if inventory_df is not None:
        missing = [c for c in ['snapshot_date', 'product_id', 'store_id', 'stock_on_hand'] if c not in inventory_df.columns]
        if missing: errors.append(f"Inventory missing: {missing}")
    return errors

def add_defaults(df, dtype):
    if dtype == 'products':
        if 'brand' not in df.columns: df['brand'] = 'Unknown'
        if 'tax_rate' not in df.columns: df['tax_rate'] = 0.05
        if 'launch_flag' not in df.columns: df['launch_flag'] = 'Regular'
    elif dtype == 'stores':
        if 'fulfillment_type' not in df.columns: df['fulfillment_type'] = 'Own'
    elif dtype == 'sales':
        if 'discount_pct' not in df.columns: df['discount_pct'] = 0
        if 'return_flag' not in df.columns: df['return_flag'] = 0
    elif dtype == 'inventory':
        if 'reorder_point' not in df.columns: df['reorder_point'] = 20
        if 'lead_time_days' not in df.columns: df['lead_time_days'] = 7
    return df

# ============================================================================
# SIDEBAR
# ============================================================================
render_sidebar_logo()

st.sidebar.markdown("## 🎛️ Controls")
view = st.sidebar.radio("View", ["Executive", "Manager"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Data Source")
data_source = st.sidebar.radio("Choose:", ["Generate Sample Data", "Upload Your Own Data"])

if data_source == "Generate Sample Data":
    st.session_state.upload_mode = False
    if st.sidebar.button("🎲 Generate Data", use_container_width=True):
        with st.spinner("Generating..."):
            st.session_state.raw_data = generate_all_data('data/raw')
            st.session_state.raw_data_generated = True
            st.session_state.data_loaded = False
        st.rerun()
    
    if st.session_state.raw_data_generated and not st.session_state.upload_mode:
        st.sidebar.success("✅ Raw data ready!")
    
    if st.sidebar.button("🧹 Clean Data", use_container_width=True, disabled=not st.session_state.raw_data_generated):
        if st.session_state.raw_data:
            with st.spinner("Cleaning..."):
                orig = {k: len(v) for k, v in st.session_state.raw_data.items()}
                cleaner = DataCleaner()
                cleaned = cleaner.clean_all(st.session_state.raw_data['products'], st.session_state.raw_data['stores'],
                                           st.session_state.raw_data['sales'], st.session_state.raw_data['inventory'], 'data/cleaned')
                st.session_state.data = cleaned
                cc = {k: len(v) for k, v in cleaned.items() if k != 'issues'}
                st.session_state.cleaning_stats = {'original': orig, 'cleaned': cc,
                    'removed': {k: orig.get(k, 0) - cc.get(k, 0) for k in cc}, 'total_issues': len(cleaned['issues']),
                    'issues_summary': cleaned['issues']['issue_type'].value_counts().to_dict() if len(cleaned['issues']) > 0 else {}}
                st.session_state.data_loaded = True
            st.rerun()
    
    if st.session_state.data_loaded and not st.session_state.upload_mode:
        st.sidebar.success("✅ Data cleaned!")
else:
    st.session_state.upload_mode = True
    st.sidebar.markdown("#### Upload CSV Files")
    products_file = st.sidebar.file_uploader("📦 Products", type=['csv'], key='p')
    stores_file = st.sidebar.file_uploader("🏪 Stores", type=['csv'], key='s')
    sales_file = st.sidebar.file_uploader("💰 Sales", type=['csv'], key='sa')
    inventory_file = st.sidebar.file_uploader("📊 Inventory", type=['csv'], key='i')
    
    if st.sidebar.button("📤 Load Data", use_container_width=True):
        if all([products_file, stores_file, sales_file, inventory_file]):
            try:
                products_df = pd.read_csv(products_file)
                stores_df = pd.read_csv(stores_file)
                sales_df = pd.read_csv(sales_file)
                inventory_df = pd.read_csv(inventory_file)
                
                errors = validate_data(products_df, stores_df, sales_df, inventory_df)
                if errors:
                    for e in errors: st.sidebar.error(e)
                else:
                    products_df = add_defaults(products_df, 'products')
                    stores_df = add_defaults(stores_df, 'stores')
                    sales_df = add_defaults(sales_df, 'sales')
                    inventory_df = add_defaults(inventory_df, 'inventory')
                    
                    orig = {'products': len(products_df), 'stores': len(stores_df), 'sales': len(sales_df), 'inventory': len(inventory_df)}
                    with st.spinner("Processing..."):
                        cleaner = DataCleaner()
                        cleaned = cleaner.clean_all(products_df, stores_df, sales_df, inventory_df, 'data/cleaned')
                        st.session_state.data = cleaned
                        cc = {k: len(v) for k, v in cleaned.items() if k != 'issues'}
                        st.session_state.cleaning_stats = {'original': orig, 'cleaned': cc,
                            'removed': {k: orig.get(k, 0) - cc.get(k, 0) for k in cc}, 'total_issues': len(cleaned['issues']),
                            'issues_summary': cleaned['issues']['issue_type'].value_counts().to_dict() if len(cleaned['issues']) > 0 else {}}
                        st.session_state.data_loaded = True
                        st.session_state.raw_data_generated = False
                    st.sidebar.success("✅ Uploaded!")
                    st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
        else:
            st.sidebar.warning("Upload all 4 files")

# Filters
st.sidebar.markdown("---")
filters = {}
if st.session_state.data_loaded and st.session_state.data:
    st.sidebar.markdown("### 🔍 Filters")
    d = st.session_state.data
    if 'stores' in d:
        filters['city'] = st.sidebar.selectbox("City", ['All'] + sorted(d['stores']['city'].unique().tolist()))
        filters['channel'] = st.sidebar.selectbox("Channel", ['All'] + sorted(d['stores']['channel'].unique().tolist()))
    if 'products' in d:
        filters['category'] = st.sidebar.selectbox("Category", ['All'] + sorted(d['products']['category'].unique().tolist()))

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎮 Simulation")
sim = {
    'discount_pct': st.sidebar.slider("Discount %", 0, 50, 15),
    'promo_budget': st.sidebar.number_input("Budget (AED)", 10000, 500000, 50000, step=5000),
    'margin_floor': st.sidebar.slider("Margin Floor %", 5, 30, 15),
    'simulation_days': st.sidebar.selectbox("Days", [7, 14], index=1)
}

# ============================================================================
# MAIN CONTENT
# ============================================================================
render_logo()
st.markdown('<p class="main-header">Advanced Retail Analytics & Promotion Simulator for UAE Market</p>', unsafe_allow_html=True)

if not st.session_state.data_loaded:
    st.markdown("---")
    if st.session_state.upload_mode:
        st.info("👈 Upload your CSV files to begin")
        with st.expander("📋 Required Format", expanded=True):
            st.markdown("""
            **Products:** product_id, category, base_price_aed, unit_cost_aed  
            **Stores:** store_id, city, channel  
            **Sales:** order_id, order_time, product_id, store_id, qty, selling_price_aed, payment_status  
            **Inventory:** snapshot_date, product_id, store_id, stock_on_hand
            """)
            
            cols = st.columns(4)
            with cols[0]:
                st.download_button("📦 Products", pd.DataFrame({'product_id': ['PROD_0001'], 'category': ['Electronics'], 'base_price_aed': [1500], 'unit_cost_aed': [900]}).to_csv(index=False), "products_template.csv")
            with cols[1]:
                st.download_button("🏪 Stores", pd.DataFrame({'store_id': ['STORE_01'], 'city': ['Dubai'], 'channel': ['App']}).to_csv(index=False), "stores_template.csv")
            with cols[2]:
                st.download_button("💰 Sales", pd.DataFrame({'order_id': ['ORD_000001'], 'order_time': ['2024-01-15 10:30:00'], 'product_id': ['PROD_0001'], 'store_id': ['STORE_01'], 'qty': [1], 'selling_price_aed': [1350], 'payment_status': ['Paid']}).to_csv(index=False), "sales_template.csv")
            with cols[3]:
                st.download_button("📊 Inventory", pd.DataFrame({'snapshot_date': ['2024-01-15'], 'product_id': ['PROD_0001'], 'store_id': ['STORE_01'], 'stock_on_hand': [50]}).to_csv(index=False), "inventory_template.csv")
    elif st.session_state.raw_data_generated:
        st.success("✅ Raw data ready! Click 'Clean Data'")
    else:
        st.info("👈 Generate or upload data to start")
        
        # Feature showcase with logo theme
        st.markdown("---")
        st.markdown("### 🚀 Dashboard Features")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1E3A5F 0%, #2E5A8F 100%); padding: 1.5rem; border-radius: 10px; color: white; height: 200px;">
                <h4 style="color: #00D4AA;">📊 Analytics</h4>
                <ul style="font-size: 0.85rem;">
                    <li>14+ Interactive Charts</li>
                    <li>Real-time KPI Tracking</li>
                    <li>Trend Analysis</li>
                    <li>Pareto Analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1E3A5F 0%, #2E5A8F 100%); padding: 1.5rem; border-radius: 10px; color: white; height: 200px;">
                <h4 style="color: #00D4AA;">🎯 Simulation</h4>
                <ul style="font-size: 0.85rem;">
                    <li>What-If Scenarios</li>
                    <li>Promotion Planning</li>
                    <li>Margin Optimization</li>
                    <li>Risk Assessment</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1E3A5F 0%, #2E5A8F 100%); padding: 1.5rem; border-radius: 10px; color: white; height: 200px;">
                <h4 style="color: #00D4AA;">💡 Insights</h4>
                <ul style="font-size: 0.85rem;">
                    <li>Business Recommendations</li>
                    <li>Actionable Insights</li>
                    <li>Data Quality Reports</li>
                    <li>Export Reports</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

else:
    data = st.session_state.data
    kpi_calc = KPICalculator(data['sales'], data['products'], data['stores'], data['inventory'])
    simulator = PromoSimulator(data['sales'], data['products'], data['stores'], data['inventory'])
    
    fdf = kpi_calc.filter_data(filters.get('city'), filters.get('channel'), filters.get('category'))
    kpis = kpi_calc.compute_kpis(fdf)
    daily = kpi_calc.compute_daily(fdf)
    sim_res = simulator.run_simulation(sim['discount_pct'], sim['promo_budget'], sim['margin_floor'], sim['simulation_days'],
                                       filters.get('city'), filters.get('channel'), filters.get('category'))
    
    # Cleaning Stats
    if st.session_state.cleaning_stats:
        with st.expander("📊 Data Quality Report", expanded=(view == "Manager")):
            s = st.session_state.cleaning_stats
            cols = st.columns(4)
            cols[0].metric("Original", f"{sum(s['original'].values()):,}")
            cols[1].metric("Removed", f"{sum(s['removed'].values()):,}")
            cols[2].metric("Cleaned", f"{sum(s['cleaned'].values()):,}")
            cols[3].metric("Issues", f"{s['total_issues']:,}")
    
    # KPIs
    st.markdown("### 💰 Key Metrics")
    cols = st.columns(5)
    cols[0].metric("Revenue", f"AED {kpis['net_revenue']:,.0f}")
    cols[1].metric("Margin", f"{kpis['gross_margin_pct']:.1f}%")
    cols[2].metric("Orders", f"{kpis['total_orders']:,}")
    cols[3].metric("Returns", f"{kpis['return_rate']:.1f}%")
    cols[4].metric("Sim Profit", f"AED {sim_res['results']['profit_proxy']:,.0f}" if sim_res['results'] else "N/A")
    
    st.markdown("---")
    
    # Row 1: Pareto & Scatter
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Pareto Analysis")
        ps = fdf.groupby('product_id').agg({'selling_price_aed': 'sum', 'qty': 'sum'}).reset_index()
        ps.columns = ['product_id', 'revenue', 'units']
        ps = ps.merge(data['products'][['product_id', 'category']], on='product_id', how='left')
        ps = ps.sort_values('revenue', ascending=False).reset_index(drop=True)
        total = ps['revenue'].sum()
        ps['cum_pct'] = ps['revenue'].cumsum() / total * 100 if total > 0 else 0
        ps['rank'] = range(1, len(ps) + 1)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=ps['rank'].head(50), y=ps['revenue'].head(50), name='Revenue', marker_color='#1E3A5F'), secondary_y=False)
        fig.add_trace(go.Scatter(x=ps['rank'].head(50), y=ps['cum_pct'].head(50), name='Cumulative %', line=dict(color='#00D4AA', width=3)), secondary_y=True)
        fig.add_hline(y=80, line_dash="dash", line_color="#f5576c", secondary_y=True)
        fig.update_layout(height=350, legend=dict(orientation="h", y=1.1), plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        p80 = len(ps[ps['cum_pct'] <= 80])
        pct = p80 / len(ps) * 100 if len(ps) > 0 else 0
        top_cat = ps.head(p80)['category'].value_counts().index[0] if len(ps) > 0 and p80 > 0 else "N/A"
        
        detailed_insight("Pareto (80/20) Analysis",
            f"<strong>{pct:.1f}%</strong> of products ({p80} SKUs) generate 80% of revenue.",
            [f"Top category: <strong>{top_cat}</strong>", f"Total products: {len(ps)}", f"Focus on top {p80} products"],
            ["Prioritize marketing for top 20%", "Review bottom performers", "Ensure stock for top sellers"],
            "success" if pct < 30 else "info")
    
    with col2:
        st.markdown("#### 🔵 Price vs Quantity")
        pa = fdf.groupby('product_id').agg({'selling_price_aed': 'mean', 'qty': 'sum', 'discount_pct': 'mean'}).reset_index()
        pa = pa.merge(data['products'][['product_id', 'category']], on='product_id', how='left')
        
        fig = px.scatter(pa, x='selling_price_aed', y='qty', color='category', size='discount_pct',
                        labels={'selling_price_aed': 'Price', 'qty': 'Quantity'},
                        color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=350, plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        corr = pa['selling_price_aed'].corr(pa['qty'])
        detailed_insight("Price-Demand Analysis",
            f"Correlation: <strong>{corr:.2f}</strong> ({('Negative' if corr < 0 else 'Positive')})",
            [f"Avg price: AED {pa['selling_price_aed'].mean():.0f}", "Larger bubbles = higher discounts"],
            ["Test price elasticity", "Optimize discount levels", "Consider dynamic pricing"],
            "info")
    
    # Row 2: Forecast & Dual Axis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Revenue Forecast")
        if len(daily) > 7:
            df = daily.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            df['ma7'] = df['revenue'].rolling(7, min_periods=1).mean()
            trend = (df['ma7'].iloc[-1] - df['ma7'].iloc[-7]) / 7 if len(df) > 7 else 0
            
            last = df['date'].max()
            future = [last + timedelta(days=i) for i in range(1, 15)]
            pred = [df['ma7'].iloc[-1] + trend * i for i in range(1, 15)]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['date'], y=df['revenue'], name='Historical', line=dict(color='#1E3A5F', width=2)))
            fig.add_trace(go.Scatter(x=df['date'], y=df['ma7'], name='7-Day MA', line=dict(color='#00D4AA', dash='dot', width=2)))
            fig.add_trace(go.Scatter(x=future, y=pred, name='Forecast', line=dict(color='#f5576c', dash='dash', width=2)))
            fig.update_layout(height=350, legend=dict(orientation="h", y=1.1), plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            detailed_insight("Forecast Analysis",
                f"Trend: <strong>{'↑ Increasing' if trend > 0 else '↓ Decreasing'}</strong> by AED {abs(trend):,.0f}/day",
                [f"14-day forecast: AED {sum(pred):,.0f}", f"Daily average: AED {sum(pred)/14:,.0f}"],
                ["Scale inventory" if trend > 0 else "Launch promotions", "Monitor actuals vs forecast"],
                "success" if trend > 0 else "warning")
        else:
            st.info("Need 7+ days of data")
    
    with col2:
        st.markdown("#### 📉 Revenue vs Orders")
        if len(daily) > 0:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=daily['date'], y=daily['revenue'], name='Revenue', marker_color='#1E3A5F', opacity=0.7), secondary_y=False)
            fig.add_trace(go.Scatter(x=daily['date'], y=daily['orders'], name='Orders', line=dict(color='#00D4AA', width=3)), secondary_y=True)
            fig.update_layout(height=350, legend=dict(orientation="h", y=1.1), plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            corr = daily['revenue'].corr(daily['orders'])
            aov = daily['revenue'].sum() / daily['orders'].sum() if daily['orders'].sum() > 0 else 0
            detailed_insight("Volume-Value Analysis",
                f"Correlation: <strong>{corr:.2f}</strong> | AOV: <strong>AED {aov:,.0f}</strong>",
                [f"Total orders: {daily['orders'].sum():,}", "Strong correlation = volume drives revenue"],
                ["Focus on order volume" if corr > 0.7 else "Increase basket size", "Implement upselling"],
                "success" if corr > 0.7 else "info")
    
    # Row 3: Waterfall & Donut
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💧 Revenue Waterfall")
        ret_amt = fdf[fdf['return_flag'] == 1]['line_total'].sum() if 'line_total' in fdf.columns else 0
        fig = go.Figure(go.Waterfall(
            orientation="v", measure=["absolute", "relative", "relative", "relative", "total"],
            x=["Gross", "Refunds", "Returns", "COGS", "Profit"],
            y=[kpis['gross_revenue'], -kpis['refund_amount'], -ret_amt, -kpis['cogs'], 0],
            decreasing={"marker": {"color": "#f5576c"}}, increasing={"marker": {"color": "#00D4AA"}},
            totals={"marker": {"color": "#1E3A5F"}}
        ))
        fig.update_layout(height=350, plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        profit_pct = (kpis['gross_margin'] / kpis['gross_revenue'] * 100) if kpis['gross_revenue'] > 0 else 0
        detailed_insight("Profit Analysis",
            f"Net margin: <strong>{profit_pct:.1f}%</strong>",
            [f"COGS: {kpis['cogs']/kpis['gross_revenue']*100:.1f}%" if kpis['gross_revenue'] > 0 else "N/A",
             f"Refunds: {kpis['refund_amount']/kpis['gross_revenue']*100:.1f}%" if kpis['gross_revenue'] > 0 else "N/A"],
            ["Reduce COGS via negotiations", "Lower refunds by improving quality"],
            "success" if profit_pct > 20 else "warning")
    
    with col2:
        st.markdown("#### 🍩 Revenue by City")
        cbd = kpi_calc.compute_breakdown(fdf, 'city')
        if len(cbd) > 0:
            fig = go.Figure(data=[go.Pie(labels=cbd['city'], values=cbd['revenue'], hole=0.5,
                                        marker=dict(colors=['#1E3A5F', '#00D4AA', '#f5576c', '#667eea']))])
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            top = cbd.iloc[0]
            share = top['revenue'] / cbd['revenue'].sum() * 100
            detailed_insight("Geographic Distribution",
                f"<strong>{top['city']}</strong> leads with <strong>{share:.1f}%</strong>",
                [f"Number of cities: {len(cbd)}", "Concentration risk" if share > 60 else "Good balance"],
                ["Diversify markets" if share > 60 else "Maintain balance", "Tailor local campaigns"],
                "warning" if share > 60 else "success")
    
    # Row 4: Outliers & Growth
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔴 Outlier Detection")
        df_out = fdf.copy()
        Q1, Q3 = df_out['qty'].quantile(0.25), df_out['qty'].quantile(0.75)
        IQR = Q3 - Q1
        df_out['outlier'] = (df_out['qty'] < Q1 - 1.5*IQR) | (df_out['qty'] > Q3 + 1.5*IQR)
        sample = df_out.sample(min(2000, len(df_out)))
        
        fig = px.scatter(sample, x='qty', y='selling_price_aed', color='outlier',
                        color_discrete_map={True: '#f5576c', False: '#1E3A5F'}, opacity=0.6)
        fig.update_layout(height=350, plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        cnt = df_out['outlier'].sum()
        pct = cnt / len(df_out) * 100
        detailed_insight("Data Anomalies",
            f"<strong>{cnt:,}</strong> outliers (<strong>{pct:.2f}%</strong>)",
            ["Red points = unusual transactions", "May indicate fraud or errors"],
            ["Investigate high-value outliers" if pct > 1 else "Outlier rate acceptable", "Set up automated alerts"],
            "warning" if pct > 1 else "success")
    
    with col2:
        st.markdown("#### 📈 Growth Trends")
        if len(daily) > 7:
            df_gr = daily.copy()
            df_gr['date'] = pd.to_datetime(df_gr['date'])
            df_gr = df_gr.sort_values('date')
            df_gr['growth'] = df_gr['revenue'].pct_change() * 100
            df_gr['ma7'] = df_gr['revenue'].rolling(7, min_periods=1).mean()
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4])
            fig.add_trace(go.Scatter(x=df_gr['date'], y=df_gr['revenue'], name='Revenue', line=dict(color='#a8d5e5')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_gr['date'], y=df_gr['ma7'], name='MA7', line=dict(color='#1E3A5F', width=2)), row=1, col=1)
            colors = ['#00D4AA' if g > 0 else '#f5576c' for g in df_gr['growth'].fillna(0)]
            fig.add_trace(go.Bar(x=df_gr['date'], y=df_gr['growth'], marker_color=colors, showlegend=False), row=2, col=1)
            fig.update_layout(height=380, legend=dict(orientation="h", y=1.1), plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            avg = df_gr['growth'].mean()
            detailed_insight("Growth Analysis",
                f"Avg daily growth: <strong>{avg:.2f}%</strong>",
                [f"Positive days: {(df_gr['growth'] > 0).sum()}", f"Negative days: {(df_gr['growth'] < 0).sum()}"],
                ["Maintain momentum" if avg > 0 else "Diagnose decline", "Set growth targets"],
                "success" if avg > 0 else "warning")
    
    # Row 5: Margin Quadrant & Priority
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💰 Portfolio Quadrant")
        pp = fdf.groupby('product_id').agg({'selling_price_aed': 'sum', 'qty': 'sum'}).reset_index()
        pp.columns = ['product_id', 'revenue', 'units']
        pp = pp.merge(data['products'][['product_id', 'category', 'unit_cost_aed']], on='product_id', how='left')
        pp['margin'] = pp['revenue'] - (pp['units'] * pp['unit_cost_aed'])
        pp['margin_pct'] = (pp['margin'] / pp['revenue'] * 100).fillna(0)
        
        med_r, med_m = pp['revenue'].median(), pp['margin_pct'].median()
        pp['quad'] = 'Dogs'
        pp.loc[(pp['revenue'] > med_r) & (pp['margin_pct'] > med_m), 'quad'] = 'Stars'
        pp.loc[(pp['revenue'] <= med_r) & (pp['margin_pct'] > med_m), 'quad'] = 'Question'
        pp.loc[(pp['revenue'] > med_r) & (pp['margin_pct'] <= med_m), 'quad'] = 'Cash Cows'
        
        fig = px.scatter(pp, x='revenue', y='margin_pct', color='quad',
                        color_discrete_map={'Stars': '#FFD700', 'Cash Cows': '#00D4AA', 'Question': '#667eea', 'Dogs': '#f5576c'})
        fig.add_hline(y=med_m, line_dash="dash", line_color="gray")
        fig.add_vline(x=med_r, line_dash="dash", line_color="gray")
        fig.update_layout(height=350, plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        counts = pp['quad'].value_counts()
        detailed_insight("BCG Matrix",
            f"Stars: <strong>{counts.get('Stars', 0)}</strong> | Dogs: <strong>{counts.get('Dogs', 0)}</strong>",
            [f"Cash Cows: {counts.get('Cash Cows', 0)}", f"Question Marks: {counts.get('Question', 0)}"],
            ["Invest in Stars", "Review Dogs for removal", "Grow Question Marks"],
            "success" if counts.get('Stars', 0) > counts.get('Dogs', 0) else "warning")
    
    with col2:
        st.markdown("#### 🎯 Priority Products")
        pm = fdf.groupby('product_id').agg({'selling_price_aed': 'sum', 'qty': 'sum'}).reset_index()
        pm.columns = ['product_id', 'revenue', 'units']
        pm = pm.merge(data['products'][['product_id', 'category', 'unit_cost_aed']], on='product_id', how='left')
        pm['margin'] = pm['revenue'] - (pm['units'] * pm['unit_cost_aed'])
        pm['margin_pct'] = (pm['margin'] / pm['revenue'] * 100).fillna(0)
        
        for col in ['revenue', 'margin_pct']:
            pm[f'{col}_n'] = (pm[col] - pm[col].min()) / (pm[col].max() - pm[col].min() + 0.001)
        pm['score'] = (pm['revenue_n'] * 0.5 + pm['margin_pct_n'] * 0.5) * 100
        
        top = pm.nlargest(15, 'score')
        fig = px.bar(top, x='product_id', y='score', color='category',
                    color_discrete_sequence=['#1E3A5F', '#00D4AA', '#667eea', '#f5576c', '#FFD700', '#a8d5e5'])
        fig.update_layout(height=350, xaxis_tickangle=-45, plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        detailed_insight("Priority Algorithm",
            "Score = 50% Revenue + 50% Margin",
            [f"Top category: {top['category'].value_counts().index[0] if len(top) > 0 else 'N/A'}", f"Highest score: {top['score'].max():.1f}" if len(top) > 0 else "N/A"],
            ["Focus on top 15 products", "Ensure 99% availability"],
            "success")
    
    # Row 6: Matrix & What-If
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔥 City × Channel Matrix")
        df_m = fdf.merge(data['stores'][['store_id', 'city', 'channel']], on='store_id', how='left', suffixes=('', '_y'))
        df_m['revenue'] = df_m['qty'] * df_m['selling_price_aed']
        pivot = df_m.groupby(['city', 'channel'])['revenue'].sum().reset_index().pivot(index='city', columns='channel', values='revenue').fillna(0)
        
        fig = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                                        colorscale=[[0, '#1E3A5F'], [0.5, '#00D4AA'], [1, '#FFD700']],
                                        text=[[f'{v:,.0f}' for v in row] for row in pivot.values], texttemplate='%{text}'))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        best = pivot.stack().idxmax() if len(pivot) > 0 else ("N/A", "N/A")
        detailed_insight("Performance Matrix",
            f"Best: <strong>{best[0]} × {best[1]}</strong>",
            [f"Channels: {len(pivot.columns)}", f"Cities: {len(pivot.index)}"],
            ["Double down on winner", "Investigate weak cells"],
            "success")
    
    with col2:
        st.markdown("#### 🔮 What-If Analysis")
        discs = [5, 10, 15, 20, 25, 30]
        budgets = [25000, 50000, 75000, 100000]
        matrix = []
        for b in budgets:
            row = []
            for d in discs:
                r = simulator.run_simulation(d, b, sim['margin_floor'], sim['simulation_days'],
                                            filters.get('city'), filters.get('channel'), filters.get('category'))
                row.append(r['results']['profit_proxy'] if r['results'] else 0)
            matrix.append(row)
        
        fig = go.Figure(data=go.Heatmap(z=matrix, x=[f'{d}%' for d in discs], y=[f'{b:,}' for b in budgets],
                                        colorscale='RdYlGn', text=[[f'{v:,.0f}' for v in row] for row in matrix], texttemplate='%{text}'))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        max_p = max(max(row) for row in matrix) if matrix else 0
        detailed_insight("Scenario Analysis",
            f"Max profit: <strong>AED {max_p:,.0f}</strong>",
            ["Green = profit", "Red = loss", "Find optimal combo"],
            ["Use recommended discount", "Avoid high discount + low budget"],
            "success" if max_p > 0 else "warning")
    
    # Row 7: Seasonality & Category
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📅 Seasonality")
        df_s = fdf.copy()
        df_s['order_time'] = pd.to_datetime(df_s['order_time'], errors='coerce')
        df_s = df_s.dropna(subset=['order_time'])
        df_s['day'] = df_s['order_time'].dt.day_name()
        df_s['hour'] = df_s['order_time'].dt.hour
        df_s['revenue'] = df_s['qty'] * df_s['selling_price_aed']
        
        pivot_s = df_s.groupby(['day', 'hour'])['revenue'].sum().reset_index().pivot(index='day', columns='hour', values='revenue').fillna(0)
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_s = pivot_s.reindex([d for d in days if d in pivot_s.index])
        
        fig = go.Figure(data=go.Heatmap(z=pivot_s.values, x=pivot_s.columns.tolist(), y=pivot_s.index.tolist(),
                                        colorscale=[[0, '#f8f9fa'], [0.5, '#00D4AA'], [1, '#1E3A5F']]))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        if len(pivot_s) > 0 and pivot_s.values.size > 0:
            max_idx = np.unravel_index(np.argmax(pivot_s.values), pivot_s.values.shape)
            peak_day = pivot_s.index[max_idx[0]]
            peak_hour = pivot_s.columns[max_idx[1]]
            detailed_insight("Timing Patterns",
                f"Peak: <strong>{peak_day} {peak_hour}:00</strong>",
                ["Dark = high sales", "Light = low sales"],
                ["Schedule promos at peak", "Staff up during busy times"],
                "success")
    
    with col2:
        st.markdown("#### 📊 Category Margins")
        catbd = kpi_calc.compute_breakdown(fdf, 'category')
        if len(catbd) > 0:
            fig = px.bar(catbd, x='category', y='margin_pct', color='margin_pct',
                        color_continuous_scale=[[0, '#f5576c'], [0.5, '#FFD700'], [1, '#00D4AA']])
            fig.add_hline(y=sim['margin_floor'], line_dash="dash", line_color="#f5576c")
            fig.update_layout(height=350, plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            low = catbd[catbd['margin_pct'] < sim['margin_floor']]['category'].tolist()
            detailed_insight("Category Health",
                f"Avg margin: <strong>{catbd['margin_pct'].mean():.1f}%</strong>",
                [f"Below floor: {', '.join(low) if low else 'None'}", f"Best: {catbd.iloc[catbd['margin_pct'].argmax()]['category']}" if len(catbd) > 0 else "N/A"],
                ["Review pricing for low-margin categories" if low else "All healthy", "Expand high-margin categories"],
                "warning" if low else "success")
    
    # Recommendations
    st.markdown("---")
    st.markdown("### 💡 Recommendations")
    recs = generate_recommendation(kpis, sim_res['results'] if sim_res['results'] else {}, sim_res['violations'])
    cols = st.columns(2)
    for i, r in enumerate(recs):
        with cols[i % 2]:
            if r.startswith("✅"): st.success(r)
            elif r.startswith("⚠️") or r.startswith("💡"): st.warning(r)
            elif r.startswith("🚫"): st.error(r)
            else: st.info(r)
    
    # Downloads
    st.markdown("---")
    st.markdown("### 📥 Export")
    cols = st.columns(4)
    with cols[0]: st.download_button("📄 Sales", data['sales'].to_csv(index=False), "sales.csv")
    with cols[1]:
        if len(data['issues']) > 0: st.download_button("📄 Issues", data['issues'].to_csv(index=False), "issues.csv")
    with cols[2]:
        if sim_res['results']: st.download_button("📄 Simulation", pd.DataFrame([sim_res['results']]).to_csv(index=False), "sim.csv")
    with cols[3]:
        if sim_res['top_risk_items'] is not None: st.download_button("📄 Risk", sim_res['top_risk_items'].to_csv(index=False), "risk.csv")

# Footer with mini logo
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: center; align-items: center; padding: 1rem;">
    <svg width="150" height="40" viewBox="0 0 150 40" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="footerGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" style="stop-color:#00D4AA;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#00F5CC;stop-opacity:1" />
            </linearGradient>
        </defs>
        <text x="10" y="25" font-family="Arial, sans-serif" font-size="12" fill="#888">UAE</text>
        <text x="40" y="25" font-family="Arial, sans-serif" font-size="12" fill="url(#footerGrad)">PROMO PULSE</text>
        <text x="130" y="25" font-family="Arial, sans-serif" font-size="10" fill="#888">v2.0</text>
    </svg>
</div>
""", unsafe_allow_html=True)

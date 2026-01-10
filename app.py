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

# ============================================================================
# PAGE CONFIG & STYLES
# ============================================================================
st.set_page_config(page_title="UAE Promo Pulse", page_icon="🛒", layout="wide")

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1E3A5F; text-align: center; margin-bottom: 0.5rem;}
    .sub-header {font-size: 1rem; color: #666; text-align: center; margin-bottom: 1.5rem;}
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.2rem; border-radius: 10px; color: white;
        margin: 0.5rem 0 1.5rem 0; font-size: 0.9rem; line-height: 1.5;
    }
    .insight-success {background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);}
    .insight-warning {background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);}
    .insight-title {font-weight: bold; font-size: 1rem; margin-bottom: 0.3rem;}
    .insight-detail {font-size: 0.85rem; margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid rgba(255,255,255,0.3);}
    .chart-title {font-size: 1.1rem; font-weight: 600; color: #1E3A5F; margin-bottom: 0.5rem;}
    .upload-section {background: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 1rem 0;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================
for k in ['data_loaded', 'data', 'raw_data', 'raw_data_generated', 'cleaning_stats', 'upload_mode']:
    if k not in st.session_state:
        st.session_state[k] = None if k in ['data', 'raw_data', 'cleaning_stats'] else False

# ============================================================================
# DETAILED INSIGHT FUNCTION
# ============================================================================
def detailed_insight(title, main_text, details, actions, itype="info"):
    """Display detailed business insight with recommendations"""
    css = f"insight-box insight-{itype}" if itype != "info" else "insight-box"
    icon = {"info": "💡", "success": "✅", "warning": "⚠️"}.get(itype, "💡")
    
    details_html = "".join([f"<br>• {d}" for d in details]) if details else ""
    actions_html = ""
    if actions:
        actions_html = f"<div class='insight-detail'><strong>🎯 Recommended Actions:</strong>{''.join([f'<br>→ {a}' for a in actions])}</div>"
    
    st.markdown(f'''
    <div class="{css}">
        <div class="insight-title">{icon} {title}</div>
        {main_text}{details_html}
        {actions_html}
    </div>
    ''', unsafe_allow_html=True)

# ============================================================================
# FILE UPLOAD & VALIDATION
# ============================================================================
def validate_uploaded_data(products_df, stores_df, sales_df, inventory_df):
    """Validate uploaded data has required columns"""
    errors = []
    
    # Required columns
    product_cols = ['product_id', 'category', 'base_price_aed', 'unit_cost_aed']
    store_cols = ['store_id', 'city', 'channel']
    sales_cols = ['order_id', 'order_time', 'product_id', 'store_id', 'qty', 'selling_price_aed', 'payment_status']
    inventory_cols = ['snapshot_date', 'product_id', 'store_id', 'stock_on_hand']
    
    if products_df is not None:
        missing = [c for c in product_cols if c not in products_df.columns]
        if missing:
            errors.append(f"Products missing columns: {missing}")
    
    if stores_df is not None:
        missing = [c for c in store_cols if c not in stores_df.columns]
        if missing:
            errors.append(f"Stores missing columns: {missing}")
    
    if sales_df is not None:
        missing = [c for c in sales_cols if c not in sales_df.columns]
        if missing:
            errors.append(f"Sales missing columns: {missing}")
    
    if inventory_df is not None:
        missing = [c for c in inventory_cols if c not in inventory_df.columns]
        if missing:
            errors.append(f"Inventory missing columns: {missing}")
    
    return errors

def add_missing_columns(df, df_type):
    """Add missing optional columns with default values"""
    if df_type == 'products':
        if 'brand' not in df.columns:
            df['brand'] = 'Unknown'
        if 'tax_rate' not in df.columns:
            df['tax_rate'] = 0.05
        if 'launch_flag' not in df.columns:
            df['launch_flag'] = 'Regular'
    
    elif df_type == 'stores':
        if 'fulfillment_type' not in df.columns:
            df['fulfillment_type'] = 'Own'
    
    elif df_type == 'sales':
        if 'discount_pct' not in df.columns:
            df['discount_pct'] = 0
        if 'return_flag' not in df.columns:
            df['return_flag'] = 0
    
    elif df_type == 'inventory':
        if 'reorder_point' not in df.columns:
            df['reorder_point'] = 20
        if 'lead_time_days' not in df.columns:
            df['lead_time_days'] = 7
    
    return df

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.markdown("## 🎛️ Controls")
view = st.sidebar.radio("View", ["Executive", "Manager"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Data Source")

data_source = st.sidebar.radio("Choose data source:", ["Generate Sample Data", "Upload Your Own Data"])

if data_source == "Generate Sample Data":
    st.session_state.upload_mode = False
    
    if st.sidebar.button("🎲 Generate Data", use_container_width=True):
        with st.spinner("Generating sample data..."):
            st.session_state.raw_data = generate_all_data('data/raw')
            st.session_state.raw_data_generated = True
            st.session_state.data_loaded = False
        st.rerun()
    
    if st.session_state.raw_data_generated and not st.session_state.upload_mode:
        st.sidebar.success("✅ Raw data ready!")
    
    if st.sidebar.button("🧹 Clean Data", use_container_width=True, disabled=not st.session_state.raw_data_generated):
        if st.session_state.raw_data:
            with st.spinner("Cleaning data..."):
                orig = {k: len(v) for k, v in st.session_state.raw_data.items()}
                cleaner = DataCleaner()
                cleaned = cleaner.clean_all(
                    st.session_state.raw_data['products'],
                    st.session_state.raw_data['stores'],
                    st.session_state.raw_data['sales'],
                    st.session_state.raw_data['inventory'],
                    'data/cleaned'
                )
                st.session_state.data = cleaned
                cc = {k: len(v) for k, v in cleaned.items() if k != 'issues'}
                st.session_state.cleaning_stats = {
                    'original': orig, 'cleaned': cc,
                    'removed': {k: orig.get(k, 0) - cc.get(k, 0) for k in cc},
                    'total_issues': len(cleaned['issues']),
                    'issues_summary': cleaned['issues']['issue_type'].value_counts().to_dict() if len(cleaned['issues']) > 0 else {}
                }
                st.session_state.data_loaded = True
            st.rerun()
    
    if st.session_state.data_loaded and not st.session_state.upload_mode:
        st.sidebar.success("✅ Data cleaned & ready!")

else:
    st.session_state.upload_mode = True
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Upload CSV Files")
    
    products_file = st.sidebar.file_uploader("📦 Products CSV", type=['csv'], key='products_upload')
    stores_file = st.sidebar.file_uploader("🏪 Stores CSV", type=['csv'], key='stores_upload')
    sales_file = st.sidebar.file_uploader("💰 Sales CSV", type=['csv'], key='sales_upload')
    inventory_file = st.sidebar.file_uploader("📊 Inventory CSV", type=['csv'], key='inventory_upload')
    
    if st.sidebar.button("📤 Load Uploaded Data", use_container_width=True):
        if all([products_file, stores_file, sales_file, inventory_file]):
            try:
                products_df = pd.read_csv(products_file)
                stores_df = pd.read_csv(stores_file)
                sales_df = pd.read_csv(sales_file)
                inventory_df = pd.read_csv(inventory_file)
                
                # Validate
                errors = validate_uploaded_data(products_df, stores_df, sales_df, inventory_df)
                
                if errors:
                    for e in errors:
                        st.sidebar.error(e)
                else:
                    # Add missing optional columns
                    products_df = add_missing_columns(products_df, 'products')
                    stores_df = add_missing_columns(stores_df, 'stores')
                    sales_df = add_missing_columns(sales_df, 'sales')
                    inventory_df = add_missing_columns(inventory_df, 'inventory')
                    
                    # Store original counts
                    orig = {
                        'products': len(products_df),
                        'stores': len(stores_df),
                        'sales': len(sales_df),
                        'inventory': len(inventory_df)
                    }
                    
                    # Clean data
                    with st.spinner("Processing uploaded data..."):
                        cleaner = DataCleaner()
                        cleaned = cleaner.clean_all(products_df, stores_df, sales_df, inventory_df, 'data/cleaned')
                        
                        st.session_state.data = cleaned
                        cc = {k: len(v) for k, v in cleaned.items() if k != 'issues'}
                        st.session_state.cleaning_stats = {
                            'original': orig, 'cleaned': cc,
                            'removed': {k: orig.get(k, 0) - cc.get(k, 0) for k in cc},
                            'total_issues': len(cleaned['issues']),
                            'issues_summary': cleaned['issues']['issue_type'].value_counts().to_dict() if len(cleaned['issues']) > 0 else {}
                        }
                        st.session_state.data_loaded = True
                        st.session_state.raw_data_generated = False
                    
                    st.sidebar.success("✅ Data uploaded & cleaned!")
                    st.rerun()
                    
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
        else:
            st.sidebar.warning("Please upload all 4 CSV files")

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
st.markdown('<p class="main-header">🛒 UAE Promo Pulse Simulator</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Retail Analytics with Business Insights</p>', unsafe_allow_html=True)

if not st.session_state.data_loaded:
    st.markdown("---")
    
    if st.session_state.upload_mode:
        st.info("👈 Upload your CSV files in the sidebar to begin analysis")
        
        with st.expander("📋 Required CSV Format", expanded=True):
            st.markdown("""
            ### Products CSV (Required Columns)
            | Column | Type | Description |
            |--------|------|-------------|
            | `product_id` | string | Unique product identifier |
            | `category` | string | Product category |
            | `base_price_aed` | float | Base selling price |
            | `unit_cost_aed` | float | Cost per unit |
            | `brand` | string | (Optional) Brand name |
            
            ### Stores CSV (Required Columns)
            | Column | Type | Description |
            |--------|------|-------------|
            | `store_id` | string | Unique store identifier |
            | `city` | string | City name |
            | `channel` | string | Sales channel (App/Web/Store) |
            
            ### Sales CSV (Required Columns)
            | Column | Type | Description |
            |--------|------|-------------|
            | `order_id` | string | Unique order identifier |
            | `order_time` | datetime | Order timestamp |
            | `product_id` | string | Product identifier |
            | `store_id` | string | Store identifier |
            | `qty` | int | Quantity sold |
            | `selling_price_aed` | float | Actual selling price |
            | `payment_status` | string | Paid/Failed/Refunded |
            | `discount_pct` | float | (Optional) Discount percentage |
            | `return_flag` | int | (Optional) 0 or 1 |
            
            ### Inventory CSV (Required Columns)
            | Column | Type | Description |
            |--------|------|-------------|
            | `snapshot_date` | date | Inventory date |
            | `product_id` | string | Product identifier |
            | `store_id` | string | Store identifier |
            | `stock_on_hand` | int | Current stock quantity |
            """)
            
            st.markdown("---")
            st.markdown("### 📥 Download Sample Templates")
            
            # Create sample templates
            sample_products = pd.DataFrame({
                'product_id': ['PROD_0001', 'PROD_0002'],
                'category': ['Electronics', 'Fashion'],
                'brand': ['Samsung', 'Nike'],
                'base_price_aed': [1500.00, 350.00],
                'unit_cost_aed': [900.00, 175.00]
            })
            
            sample_stores = pd.DataFrame({
                'store_id': ['STORE_01', 'STORE_02'],
                'city': ['Dubai', 'Abu Dhabi'],
                'channel': ['App', 'Web'],
                'fulfillment_type': ['Own', '3PL']
            })
            
            sample_sales = pd.DataFrame({
                'order_id': ['ORD_000001', 'ORD_000002'],
                'order_time': ['2024-01-15 10:30:00', '2024-01-15 14:45:00'],
                'product_id': ['PROD_0001', 'PROD_0002'],
                'store_id': ['STORE_01', 'STORE_02'],
                'qty': [1, 2],
                'selling_price_aed': [1350.00, 297.50],
                'discount_pct': [10, 15],
                'payment_status': ['Paid', 'Paid'],
                'return_flag': [0, 0]
            })
            
            sample_inventory = pd.DataFrame({
                'snapshot_date': ['2024-01-15', '2024-01-15'],
                'product_id': ['PROD_0001', 'PROD_0002'],
                'store_id': ['STORE_01', 'STORE_02'],
                'stock_on_hand': [50, 120],
                'reorder_point': [10, 20]
            })
            
            cols = st.columns(4)
            with cols[0]:
                st.download_button("📦 Products Template", sample_products.to_csv(index=False), "products_template.csv")
            with cols[1]:
                st.download_button("🏪 Stores Template", sample_stores.to_csv(index=False), "stores_template.csv")
            with cols[2]:
                st.download_button("💰 Sales Template", sample_sales.to_csv(index=False), "sales_template.csv")
            with cols[3]:
                st.download_button("📊 Inventory Template", sample_inventory.to_csv(index=False), "inventory_template.csv")
    
    elif st.session_state.raw_data_generated:
        st.success("✅ Raw data generated! Click '🧹 Clean Data' to proceed.")
    else:
        st.info("👈 Select a data source and click 'Generate Data' or upload your own files")
        
        with st.expander("📊 Dashboard Features", expanded=True):
            st.markdown("""
            ### Charts & Analytics Included:
            
            | Chart | Purpose | Business Value |
            |-------|---------|----------------|
            | 📊 **Pareto Analysis** | 80/20 product analysis | Identify top revenue drivers |
            | 🔵 **Scatter Plot** | Price vs Quantity | Optimize pricing strategy |
            | 📈 **Revenue Forecast** | 14-day prediction | Plan inventory & staffing |
            | 📉 **Dual Axis** | Revenue vs Orders | Understand volume-value relationship |
            | 💧 **Waterfall** | Revenue breakdown | Identify profit leakages |
            | 🍩 **Donut Chart** | Distribution analysis | Geographic performance |
            | 🔴 **Outlier Detection** | Anomaly identification | Data quality & fraud detection |
            | 📈 **Growth Trends** | Daily growth patterns | Track momentum |
            | 💰 **Margin Quadrant** | Product portfolio | Strategic product decisions |
            | 🎯 **Priority Algorithm** | Product scoring | Focus resources effectively |
            | 🔥 **Performance Matrix** | City × Channel | Optimize channel mix |
            | 🔮 **What-If Heatmap** | Scenario analysis | Optimize promotions |
            | 📅 **Seasonality** | Day × Hour patterns | Time promotions optimally |
            """)

else:
    # ==================== DASHBOARD WITH DATA ====================
    data = st.session_state.data
    kpi_calc = KPICalculator(data['sales'], data['products'], data['stores'], data['inventory'])
    simulator = PromoSimulator(data['sales'], data['products'], data['stores'], data['inventory'])
    
    fdf = kpi_calc.filter_data(filters.get('city'), filters.get('channel'), filters.get('category'))
    kpis = kpi_calc.compute_kpis(fdf)
    daily = kpi_calc.compute_daily(fdf)
    sim_res = simulator.run_simulation(
        sim['discount_pct'], sim['promo_budget'], sim['margin_floor'],
        sim['simulation_days'], filters.get('city'), filters.get('channel'), filters.get('category')
    )
    
    # ==================== CLEANING STATISTICS ====================
    if st.session_state.cleaning_stats:
        with st.expander("📊 Data Quality Report", expanded=(view == "Manager")):
            s = st.session_state.cleaning_stats
            
            cols = st.columns(4)
            cols[0].metric("Original Records", f"{sum(s['original'].values()):,}")
            cols[1].metric("Records Removed", f"{sum(s['removed'].values()):,}", delta=f"-{sum(s['removed'].values())/sum(s['original'].values())*100:.1f}%", delta_color="inverse")
            cols[2].metric("Clean Records", f"{sum(s['cleaned'].values()):,}")
            cols[3].metric("Issues Found", f"{s['total_issues']:,}")
            
            if s.get('issues_summary'):
                st.markdown("#### Issue Breakdown")
                issues_df = pd.DataFrame([{'Issue Type': k, 'Count': v} for k, v in s['issues_summary'].items()])
                fig = px.bar(issues_df, x='Issue Type', y='Count', color='Count', color_continuous_scale='Reds')
                fig.update_layout(height=250, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
    
    # ==================== KPI CARDS ====================
    st.markdown("### 💰 Key Performance Indicators")
    cols = st.columns(5)
    cols[0].metric("Net Revenue", f"AED {kpis['net_revenue']:,.0f}")
    cols[1].metric("Gross Margin", f"{kpis['gross_margin_pct']:.1f}%", delta="Healthy" if kpis['gross_margin_pct'] > 20 else "Low", delta_color="normal" if kpis['gross_margin_pct'] > 20 else "inverse")
    cols[2].metric("Total Orders", f"{kpis['total_orders']:,}")
    cols[3].metric("Return Rate", f"{kpis['return_rate']:.1f}%", delta="Good" if kpis['return_rate'] < 8 else "High", delta_color="normal" if kpis['return_rate'] < 8 else "inverse")
    cols[4].metric("Sim Profit", f"AED {sim_res['results']['profit_proxy']:,.0f}" if sim_res['results'] else "N/A", delta="Viable" if sim_res['results'] and sim_res['results']['profit_proxy'] > 0 else "Loss", delta_color="normal" if sim_res['results'] and sim_res['results']['profit_proxy'] > 0 else "inverse")
    
    st.markdown("---")
    
    # ==================== ROW 1: PARETO & SCATTER ====================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Pareto Analysis (80/20 Rule)")
        ps = fdf.groupby('product_id').agg({'selling_price_aed': 'sum', 'qty': 'sum'}).reset_index()
        ps.columns = ['product_id', 'revenue', 'units']
        ps = ps.merge(data['products'][['product_id', 'category']], on='product_id', how='left')
        ps = ps.sort_values('revenue', ascending=False).reset_index(drop=True)
        total = ps['revenue'].sum()
        ps['cum_pct'] = ps['revenue'].cumsum() / total * 100
        ps['rank'] = range(1, len(ps) + 1)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=ps['rank'].head(50), y=ps['revenue'].head(50), name='Revenue', marker_color='steelblue'), secondary_y=False)
        fig.add_trace(go.Scatter(x=ps['rank'].head(50), y=ps['cum_pct'].head(50), name='Cumulative %', line=dict(color='red', width=2)), secondary_y=True)
        fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="80% Line", secondary_y=True)
        fig.update_layout(height=380, showlegend=True, legend=dict(orientation="h", y=1.1))
        fig.update_yaxes(title_text="Revenue (AED)", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative %", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed insight
        p80 = len(ps[ps['cum_pct'] <= 80])
        pct = p80 / len(ps) * 100
        top_cat = ps.head(p80)['category'].value_counts().index[0] if len(ps) > 0 else "N/A"
        top_revenue = ps['revenue'].iloc[0] if len(ps) > 0 else 0
        
        detailed_insight(
            "Pareto Principle Analysis",
            f"<strong>{pct:.1f}%</strong> of your products ({p80} SKUs) generate <strong>80%</strong> of total revenue.",
            [
                f"Top performing category: <strong>{top_cat}</strong>",
                f"#1 product revenue: <strong>AED {top_revenue:,.0f}</strong>",
                f"Bottom {100-pct:.0f}% of products contribute only 20% of revenue",
                f"Total products analyzed: {len(ps)}"
            ],
            [
                "Focus marketing budget on top 20% products for maximum ROI",
                "Review bottom performers for discontinuation or repositioning",
                f"Ensure adequate inventory for top {p80} SKUs",
                "Consider bundling low performers with top sellers"
            ],
            "success" if pct < 30 else "info"
        )
    
    with col2:
        st.markdown("#### 🔵 Price-Quantity Relationship")
        pa = fdf.groupby('product_id').agg({'selling_price_aed': 'mean', 'qty': 'sum', 'discount_pct': 'mean'}).reset_index()
        pa = pa.merge(data['products'][['product_id', 'category', 'brand']], on='product_id', how='left')
        
        fig = px.scatter(pa, x='selling_price_aed', y='qty', color='category', size='discount_pct',
                        hover_data=['brand', 'product_id'],
                        labels={'selling_price_aed': 'Avg Price (AED)', 'qty': 'Total Quantity Sold'})
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate price-quantity correlation
        corr = pa['selling_price_aed'].corr(pa['qty'])
        avg_price = pa['selling_price_aed'].mean()
        avg_qty = pa['qty'].mean()
        high_vol_low_price = len(pa[(pa['selling_price_aed'] < avg_price) & (pa['qty'] > avg_qty)])
        
        detailed_insight(
            "Price Elasticity Analysis",
            f"Price-Quantity correlation: <strong>{corr:.2f}</strong> ({'Negative' if corr < 0 else 'Positive'} relationship)",
            [
                f"Average selling price: <strong>AED {avg_price:.0f}</strong>",
                f"Average quantity per product: <strong>{avg_qty:.0f}</strong> units",
                f"Products with high volume & low price: <strong>{high_vol_low_price}</strong>",
                "Larger bubbles indicate higher discount levels"
            ],
            [
                "Products in upper-left quadrant are volume drivers - protect pricing",
                "Lower-right products may need promotional support",
                "Consider dynamic pricing for price-sensitive categories",
                "Test price elasticity with A/B pricing experiments"
            ],
            "info"
        )
    
    # ==================== ROW 2: FORECAST & DUAL AXIS ====================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Revenue Forecast (14-Day)")
        if len(daily) > 7:
            df = daily.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            df['ma7'] = df['revenue'].rolling(7, min_periods=1).mean()
            trend = (df['ma7'].iloc[-1] - df['ma7'].iloc[-7]) / 7 if len(df) > 7 else 0
            
            last = df['date'].max()
            future = [last + timedelta(days=i) for i in range(1, 15)]
            pred = [df['ma7'].iloc[-1] + trend * i for i in range(1, 15)]
            
            # Confidence band
            std = df['revenue'].std() * 0.5
            upper = [p + std for p in pred]
            lower = [p - std for p in pred]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['date'], y=df['revenue'], name='Historical', line=dict(color='steelblue', width=2)))
            fig.add_trace(go.Scatter(x=df['date'], y=df['ma7'], name='7-Day MA', line=dict(color='orange', dash='dot')))
            fig.add_trace(go.Scatter(x=future, y=pred, name='Forecast', line=dict(color='red', dash='dash', width=2)))
            fig.add_trace(go.Scatter(x=future + future[::-1], y=upper + lower[::-1], fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(255,0,0,0)'), name='Confidence Band'))
            fig.update_layout(height=380, legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate forecast details
            forecast_total = sum(pred)
            current_monthly = df['revenue'].sum()
            growth_pct = (forecast_total / (current_monthly / len(df) * 14) - 1) * 100 if current_monthly > 0 else 0
            
            detailed_insight(
                "Revenue Forecast Analysis",
                f"Trend: Revenue is <strong>{'increasing' if trend > 0 else 'decreasing'}</strong> by <strong>AED {abs(trend):,.0f}</strong> per day",
                [
                    f"14-day forecast total: <strong>AED {forecast_total:,.0f}</strong>",
                    f"Expected daily average: <strong>AED {forecast_total/14:,.0f}</strong>",
                    f"Projected growth: <strong>{growth_pct:+.1f}%</strong> vs current pace",
                    f"Forecast confidence: ±AED {std:,.0f} daily variance"
                ],
                [
                    f"{'Scale up inventory and staffing' if trend > 0 else 'Launch retention campaigns'}",
                    f"{'Capitalize on momentum with upselling' if trend > 0 else 'Review pricing and promotions'}",
                    "Monitor daily actuals vs forecast for early warning",
                    "Update forecast weekly with new data"
                ],
                "success" if trend > 0 else "warning"
            )
        else:
            st.info("Need at least 7 days of data for forecasting")
    
    with col2:
        st.markdown("#### 📉 Revenue vs Order Volume")
        if len(daily) > 0:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=daily['date'], y=daily['revenue'], name='Revenue', marker_color='steelblue', opacity=0.7), secondary_y=False)
            fig.add_trace(go.Scatter(x=daily['date'], y=daily['orders'], name='Orders', line=dict(color='red', width=2), mode='lines+markers'), secondary_y=True)
            fig.update_layout(height=380, legend=dict(orientation="h", y=1.1))
            fig.update_yaxes(title_text="Revenue (AED)", secondary_y=False)
            fig.update_yaxes(title_text="Orders", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
            
            corr = daily['revenue'].corr(daily['orders'])
            aov = daily['revenue'].sum() / daily['orders'].sum() if daily['orders'].sum() > 0 else 0
            peak_day = daily.loc[daily['revenue'].idxmax(), 'date'] if len(daily) > 0 else "N/A"
            
            detailed_insight(
                "Volume-Value Relationship",
                f"Revenue-Orders correlation: <strong>{corr:.2f}</strong> ({('Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.4 else 'Weak')})",
                [
                    f"Average Order Value (AOV): <strong>AED {aov:,.0f}</strong>",
                    f"Peak revenue day: <strong>{peak_day}</strong>",
                    f"Total orders: <strong>{daily['orders'].sum():,}</strong>",
                    f"Revenue per order range: AED {daily['revenue'].min()/max(daily['orders'].min(),1):,.0f} - {daily['revenue'].max()/max(daily['orders'].max(),1):,.0f}"
                ],
                [
                    f"{'Focus on order volume to drive revenue' if corr > 0.7 else 'Increase AOV through bundling/upselling'}",
                    "Implement cross-selling at checkout",
                    "Set minimum order value for free shipping",
                    "Create tiered loyalty rewards based on order value"
                ],
                "success" if corr > 0.7 else "info"
            )
    
    # ==================== ROW 3: WATERFALL & DONUT ====================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💧 Revenue Waterfall Breakdown")
        ret_amt = fdf[fdf['return_flag'] == 1]['line_total'].sum() if 'line_total' in fdf.columns else 0
        
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=["Gross Revenue", "Refunds", "Returns", "COGS", "Net Profit"],
            y=[kpis['gross_revenue'], -kpis['refund_amount'], -ret_amt, -kpis['cogs'], 0],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#ef553b"}},
            increasing={"marker": {"color": "#00cc96"}},
            totals={"marker": {"color": "#636efa"}},
            text=[f"AED {kpis['gross_revenue']:,.0f}", f"-AED {kpis['refund_amount']:,.0f}", f"-AED {ret_amt:,.0f}", f"-AED {kpis['cogs']:,.0f}", f"AED {kpis['gross_margin']:,.0f}"],
            textposition="outside"
        ))
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
        
        profit_pct = (kpis['gross_margin'] / kpis['gross_revenue'] * 100) if kpis['gross_revenue'] > 0 else 0
        refund_pct = (kpis['refund_amount'] / kpis['gross_revenue'] * 100) if kpis['gross_revenue'] > 0 else 0
        cogs_pct = (kpis['cogs'] / kpis['gross_revenue'] * 100) if kpis['gross_revenue'] > 0 else 0
        
        detailed_insight(
            "Profit Leakage Analysis",
            f"Net profit margin: <strong>{profit_pct:.1f}%</strong> of gross revenue",
            [
                f"COGS consumes: <strong>{cogs_pct:.1f}%</strong> of revenue",
                f"Refunds consume: <strong>{refund_pct:.1f}%</strong> of revenue",
                f"Returns consume: <strong>{ret_amt/kpis['gross_revenue']*100 if kpis['gross_revenue'] > 0 else 0:.1f}%</strong> of revenue",
                f"Total leakage: <strong>{100-profit_pct:.1f}%</strong>"
            ],
            [
                "Negotiate better supplier terms to reduce COGS",
                f"{'Investigate high refund rate - check product quality' if refund_pct > 5 else 'Refund rate is healthy'}",
                "Implement stricter return policy or improve product descriptions",
                "Target 25%+ net margin through operational efficiency"
            ],
            "success" if profit_pct > 20 else "warning"
        )
    
    with col2:
        st.markdown("#### 🍩 Geographic Revenue Distribution")
        cbd = kpi_calc.compute_breakdown(fdf, 'city')
        if len(cbd) > 0:
            fig = go.Figure(data=[go.Pie(
                labels=cbd['city'], values=cbd['revenue'], hole=0.5,
                textinfo='label+percent', textposition='outside',
                marker=dict(colors=px.colors.qualitative.Set2)
            )])
            total = cbd['revenue'].sum()
            fig.add_annotation(text=f"Total<br>AED {total:,.0f}", x=0.5, y=0.5, font_size=12, showarrow=False)
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
            
            top = cbd.iloc[0]
            share = top['revenue'] / total * 100
            second = cbd.iloc[1] if len(cbd) > 1 else None
            hhi = sum((c['revenue']/total*100)**2 for _, c in cbd.iterrows())  # Concentration index
            
            detailed_insight(
                "Geographic Performance Analysis",
                f"<strong>{top['city']}</strong> leads with <strong>{share:.1f}%</strong> of revenue (AED {top['revenue']:,.0f})",
                [
                    f"Second place: <strong>{second['city'] if second is not None else 'N/A'}</strong> ({second['revenue']/total*100:.1f}% share)" if second is not None else "Single city market",
                    f"Market concentration (HHI): <strong>{hhi:.0f}</strong> ({'High' if hhi > 5000 else 'Moderate' if hhi > 2500 else 'Low'})",
                    f"Number of active cities: <strong>{len(cbd)}</strong>",
                    f"Average revenue per city: <strong>AED {total/len(cbd):,.0f}</strong>"
                ],
                [
                    f"{'Diversify revenue - over-reliance on single city is risky' if share > 60 else 'Good geographic balance maintained'}",
                    "Invest in underperforming cities with high potential",
                    "Tailor marketing campaigns to local preferences",
                    "Consider city-specific promotions during local events"
                ],
                "warning" if share > 60 else "success"
            )
    
    # ==================== ROW 4: OUTLIERS & GROWTH ====================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔴 Outlier Detection (Data Quality)")
        df_out = fdf.copy()
        Q1_q, Q3_q = df_out['qty'].quantile(0.25), df_out['qty'].quantile(0.75)
        IQR_q = Q3_q - Q1_q
        Q1_p, Q3_p = df_out['selling_price_aed'].quantile(0.25), df_out['selling_price_aed'].quantile(0.75)
        IQR_p = Q3_p - Q1_p
        
        df_out['outlier_type'] = 'Normal'
        df_out.loc[(df_out['qty'] < Q1_q - 1.5*IQR_q) | (df_out['qty'] > Q3_q + 1.5*IQR_q), 'outlier_type'] = 'Qty Outlier'
        df_out.loc[(df_out['selling_price_aed'] < Q1_p - 1.5*IQR_p) | (df_out['selling_price_aed'] > Q3_p + 1.5*IQR_p), 'outlier_type'] = 'Price Outlier'
        df_out.loc[(df_out['outlier_type'] == 'Qty Outlier') & ((df_out['selling_price_aed'] < Q1_p - 1.5*IQR_p) | (df_out['selling_price_aed'] > Q3_p + 1.5*IQR_p)), 'outlier_type'] = 'Both'
        
        sample = df_out.sample(min(2000, len(df_out)))
        color_map = {'Normal': 'steelblue', 'Qty Outlier': 'orange', 'Price Outlier': 'red', 'Both': 'purple'}
        fig = px.scatter(sample, x='qty', y='selling_price_aed', color='outlier_type', color_discrete_map=color_map, opacity=0.6,
                        labels={'qty': 'Quantity', 'selling_price_aed': 'Price (AED)'})
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
        
        outlier_counts = df_out['outlier_type'].value_counts()
        total_outliers = len(df_out) - outlier_counts.get('Normal', 0)
        pct = total_outliers / len(df_out) * 100
        
        detailed_insight(
            "Data Anomaly Analysis",
            f"<strong>{total_outliers:,}</strong> outliers detected (<strong>{pct:.2f}%</strong> of transactions)",
            [
                f"Quantity outliers: <strong>{outlier_counts.get('Qty Outlier', 0):,}</strong> (unusual order sizes)",
                f"Price outliers: <strong>{outlier_counts.get('Price Outlier', 0):,}</strong> (unusual prices)",
                f"Both qty & price: <strong>{outlier_counts.get('Both', 0):,}</strong> (investigate for fraud)",
                f"Normal transactions: <strong>{outlier_counts.get('Normal', 0):,}</strong>"
            ],
            [
                f"{'Investigate outliers for potential fraud or data errors' if pct > 1 else 'Outlier rate is within acceptable range'}",
                "Review bulk orders (qty outliers) for B2B vs B2C classification",
                "Check price outliers for manual entry errors",
                "Set up automated alerts for real-time anomaly detection"
            ],
            "warning" if pct > 1 else "success"
        )
    
    with col2:
        st.markdown("#### 📈 Revenue Growth Trends")
        if len(daily) > 7:
            df_gr = daily.copy()
            df_gr['date'] = pd.to_datetime(df_gr['date'])
            df_gr = df_gr.sort_values('date')
            df_gr['growth'] = df_gr['revenue'].pct_change() * 100
            df_gr['ma7'] = df_gr['revenue'].rolling(7, min_periods=1).mean()
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], subplot_titles=('Revenue & 7-Day MA', 'Daily Growth %'))
            fig.add_trace(go.Scatter(x=df_gr['date'], y=df_gr['revenue'], name='Revenue', line=dict(color='lightblue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_gr['date'], y=df_gr['ma7'], name='7-Day MA', line=dict(color='steelblue', width=2)), row=1, col=1)
            
            colors = ['green' if g > 0 else 'red' for g in df_gr['growth'].fillna(0)]
            fig.add_trace(go.Bar(x=df_gr['date'], y=df_gr['growth'], marker_color=colors, name='Growth %', showlegend=False), row=2, col=1)
            fig.update_layout(height=420, legend=dict(orientation="h", y=1.15))
            st.plotly_chart(fig, use_container_width=True)
            
            avg_growth = df_gr['growth'].mean()
            positive_days = (df_gr['growth'] > 0).sum()
            negative_days = (df_gr['growth'] < 0).sum()
            max_growth = df_gr['growth'].max()
            min_growth = df_gr['growth'].min()
            volatility = df_gr['growth'].std()
            
            detailed_insight(
                "Growth Momentum Analysis",
                f"Average daily growth: <strong>{avg_growth:.2f}%</strong>",
                [
                    f"Positive growth days: <strong>{positive_days}</strong> ({positive_days/len(df_gr)*100:.0f}%)",
                    f"Negative growth days: <strong>{negative_days}</strong> ({negative_days/len(df_gr)*100:.0f}%)",
                    f"Best day: <strong>+{max_growth:.1f}%</strong> | Worst day: <strong>{min_growth:.1f}%</strong>",
                    f"Volatility (std dev): <strong>{volatility:.1f}%</strong> ({'High' if volatility > 20 else 'Moderate' if volatility > 10 else 'Low'})"
                ],
                [
                    f"{'Maintain momentum with consistent execution' if avg_growth > 0 else 'Diagnose declining trend immediately'}",
                    f"{'High volatility - stabilize with loyalty programs' if volatility > 20 else 'Stable growth pattern - optimize further'}",
                    "Analyze best/worst days to identify success factors",
                    "Set weekly growth targets with team accountability"
                ],
                "success" if avg_growth > 0 else "warning"
            )
        else:
            st.info("Need more data for growth analysis")
    
    # ==================== ROW 5: MARGIN QUADRANT & PRIORITY ====================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💰 Product Portfolio Quadrant")
        pp = fdf.groupby('product_id').agg({'selling_price_aed': 'sum', 'qty': 'sum'}).reset_index()
        pp.columns = ['product_id', 'revenue', 'units']
        pp = pp.merge(data['products'][['product_id', 'category', 'unit_cost_aed']], on='product_id', how='left')
        pp['margin'] = pp['revenue'] - (pp['units'] * pp['unit_cost_aed'])
        pp['margin_pct'] = (pp['margin'] / pp['revenue'] * 100).fillna(0)
        
        med_r = pp['revenue'].median()
        med_m = pp['margin_pct'].median()
        
        pp['quadrant'] = 'Dogs'
        pp.loc[(pp['revenue'] > med_r) & (pp['margin_pct'] > med_m), 'quadrant'] = 'Stars'
        pp.loc[(pp['revenue'] <= med_r) & (pp['margin_pct'] > med_m), 'quadrant'] = 'Question Marks'
        pp.loc[(pp['revenue'] > med_r) & (pp['margin_pct'] <= med_m), 'quadrant'] = 'Cash Cows'
        
        fig = px.scatter(pp, x='revenue', y='margin_pct', color='quadrant', size='units',
                        color_discrete_map={'Stars': 'gold', 'Cash Cows': 'green', 'Question Marks': 'blue', 'Dogs': 'red'},
                        labels={'revenue': 'Revenue (AED)', 'margin_pct': 'Margin %'})
        fig.add_hline(y=med_m, line_dash="dash", line_color="gray")
        fig.add_vline(x=med_r, line_dash="dash", line_color="gray")
        fig.add_annotation(x=med_r*2, y=med_m*1.5, text="⭐ Stars", showarrow=False, font=dict(size=12, color="gold"))
        fig.add_annotation(x=med_r*0.3, y=med_m*1.5, text="❓ Question", showarrow=False, font=dict(size=12, color="blue"))
        fig.add_annotation(x=med_r*2, y=med_m*0.3, text="🐄 Cash Cows", showarrow=False, font=dict(size=12, color="green"))
        fig.add_annotation(x=med_r*0.3, y=med_m*0.3, text="🐕 Dogs", showarrow=False, font=dict(size=12, color="red"))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        quad_counts = pp['quadrant'].value_counts()
        stars = quad_counts.get('Stars', 0)
        cows = quad_counts.get('Cash Cows', 0)
        questions = quad_counts.get('Question Marks', 0)
        dogs = quad_counts.get('Dogs', 0)
        
        detailed_insight(
            "BCG Portfolio Matrix",
            f"Portfolio composition: <strong>{stars} Stars</strong>, <strong>{cows} Cash Cows</strong>, <strong>{questions} Question Marks</strong>, <strong>{dogs} Dogs</strong>",
            [
                f"⭐ Stars (High Revenue + High Margin): {stars} products - INVEST",
                f"🐄 Cash Cows (High Revenue + Low Margin): {cows} products - MILK",
                f"❓ Question Marks (Low Revenue + High Margin): {questions} products - GROW",
                f"🐕 Dogs (Low Revenue + Low Margin): {dogs} products - DIVEST"
            ],
            [
                f"Allocate {stars/(stars+cows+questions+dogs)*100:.0f}% of marketing to Stars",
                "Protect Cash Cows - they fund growth initiatives",
                "Invest in Question Marks with highest potential",
                f"Review {dogs} Dogs for discontinuation to reduce complexity"
            ],
            "success" if stars > dogs else "warning"
        )
    
    with col2:
        st.markdown("#### 🎯 Product Priority Algorithm")
        pm = fdf.groupby('product_id').agg({'selling_price_aed': 'sum', 'qty': 'sum', 'order_id': 'nunique'}).reset_index()
        pm.columns = ['product_id', 'revenue', 'units', 'orders']
        pm = pm.merge(data['products'][['product_id', 'category', 'unit_cost_aed']], on='product_id', how='left')
        pm['margin'] = pm['revenue'] - (pm['units'] * pm['unit_cost_aed'])
        pm['margin_pct'] = (pm['margin'] / pm['revenue'] * 100).fillna(0)
        pm['velocity'] = pm['orders'] / max(1, (pd.to_datetime(fdf['order_time']).max() - pd.to_datetime(fdf['order_time']).min()).days)
        
        for col in ['revenue', 'margin_pct', 'velocity']:
            pm[f'{col}_n'] = (pm[col] - pm[col].min()) / (pm[col].max() - pm[col].min() + 0.001)
        
        # Priority score with weights
        pm['score'] = (pm['revenue_n'] * 0.4 + pm['margin_pct_n'] * 0.35 + pm['velocity_n'] * 0.25) * 100
        
        top = pm.nlargest(15, 'score')
        fig = px.bar(top, x='product_id', y='score', color='category',
                    hover_data=['revenue', 'margin_pct', 'velocity'],
                    labels={'score': 'Priority Score', 'product_id': 'Product'})
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        top_cat = top['category'].value_counts().index[0]
        avg_score = pm['score'].mean()
        top_score = top['score'].iloc[0]
        
        detailed_insight(
            "Priority Scoring Algorithm",
            f"Score = <strong>40% Revenue</strong> + <strong>35% Margin</strong> + <strong>25% Velocity</strong>",
            [
                f"Top priority category: <strong>{top_cat}</strong>",
                f"Highest score: <strong>{top_score:.1f}</strong> | Average: <strong>{avg_score:.1f}</strong>",
                f"Score range: {pm['score'].min():.1f} - {pm['score'].max():.1f}",
                f"Products above average: <strong>{len(pm[pm['score'] > avg_score])}</strong>"
            ],
            [
                "Focus inventory investment on top 15 priority products",
                "Ensure 99%+ availability for high-priority items",
                "Assign dedicated account management for top SKUs",
                "Review bottom 20% monthly for optimization"
            ],
            "success"
        )
    
    # ==================== ROW 6: MATRIX & WHAT-IF ====================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔥 City × Channel Performance Matrix")
        df_m = fdf.merge(data['stores'][['store_id', 'city', 'channel']], on='store_id', how='left', suffixes=('', '_y'))
        df_m['revenue'] = df_m['qty'] * df_m['selling_price_aed']
        pivot = df_m.groupby(['city', 'channel'])['revenue'].sum().reset_index().pivot(index='city', columns='channel', values='revenue').fillna(0)
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            colorscale='Viridis',
            text=[[f'AED {v:,.0f}' for v in row] for row in pivot.values],
            texttemplate='%{text}', textfont={"size": 10}
        ))
        fig.update_layout(height=400, xaxis_title='Channel', yaxis_title='City')
        st.plotly_chart(fig, use_container_width=True)
        
        best = pivot.stack().idxmax()
        worst = pivot.stack().idxmin()
        best_val = pivot.loc[best[0], best[1]]
        worst_val = pivot.loc[worst[0], worst[1]]
        channel_totals = pivot.sum()
        best_channel = channel_totals.idxmax()
        
        detailed_insight(
            "Channel-Geography Matrix",
            f"Best combination: <strong>{best[0]} × {best[1]}</strong> (AED {best_val:,.0f})",
            [
                f"Weakest combination: <strong>{worst[0]} × {worst[1]}</strong> (AED {worst_val:,.0f})",
                f"Top performing channel overall: <strong>{best_channel}</strong> (AED {channel_totals[best_channel]:,.0f})",
                f"Channel mix: {' | '.join([f'{c}: {v/channel_totals.sum()*100:.0f}%' for c, v in channel_totals.items()])}",
                f"Cross-sell opportunity: {worst_val/best_val*100:.0f}% potential in weakest vs best"
            ],
            [
                f"Double down on {best[0]} × {best[1]} - proven winner",
                f"Investigate why {worst[0]} × {worst[1]} underperforms",
                "Test targeted campaigns in underperforming segments",
                "Align channel strategy with city demographics"
            ],
            "success"
        )
    
    with col2:
        st.markdown("#### 🔮 What-If Scenario Analysis")
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
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix, x=[f'{d}%' for d in discs], y=[f'AED {b:,}' for b in budgets],
            colorscale='RdYlGn',
            text=[[f'AED {v:,.0f}' for v in row] for row in matrix],
            texttemplate='%{text}', textfont={"size": 9}
        ))
        fig.update_layout(height=400, xaxis_title='Discount Level', yaxis_title='Budget')
        st.plotly_chart(fig, use_container_width=True)
        
        max_profit = max(max(row) for row in matrix)
        min_profit = min(min(row) for row in matrix)
        optimal_idx = [(i, j) for i, row in enumerate(matrix) for j, v in enumerate(row) if v == max_profit][0]
        optimal_budget = budgets[optimal_idx[0]]
        optimal_discount = discs[optimal_idx[1]]
        
        detailed_insight(
            "Promotion Optimization",
            f"Optimal scenario: <strong>{optimal_discount}% discount</strong> with <strong>AED {optimal_budget:,} budget</strong>",
            [
                f"Maximum profit potential: <strong>AED {max_profit:,.0f}</strong>",
                f"Minimum scenario (worst case): <strong>AED {min_profit:,.0f}</strong>",
                f"Profit range: AED {min_profit:,.0f} to AED {max_profit:,.0f}",
                f"Current selection ({sim['discount_pct']}%, AED {sim['promo_budget']:,}): AED {sim_res['results']['profit_proxy']:,.0f}" if sim_res['results'] else "N/A"
            ],
            [
                f"{'Use recommended {optimal_discount}% discount for best ROI' if optimal_discount != sim['discount_pct'] else 'Current discount level is optimal'}",
                "Avoid high discount + low budget combinations",
                "Test scenarios in limited markets before full rollout",
                "Monitor margin floor constraint at higher discounts"
            ],
            "success" if max_profit > 0 else "warning"
        )
    
    # ==================== ROW 7: SEASONALITY & CATEGORY MARGINS ====================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📅 Sales Seasonality (Day × Hour)")
        df_s = fdf.copy()
        df_s['order_time'] = pd.to_datetime(df_s['order_time'], errors='coerce')
        df_s = df_s.dropna(subset=['order_time'])
        df_s['day'] = df_s['order_time'].dt.day_name()
        df_s['hour'] = df_s['order_time'].dt.hour
        df_s['revenue'] = df_s['qty'] * df_s['selling_price_aed']
        
        pivot_s = df_s.groupby(['day', 'hour'])['revenue'].sum().reset_index().pivot(index='day', columns='hour', values='revenue').fillna(0)
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_s = pivot_s.reindex([d for d in days if d in pivot_s.index])
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_s.values, x=pivot_s.columns.tolist(), y=pivot_s.index.tolist(),
            colorscale='Blues'
        ))
        fig.update_layout(height=400, xaxis_title='Hour of Day', yaxis_title='Day of Week')
        st.plotly_chart(fig, use_container_width=True)
        
        max_idx = np.unravel_index(np.argmax(pivot_s.values), pivot_s.values.shape)
        peak_day = pivot_s.index[max_idx[0]]
        peak_hour = pivot_s.columns[max_idx[1]]
        peak_revenue = pivot_s.values[max_idx[0], max_idx[1]]
        
        # Find slow periods
        min_idx = np.unravel_index(np.argmin(pivot_s.values[pivot_s.values > 0]), pivot_s.values.shape) if (pivot_s.values > 0).any() else (0, 0)
        slow_day = pivot_s.index[min_idx[0]]
        slow_hour = pivot_s.columns[min_idx[1]]
        
        # Weekend vs weekday
        weekend_rev = pivot_s.loc[pivot_s.index.isin(['Saturday', 'Sunday'])].values.sum() if any(d in pivot_s.index for d in ['Saturday', 'Sunday']) else 0
        weekday_rev = pivot_s.loc[~pivot_s.index.isin(['Saturday', 'Sunday'])].values.sum()
        
        detailed_insight(
            "Temporal Sales Patterns",
            f"Peak sales: <strong>{peak_day} at {peak_hour}:00</strong> (AED {peak_revenue:,.0f})",
            [
                f"Slowest period: <strong>{slow_day} at {slow_hour}:00</strong>",
                f"Weekend revenue: <strong>AED {weekend_rev:,.0f}</strong> ({weekend_rev/(weekend_rev+weekday_rev)*100:.1f}%)" if weekend_rev + weekday_rev > 0 else "N/A",
                f"Weekday revenue: <strong>AED {weekday_rev:,.0f}</strong> ({weekday_rev/(weekend_rev+weekday_rev)*100:.1f}%)" if weekend_rev + weekday_rev > 0 else "N/A",
                "Darker cells = higher revenue"
            ],
            [
                f"Schedule flash sales on {peak_day} around {peak_hour}:00",
                f"Boost slow periods ({slow_day} {slow_hour}:00) with special offers",
                "Align customer service staffing with peak hours",
                "Time email/push campaigns 1-2 hours before peaks"
            ],
            "success"
        )
    
    with col2:
        st.markdown("#### 📊 Category Margin Analysis")
        catbd = kpi_calc.compute_breakdown(fdf, 'category')
        if len(catbd) > 0:
            fig = px.bar(catbd, x='category', y='margin_pct', color='margin_pct',
                        color_continuous_scale='RdYlGn', text='margin_pct',
                        labels={'margin_pct': 'Margin %', 'category': 'Category'})
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.add_hline(y=sim['margin_floor'], line_dash="dash", line_color="red",
                         annotation_text=f"Floor: {sim['margin_floor']}%")
            fig.add_hline(y=catbd['margin_pct'].mean(), line_dash="dot", line_color="blue",
                         annotation_text=f"Avg: {catbd['margin_pct'].mean():.1f}%")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            low_margin = catbd[catbd['margin_pct'] < sim['margin_floor']]['category'].tolist()
            high_margin = catbd[catbd['margin_pct'] > 25]['category'].tolist()
            avg_margin = catbd['margin_pct'].mean()
            top_margin_cat = catbd.loc[catbd['margin_pct'].idxmax(), 'category']
            low_margin_cat = catbd.loc[catbd['margin_pct'].idxmin(), 'category']
            
            detailed_insight(
                "Category Profitability",
                f"Average margin: <strong>{avg_margin:.1f}%</strong> across {len(catbd)} categories",
                [
                    f"Highest margin: <strong>{top_margin_cat}</strong> ({catbd['margin_pct'].max():.1f}%)",
                    f"Lowest margin: <strong>{low_margin_cat}</strong> ({catbd['margin_pct'].min():.1f}%)",
                    f"Categories below floor ({sim['margin_floor']}%): <strong>{', '.join(low_margin) if low_margin else 'None'}</strong>",
                    f"High-margin categories (>25%): <strong>{', '.join(high_margin) if high_margin else 'None'}</strong>"
                ],
                [
                    f"{'⚠️ Review pricing in: ' + ', '.join(low_margin) if low_margin else '✅ All categories above margin floor'}",
                    f"Expand {top_margin_cat} assortment - highest profitability",
                    "Negotiate supplier costs for low-margin categories",
                    "Consider premium product tiers in high-margin categories"
                ],
                "warning" if low_margin else "success"
            )
    
    # ==================== RECOMMENDATIONS ====================
    st.markdown("---")
    st.markdown("### 💡 Strategic Recommendations")
    
    recs = generate_recommendation(kpis, sim_res['results'] if sim_res['results'] else {}, sim_res['violations'])
    
    cols = st.columns(2)
    for i, r in enumerate(recs):
        with cols[i % 2]:
            if r.startswith("✅"):
                st.success(r)
            elif r.startswith("⚠️") or r.startswith("💡"):
                st.warning(r)
            elif r.startswith("🚫"):
                st.error(r)
            else:
                st.info(r)
    
    # ==================== DOWNLOADS ====================
    st.markdown("---")
    st.markdown("### 📥 Export Reports")
    
    cols = st.columns(4)
    with cols[0]:
        st.download_button("📄 Cleaned Sales Data", data['sales'].to_csv(index=False), "sales_cleaned.csv", "text/csv")
    with cols[1]:
        if 'issues' in data and len(data['issues']) > 0:
            st.download_button("📄 Data Quality Issues", data['issues'].to_csv(index=False), "data_issues.csv", "text/csv")
    with cols[2]:
        if sim_res['results']:
            sim_df = pd.DataFrame([sim_res['results']])
            st.download_button("📄 Simulation Results", sim_df.to_csv(index=False), "simulation_results.csv", "text/csv")
    with cols[3]:
        if sim_res['top_risk_items'] is not None:
            st.download_button("📄 Stockout Risk Items", sim_res['top_risk_items'].to_csv(index=False), "stockout_risk.csv", "text/csv")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>UAE Promo Pulse Simulator v2.0 | Advanced Retail Analytics</p>", unsafe_allow_html=True)

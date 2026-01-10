"""
app.py
UAE Promo Pulse Simulator - Streamlit Dashboard
Main application with Executive and Manager views
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

from data_generator import generate_all_data
from cleaner import DataCleaner
from simulator import KPICalculator, PromoSimulator, generate_recommendation

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="UAE Promo Pulse Simulator",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
    }
    .stats-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1E88E5;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #43a047;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'raw_data_generated' not in st.session_state:
    st.session_state.raw_data_generated = False
if 'cleaning_stats' not in st.session_state:
    st.session_state.cleaning_stats = None

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_data_from_files(raw_dir='data/raw', cleaned_dir='data/cleaned'):
    """Load data from CSV files"""
    try:
        if os.path.exists(cleaned_dir):
            sales = pd.read_csv(f'{cleaned_dir}/sales_cleaned.csv')
            products = pd.read_csv(f'{cleaned_dir}/products_cleaned.csv')
            stores = pd.read_csv(f'{cleaned_dir}/stores_cleaned.csv')
            inventory = pd.read_csv(f'{cleaned_dir}/inventory_cleaned.csv')
            issues = pd.read_csv(f'{cleaned_dir}/issues.csv') if os.path.exists(f'{cleaned_dir}/issues.csv') else pd.DataFrame()
            
            return {
                'sales': sales,
                'products': products,
                'stores': stores,
                'inventory': inventory,
                'issues': issues,
                'source': 'cleaned'
            }
    except Exception as e:
        st.error(f"Error loading data: {e}")
    return None


def generate_raw_data():
    """Generate synthetic raw data only"""
    with st.spinner("🔄 Generating synthetic raw data..."):
        raw_data = generate_all_data('data/raw')
    return raw_data


def clean_data(raw_data):
    """Clean the raw data and return cleaning statistics"""
    with st.spinner("🧹 Cleaning data..."):
        # Store original counts
        original_counts = {
            'products': len(raw_data['products']),
            'stores': len(raw_data['stores']),
            'sales': len(raw_data['sales']),
            'inventory': len(raw_data['inventory']),
            'campaigns': len(raw_data['campaigns'])
        }
        
        # Clean the data
        cleaner = DataCleaner()
        cleaned_data = cleaner.clean_all(
            raw_data['products'],
            raw_data['stores'],
            raw_data['sales'],
            raw_data['inventory'],
            'data/cleaned'
        )
        
        # Calculate cleaned counts
        cleaned_counts = {
            'products': len(cleaned_data['products']),
            'stores': len(cleaned_data['stores']),
            'sales': len(cleaned_data['sales']),
            'inventory': len(cleaned_data['inventory'])
        }
        
        # Calculate removed counts
        removed_counts = {
            'products': original_counts['products'] - cleaned_counts['products'],
            'stores': original_counts['stores'] - cleaned_counts['stores'],
            'sales': original_counts['sales'] - cleaned_counts['sales'],
            'inventory': original_counts['inventory'] - cleaned_counts['inventory']
        }
        
        # Get issues summary
        issues_summary = {}
        if len(cleaned_data['issues']) > 0:
            issues_summary = cleaned_data['issues']['issue_type'].value_counts().to_dict()
        
        cleaning_stats = {
            'original': original_counts,
            'cleaned': cleaned_counts,
            'removed': removed_counts,
            'issues_summary': issues_summary,
            'total_issues': len(cleaned_data['issues'])
        }
        
        return cleaned_data, cleaning_stats


def process_uploaded_file(uploaded_file, file_type):
    """Process uploaded file and return DataFrame"""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading {file_type}: {e}")
    return None


# ============================================================================
# CLEANING STATS DISPLAY COMPONENT
# ============================================================================

def display_cleaning_stats(stats):
    """Display cleaning statistics in a nice format"""
    if stats is None:
        return
    
    st.markdown("### 📊 Data Cleaning Statistics")
    
    # Create columns for the stats table
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**📁 Table**")
        for table in ['products', 'stores', 'sales', 'inventory']:
            st.markdown(f"`{table}`")
    
    with col2:
        st.markdown("**📥 Original Records**")
        for table in ['products', 'stores', 'sales', 'inventory']:
            st.markdown(f"**{stats['original'][table]:,}**")
    
    with col3:
        st.markdown("**🗑️ Records Removed**")
        for table in ['products', 'stores', 'sales', 'inventory']:
            removed = stats['removed'][table]
            if removed > 0:
                st.markdown(f"<span style='color: red;'>-{removed:,}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color: green;'>0</span>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("**✅ Clean Records**")
        for table in ['products', 'stores', 'sales', 'inventory']:
            st.markdown(f"**{stats['cleaned'][table]:,}**")
    
    # Summary metrics
    st.markdown("---")
    
    total_original = sum(stats['original'].values())
    total_cleaned = sum(stats['cleaned'].values())
    total_removed = sum(stats['removed'].values())
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Original Records",
            f"{total_original:,}"
        )
    
    with col2:
        st.metric(
            "Total Records Removed",
            f"{total_removed:,}",
            delta=f"-{total_removed:,}" if total_removed > 0 else "0",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Total Clean Records",
            f"{total_cleaned:,}"
        )
    
    with col4:
        st.metric(
            "Total Issues Logged",
            f"{stats['total_issues']:,}"
        )
    
    # Issues breakdown
    if stats['issues_summary']:
        st.markdown("---")
        st.markdown("### 🔍 Issues Breakdown by Type")
        
        issues_df = pd.DataFrame([
            {'Issue Type': k, 'Count': v} 
            for k, v in stats['issues_summary'].items()
        ]).sort_values('Count', ascending=False)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(issues_df, use_container_width=True, hide_index=True)
        
        with col2:
            fig = px.bar(
                issues_df, 
                x='Issue Type', 
                y='Count',
                color='Count',
                color_continuous_scale='Reds',
                title='Issues by Type'
            )
            fig.update_layout(height=300, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    """Render sidebar with filters and controls"""
    
    st.sidebar.markdown("## 🎛️ Controls")
    
    # View Toggle
    view_mode = st.sidebar.radio(
        "Dashboard View",
        ["Executive View", "Manager View"],
        help="Toggle between Executive (financial) and Manager (operational) views"
    )
    
    st.sidebar.markdown("---")
    
    # Data Source
    st.sidebar.markdown("### 📊 Data Source")
    data_source = st.sidebar.radio(
        "Select data source:",
        ["Generate & Clean Data", "Load Existing Data", "Upload Custom Data"]
    )
    
    if data_source == "Generate & Clean Data":
        st.sidebar.markdown("---")
        st.sidebar.markdown("#### Step 1: Generate Raw Data")
        
        # Generate Data Button
        if st.sidebar.button("🎲 Generate Raw Data", use_container_width=True):
            st.session_state.raw_data = generate_raw_data()
            st.session_state.raw_data_generated = True
            st.session_state.data_loaded = False
            st.session_state.cleaning_stats = None
            st.rerun()
        
        # Show status of raw data
        if st.session_state.raw_data_generated and st.session_state.raw_data is not None:
            st.sidebar.success("✅ Raw data generated!")
            st.sidebar.markdown(f"""
            **Raw Data Summary:**
            - Products: {len(st.session_state.raw_data['products']):,}
            - Stores: {len(st.session_state.raw_data['stores']):,}
            - Sales: {len(st.session_state.raw_data['sales']):,}
            - Inventory: {len(st.session_state.raw_data['inventory']):,}
            - Campaigns: {len(st.session_state.raw_data['campaigns']):,}
            """)
        else:
            st.sidebar.info("⏳ No raw data generated yet")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("#### Step 2: Clean Data")
        
        # Clean Data Button (only enabled if raw data exists)
        clean_button_disabled = not st.session_state.raw_data_generated
        
        if st.sidebar.button(
            "🧹 Clean Data", 
            use_container_width=True, 
            disabled=clean_button_disabled
        ):
            if st.session_state.raw_data is not None:
                cleaned_data, cleaning_stats = clean_data(st.session_state.raw_data)
                st.session_state.data = cleaned_data
                st.session_state.cleaning_stats = cleaning_stats
                st.session_state.data_loaded = True
                st.rerun()
        
        if clean_button_disabled:
            st.sidebar.warning("⚠️ Generate raw data first!")
        
        # Show status of cleaned data
        if st.session_state.data_loaded:
            st.sidebar.success("✅ Data cleaned and ready!")
    
    elif data_source == "Load Existing Data":
        if st.sidebar.button("📂 Load Data", use_container_width=True):
            data = load_data_from_files()
            if data:
                st.session_state.data = data
                st.session_state.data_loaded = True
                st.session_state.cleaning_stats = None  # No stats for loaded data
                st.rerun()
            else:
                st.sidebar.error("No existing data found. Generate new data first.")
    
    elif data_source == "Upload Custom Data":
        st.sidebar.markdown("#### Upload Files")
        
        sales_file = st.sidebar.file_uploader("Sales Data", type=['csv', 'xlsx'])
        products_file = st.sidebar.file_uploader("Products Data", type=['csv', 'xlsx'])
        stores_file = st.sidebar.file_uploader("Stores Data", type=['csv', 'xlsx'])
        inventory_file = st.sidebar.file_uploader("Inventory Data", type=['csv', 'xlsx'])
        
        if st.sidebar.button("Process Uploaded Data", use_container_width=True):
            if all([sales_file, products_file, stores_file, inventory_file]):
                sales = process_uploaded_file(sales_file, 'sales')
                products = process_uploaded_file(products_file, 'products')
                stores = process_uploaded_file(stores_file, 'stores')
                inventory = process_uploaded_file(inventory_file, 'inventory')
                
                if all([sales is not None, products is not None, stores is not None, inventory is not None]):
                    # Store as raw data
                    st.session_state.raw_data = {
                        'products': products,
                        'stores': stores,
                        'sales': sales,
                        'inventory': inventory,
                        'campaigns': pd.DataFrame()
                    }
                    st.session_state.raw_data_generated = True
                    
                    # Clean the data
                    cleaned_data, cleaning_stats = clean_data(st.session_state.raw_data)
                    st.session_state.data = cleaned_data
                    st.session_state.cleaning_stats = cleaning_stats
                    st.session_state.data_loaded = True
                    st.rerun()
            else:
                st.sidebar.warning("Please upload all required files.")
    
    st.sidebar.markdown("---")
    
    # Filters (only if data is loaded)
    filters = {}
    if st.session_state.data_loaded and st.session_state.data:
        st.sidebar.markdown("### 🔍 Filters")
        
        data = st.session_state.data
        
        # Date Range
        if 'sales' in data and 'order_time' in data['sales'].columns:
            sales_df = data['sales'].copy()
            sales_df['order_time'] = pd.to_datetime(sales_df['order_time'])
            min_date = sales_df['order_time'].min().date()
            max_date = sales_df['order_time'].max().date()
            
            date_range = st.sidebar.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            filters['date_range'] = date_range
        
        # City Filter
        if 'stores' in data:
            cities = ['All'] + sorted(data['stores']['city'].unique().tolist())
            filters['city'] = st.sidebar.selectbox("City", cities)
        
        # Channel Filter
        if 'stores' in data:
            channels = ['All'] + sorted(data['stores']['channel'].unique().tolist())
            filters['channel'] = st.sidebar.selectbox("Channel", channels)
        
        # Category Filter
        if 'products' in data:
            categories = ['All'] + sorted(data['products']['category'].unique().tolist())
            filters['category'] = st.sidebar.selectbox("Category", categories)
        
        # Brand Filter
        if 'products' in data:
            brands = ['All'] + sorted(data['products']['brand'].unique().tolist())
            filters['brand'] = st.sidebar.selectbox("Brand", brands)
        
        # Fulfillment Filter
        if 'stores' in data and 'fulfillment_type' in data['stores'].columns:
            fulfillments = ['All'] + sorted(data['stores']['fulfillment_type'].unique().tolist())
            filters['fulfillment'] = st.sidebar.selectbox("Fulfillment Type", fulfillments)
    
    st.sidebar.markdown("---")
    
    # Simulation Controls
    st.sidebar.markdown("### 🎮 Simulation Parameters")
    
    sim_params = {
        'discount_pct': st.sidebar.slider("Discount %", 0, 50, 15),
        'promo_budget': st.sidebar.number_input("Promo Budget (AED)", 10000, 500000, 50000, step=5000),
        'margin_floor': st.sidebar.slider("Margin Floor %", 5, 30, 15),
        'simulation_days': st.sidebar.selectbox("Simulation Window", [7, 14], index=1)
    }
    
    return view_mode, filters, sim_params


# ============================================================================
# EXECUTIVE VIEW
# ============================================================================

def render_executive_view(data, filters, sim_params):
    """Render Executive/CFO view with financial KPIs"""
    
    st.markdown("## 📊 Executive Dashboard")
    st.markdown("*Financial and strategic overview for leadership*")
    
    # Show cleaning stats if available
    if st.session_state.cleaning_stats is not None:
        with st.expander("📊 View Data Cleaning Statistics", expanded=False):
            display_cleaning_stats(st.session_state.cleaning_stats)
    
    # Initialize calculators
    kpi_calc = KPICalculator(
        data['sales'], data['products'], 
        data['stores'], data['inventory']
    )
    
    simulator = PromoSimulator(
        data['sales'], data['products'],
        data['stores'], data['inventory']
    )
    
    # Apply filters
    start_date = filters.get('date_range', [None, None])[0] if 'date_range' in filters else None
    end_date = filters.get('date_range', [None, None])[1] if 'date_range' in filters and len(filters['date_range']) > 1 else None
    
    filtered_df = kpi_calc.filter_data(
        city=filters.get('city'),
        channel=filters.get('channel'),
        category=filters.get('category'),
        start_date=start_date,
        end_date=end_date,
        brand=filters.get('brand'),
        fulfillment=filters.get('fulfillment')
    )
    
    # Compute KPIs
    kpis = kpi_calc.compute_historical_kpis(filtered_df)
    
    # Run simulation
    sim_result = simulator.run_simulation(
        discount_pct=sim_params['discount_pct'],
        promo_budget=sim_params['promo_budget'],
        margin_floor=sim_params['margin_floor'],
        simulation_days=sim_params['simulation_days'],
        city=filters.get('city'),
        channel=filters.get('channel'),
        category=filters.get('category')
    )
    
    # KPI Cards Row
    st.markdown("### Key Financial Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Net Revenue",
            f"AED {kpis['net_revenue']:,.0f}",
            help="Gross revenue minus refunds"
        )
    
    with col2:
        st.metric(
            "Gross Margin %",
            f"{kpis['gross_margin_pct']:.1f}%",
            delta=f"{kpis['gross_margin_pct'] - 25:.1f}% vs target" if kpis['gross_margin_pct'] else None
        )
    
    with col3:
        if sim_result['results']:
            st.metric(
                "Profit Proxy (Sim)",
                f"AED {sim_result['results']['profit_proxy']:,.0f}",
                delta="Viable" if sim_result['results']['profit_proxy'] > 0 else "Loss"
            )
        else:
            st.metric("Profit Proxy (Sim)", "N/A")
    
    with col4:
        if sim_result['results']:
            st.metric(
                "Budget Utilization",
                f"{sim_result['results']['budget_utilization']:.1f}%",
                delta="Over" if sim_result['results']['budget_utilization'] > 100 else "Within"
            )
        else:
            st.metric("Budget Utilization", "N/A")
    
    st.markdown("---")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Revenue Trend")
        daily_kpis = kpi_calc.compute_daily_kpis(filtered_df)
        if len(daily_kpis) > 0:
            fig = px.line(
                daily_kpis, x='date', y='revenue',
                title='Daily Net Revenue',
                labels={'revenue': 'Revenue (AED)', 'date': 'Date'}
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for trend chart")
    
    with col2:
        st.markdown("#### 🏙️ Revenue by City & Channel")
        breakdown = kpi_calc.compute_breakdown(filtered_df, 'city')
        if len(breakdown) > 0:
            fig = px.bar(
                breakdown, x='city', y='revenue',
                color='city',
                title='Revenue Distribution by City',
                labels={'revenue': 'Revenue (AED)', 'city': 'City'}
            )
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available")
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Margin % by Category")
        cat_breakdown = kpi_calc.compute_breakdown(filtered_df, 'category')
        if len(cat_breakdown) > 0:
            fig = px.bar(
                cat_breakdown, x='category', y='margin_pct',
                color='margin_pct',
                color_continuous_scale='RdYlGn',
                title='Gross Margin % by Category',
                labels={'margin_pct': 'Margin %', 'category': 'Category'}
            )
            fig.add_hline(y=sim_params['margin_floor'], line_dash="dash", 
                         annotation_text=f"Floor: {sim_params['margin_floor']}%")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available")
    
    with col2:
        st.markdown("#### 🎯 Scenario Impact Analysis")
        # Run scenarios
        scenarios = simulator.run_scenario_comparison(
            discount_levels=[5, 10, 15, 20, 25, 30],
            promo_budget=sim_params['promo_budget'],
            margin_floor=sim_params['margin_floor'],
            simulation_days=sim_params['simulation_days'],
            city=filters.get('city'),
            channel=filters.get('channel'),
            category=filters.get('category')
        )
        
        if len(scenarios) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=scenarios['discount_pct'], y=scenarios['profit_proxy'],
                mode='lines+markers', name='Profit Proxy',
                line=dict(color='green', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=scenarios['discount_pct'], y=scenarios['margin_pct'] * 1000,
                mode='lines+markers', name='Margin % (scaled)',
                line=dict(color='blue', width=2, dash='dot'),
                yaxis='y2'
            ))
            fig.update_layout(
                title='Profit vs Discount Level',
                xaxis_title='Discount %',
                yaxis_title='Profit Proxy (AED)',
                yaxis2=dict(title='Margin %', overlaying='y', side='right'),
                height=350,
                legend=dict(x=0.7, y=1.1, orientation='h')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No scenario data available")
    
    # Recommendation Box
    st.markdown("---")
    st.markdown("### 💡 Recommendations")
    
    recommendations = generate_recommendation(
        kpis, 
        sim_result['results'] if sim_result['results'] else {},
        sim_result['violations']
    )
    
    for rec in recommendations:
        if rec.startswith("✅"):
            st.success(rec)
        elif rec.startswith("⚠️") or rec.startswith("💡"):
            st.warning(rec)
        elif rec.startswith("🚫") or rec.startswith("❌"):
            st.error(rec)
        else:
            st.info(rec)
    
    # Violations Alert
    if sim_result['violations']:
        st.markdown("### ⚠️ Constraint Violations")
        for v in sim_result['violations']:
            if v['severity'] == 'HIGH':
                st.error(f"**{v['constraint']}**: {v['message']}")
            else:
                st.warning(f"**{v['constraint']}**: {v['message']}")


# ============================================================================
# MANAGER VIEW
# ============================================================================

def render_manager_view(data, filters, sim_params):
    """Render Manager/Operations view with operational KPIs"""
    
    st.markdown("## 🔧 Operations Dashboard")
    st.markdown("*Operational risks and execution insights for managers*")
    
    # Show cleaning stats if available
    if st.session_state.cleaning_stats is not None:
        with st.expander("📊 View Data Cleaning Statistics", expanded=True):
            display_cleaning_stats(st.session_state.cleaning_stats)
    
    # Initialize calculators
    kpi_calc = KPICalculator(
        data['sales'], data['products'],
        data['stores'], data['inventory']
    )
    
    simulator = PromoSimulator(
        data['sales'], data['products'],
        data['stores'], data['inventory']
    )
    
    # Apply filters
    start_date = filters.get('date_range', [None, None])[0] if 'date_range' in filters else None
    end_date = filters.get('date_range', [None, None])[1] if 'date_range' in filters and len(filters['date_range']) > 1 else None
    
    filtered_df = kpi_calc.filter_data(
        city=filters.get('city'),
        channel=filters.get('channel'),
        category=filters.get('category'),
        start_date=start_date,
        end_date=end_date,
        brand=filters.get('brand'),
        fulfillment=filters.get('fulfillment')
    )
    
    # Compute KPIs
    kpis = kpi_calc.compute_historical_kpis(filtered_df)
    
    # Run simulation
    sim_result = simulator.run_simulation(
        discount_pct=sim_params['discount_pct'],
        promo_budget=sim_params['promo_budget'],
        margin_floor=sim_params['margin_floor'],
        simulation_days=sim_params['simulation_days'],
        city=filters.get('city'),
        channel=filters.get('channel'),
        category=filters.get('category')
    )
    
    # KPI Cards Row
    st.markdown("### Key Operational Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if sim_result['results']:
            stockout_risk = sim_result['results']['stockout_risk_pct']
            st.metric(
                "Stockout Risk %",
                f"{stockout_risk:.1f}%",
                delta="High" if stockout_risk > 30 else "Normal",
                delta_color="inverse"
            )
        else:
            st.metric("Stockout Risk %", "N/A")
    
    with col2:
        st.metric(
            "Return Rate %",
            f"{kpis['return_rate']:.1f}%",
            delta="High" if kpis['return_rate'] > 10 else "Normal",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Payment Failure Rate %",
            f"{kpis['payment_failure_rate']:.1f}%",
            delta="High" if kpis['payment_failure_rate'] > 10 else "Normal",
            delta_color="inverse"
        )
    
    with col4:
        if sim_result['results']:
            st.metric(
                "High-Risk SKUs",
                f"{sim_result['results']['high_risk_skus']}",
                help="Product-store combinations with >80% demand vs stock"
            )
        else:
            st.metric("High-Risk SKUs", "N/A")
    
    st.markdown("---")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏙️ Stockout Risk by City/Channel")
        if sim_result['results'] and 'detail_data' in sim_result:
            detail_data = sim_result['detail_data']
            
            # Get store info
            stores_df = data['stores']
            risk_by_location = detail_data.merge(
                stores_df[['store_id', 'city', 'channel']],
                on='store_id', how='left', suffixes=('', '_store')
            )
            
            # Use the correct city column
            city_col = 'city' if 'city' in risk_by_location.columns else 'city_store'
            
            risk_summary = risk_by_location.groupby(city_col).agg({
                'stockout_risk': 'mean'
            }).reset_index()
            risk_summary['stockout_risk'] = risk_summary['stockout_risk'] * 100
            risk_summary.columns = ['city', 'risk_pct']
            
            fig = px.bar(
                risk_summary, x='city', y='risk_pct',
                color='risk_pct',
                color_continuous_scale='RdYlGn_r',
                title='Stockout Risk % by City',
                labels={'risk_pct': 'Risk %', 'city': 'City'}
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No simulation data available")
    
    with col2:
        st.markdown("#### 📦 Top 10 Stockout Risk Items")
        if 'top_risk_items' in sim_result and sim_result['top_risk_items'] is not None:
            st.dataframe(
                sim_result['top_risk_items'],
                use_container_width=True,
                height=350
            )
        else:
            st.info("No risk data available")
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Inventory Distribution")
        inventory_df = data['inventory'].copy()
        
        # Get latest snapshot
        inventory_df['snapshot_date'] = pd.to_datetime(inventory_df['snapshot_date'])
        latest = inventory_df[inventory_df['snapshot_date'] == inventory_df['snapshot_date'].max()]
        
        fig = px.histogram(
            latest, x='stock_on_hand',
            nbins=50,
            title='Stock on Hand Distribution',
            labels={'stock_on_hand': 'Stock Quantity', 'count': 'Count'}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🔍 Data Quality Issues (Pareto)")
        if 'issues' in data and len(data['issues']) > 0:
            issues_df = data['issues']
            issue_counts = issues_df['issue_type'].value_counts().reset_index()
            issue_counts.columns = ['issue_type', 'count']
            issue_counts = issue_counts.head(10)
            
            fig = px.bar(
                issue_counts, x='issue_type', y='count',
                color='count',
                color_continuous_scale='Reds',
                title='Top Data Quality Issues',
                labels={'count': 'Count', 'issue_type': 'Issue Type'}
            )
            fig.update_layout(height=350, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No issues logged")
    
    # Drill-down section
    st.markdown("---")
    st.markdown("### 🔍 Drill-Down Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        drill_dimension = st.selectbox(
            "Select Dimension",
            ['city', 'channel', 'category']
        )
    
    with col2:
        breakdown = kpi_calc.compute_breakdown(filtered_df, drill_dimension)
        if len(breakdown) > 0:
            st.dataframe(
                breakdown.style.format({
                    'revenue': 'AED {:,.0f}',
                    'cogs': 'AED {:,.0f}',
                    'margin': 'AED {:,.0f}',
                    'margin_pct': '{:.1f}%',
                    'avg_discount': '{:.1f}%'
                }),
                use_container_width=True
            )
        else:
            st.info("No data for breakdown")
    
    # Download Section
    st.markdown("---")
    st.markdown("### 📥 Download Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'sales' in data:
            csv = data['sales'].to_csv(index=False)
            st.download_button(
                "📄 Download Cleaned Sales",
                csv,
                "cleaned_sales.csv",
                "text/csv"
            )
    
    with col2:
        if 'issues' in data and len(data['issues']) > 0:
            csv = data['issues'].to_csv(index=False)
            st.download_button(
                "📄 Download Issues Log",
                csv,
                "issues.csv",
                "text/csv"
            )
    
    with col3:
        if sim_result['results']:
            sim_df = pd.DataFrame([sim_result['results']])
            csv = sim_df.to_csv(index=False)
            st.download_button(
                "📄 Download Simulation Results",
                csv,
                "simulation_results.csv",
                "text/csv"
            )


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point"""
    
    # Header
    st.markdown('<p class="main-header">🛒 UAE Promo Pulse Simulator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Data Rescue Dashboard | What-If Promotion Analysis</p>', unsafe_allow_html=True)
    
    # Render sidebar and get controls
    view_mode, filters, sim_params = render_sidebar()
    
    # Check if data is loaded
    if not st.session_state.data_loaded or st.session_state.data is None:
        st.markdown("---")
        
        # Show different messages based on state
        if st.session_state.raw_data_generated:
            st.success("✅ Raw data generated! Now click '🧹 Clean Data' in the sidebar to clean and process the data.")
            
            # Preview of raw data
            st.markdown("### 📋 Raw Data Preview")
            
            if st.session_state.raw_data:
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Products", "Stores", "Sales", "Inventory", "Campaigns"])
                
                with tab1:
                    st.markdown(f"**{len(st.session_state.raw_data['products']):,} records**")
                    st.dataframe(st.session_state.raw_data['products'].head(10), use_container_width=True)
                
                with tab2:
                    st.markdown(f"**{len(st.session_state.raw_data['stores']):,} records**")
                    st.dataframe(st.session_state.raw_data['stores'].head(10), use_container_width=True)
                
                with tab3:
                    st.markdown(f"**{len(st.session_state.raw_data['sales']):,} records**")
                    st.dataframe(st.session_state.raw_data['sales'].head(10), use_container_width=True)
                
                with tab4:
                    st.markdown(f"**{len(st.session_state.raw_data['inventory']):,} records**")
                    st.dataframe(st.session_state.raw_data['inventory'].head(10), use_container_width=True)
                
                with tab5:
                    st.markdown(f"**{len(st.session_state.raw_data['campaigns']):,} records**")
                    st.dataframe(st.session_state.raw_data['campaigns'], use_container_width=True)
        else:
            st.info("👈 Please generate data using the sidebar controls to begin.")
            
            # Show instructions
            with st.expander("📖 Getting Started", expanded=True):
                st.markdown("""
                ### Welcome to UAE Promo Pulse Simulator!
                
                This dashboard helps UAE retailers:
                1. **Rescue messy data** - Clean and validate real-world data exports
                2. **Compute trustworthy KPIs** - Track revenue, margins, and operational metrics
                3. **Run what-if simulations** - Test discount scenarios with budget and margin constraints
                
                #### To begin:
                
                **Step 1:** Click **"🎲 Generate Raw Data"** to create synthetic retail data with dirty data issues
                
                **Step 2:** Click **"🧹 Clean Data"** to validate and clean the generated data
                
                **Step 3:** Use filters to explore different segments
                
                **Step 4:** Adjust simulation parameters to test promotional scenarios
                
                **Step 5:** Toggle between **Executive** and **Manager** views for different perspectives
                
                ---
                
                #### Dashboard Views:
                - **Executive View**: Financial KPIs, revenue trends, margin analysis, recommendations
                - **Manager View**: Operational risks, stockout alerts, data quality, drill-down analysis
                
                #### Data Issues Simulated:
                - Inconsistent city names (Dubai/DUBAI/dubai etc.)
                - Missing values (unit_cost, discount_pct)
                - Duplicate order IDs
                - Corrupted timestamps
                - Outliers (extreme quantities and prices)
                - Invalid inventory (negative stock, extreme values)
                """)
        
        return
    
    # Main content based on view mode
    st.markdown("---")
    
    if view_mode == "Executive View":
        render_executive_view(st.session_state.data, filters, sim_params)
    else:
        render_manager_view(st.session_state.data, filters, sim_params)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>UAE Promo Pulse Simulator | "
        "Built with Streamlit, Pandas, and Plotly</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

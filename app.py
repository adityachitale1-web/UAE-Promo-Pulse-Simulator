"""
UAE Promo Pulse - Main Dashboard
================================
Streamlit dashboard with Executive and Manager views.
Includes all 12 required chart types.

Chart Types Included:
1. Scatter plot - Demand vs Stock, Outlier Detection
2. Historical prediction chart - Revenue forecast
3. Dual axis chart - Issues Pareto, Growth with Target
4. Outlier detection plot - Price/Qty anomalies
5. Waterfall chart - Revenue breakdown
6. Comparison matrix - Channel vs Category heatmap
7. Margin profitability - Enhanced margin analysis
8. Donut chart - Revenue by City
9. Growth trends - Week-over-week growth
10. Cumulative performance tracker - Running totals
11. Performance matrix - KPI heatmap
12. What-if heatmap - Discount vs Category simulation
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
    .chart-container {
        background: white; border-radius: 10px; padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 1rem;
    }
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


def section_header(title, icon="📊"):
    """Render section header"""
    st.markdown(f'<div class="section-header">{icon} {title}</div>', unsafe_allow_html=True)


def view_badge(view):
    """Render view badge"""
    css = "view-executive" if view == "Executive" else "view-manager"
    emoji = "👔" if view == "Executive" else "🔧"
    st.markdown(f'<span class="view-badge {css}">{emoji} {view} View</span>', unsafe_allow_html=True)


def safe_get_filter(filters, key, default=None):
    """Safely get filter value"""
    val = filters.get(key, default)
    if val == 'All' or val is None:
        return None
    return val


# =============================================================================
# CHART FUNCTIONS - All 12 Required Chart Types
# =============================================================================

def create_waterfall_chart(kpis):
    """5. Waterfall Chart - Revenue breakdown"""
    fig = go.Figure(go.Waterfall(
        name="Revenue Flow",
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["Gross Revenue", "Refunds", "Returns", "COGS", "Net Profit"],
        textposition="outside",
        text=[f"AED {kpis['gross_revenue']:,.0f}", 
              f"-AED {kpis['refund_amount']:,.0f}",
              f"-AED {kpis.get('returns_amount', 0):,.0f}",
              f"-AED {kpis['cogs']:,.0f}",
              f"AED {kpis['gross_margin']:,.0f}"],
        y=[kpis['gross_revenue'], 
           -kpis['refund_amount'], 
           -kpis.get('returns_amount', 0),
           -kpis['cogs'], 
           kpis['gross_margin']],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#00D4AA"}},
        decreasing={"marker": {"color": "#f5576c"}},
        totals={"marker": {"color": "#1E3A5F"}}
    ))
    fig.update_layout(
        title="Revenue Waterfall Analysis",
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def create_historical_prediction_chart(daily_df):
    """2. Historical Prediction Chart - Revenue with forecast"""
    if len(daily_df) < 7:
        return None
    
    df = daily_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate simple moving average for "prediction"
    df['ma7'] = df['revenue'].rolling(7, min_periods=1).mean()
    df['ma14'] = df['revenue'].rolling(14, min_periods=1).mean()
    
    # Generate forecast for next 7 days using trend
    last_ma = df['ma7'].iloc[-1]
    trend = (df['ma7'].iloc[-1] - df['ma7'].iloc[-7]) / 7 if len(df) >= 7 else 0
    
    future_dates = pd.date_range(start=df['date'].max() + timedelta(days=1), periods=7)
    forecast = [last_ma + trend * (i+1) for i in range(7)]
    forecast_upper = [f * 1.15 for f in forecast]
    forecast_lower = [f * 0.85 for f in forecast]
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['revenue'],
        mode='lines+markers',
        name='Actual Revenue',
        line=dict(color='#1E3A5F', width=2)
    ))
    
    # Moving average
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['ma7'],
        mode='lines',
        name='7-Day MA',
        line=dict(color='#00D4AA', width=2, dash='dot')
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=future_dates, y=forecast,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#667eea', width=2, dash='dash')
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates)[::-1],
        y=forecast_upper + forecast_lower[::-1],
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval'
    ))
    
    fig.update_layout(
        title="Historical Revenue with Forecast",
        height=400,
        legend=dict(orientation="h", y=-0.15),
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Date",
        yaxis_title="Revenue (AED)"
    )
    return fig


def create_outlier_detection_plot(sales_df):
    """4. Outlier Detection Plot - Price and Qty anomalies"""
    if len(sales_df) == 0:
        return None
    
    df = sales_df.copy()
    
    # Calculate z-scores for outlier detection
    df['price_zscore'] = (df['selling_price_aed'] - df['selling_price_aed'].mean()) / df['selling_price_aed'].std()
    df['qty_zscore'] = (df['qty'] - df['qty'].mean()) / df['qty'].std()
    
    # Flag outliers (|z| > 2)
    df['is_outlier'] = ((abs(df['price_zscore']) > 2) | (abs(df['qty_zscore']) > 2)).astype(int)
    
    # Sample for performance
    sample = df.sample(min(1000, len(df)))
    
    fig = px.scatter(
        sample,
        x='qty',
        y='selling_price_aed',
        color='is_outlier',
        color_discrete_map={0: '#00D4AA', 1: '#f5576c'},
        opacity=0.6,
        title="Outlier Detection: Price vs Quantity",
        labels={'is_outlier': 'Outlier', 'qty': 'Quantity', 'selling_price_aed': 'Price (AED)'}
    )
    
    # Add threshold lines
    price_upper = df['selling_price_aed'].mean() + 2 * df['selling_price_aed'].std()
    qty_upper = df['qty'].mean() + 2 * df['qty'].std()
    
    fig.add_hline(y=price_upper, line_dash="dash", line_color="red", 
                  annotation_text=f"Price threshold: {price_upper:.0f}")
    fig.add_vline(x=qty_upper, line_dash="dash", line_color="red",
                  annotation_text=f"Qty threshold: {qty_upper:.0f}")
    
    fig.update_layout(
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        legend_title="Is Outlier"
    )
    return fig


def create_comparison_matrix(sales_df, products_df, stores_df):
    """6. Comparison Matrix - Channel vs Category Revenue Heatmap"""
    if len(sales_df) == 0:
        return None
    
    df = sales_df.copy()
    
    # Merge if needed
    if 'category' not in df.columns:
        df = df.merge(products_df[['product_id', 'category']], on='product_id', how='left')
    if 'channel' not in df.columns:
        df = df.merge(stores_df[['store_id', 'channel']], on='store_id', how='left')
    
    # Create pivot table
    pivot = df.pivot_table(
        values='line_total',
        index='category',
        columns='channel',
        aggfunc='sum',
        fill_value=0
    )
    
    # Convert to thousands
    pivot = pivot / 1000
    
    fig = px.imshow(
        pivot,
        text_auto='.0f',
        color_continuous_scale='Blues',
        title="Revenue Matrix: Category vs Channel (in '000 AED)",
        labels=dict(x="Channel", y="Category", color="Revenue (K)")
    )
    
    fig.update_layout(height=400)
    return fig


def create_margin_profitability_chart(breakdown_df):
    """7. Enhanced Margin Profitability Chart"""
    if len(breakdown_df) == 0:
        return None
    
    df = breakdown_df.copy()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Revenue vs Margin', 'Profitability Quadrant'),
        specs=[[{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Revenue and Margin bars
    fig.add_trace(
        go.Bar(name='Revenue', x=df['category'], y=df['revenue']/1000, marker_color='#1E3A5F'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name='Margin', x=df['category'], y=df['margin']/1000, marker_color='#00D4AA'),
        row=1, col=1
    )
    
    # Profitability scatter (Revenue vs Margin %)
    fig.add_trace(
        go.Scatter(
            x=df['revenue']/1000,
            y=df['margin_pct'],
            mode='markers+text',
            text=df['category'],
            textposition='top center',
            marker=dict(size=df['orders']/df['orders'].max()*50 + 10, color='#667eea', opacity=0.7),
            name='Categories'
        ),
        row=1, col=2
    )
    
    # Add quadrant lines
    avg_revenue = df['revenue'].mean() / 1000
    avg_margin = df['margin_pct'].mean()
    fig.add_hline(y=avg_margin, line_dash="dash", line_color="gray", row=1, col=2)
    fig.add_vline(x=avg_revenue, line_dash="dash", line_color="gray", row=1, col=2)
    
    fig.update_layout(
        height=400,
        title="Margin Profitability Analysis",
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_xaxes(title_text="Revenue (K AED)", row=1, col=2)
    fig.update_yaxes(title_text="Margin %", row=1, col=2)
    
    return fig


def create_growth_trends_chart(daily_df):
    """9. Growth Trends - Week over Week"""
    if len(daily_df) < 14:
        return None
    
    df = daily_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['week'] = df['date'].dt.isocalendar().week
    df['year'] = df['date'].dt.year
    
    weekly = df.groupby(['year', 'week']).agg({
        'revenue': 'sum',
        'orders': 'sum',
        'units': 'sum'
    }).reset_index()
    
    weekly['week_label'] = weekly['year'].astype(str) + '-W' + weekly['week'].astype(str)
    weekly['revenue_growth'] = weekly['revenue'].pct_change() * 100
    weekly['orders_growth'] = weekly['orders'].pct_change() * 100
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Revenue bars
    fig.add_trace(
        go.Bar(
            x=weekly['week_label'],
            y=weekly['revenue']/1000,
            name='Revenue (K)',
            marker_color='#1E3A5F'
        ),
        secondary_y=False
    )
    
    # Growth line
    fig.add_trace(
        go.Scatter(
            x=weekly['week_label'],
            y=weekly['revenue_growth'],
            name='WoW Growth %',
            line=dict(color='#00D4AA', width=3),
            mode='lines+markers'
        ),
        secondary_y=True
    )
    
    # Zero line for growth
    fig.add_hline(y=0, line_dash="dash", line_color="gray", secondary_y=True)
    
    fig.update_layout(
        title="Weekly Growth Trends",
        height=400,
        legend=dict(orientation="h", y=-0.15),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_yaxes(title_text="Revenue (K AED)", secondary_y=False)
    fig.update_yaxes(title_text="Growth %", secondary_y=True)
    
    return fig


def create_cumulative_performance(daily_df):
    """10. Cumulative Performance Tracker"""
    if len(daily_df) == 0:
        return None
    
    df = daily_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Calculate cumulative metrics
    df['cum_revenue'] = df['revenue'].cumsum()
    df['cum_orders'] = df['orders'].cumsum()
    df['cum_units'] = df['units'].cumsum()
    
    # Create target line (assume 10% above actual as target)
    total_days = len(df)
    target_daily = df['revenue'].sum() * 1.1 / total_days
    df['target'] = target_daily * (df.index - df.index[0] + 1)
    
    fig = go.Figure()
    
    # Actual cumulative
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['cum_revenue'],
        mode='lines',
        name='Actual Cumulative',
        line=dict(color='#1E3A5F', width=3),
        fill='tozeroy',
        fillcolor='rgba(30,58,95,0.1)'
    ))
    
    # Target line
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['target'],
        mode='lines',
        name='Target',
        line=dict(color='#f5576c', width=2, dash='dash')
    ))
    
    # Add achievement markers
    df['achievement'] = (df['cum_revenue'] / df['target'] * 100).round(1)
    
    fig.update_layout(
        title="Cumulative Revenue Performance vs Target",
        height=400,
        legend=dict(orientation="h", y=-0.15),
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Date",
        yaxis_title="Cumulative Revenue (AED)"
    )
    
    return fig


def create_performance_matrix(kpis_by_dimension):
    """11. Performance Matrix - KPI Heatmap by City/Channel"""
    if len(kpis_by_dimension) == 0:
        return None
    
    # Normalize KPIs for comparison
    df = kpis_by_dimension.copy()
    
    # Select numeric columns
    metrics = ['revenue', 'margin_pct', 'orders', 'avg_discount']
    for m in metrics:
        if m in df.columns:
            df[f'{m}_norm'] = (df[m] - df[m].min()) / (df[m].max() - df[m].min() + 0.001) * 100
    
    norm_cols = [c for c in df.columns if '_norm' in c]
    
    if len(norm_cols) == 0:
        return None
    
    # Create heatmap data
    dimension_col = df.columns[0]  # First column is the dimension
    heat_data = df.set_index(dimension_col)[norm_cols]
    heat_data.columns = [c.replace('_norm', '').title() for c in heat_data.columns]
    
    fig = px.imshow(
        heat_data,
        text_auto='.0f',
        color_continuous_scale='RdYlGn',
        title="Performance Matrix (Normalized Scores 0-100)",
        labels=dict(x="Metric", y=dimension_col.title(), color="Score")
    )
    
    fig.update_layout(height=350)
    return fig


def create_whatif_heatmap(simulator, sim_params, filters):
    """12. What-If Analysis Heatmap - Discount vs Category"""
    discounts = [5, 10, 15, 20, 25, 30]
    categories = ['Electronics', 'Fashion', 'Grocery', 'Home & Garden', 'Beauty', 'Sports']
    
    # Build matrix
    profit_matrix = []
    
    for cat in categories:
        row = []
        for disc in discounts:
            try:
                result = simulator.run_simulation(
                    disc,
                    sim_params['promo_budget'],
                    sim_params['margin_floor'],
                    sim_params['simulation_days'],
                    city=safe_get_filter(filters, 'city'),
                    channel=safe_get_filter(filters, 'channel'),
                    category=cat
                )
                profit = result['results'].get('profit_proxy', 0) if result.get('results') else 0
                row.append(profit / 1000)  # Convert to thousands
            except Exception:
                row.append(0)
        profit_matrix.append(row)
    
    fig = px.imshow(
        profit_matrix,
        x=[f'{d}%' for d in discounts],
        y=categories,
        color_continuous_scale='RdYlGn',
        text_auto='.0f',
        title="What-If Analysis: Profit by Category & Discount (K AED)",
        labels=dict(x="Discount Level", y="Category", color="Profit (K)")
    )
    
    fig.update_layout(height=400)
    return fig


def create_donut_chart(breakdown_df, dimension, value_col='revenue'):
    """8. Donut Chart"""
    if len(breakdown_df) == 0:
        return None
    
    fig = px.pie(
        breakdown_df,
        values=value_col,
        names=dimension,
        hole=0.5,
        title=f"Distribution by {dimension.title()}",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=350, showlegend=True)
    
    return fig


def create_dual_axis_growth_target(daily_df, target_multiplier=1.1):
    """3. Dual Axis Chart - Revenue with Growth Rate"""
    if len(daily_df) < 7:
        return None
    
    df = daily_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Calculate daily growth rate
    df['growth_rate'] = df['revenue'].pct_change() * 100
    df['target'] = df['revenue'].mean() * target_multiplier
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Revenue area
    fig.add_trace(
        go.Scatter(
            x=df['date'], y=df['revenue'],
            name='Revenue',
            fill='tozeroy',
            fillcolor='rgba(30,58,95,0.3)',
            line=dict(color='#1E3A5F', width=2)
        ),
        secondary_y=False
    )
    
    # Target line
    fig.add_trace(
        go.Scatter(
            x=df['date'], y=[df['target'].iloc[0]] * len(df),
            name='Target',
            line=dict(color='#f5576c', width=2, dash='dash')
        ),
        secondary_y=False
    )
    
    # Growth rate line
    fig.add_trace(
        go.Scatter(
            x=df['date'], y=df['growth_rate'],
            name='Daily Growth %',
            line=dict(color='#00D4AA', width=2),
            mode='lines+markers'
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Revenue Performance with Growth Rate (Dual Axis)",
        height=400,
        legend=dict(orientation="h", y=-0.15),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_yaxes(title_text="Revenue (AED)", secondary_y=False)
    fig.update_yaxes(title_text="Growth Rate %", secondary_y=True)
    
    return fig


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
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
                orig = {}
                for k, v in st.session_state.raw_data.items():
                    if isinstance(v, pd.DataFrame):
                        orig[k] = len(v)
                
                cleaner = DataCleaner()
                cleaned = cleaner.clean_all(
                    st.session_state.raw_data['products'],
                    st.session_state.raw_data['stores'],
                    st.session_state.raw_data['sales'],
                    st.session_state.raw_data['inventory'],
                    'data/cleaned'
                )
                st.session_state.data = cleaned
                
                cc = {}
                for k, v in cleaned.items():
                    if isinstance(v, pd.DataFrame) and k != 'issues':
                        cc[k] = len(v)
                
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

# =============================================================================
# FILTERS
# =============================================================================
st.sidebar.markdown("---")
filters = {}

if st.session_state.data_loaded and st.session_state.data:
    st.sidebar.markdown("### 🔍 Filters")
    data = st.session_state.data
    
    # Date Range
    if 'sales' in data and data['sales'] is not None:
        try:
            sales_dates = pd.to_datetime(data['sales']['order_time'], errors='coerce')
            valid_dates = sales_dates.dropna()
            if len(valid_dates) > 0:
                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()
                date_range = st.sidebar.date_input(
                    "📅 Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    filters['date_from'] = date_range[0]
                    filters['date_to'] = date_range[1]
        except Exception:
            pass
    
    # City
    if 'stores' in data and 'city' in data['stores'].columns:
        cities = ['All'] + sorted(data['stores']['city'].dropna().unique().tolist())
        filters['city'] = st.sidebar.selectbox("🏙️ City", cities)
    
    # Channel
    if 'stores' in data and 'channel' in data['stores'].columns:
        channels = ['All'] + sorted(data['stores']['channel'].dropna().unique().tolist())
        filters['channel'] = st.sidebar.selectbox("📱 Channel", channels)
    
    # Category
    if 'products' in data and 'category' in data['products'].columns:
        categories = ['All'] + sorted(data['products']['category'].dropna().unique().tolist())
        filters['category'] = st.sidebar.selectbox("📦 Category", categories)
    
    # Brand
    if 'products' in data and 'brand' in data['products'].columns:
        brands = ['All'] + sorted(data['products']['brand'].dropna().unique().tolist())
        filters['brand'] = st.sidebar.selectbox("🏷️ Brand", brands)

# Simulation Parameters
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎮 Simulation Parameters")

sim_params = {
    'discount_pct': st.sidebar.slider("💸 Discount %", 0, 50, 15),
    'promo_budget': st.sidebar.number_input("💰 Promo Budget (AED)", 10000, 500000, 50000, step=5000),
    'margin_floor': st.sidebar.slider("📉 Margin Floor %", 5, 30, 15),
    'simulation_days': st.sidebar.selectbox("📆 Simulation Days", [7, 14], index=1)
}


# =============================================================================
# MAIN CONTENT
# =============================================================================
render_logo()
st.markdown('<p class="main-header">Advanced Retail Analytics & Promotion Simulator for UAE Market</p>', unsafe_allow_html=True)

view_badge(view_type)

if not st.session_state.data_loaded:
    # Welcome Screen
    st.markdown("---")
    
    if st.session_state.raw_data_generated:
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
    else:
        st.info("👈 Click 'Generate' in the sidebar to create sample data, then 'Clean' to process it")
        
        # Feature cards
        st.markdown("### 🚀 Dashboard Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1E3A5F 0%, #2E5A8F 100%); 
                        padding: 1.5rem; border-radius: 10px; color: white;">
                <h4 style="color: #00D4AA;">📊 12 Chart Types</h4>
                <ul style="font-size: 0.85rem;">
                    <li>Waterfall & Heatmaps</li>
                    <li>Scatter & Outlier Detection</li>
                    <li>Growth Trends</li>
                    <li>Performance Matrix</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1E3A5F 0%, #2E5A8F 100%); 
                        padding: 1.5rem; border-radius: 10px; color: white;">
                <h4 style="color: #00D4AA;">🎯 What-If Simulation</h4>
                <ul style="font-size: 0.85rem;">
                    <li>Discount Scenario Analysis</li>
                    <li>Category Heatmaps</li>
                    <li>Constraint Checking</li>
                    <li>Profit Forecasting</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1E3A5F 0%, #2E5A8F 100%); 
                        padding: 1.5rem; border-radius: 10px; color: white;">
                <h4 style="color: #00D4AA;">🧹 Data Quality</h4>
                <ul style="font-size: 0.85rem;">
                    <li>7+ Validation Rules</li>
                    <li>Issue Logging</li>
                    <li>Auto-Cleaning</li>
                    <li>Quality Reports</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

else:
    # ==========================================================================
    # DASHBOARD WITH DATA
    # ==========================================================================
    data = st.session_state.data
    
    # Initialize calculators
    try:
        kpi_calc = KPICalculator(
            data['sales'], data['products'], data['stores'], data['inventory']
        )
        simulator = PromoSimulator(
            data['sales'], data['products'], data['stores'], data['inventory']
        )
    except Exception as e:
        st.error(f"Error initializing: {e}")
        st.stop()
    
    # Apply filters
    filtered_df = kpi_calc.filter_data(
        city=safe_get_filter(filters, 'city'),
        channel=safe_get_filter(filters, 'channel'),
        category=safe_get_filter(filters, 'category'),
        brand=safe_get_filter(filters, 'brand'),
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
        city=safe_get_filter(filters, 'city'),
        channel=safe_get_filter(filters, 'channel'),
        category=safe_get_filter(filters, 'category')
    )
    
    # ==========================================================================
    # DATA QUALITY REPORT
    # ==========================================================================
    if st.session_state.cleaning_stats:
        with st.expander("📊 Data Quality Report", expanded=False):
            stats = st.session_state.cleaning_stats
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Original Records", f"{sum(stats['original'].values()):,}")
            col2.metric("Records Removed", f"{sum(stats['removed'].values()):,}")
            col3.metric("Cleaned Records", f"{sum(stats['cleaned'].values()):,}")
            col4.metric("Issues Logged", f"{stats['total_issues']:,}")
    
    # ==========================================================================
    # EXECUTIVE VIEW
    # ==========================================================================
    if view_type == "Executive":
        
        # KPI Cards
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
            profit = sim_results['results'].get('profit_proxy', 0) if sim_results.get('results') else 0
            kpi_card("Profit Proxy", f"AED {profit:,.0f}",
                    "Profit" if profit > 0 else "Loss",
                    "positive" if profit > 0 else "negative")
        
        st.markdown("")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            kpi_card("Gross Revenue", f"AED {kpis['gross_revenue']:,.0f}")
        with col2:
            kpi_card("Refunds", f"AED {kpis['refund_amount']:,.0f}")
        with col3:
            kpi_card("Total Orders", f"{kpis['total_orders']:,}")
        with col4:
            kpi_card("AOV", f"AED {kpis['aov']:,.0f}")
        with col5:
            budget_util = sim_results['results'].get('budget_utilization', 0) if sim_results.get('results') else 0
            kpi_card("Budget Util", f"{budget_util:.1f}%")
        
        st.markdown("---")
        
        # ==========================================================================
        # CHART ROW 1: Waterfall & Historical Prediction
        # ==========================================================================
        section_header("Revenue Analysis", "📈")
        col1, col2 = st.columns(2)
        
        with col1:
            # 5. WATERFALL CHART
            waterfall_fig = create_waterfall_chart(kpis)
            st.plotly_chart(waterfall_fig, use_container_width=True)
        
        with col2:
            # 2. HISTORICAL PREDICTION CHART
            prediction_fig = create_historical_prediction_chart(daily)
            if prediction_fig:
                st.plotly_chart(prediction_fig, use_container_width=True)
            else:
                st.info("Insufficient data for forecast (need 7+ days)")
        
        # ==========================================================================
        # CHART ROW 2: Growth Trends & Cumulative Performance
        # ==========================================================================
        section_header("Growth & Performance", "📊")
        col1, col2 = st.columns(2)
        
        with col1:
            # 9. GROWTH TRENDS
            growth_fig = create_growth_trends_chart(daily)
            if growth_fig:
                st.plotly_chart(growth_fig, use_container_width=True)
            else:
                st.info("Insufficient data for growth analysis")
        
        with col2:
            # 10. CUMULATIVE PERFORMANCE TRACKER
            cumulative_fig = create_cumulative_performance(daily)
            if cumulative_fig:
                st.plotly_chart(cumulative_fig, use_container_width=True)
            else:
                st.info("No data for cumulative tracking")
        
        # ==========================================================================
        # CHART ROW 3: Donut & Comparison Matrix
        # ==========================================================================
        section_header("Distribution & Comparison", "🔄")
        col1, col2 = st.columns(2)
        
        with col1:
            # 8. DONUT CHART
            city_breakdown = kpi_calc.compute_breakdown(filtered_df, 'city')
            donut_fig = create_donut_chart(city_breakdown, 'city')
            if donut_fig:
                st.plotly_chart(donut_fig, use_container_width=True)
            else:
                st.info("No data for distribution")
        
        with col2:
            # 6. COMPARISON MATRIX
            matrix_fig = create_comparison_matrix(filtered_df, data['products'], data['stores'])
            if matrix_fig:
                st.plotly_chart(matrix_fig, use_container_width=True)
            else:
                st.info("No data for comparison matrix")
        
        # ==========================================================================
        # CHART ROW 4: Margin Profitability & Performance Matrix
        # ==========================================================================
        section_header("Profitability Analysis", "💎")
        col1, col2 = st.columns(2)
        
        with col1:
            # 7. MARGIN PROFITABILITY
            cat_breakdown = kpi_calc.compute_breakdown(filtered_df, 'category')
            margin_fig = create_margin_profitability_chart(cat_breakdown)
            if margin_fig:
                st.plotly_chart(margin_fig, use_container_width=True)
            else:
                st.info("No data for margin analysis")
        
        with col2:
            # 11. PERFORMANCE MATRIX
            channel_breakdown = kpi_calc.compute_breakdown(filtered_df, 'channel')
            perf_matrix_fig = create_performance_matrix(channel_breakdown)
            if perf_matrix_fig:
                st.plotly_chart(perf_matrix_fig, use_container_width=True)
            else:
                st.info("No data for performance matrix")
        
        # ==========================================================================
        # CHART ROW 5: What-If Heatmap
        # ==========================================================================
        section_header("What-If Analysis", "🎯")
        
        # 12. WHAT-IF HEATMAP
        whatif_fig = create_whatif_heatmap(simulator, sim_params, filters)
        st.plotly_chart(whatif_fig, use_container_width=True)
        
        st.markdown("""
        <div style="background: #e8f4f8; padding: 1rem; border-radius: 8px; margin-top: 0.5rem;">
            <strong>📊 Interpretation:</strong> Green cells indicate profitable scenarios. 
            Use this heatmap to identify optimal discount levels per category.
        </div>
        """, unsafe_allow_html=True)
        
        # ==========================================================================
        # RECOMMENDATIONS
        # ==========================================================================
        section_header("Auto-Generated Recommendations", "💡")
        recommendations = generate_recommendation(kpis, sim_results.get('results'), sim_results.get('violations', []))
        
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
    else:
        # Operational KPIs
        section_header("Operational KPIs", "🔧")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        stockout_risk = sim_results['results'].get('stockout_risk_pct', 0) if sim_results.get('results') else 0
        high_risk_skus = sim_results['results'].get('high_risk_skus', 0) if sim_results.get('results') else 0
        
        with col1:
            kpi_card("Stockout Risk", f"{stockout_risk:.1f}%",
                    "High" if stockout_risk > 30 else "OK",
                    "negative" if stockout_risk > 30 else "positive")
        with col2:
            kpi_card("High Risk SKUs", f"{high_risk_skus:,}")
        with col3:
            kpi_card("Return Rate", f"{kpis['return_rate']:.1f}%",
                    "High" if kpis['return_rate'] > 10 else "Normal",
                    "negative" if kpis['return_rate'] > 10 else "positive")
        with col4:
            kpi_card("Payment Failure", f"{kpis['payment_failure_rate']:.1f}%")
        with col5:
            kpi_card("Total Units", f"{kpis['total_units']:,}")
        
        st.markdown("---")
        
        # ==========================================================================
        # CHART ROW 1: Outlier Detection & Dual Axis
        # ==========================================================================
        section_header("Data Quality & Trends", "🔍")
        col1, col2 = st.columns(2)
        
        with col1:
            # 4. OUTLIER DETECTION PLOT
            outlier_fig = create_outlier_detection_plot(filtered_df)
            if outlier_fig:
                st.plotly_chart(outlier_fig, use_container_width=True)
            else:
                st.info("No data for outlier detection")
        
        with col2:
            # 3. DUAL AXIS CHART
            dual_fig = create_dual_axis_growth_target(daily)
            if dual_fig:
                st.plotly_chart(dual_fig, use_container_width=True)
            else:
                st.info("Insufficient data for dual axis chart")
        
        # ==========================================================================
        # CHART ROW 2: Scatter & Risk Table
        # ==========================================================================
        section_header("Inventory & Risk Analysis", "📦")
        col1, col2 = st.columns(2)
        
        with col1:
            # 1. SCATTER PLOT - Demand vs Stock
            sim_detail = sim_results.get('simulation_detail')
            if sim_detail is not None and len(sim_detail) > 0:
                sample = sim_detail.sample(min(500, len(sim_detail)))
                
                fig = px.scatter(
                    sample,
                    x='stock_on_hand',
                    y='sim_demand',
                    color='stockout_risk',
                    color_discrete_map={0: '#00D4AA', 1: '#f5576c'},
                    opacity=0.6,
                    title="Scatter: Demand vs Stock"
                )
                max_val = max(sample['stock_on_hand'].max(), sample['sim_demand'].max())
                fig.add_trace(go.Scatter(
                    x=[0, max_val], y=[0, max_val],
                    mode='lines', name='Demand = Stock',
                    line=dict(color='gray', dash='dash')
                ))
                fig.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No simulation data for scatter plot")
        
        with col2:
            # Top Risk Items Table
            st.markdown("#### Top 10 Stockout Risk Items")
            top_risk = sim_results.get('top_risk_items')
            if top_risk is not None and len(top_risk) > 0:
                st.dataframe(top_risk, use_container_width=True, height=350)
            else:
                st.info("No risk items to display")
        
        # ==========================================================================
        # CHART ROW 3: Issues Pareto (Dual Axis) & Inventory Distribution
        # ==========================================================================
        section_header("Issues & Inventory", "⚠️")
        col1, col2 = st.columns(2)
        
        with col1:
            # 3. DUAL AXIS - Issues Pareto
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
                    title="Issues Pareto (Dual Axis)",
                    height=400,
                    legend=dict(orientation="h", y=-0.2),
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No issues logged")
        
        with col2:
            # Inventory Distribution
            inv = data['inventory'].copy()
            inv['snapshot_date'] = pd.to_datetime(inv['snapshot_date'], errors='coerce')
            latest_date = inv['snapshot_date'].max()
            latest_inv = inv[inv['snapshot_date'] == latest_date] if not pd.isna(latest_date) else inv
            
            if len(latest_inv) > 0:
                fig = px.histogram(
                    latest_inv, x='stock_on_hand',
                    nbins=30,
                    title="Inventory Distribution",
                    color_discrete_sequence=['#1E3A5F']
                )
                fig.add_vline(x=latest_inv['stock_on_hand'].median(), line_dash="dash", 
                             line_color="#00D4AA", annotation_text="Median")
                fig.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        
        # ==========================================================================
        # CHART ROW 4: What-If Seasonality & Constraint Violations
        # ==========================================================================
        section_header("Simulation & Constraints", "🎮")
        col1, col2 = st.columns(2)
        
        with col1:
            # 12. WHAT-IF HEATMAP (Manager version - by channel)
            channels = ['App', 'Web', 'Marketplace']
            discounts = [5, 10, 15, 20, 25, 30]
            
            profit_matrix = []
            for ch in channels:
                row = []
                for disc in discounts:
                    try:
                        result = simulator.run_simulation(
                            disc, sim_params['promo_budget'], sim_params['margin_floor'],
                            sim_params['simulation_days'],
                            safe_get_filter(filters, 'city'),
                            ch, safe_get_filter(filters, 'category')
                        )
                        profit = result['results'].get('profit_proxy', 0) if result.get('results') else 0
                        row.append(profit / 1000)
                    except:
                        row.append(0)
                profit_matrix.append(row)
            
            fig = px.imshow(
                profit_matrix,
                x=[f'{d}%' for d in discounts],
                y=channels,
                color_continuous_scale='RdYlGn',
                text_auto='.0f',
                title="What-If: Profit by Channel & Discount (K AED)"
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Constraint Violations
            st.markdown("#### Constraint Violations")
            violations = sim_results.get('violations', [])
            
            if violations:
                for v in violations:
                    severity_class = "constraint-card-error" if v.get('severity') == 'HIGH' else "constraint-card"
                    icon = "🚫" if v.get('severity') == 'HIGH' else "⚠️"
                    st.markdown(f"""
                    <div class="{severity_class}">
                        <strong>{icon} {v.get('constraint', 'CONSTRAINT')}</strong><br>
                        {v.get('message', '')}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ All constraints satisfied!")
    
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
    
    with col3:
        if sim_results.get('results'):
            st.download_button(
                "📄 Simulation",
                pd.DataFrame([sim_results['results']]).to_csv(index=False),
                "simulation_results.csv",
                "text/csv",
                use_container_width=True
            )
    
    with col4:
        top_risk = sim_results.get('top_risk_items')
        if top_risk is not None and len(top_risk) > 0:
            st.download_button(
                "📄 Risk Items",
                top_risk.to_csv(index=False),
                "risk_items.csv",
                "text/csv",
                use_container_width=True
            )


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #888;">
    <strong>UAE PROMO PULSE</strong> v2.0 | Retail Analytics Dashboard | 12 Chart Types
</div>
""", unsafe_allow_html=True)

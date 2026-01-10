"""
app.py
UAE Promo Pulse Simulator - Streamlit Dashboard
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

# Page config
st.set_page_config(page_title="UAE Promo Pulse", page_icon="🛒", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1E3A5F; text-align: center;}
    .insight-box {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; margin: 0.5rem 0;}
    .insight-box-success {background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);}
    .insight-box-warning {background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);}
</style>
""", unsafe_allow_html=True)

# Session state
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


def display_insight(title, text, insight_type="info"):
    """Display insight box"""
    css = f"insight-box insight-box-{insight_type}" if insight_type != "info" else "insight-box"
    icon = "💡" if insight_type == "info" else "✅" if insight_type == "success" else "⚠️"
    st.markdown(f'<div class="{css}"><strong>{icon} {title}:</strong> {text}</div>', unsafe_allow_html=True)


def display_cleaning_stats(stats):
    """Display cleaning statistics"""
    if not stats:
        return
    
    st.markdown("### 📊 Cleaning Statistics")
    cols = st.columns(4)
    cols[0].metric("Original Records", f"{sum(stats['original'].values()):,}")
    cols[1].metric("Records Removed", f"{sum(stats['removed'].values()):,}")
    cols[2].metric("Clean Records", f"{sum(stats['cleaned'].values()):,}")
    cols[3].metric("Issues Found", f"{stats['total_issues']:,}")


# Sidebar
st.sidebar.markdown("## 🎛️ Controls")

view_mode = st.sidebar.radio("View", ["Executive View", "Manager View"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Data")

if st.sidebar.button("🎲 Generate Data", use_container_width=True):
    with st.spinner("Generating..."):
        st.session_state.raw_data = generate_all_data('data/raw')
        st.session_state.raw_data_generated = True
        st.session_state.data_loaded = False
    st.rerun()

if st.session_state.raw_data_generated:
    st.sidebar.success("✅ Raw data ready!")

if st.sidebar.button("🧹 Clean Data", use_container_width=True, disabled=not st.session_state.raw_data_generated):
    if st.session_state.raw_data:
        with st.spinner("Cleaning..."):
            original = {k: len(v) for k, v in st.session_state.raw_data.items()}
            cleaner = DataCleaner()
            cleaned = cleaner.clean_all(
                st.session_state.raw_data['products'],
                st.session_state.raw_data['stores'],
                st.session_state.raw_data['sales'],
                st.session_state.raw_data['inventory'],
                'data/cleaned'
            )
            st.session_state.data = cleaned
            cleaned_counts = {k: len(v) for k, v in cleaned.items() if k != 'issues'}
            st.session_state.cleaning_stats = {
                'original': original,
                'cleaned': cleaned_counts,
                'removed': {k: original.get(k, 0) - cleaned_counts.get(k, 0) for k in cleaned_counts},
                'total_issues': len(cleaned['issues']),
                'issues_summary': cleaned['issues']['issue_type'].value_counts().to_dict() if len(cleaned['issues']) > 0 else {}
            }
            st.session_state.data_loaded = True
        st.rerun()

if st.session_state.data_loaded:
    st.sidebar.success("✅ Data cleaned!")

st.sidebar.markdown("---")

# Filters
filters = {}
if st.session_state.data_loaded and st.session_state.data:
    st.sidebar.markdown("### Filters")
    data = st.session_state.data
    
    if 'stores' in data:
        filters['city'] = st.sidebar.selectbox("City", ['All'] + sorted(data['stores']['city'].unique().tolist()))
        filters['channel'] = st.sidebar.selectbox("Channel", ['All'] + sorted(data['stores']['channel'].unique().tolist()))
    
    if 'products' in data:
        filters['category'] = st.sidebar.selectbox("Category", ['All'] + sorted(data['products']['category'].unique().tolist()))

st.sidebar.markdown("---")
st.sidebar.markdown("### Simulation")

sim_params = {
    'discount_pct': st.sidebar.slider("Discount %", 0, 50, 15),
    'promo_budget': st.sidebar.number_input("Budget (AED)", 10000, 500000, 50000, step=5000),
    'margin_floor': st.sidebar.slider("Margin Floor %", 5, 30, 15),
    'simulation_days': st.sidebar.selectbox("Days", [7, 14], index=1)
}

# Main content
st.markdown('<p class="main-header">🛒 UAE Promo Pulse Simulator</p>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Data Rescue Dashboard | What-If Analysis</p>", unsafe_allow_html=True)

if not st.session_state.data_loaded:
    st.markdown("---")
    if st.session_state.raw_data_generated:
        st.success("✅ Raw data generated! Click 'Clean Data' to proceed.")
    else:
        st.info("👈 Click 'Generate Data' in the sidebar to begin.")
        with st.expander("📖 About", expanded=True):
            st.markdown("""
            **UAE Promo Pulse Simulator** helps retailers:
            - 🧹 Clean messy data with quality issues
            - 📊 Calculate trustworthy KPIs
            - 🔮 Run what-if promotion simulations
            - 💡 Get actionable recommendations
            """)
else:
    st.markdown("---")
    
    data = st.session_state.data
    kpi_calc = KPICalculator(data['sales'], data['products'], data['stores'], data['inventory'])
    simulator = PromoSimulator(data['sales'], data['products'], data['stores'], data['inventory'])
    
    filtered_df = kpi_calc.filter_data(
        city=filters.get('city'), channel=filters.get('channel'), category=filters.get('category')
    )
    
    kpis = kpi_calc.compute_historical_kpis(filtered_df)
    daily_kpis = kpi_calc.compute_daily_kpis(filtered_df)
    
    sim_result = simulator.run_simulation(
        discount_pct=sim_params['discount_pct'],
        promo_budget=sim_params['promo_budget'],
        margin_floor=sim_params['margin_floor'],
        simulation_days=sim_params['simulation_days'],
        city=filters.get('city'), channel=filters.get('channel'), category=filters.get('category')
    )
    
    # Cleaning stats
    if st.session_state.cleaning_stats:
        with st.expander("📊 Data Cleaning Statistics", expanded=(view_mode == "Manager View")):
            display_cleaning_stats(st.session_state.cleaning_stats)
            
            if st.session_state.cleaning_stats.get('issues_summary'):
                issues_df = pd.DataFrame([
                    {'Issue': k, 'Count': v} for k, v in st.session_state.cleaning_stats['issues_summary'].items()
                ]).sort_values('Count', ascending=False)
                
                fig = px.bar(issues_df, x='Issue', y='Count', color='Count', color_continuous_scale='Reds', title='Issues by Type')
                fig.update_layout(height=300, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
    
    # KPIs
    st.markdown("### 💰 Key Metrics")
    cols = st.columns(5)
    cols[0].metric("Net Revenue", f"AED {kpis['net_revenue']:,.0f}")
    cols[1].metric("Margin %", f"{kpis['gross_margin_pct']:.1f}%")
    cols[2].metric("Orders", f"{kpis['total_orders']:,}")
    cols[3].metric("Return Rate", f"{kpis['return_rate']:.1f}%")
    if sim_result['results']:
        cols[4].metric("Sim Profit", f"AED {sim_result['results']['profit_proxy']:,.0f}")
    
    st.markdown("---")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Revenue Trend")
        if len(daily_kpis) > 0:
            fig = px.line(daily_kpis, x='date', y='revenue', title='Daily Revenue')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            # Insight
            if len(daily_kpis) > 7:
                recent = daily_kpis.tail(7)['revenue'].mean()
                prev = daily_kpis.tail(14).head(7)['revenue'].mean()
                change = ((recent - prev) / prev * 100) if prev > 0 else 0
                
                if change > 10:
                    display_insight("Trend", f"Revenue up {change:.1f}% week-over-week. Consider increasing inventory.", "success")
                elif change < -10:
                    display_insight("Trend", f"Revenue down {abs(change):.1f}%. Consider promotional activities.", "warning")
                else:
                    display_insight("Trend", f"Revenue stable ({change:.1f}% change).", "info")
    
    with col2:
        st.markdown("#### 🏙️ Revenue by City")
        city_bd = kpi_calc.compute_breakdown(filtered_df, 'city')
        if len(city_bd) > 0:
            fig = go.Figure(data=[go.Pie(labels=city_bd['city'], values=city_bd['revenue'], hole=0.5)])
            fig.update_layout(title='Revenue Distribution', height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            top_city = city_bd.iloc[0]
            share = top_city['revenue'] / city_bd['revenue'].sum() * 100
            display_insight("Distribution", f"{top_city['city']} leads with {share:.1f}% of revenue.", "success" if share < 60 else "warning")
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Margin by Category")
        cat_bd = kpi_calc.compute_breakdown(filtered_df, 'category')
        if len(cat_bd) > 0:
            fig = px.bar(cat_bd, x='category', y='margin_pct', color='margin_pct', color_continuous_scale='RdYlGn', title='Gross Margin %')
            fig.add_hline(y=sim_params['margin_floor'], line_dash="dash", annotation_text=f"Floor: {sim_params['margin_floor']}%")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            low_margin = cat_bd[cat_bd['margin_pct'] < 15]['category'].tolist()
            if low_margin:
                display_insight("Margin Alert", f"{', '.join(low_margin)} have margins below 15%.", "warning")
            else:
                display_insight("Margin", "All categories have healthy margins.", "success")
    
    with col2:
        st.markdown("#### 🔮 What-If Heatmap")
        
        # Create simple heatmap
        discounts = [5, 10, 15, 20, 25, 30]
        budgets = [25000, 50000, 100000]
        
        matrix = []
        for budget in budgets:
            row = []
            for disc in discounts:
                res = simulator.run_simulation(disc, budget, sim_params['margin_floor'], sim_params['simulation_days'],
                                              filters.get('city'), filters.get('channel'), filters.get('category'))
                row.append(res['results']['profit_proxy'] if res['results'] else 0)
            matrix.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix, x=[f'{d}%' for d in discounts], y=[f'AED {b:,}' for b in budgets],
            colorscale='RdYlGn', text=[[f'AED {v:,.0f}' for v in row] for row in matrix], texttemplate='%{text}'
        ))
        fig.update_layout(title='Profit by Discount & Budget', height=350, xaxis_title='Discount', yaxis_title='Budget')
        st.plotly_chart(fig, use_container_width=True)
        
        max_profit = max(max(row) for row in matrix)
        display_insight("Optimal", f"Max profit potential: AED {max_profit:,.0f}", "success" if max_profit > 0 else "warning")
    
    # Charts Row 3
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📅 Seasonality")
        df_temp = filtered_df.copy()
        df_temp['order_time'] = pd.to_datetime(df_temp['order_time'], errors='coerce')
        df_temp = df_temp.dropna(subset=['order_time'])
        
        if len(df_temp) > 0:
            df_temp['day'] = df_temp['order_time'].dt.day_name()
            df_temp['hour'] = df_temp['order_time'].dt.hour
            
            heat_data = df_temp.groupby(['day', 'hour'])['selling_price_aed'].sum().reset_index()
            pivot = heat_data.pivot(index='day', columns='hour', values='selling_price_aed').fillna(0)
            
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            pivot = pivot.reindex([d for d in day_order if d in pivot.index])
            
            fig = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale='Blues'))
            fig.update_layout(title='Sales by Day & Hour', height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            display_insight("Timing", "Use this heatmap to time your promotions for maximum impact.", "info")
    
    with col2:
        st.markdown("#### ⚠️ Stockout Risk")
        if sim_result['results']:
            risk = sim_result['results']['stockout_risk_pct']
            skus = sim_result['results']['high_risk_skus']
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Stockout Risk %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 60], 'color': "yellow"},
                        {'range': [60, 100], 'color': "red"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 80}
                }
            ))
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            if risk > 30:
                display_insight("Inventory Alert", f"{skus} SKUs at high risk. Replenish before promotion.", "warning")
            else:
                display_insight("Inventory", "Stock levels are adequate for the promotion.", "success")
    
    # Recommendations
    st.markdown("---")
    st.markdown("### 💡 Recommendations")
    
    recs = generate_recommendation(kpis, sim_result['results'] if sim_result['results'] else {}, sim_result['violations'])
    
    for rec in recs:
        if rec.startswith("✅"):
            st.success(rec)
        elif rec.startswith("⚠️") or rec.startswith("💡"):
            st.warning(rec)
        elif rec.startswith("🚫"):
            st.error(rec)
        else:
            st.info(rec)
    
    # Downloads
    st.markdown("---")
    st.markdown("### 📥 Downloads")
    
    cols = st.columns(3)
    with cols[0]:
        st.download_button("📄 Cleaned Sales", data['sales'].to_csv(index=False), "sales_cleaned.csv")
    with cols[1]:
        if 'issues' in data and len(data['issues']) > 0:
            st.download_button("📄 Issues Log", data['issues'].to_csv(index=False), "issues.csv")
    with cols[2]:
        if sim_result['top_risk_items'] is not None:
            st.download_button("📄 Risk Items", sim_result['top_risk_items'].to_csv(index=False), "risk_items.csv")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>UAE Promo Pulse Simulator v1.0</p>", unsafe_allow_html=True)

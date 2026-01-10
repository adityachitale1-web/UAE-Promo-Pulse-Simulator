"""
app.py
UAE Promo Pulse Simulator - Streamlit Dashboard
Enhanced with Advanced Analytics, Business Insights, and Multiple Chart Types
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from itertools import combinations
from collections import Counter
import os
import warnings
warnings.filterwarnings('ignore')

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
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.2rem;
        border-radius: 10px;
        margin: 0.8rem 0 1.5rem 0;
        font-size: 0.9rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .insight-box-success {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .insight-box-warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .insight-box-info {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .insight-title {
        font-weight: bold;
        font-size: 1rem;
        margin-bottom: 0.3rem;
    }
    .chart-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
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
# INSIGHT GENERATOR CLASS
# ============================================================================

class InsightGenerator:
    """Generate business insights from data analysis"""
    
    @staticmethod
    def display_insight(title, insight_text, insight_type="info"):
        """Display styled insight box"""
        css_class = f"insight-box insight-box-{insight_type}"
        icon = "💡" if insight_type == "info" else "✅" if insight_type == "success" else "⚠️"
        
        st.markdown(f'''
        <div class="{css_class}">
            <div class="insight-title">{icon} {title}</div>
            {insight_text}
        </div>
        ''', unsafe_allow_html=True)
    
    @staticmethod
    def revenue_trend_insight(daily_data):
        """Generate insight for revenue trend"""
        if len(daily_data) < 7:
            return "Insufficient data for trend analysis.", "info"
        
        recent_7d = daily_data.tail(7)['revenue'].mean()
        previous_7d = daily_data.tail(14).head(7)['revenue'].mean()
        
        if previous_7d > 0:
            change_pct = ((recent_7d - previous_7d) / previous_7d) * 100
        else:
            change_pct = 0
        
        if change_pct > 10:
            insight = f"Revenue is <strong>trending up {change_pct:.1f}%</strong> week-over-week. Consider increasing inventory for high-demand products to capitalize on momentum."
            return insight, "success"
        elif change_pct < -10:
            insight = f"Revenue is <strong>declining {abs(change_pct):.1f}%</strong> week-over-week. Recommend launching targeted promotions or investigating potential issues."
            return insight, "warning"
        else:
            insight = f"Revenue is <strong>stable</strong> with {change_pct:.1f}% change. Current strategies are maintaining performance."
            return insight, "info"
    
    @staticmethod
    def city_performance_insight(breakdown_df):
        """Generate insight for city performance"""
        if len(breakdown_df) == 0:
            return "No data available for analysis.", "info"
        
        top_city = breakdown_df.loc[breakdown_df['revenue'].idxmax()]
        total_revenue = breakdown_df['revenue'].sum()
        top_share = (top_city['revenue'] / total_revenue * 100) if total_revenue > 0 else 0
        
        insight = f"<strong>{top_city['city']}</strong> leads with <strong>{top_share:.1f}%</strong> of total revenue (AED {top_city['revenue']:,.0f}). "
        
        if top_share > 60:
            insight += "Consider expanding marketing efforts in underperforming cities to reduce concentration risk."
            return insight, "warning"
        else:
            insight += "Revenue distribution across cities is healthy."
            return insight, "success"
    
    @staticmethod
    def margin_insight(cat_breakdown):
        """Generate insight for margin analysis"""
        if len(cat_breakdown) == 0:
            return "No data available for analysis.", "info"
        
        avg_margin = cat_breakdown['margin_pct'].mean()
        low_margin_cats = cat_breakdown[cat_breakdown['margin_pct'] < 15]['category'].tolist()
        high_margin_cats = cat_breakdown[cat_breakdown['margin_pct'] > 30]['category'].tolist()
        
        insight = f"Average margin across categories is <strong>{avg_margin:.1f}%</strong>. "
        
        if low_margin_cats:
            insight += f"<strong>{', '.join(low_margin_cats)}</strong> have margins below 15% - review pricing strategy. "
            insight_type = "warning"
        elif high_margin_cats:
            insight += f"<strong>{', '.join(high_margin_cats)}</strong> show strong margins above 30%."
            insight_type = "success"
        else:
            insight_type = "info"
        
        return insight, insight_type
    
    @staticmethod
    def pareto_insight(pareto_data, threshold=80):
        """Generate insight for Pareto analysis"""
        products_for_80 = len(pareto_data[pareto_data['cumulative_pct'] <= threshold])
        total_products = len(pareto_data)
        pct_products = (products_for_80 / total_products * 100) if total_products > 0 else 0
        
        insight = f"<strong>{pct_products:.1f}%</strong> of products generate <strong>{threshold}%</strong> of revenue. "
        
        if pct_products < 25:
            insight += "Strong concentration - focus inventory and marketing on top performers while reviewing tail products."
            return insight, "success"
        else:
            insight += "Revenue is well-distributed across products, indicating a healthy product mix."
            return insight, "info"
    
    @staticmethod
    def stockout_insight(risk_pct, high_risk_count):
        """Generate insight for stockout risk"""
        if risk_pct > 30:
            insight = f"<strong>Critical Alert:</strong> {risk_pct:.1f}% stockout risk with {high_risk_count} high-risk SKUs. Immediate inventory replenishment recommended."
            return insight, "warning"
        elif risk_pct > 15:
            insight = f"<strong>Moderate Risk:</strong> {risk_pct:.1f}% stockout risk. Monitor {high_risk_count} SKUs closely and prepare reorder plans."
            return insight, "info"
        else:
            insight = f"Stockout risk is <strong>under control</strong> at {risk_pct:.1f}%. Inventory levels are healthy."
            return insight, "success"
    
    @staticmethod
    def hvc_insight(hvc_count, total_customers, hvc_revenue_pct):
        """Generate insight for High Value Customer analysis"""
        hvc_pct = (hvc_count / total_customers * 100) if total_customers > 0 else 0
        
        insight = f"<strong>{hvc_count:,}</strong> High Value Customers ({hvc_pct:.1f}% of base) generate <strong>{hvc_revenue_pct:.1f}%</strong> of revenue. "
        
        if hvc_revenue_pct > 50:
            insight += "Strong HVC segment - implement loyalty programs and personalized offers to retain them."
            return insight, "success"
        else:
            insight += "Opportunity to grow HVC segment through targeted campaigns and premium services."
            return insight, "info"
    
    @staticmethod
    def basket_insight(top_pairs):
        """Generate insight for market basket analysis"""
        if len(top_pairs) == 0:
            return "Insufficient data for basket analysis.", "info"
        
        top_pair = top_pairs[0]
        insight = f"<strong>{top_pair[0][0]}</strong> and <strong>{top_pair[0][1]}</strong> are most frequently purchased together ({top_pair[1]} times). "
        insight += "Consider cross-selling bundles and strategic product placement."
        return insight, "success"


# ============================================================================
# ADVANCED ANALYTICS FUNCTIONS
# ============================================================================

class AdvancedAnalytics:
    """Advanced analytics and chart generation"""
    
    @staticmethod
    def create_pareto_chart(sales_df, products_df):
        """Create Pareto chart (80/20 analysis)"""
        # Aggregate sales by product
        product_sales = sales_df.groupby('product_id').agg({
            'selling_price_aed': 'sum',
            'qty': 'sum'
        }).reset_index()
        product_sales.columns = ['product_id', 'revenue', 'units']
        
        # Merge with product info
        product_sales = product_sales.merge(
            products_df[['product_id', 'category', 'brand']],
            on='product_id', how='left'
        )
        
        # Sort by revenue descending
        product_sales = product_sales.sort_values('revenue', ascending=False).reset_index(drop=True)
        
        # Calculate cumulative percentage
        total_revenue = product_sales['revenue'].sum()
        product_sales['cumulative_revenue'] = product_sales['revenue'].cumsum()
        product_sales['cumulative_pct'] = (product_sales['cumulative_revenue'] / total_revenue * 100)
        product_sales['rank'] = range(1, len(product_sales) + 1)
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add bars for revenue
        fig.add_trace(
            go.Bar(
                x=product_sales['rank'].head(50),
                y=product_sales['revenue'].head(50),
                name='Revenue',
                marker_color='steelblue'
            ),
            secondary_y=False
        )
        
        # Add line for cumulative percentage
        fig.add_trace(
            go.Scatter(
                x=product_sales['rank'].head(50),
                y=product_sales['cumulative_pct'].head(50),
                name='Cumulative %',
                line=dict(color='red', width=2),
                mode='lines'
            ),
            secondary_y=True
        )
        
        # Add 80% reference line
        fig.add_hline(y=80, line_dash="dash", line_color="green",
                      annotation_text="80% Revenue Line", secondary_y=True)
        
        fig.update_layout(
            title='Pareto Analysis - Top 50 Products',
            xaxis_title='Product Rank',
            height=400,
            showlegend=True,
            legend=dict(x=0.7, y=1.1, orientation='h')
        )
        fig.update_yaxes(title_text="Revenue (AED)", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative %", secondary_y=True)
        
        return fig, product_sales
    
    @staticmethod
    def create_waterfall_chart(kpis):
        """Create waterfall chart for revenue breakdown"""
        fig = go.Figure(go.Waterfall(
            name="Revenue Breakdown",
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=["Gross Revenue", "Refunds", "COGS", "Other Costs", "Net Profit"],
            y=[
                kpis['gross_revenue'],
                -kpis['refund_amount'],
                -kpis['cogs'],
                -kpis.get('other_costs', 0),
                0
            ],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#ef553b"}},
            increasing={"marker": {"color": "#00cc96"}},
            totals={"marker": {"color": "#636efa"}}
        ))
        
        fig.update_layout(
            title="Revenue Waterfall Analysis",
            showlegend=False,
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_scatter_plot(sales_df, products_df):
        """Create scatter plot for price vs quantity analysis"""
        # Aggregate by product
        product_analysis = sales_df.groupby('product_id').agg({
            'selling_price_aed': 'mean',
            'qty': 'sum',
            'discount_pct': 'mean'
        }).reset_index()
        
        product_analysis = product_analysis.merge(
            products_df[['product_id', 'category', 'brand']],
            on='product_id', how='left'
        )
        
        fig = px.scatter(
            product_analysis,
            x='selling_price_aed',
            y='qty',
            color='category',
            size='discount_pct',
            hover_data=['brand', 'product_id'],
            title='Price vs Quantity by Category',
            labels={
                'selling_price_aed': 'Average Selling Price (AED)',
                'qty': 'Total Quantity Sold',
                'discount_pct': 'Avg Discount %'
            }
        )
        
        fig.update_layout(height=400)
        return fig, product_analysis
    
    @staticmethod
    def create_dual_axis_chart(daily_kpis):
        """Create dual axis chart for revenue and orders"""
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(
                x=daily_kpis['date'],
                y=daily_kpis['revenue'],
                name='Revenue',
                marker_color='steelblue',
                opacity=0.7
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=daily_kpis['date'],
                y=daily_kpis['orders'],
                name='Orders',
                line=dict(color='red', width=2),
                mode='lines+markers'
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title='Revenue vs Order Volume (Dual Axis)',
            height=400,
            legend=dict(x=0.7, y=1.15, orientation='h')
        )
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Revenue (AED)", secondary_y=False)
        fig.update_yaxes(title_text="Number of Orders", secondary_y=True)
        
        return fig
    
    @staticmethod
    def create_historical_prediction_chart(daily_kpis):
        """Create historical data with simple prediction"""
        df = daily_kpis.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['day_num'] = (df['date'] - df['date'].min()).dt.days
        
        # Simple linear regression for prediction
        X = df['day_num'].values.reshape(-1, 1)
        y = df['revenue'].values
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict next 14 days
        last_day = df['day_num'].max()
        future_days = np.array(range(int(last_day) + 1, int(last_day) + 15)).reshape(-1, 1)
        future_revenue = model.predict(future_days)
        
        # Create future dates
        last_date = df['date'].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, 15)]
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['revenue'],
            mode='lines',
            name='Historical Revenue',
            line=dict(color='steelblue', width=2)
        ))
        
        # Prediction
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_revenue,
            mode='lines',
            name='Predicted Revenue',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Confidence band
        std_dev = df['revenue'].std()
        fig.add_trace(go.Scatter(
            x=future_dates + future_dates[::-1],
            y=list(future_revenue + std_dev) + list((future_revenue - std_dev)[::-1]),
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.1)',
            line=dict(color='rgba(255, 0, 0, 0)'),
            name='Confidence Band'
        ))
        
        fig.update_layout(
            title='Historical Revenue with 14-Day Forecast',
            xaxis_title='Date',
            yaxis_title='Revenue (AED)',
            height=400,
            legend=dict(x=0.6, y=1.15, orientation='h')
        )
        
        return fig, future_revenue
    
    @staticmethod
    def create_outlier_detection_plot(sales_df):
        """Create outlier detection visualization"""
        # Calculate Z-scores for quantity and price
        sales_analysis = sales_df.copy()
        sales_analysis['qty_zscore'] = stats.zscore(sales_analysis['qty'].fillna(0))
        sales_analysis['price_zscore'] = stats.zscore(sales_analysis['selling_price_aed'].fillna(0))
        
        # Identify outliers (Z-score > 3)
        sales_analysis['is_outlier'] = (
            (abs(sales_analysis['qty_zscore']) > 3) | 
            (abs(sales_analysis['price_zscore']) > 3)
        )
        
        fig = px.scatter(
            sales_analysis.sample(min(5000, len(sales_analysis))),
            x='qty_zscore',
            y='price_zscore',
            color='is_outlier',
            color_discrete_map={True: 'red', False: 'steelblue'},
            title='Outlier Detection (Z-Score Analysis)',
            labels={
                'qty_zscore': 'Quantity Z-Score',
                'price_zscore': 'Price Z-Score',
                'is_outlier': 'Is Outlier'
            },
            opacity=0.6
        )
        
        # Add reference lines
        fig.add_hline(y=3, line_dash="dash", line_color="red", annotation_text="Upper Threshold")
        fig.add_hline(y=-3, line_dash="dash", line_color="red", annotation_text="Lower Threshold")
        fig.add_vline(x=3, line_dash="dash", line_color="red")
        fig.add_vline(x=-3, line_dash="dash", line_color="red")
        
        fig.update_layout(height=400)
        
        outlier_count = sales_analysis['is_outlier'].sum()
        return fig, outlier_count
    
    @staticmethod
    def create_donut_chart(breakdown_df, value_col, name_col, title):
        """Create donut chart"""
        fig = go.Figure(data=[go.Pie(
            labels=breakdown_df[name_col],
            values=breakdown_df[value_col],
            hole=0.5,
            textinfo='label+percent',
            textposition='outside',
            marker=dict(colors=px.colors.qualitative.Set3)
        )])
        
        fig.update_layout(
            title=title,
            height=400,
            showlegend=True,
            legend=dict(x=1, y=0.5)
        )
        
        # Add center text
        total = breakdown_df[value_col].sum()
        fig.add_annotation(
            text=f'Total<br>AED {total:,.0f}',
            x=0.5, y=0.5,
            font_size=14,
            showarrow=False
        )
        
        return fig
    
    @staticmethod
    def create_growth_trend_chart(daily_kpis):
        """Create growth trend analysis"""
        df = daily_kpis.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Calculate rolling metrics
        df['revenue_7d_ma'] = df['revenue'].rolling(7, min_periods=1).mean()
        df['revenue_growth'] = df['revenue'].pct_change() * 100
        df['revenue_growth_7d_ma'] = df['revenue_growth'].rolling(7, min_periods=1).mean()
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           subplot_titles=('Revenue with 7-Day Moving Average', 
                                          'Daily Growth Rate (%)'))
        
        # Revenue plot
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['revenue'], name='Daily Revenue',
                      line=dict(color='lightblue'), opacity=0.5),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['revenue_7d_ma'], name='7-Day MA',
                      line=dict(color='steelblue', width=2)),
            row=1, col=1
        )
        
        # Growth rate plot
        colors = ['green' if x > 0 else 'red' for x in df['revenue_growth'].fillna(0)]
        fig.add_trace(
            go.Bar(x=df['date'], y=df['revenue_growth'], name='Daily Growth %',
                  marker_color=colors, opacity=0.7),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['revenue_growth_7d_ma'], name='Growth MA',
                      line=dict(color='purple', width=2)),
            row=2, col=1
        )
        
        fig.update_layout(height=500, showlegend=True,
                         legend=dict(x=0.7, y=1.15, orientation='h'))
        
        return fig
    
    @staticmethod
    def create_cumulative_performance_chart(daily_kpis):
        """Create cumulative performance tracker"""
        df = daily_kpis.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Calculate cumulative metrics
        df['cumulative_revenue'] = df['revenue'].cumsum()
        df['cumulative_orders'] = df['orders'].cumsum()
        df['cumulative_units'] = df['units'].cumsum()
        
        # Create targets (linear growth assumption)
        total_days = len(df)
        end_target = df['cumulative_revenue'].iloc[-1] * 1.2  # 20% above actual
        df['target_revenue'] = [(end_target / total_days) * (i + 1) for i in range(total_days)]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['cumulative_revenue'],
            name='Actual Cumulative Revenue',
            fill='tozeroy',
            line=dict(color='steelblue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['target_revenue'],
            name='Target (Linear)',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Cumulative Revenue Performance vs Target',
            xaxis_title='Date',
            yaxis_title='Cumulative Revenue (AED)',
            height=400,
            legend=dict(x=0.1, y=1.1, orientation='h')
        )
        
        return fig
    
    @staticmethod
    def create_whatif_heatmap(simulator, base_params, filters):
        """Create what-if analysis heatmap"""
        discount_levels = [5, 10, 15, 20, 25, 30]
        budget_levels = [25000, 50000, 75000, 100000, 150000]
        
        # Create matrix for profit proxy
        profit_matrix = []
        
        for budget in budget_levels:
            row = []
            for discount in discount_levels:
                result = simulator.run_simulation(
                    discount_pct=discount,
                    promo_budget=budget,
                    margin_floor=base_params['margin_floor'],
                    simulation_days=base_params['simulation_days'],
                    city=filters.get('city'),
                    channel=filters.get('channel'),
                    category=filters.get('category')
                )
                if result['results']:
                    row.append(result['results']['profit_proxy'])
                else:
                    row.append(0)
            profit_matrix.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=profit_matrix,
            x=[f'{d}%' for d in discount_levels],
            y=[f'AED {b:,}' for b in budget_levels],
            colorscale='RdYlGn',
            text=[[f'AED {v:,.0f}' for v in row] for row in profit_matrix],
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='Discount: %{x}<br>Budget: %{y}<br>Profit: %{z:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='What-If Analysis: Profit by Discount & Budget',
            xaxis_title='Discount Level',
            yaxis_title='Promo Budget',
            height=400
        )
        
        return fig, profit_matrix
    
    @staticmethod
    def create_seasonality_heatmap(sales_df):
        """Create seasonality heatmap by day of week and hour"""
        df = sales_df.copy()
        df['order_time'] = pd.to_datetime(df['order_time'], errors='coerce')
        df = df.dropna(subset=['order_time'])
        
        df['day_of_week'] = df['order_time'].dt.day_name()
        df['hour'] = df['order_time'].dt.hour
        
        # Aggregate by day and hour
        heatmap_data = df.groupby(['day_of_week', 'hour']).agg({
            'selling_price_aed': 'sum'
        }).reset_index()
        
        # Pivot for heatmap
        pivot_data = heatmap_data.pivot(
            index='day_of_week',
            columns='hour',
            values='selling_price_aed'
        ).fillna(0)
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_data = pivot_data.reindex([d for d in day_order if d in pivot_data.index])
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='Blues',
            hovertemplate='Day: %{y}<br>Hour: %{x}<br>Revenue: AED %{z:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Sales Seasonality: Day of Week vs Hour',
            xaxis_title='Hour of Day',
            yaxis_title='Day of Week',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_margin_profitability_chart(sales_df, products_df):
        """Create margin profitability quadrant chart"""
        # Aggregate by product
        product_perf = sales_df.groupby('product_id').agg({
            'selling_price_aed': 'sum',
            'qty': 'sum'
        }).reset_index()
        product_perf.columns = ['product_id', 'revenue', 'units']
        
        # Merge with product cost info
        product_perf = product_perf.merge(
            products_df[['product_id', 'category', 'unit_cost_aed', 'base_price_aed']],
            on='product_id', how='left'
        )
        
        # Calculate margin
        product_perf['margin'] = product_perf['revenue'] - (product_perf['units'] * product_perf['unit_cost_aed'])
        product_perf['margin_pct'] = (product_perf['margin'] / product_perf['revenue'] * 100).fillna(0)
        
        # Calculate medians for quadrants
        median_revenue = product_perf['revenue'].median()
        median_margin = product_perf['margin_pct'].median()
        
        fig = px.scatter(
            product_perf,
            x='revenue',
            y='margin_pct',
            color='category',
            size='units',
            hover_data=['product_id'],
            title='Margin-Revenue Quadrant Analysis',
            labels={
                'revenue': 'Revenue (AED)',
                'margin_pct': 'Margin %',
                'units': 'Units Sold'
            }
        )
        
        # Add quadrant lines
        fig.add_hline(y=median_margin, line_dash="dash", line_color="gray",
                      annotation_text=f"Median Margin: {median_margin:.1f}%")
        fig.add_vline(x=median_revenue, line_dash="dash", line_color="gray",
                      annotation_text=f"Median Revenue")
        
        # Add quadrant labels
        fig.add_annotation(x=median_revenue*2, y=median_margin*1.5, text="⭐ Stars",
                          showarrow=False, font=dict(size=12, color="green"))
        fig.add_annotation(x=median_revenue*0.3, y=median_margin*1.5, text="❓ Question Marks",
                          showarrow=False, font=dict(size=12, color="blue"))
        fig.add_annotation(x=median_revenue*2, y=median_margin*0.5, text="🐄 Cash Cows",
                          showarrow=False, font=dict(size=12, color="orange"))
        fig.add_annotation(x=median_revenue*0.3, y=median_margin*0.5, text="🐕 Dogs",
                          showarrow=False, font=dict(size=12, color="red"))
        
        fig.update_layout(height=450)
        
        return fig, product_perf
    
    @staticmethod
    def create_performance_matrix(breakdown_df, dim1, dim2, value_col='revenue'):
        """Create comparison matrix heatmap"""
        if len(breakdown_df) == 0:
            return None
        
        # Create pivot table
        pivot = breakdown_df.pivot_table(
            values=value_col,
            index=dim1,
            columns=dim2,
            aggfunc='sum',
            fill_value=0
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale='Viridis',
            text=[[f'AED {v:,.0f}' for v in row] for row in pivot.values],
            texttemplate='%{text}',
            textfont={"size": 9}
        ))
        
        fig.update_layout(
            title=f'Performance Matrix: {dim1.title()} vs {dim2.title()}',
            xaxis_title=dim2.title(),
            yaxis_title=dim1.title(),
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_roc_curve_hvc(sales_df, products_df, stores_df):
        """Create ROC curve for High Value Customer prediction"""
        # Create customer-level data
        df = sales_df.copy()
        df['line_total'] = df['qty'] * df['selling_price_aed']
        
        # Simulate customer ID (using store_id + product combination as proxy)
        df['customer_id'] = df['store_id'] + '_' + df['product_id'].str[-4:]
        
        customer_data = df.groupby('customer_id').agg({
            'line_total': 'sum',
            'qty': 'sum',
            'order_id': 'nunique',
            'discount_pct': 'mean'
        }).reset_index()
        customer_data.columns = ['customer_id', 'total_spent', 'total_qty', 'order_count', 'avg_discount']
        
        # Define HVC (top 20% by spending)
        threshold = customer_data['total_spent'].quantile(0.8)
        customer_data['is_hvc'] = (customer_data['total_spent'] >= threshold).astype(int)
        
        # Features for prediction
        X = customer_data[['total_qty', 'order_count', 'avg_discount']].fillna(0)
        y = customer_data['is_hvc']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train logistic regression
        model = LogisticRegression()
        model.fit(X_scaled, y)
        
        # Get probabilities
        y_prob = model.predict_proba(X_scaled)[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='steelblue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title='ROC Curve: High Value Customer Prediction',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400,
            legend=dict(x=0.5, y=0.1)
        )
        
        hvc_count = customer_data['is_hvc'].sum()
        total_customers = len(customer_data)
        hvc_revenue = customer_data[customer_data['is_hvc'] == 1]['total_spent'].sum()
        total_revenue = customer_data['total_spent'].sum()
        hvc_revenue_pct = (hvc_revenue / total_revenue * 100) if total_revenue > 0 else 0
        
        return fig, roc_auc, hvc_count, total_customers, hvc_revenue_pct
    
    @staticmethod
    def apriori_analysis(sales_df, min_support=0.01):
        """Simple market basket analysis (Apriori-like)"""
        # Group products by order
        order_products = sales_df.groupby('order_id')['product_id'].apply(list).reset_index()
        
        # Find frequent pairs
        pair_counts = Counter()
        for products in order_products['product_id']:
            if len(products) >= 2:
                for pair in combinations(set(products), 2):
                    pair_counts[tuple(sorted(pair))] += 1
        
        # Get top pairs
        total_orders = len(order_products)
        top_pairs = [(pair, count) for pair, count in pair_counts.most_common(10)
                     if count / total_orders >= min_support]
        
        # Create visualization
        if top_pairs:
            pairs_df = pd.DataFrame([
                {'pair': f"{p[0][:9]}...\n{p[1][:9]}...", 
                 'count': c,
                 'support': c / total_orders * 100}
                for p, c in top_pairs
            ])
            
            fig = px.bar(
                pairs_df,
                x='pair',
                y='count',
                color='support',
                color_continuous_scale='Viridis',
                title='Market Basket Analysis: Frequently Bought Together',
                labels={'pair': 'Product Pairs', 'count': 'Co-occurrence Count', 'support': 'Support %'}
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
        else:
            fig = None
        
        return fig, top_pairs


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
        original_counts = {
            'products': len(raw_data['products']),
            'stores': len(raw_data['stores']),
            'sales': len(raw_data['sales']),
            'inventory': len(raw_data['inventory']),
            'campaigns': len(raw_data['campaigns'])
        }
        
        cleaner = DataCleaner()
        cleaned_data = cleaner.clean_all(
            raw_data['products'],
            raw_data['stores'],
            raw_data['sales'],
            raw_data['inventory'],
            'data/cleaned'
        )
        
        cleaned_counts = {
            'products': len(cleaned_data['products']),
            'stores': len(cleaned_data['stores']),
            'sales': len(cleaned_data['sales']),
            'inventory': len(cleaned_data['inventory'])
        }
        
        removed_counts = {
            'products': original_counts['products'] - cleaned_counts['products'],
            'stores': original_counts['stores'] - cleaned_counts['stores'],
            'sales': original_counts['sales'] - cleaned_counts['sales'],
            'inventory': original_counts['inventory'] - cleaned_counts['inventory']
        }
        
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
# CLEANING STATS DISPLAY
# ============================================================================

def display_cleaning_stats(stats):
    """Display cleaning statistics"""
    if stats is None:
        return
    
    st.markdown("### 📊 Data Cleaning Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**📁 Table**")
        for table in ['products', 'stores', 'sales', 'inventory']:
            st.markdown(f"`{table}`")
    
    with col2:
        st.markdown("**📥 Original**")
        for table in ['products', 'stores', 'sales', 'inventory']:
            st.markdown(f"**{stats['original'][table]:,}**")
    
    with col3:
        st.markdown("**🗑️ Removed**")
        for table in ['products', 'stores', 'sales', 'inventory']:
            removed = stats['removed'][table]
            color = "red" if removed > 0 else "green"
            st.markdown(f"<span style='color: {color};'>{removed:,}</span>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("**✅ Clean**")
        for table in ['products', 'stores', 'sales', 'inventory']:
            st.markdown(f"**{stats['cleaned'][table]:,}**")
    
    st.markdown("---")
    
    total_original = sum(stats['original'].values())
    total_cleaned = sum(stats['cleaned'].values())
    total_removed = sum(stats['removed'].values())
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Original", f"{total_original:,}")
    col2.metric("Total Removed", f"{total_removed:,}", delta=f"-{total_removed:,}" if total_removed > 0 else "0", delta_color="inverse")
    col3.metric("Total Clean", f"{total_cleaned:,}")
    col4.metric("Issues Logged", f"{stats['total_issues']:,}")


# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    """Render sidebar with controls"""
    st.sidebar.markdown("## 🎛️ Controls")
    
    view_mode = st.sidebar.radio(
        "Dashboard View",
        ["Executive View", "Manager View"],
        help="Toggle between Executive and Manager views"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Source")
    
    data_source = st.sidebar.radio(
        "Select data source:",
        ["Generate & Clean Data", "Load Existing Data", "Upload Custom Data"]
    )
    
    if data_source == "Generate & Clean Data":
        st.sidebar.markdown("---")
        st.sidebar.markdown("#### Step 1: Generate Raw Data")
        
        if st.sidebar.button("🎲 Generate Raw Data", use_container_width=True):
            st.session_state.raw_data = generate_raw_data()
            st.session_state.raw_data_generated = True
            st.session_state.data_loaded = False
            st.session_state.cleaning_stats = None
            st.rerun()
        
        if st.session_state.raw_data_generated and st.session_state.raw_data is not None:
            st.sidebar.success("✅ Raw data generated!")
            st.sidebar.markdown(f"""
            **Raw Data Summary:**
            - Products: {len(st.session_state.raw_data['products']):,}
            - Stores: {len(st.session_state.raw_data['stores']):,}
            - Sales: {len(st.session_state.raw_data['sales']):,}
            - Inventory: {len(st.session_state.raw_data['inventory']):,}
            """)
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("#### Step 2: Clean Data")
        
        clean_disabled = not st.session_state.raw_data_generated
        
        if st.sidebar.button("🧹 Clean Data", use_container_width=True, disabled=clean_disabled):
            if st.session_state.raw_data is not None:
                cleaned_data, cleaning_stats = clean_data(st.session_state.raw_data)
                st.session_state.data = cleaned_data
                st.session_state.cleaning_stats = cleaning_stats
                st.session_state.data_loaded = True
                st.rerun()
        
        if clean_disabled:
            st.sidebar.warning("⚠️ Generate raw data first!")
        
        if st.session_state.data_loaded:
            st.sidebar.success("✅ Data cleaned!")
    
    elif data_source == "Load Existing Data":
        if st.sidebar.button("📂 Load Data", use_container_width=True):
            data = load_data_from_files()
            if data:
                st.session_state.data = data
                st.session_state.data_loaded = True
                st.session_state.cleaning_stats = None
                st.rerun()
            else:
                st.sidebar.error("No existing data found.")
    
    elif data_source == "Upload Custom Data":
        st.sidebar.markdown("#### Upload Files")
        sales_file = st.sidebar.file_uploader("Sales", type=['csv', 'xlsx'])
        products_file = st.sidebar.file_uploader("Products", type=['csv', 'xlsx'])
        stores_file = st.sidebar.file_uploader("Stores", type=['csv', 'xlsx'])
        inventory_file = st.sidebar.file_uploader("Inventory", type=['csv', 'xlsx'])
        
        if st.sidebar.button("Process Data", use_container_width=True):
            if all([sales_file, products_file, stores_file, inventory_file]):
                sales = process_uploaded_file(sales_file, 'sales')
                products = process_uploaded_file(products_file, 'products')
                stores = process_uploaded_file(stores_file, 'stores')
                inventory = process_uploaded_file(inventory_file, 'inventory')
                
                if all([sales is not None, products is not None, stores is not None, inventory is not None]):
                    st.session_state.raw_data = {
                        'products': products, 'stores': stores,
                        'sales': sales, 'inventory': inventory,
                        'campaigns': pd.DataFrame()
                    }
                    st.session_state.raw_data_generated = True
                    cleaned_data, cleaning_stats = clean_data(st.session_state.raw_data)
                    st.session_state.data = cleaned_data
                    st.session_state.cleaning_stats = cleaning_stats
                    st.session_state.data_loaded = True
                    st.rerun()
    
    st.sidebar.markdown("---")
    
    # Filters
    filters = {}
    if st.session_state.data_loaded and st.session_state.data:
        st.sidebar.markdown("### 🔍 Filters")
        data = st.session_state.data
        
        if 'sales' in data and 'order_time' in data['sales'].columns:
            sales_df = data['sales'].copy()
            sales_df['order_time'] = pd.to_datetime(sales_df['order_time'], errors='coerce')
            sales_df = sales_df.dropna(subset=['order_time'])
            if len(sales_df) > 0:
                min_date = sales_df['order_time'].min().date()
                max_date = sales_df['order_time'].max().date()
                date_range = st.sidebar.date_input("Date Range", value=(min_date, max_date),
                                                   min_value=min_date, max_value=max_date)
                filters['date_range'] = date_range
        
        if 'stores' in data:
            cities = ['All'] + sorted(data['stores']['city'].unique().tolist())
            filters['city'] = st.sidebar.selectbox("City", cities)
            channels = ['All'] + sorted(data['stores']['channel'].unique().tolist())
            filters['channel'] = st.sidebar.selectbox("Channel", channels)
        
        if 'products' in data:
            categories = ['All'] + sorted(data['products']['category'].unique().tolist())
            filters['category'] = st.sidebar.selectbox("Category", categories)
            brands = ['All'] + sorted(data['products']['brand'].unique().tolist())
            filters['brand'] = st.sidebar.selectbox("Brand", brands)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎮 Simulation Parameters")
    
    sim_params = {
        'discount_pct': st.sidebar.slider("Discount %", 0, 50, 15),
        'promo_budget': st.sidebar.number_input("Budget (AED)", 10000, 500000, 50000, step=5000),
        'margin_floor': st.sidebar.slider("Margin Floor %", 5, 30, 15),
        'simulation_days': st.sidebar.selectbox("Days", [7, 14], index=1)
    }
    
    return view_mode, filters, sim_params


# ============================================================================
# EXECUTIVE VIEW
# ============================================================================

def render_executive_view(data, filters, sim_params):
    """Render Executive Dashboard with advanced analytics"""
    
    st.markdown("## 📊 Executive Dashboard")
    st.markdown("*Financial analytics and strategic insights for leadership*")
    
    # Show cleaning stats
    if st.session_state.cleaning_stats is not None:
        with st.expander("📊 Data Cleaning Statistics", expanded=False):
            display_cleaning_stats(st.session_state.cleaning_stats)
    
    # Initialize
    kpi_calc = KPICalculator(data['sales'], data['products'], data['stores'], data['inventory'])
    simulator = PromoSimulator(data['sales'], data['products'], data['stores'], data['inventory'])
    insight_gen = InsightGenerator()
    analytics = AdvancedAnalytics()
    
    # Apply filters
    start_date = filters.get('date_range', [None, None])[0] if 'date_range' in filters else None
    end_date = filters.get('date_range', [None, None])[1] if 'date_range' in filters and len(filters['date_range']) > 1 else None
    
    filtered_df = kpi_calc.filter_data(
        city=filters.get('city'), channel=filters.get('channel'),
        category=filters.get('category'), start_date=start_date,
        end_date=end_date, brand=filters.get('brand')
    )
    
    kpis = kpi_calc.compute_historical_kpis(filtered_df)
    daily_kpis = kpi_calc.compute_daily_kpis(filtered_df)
    
    sim_result = simulator.run_simulation(
        discount_pct=sim_params['discount_pct'],
        promo_budget=sim_params['promo_budget'],
        margin_floor=sim_params['margin_floor'],
        simulation_days=sim_params['simulation_days'],
        city=filters.get('city'), channel=filters.get('channel'),
        category=filters.get('category')
    )
    
    # KPI Cards
    st.markdown("### 💰 Key Financial Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Net Revenue", f"AED {kpis['net_revenue']:,.0f}")
    col2.metric("Gross Margin %", f"{kpis['gross_margin_pct']:.1f}%")
    col3.metric("AOV", f"AED {kpis['aov']:,.0f}")
    col4.metric("Total Orders", f"{kpis['total_orders']:,}")
    if sim_result['results']:
        col5.metric("Sim Profit", f"AED {sim_result['results']['profit_proxy']:,.0f}")
    else:
        col5.metric("Sim Profit", "N/A")
    
    st.markdown("---")
    
    # ==================== ROW 1 ====================
    st.markdown("### 📈 Revenue Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Historical + Prediction Chart
        if len(daily_kpis) > 7:
            fig, predictions = analytics.create_historical_prediction_chart(daily_kpis)
            st.plotly_chart(fig, use_container_width=True)
            
            trend_insight, trend_type = insight_gen.revenue_trend_insight(daily_kpis)
            insight_gen.display_insight("Revenue Trend Analysis", trend_insight, trend_type)
        else:
            st.info("Insufficient data for prediction")
    
    with col2:
        # Donut Chart - Revenue by City
        city_breakdown = kpi_calc.compute_breakdown(filtered_df, 'city')
        if len(city_breakdown) > 0:
            fig = analytics.create_donut_chart(city_breakdown, 'revenue', 'city', 'Revenue Distribution by City')
            st.plotly_chart(fig, use_container_width=True)
            
            city_insight, city_type = insight_gen.city_performance_insight(city_breakdown)
            insight_gen.display_insight("City Performance", city_insight, city_type)
    
    # ==================== ROW 2 ====================
    st.markdown("### 📊 Profitability Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Waterfall Chart
        fig = analytics.create_waterfall_chart(kpis)
        st.plotly_chart(fig, use_container_width=True)
        
        profit_pct = (kpis['gross_margin'] / kpis['gross_revenue'] * 100) if kpis['gross_revenue'] > 0 else 0
        insight_gen.display_insight(
            "Profitability Breakdown",
            f"After deducting <strong>AED {kpis['refund_amount']:,.0f}</strong> in refunds and <strong>AED {kpis['cogs']:,.0f}</strong> in COGS, "
            f"gross margin stands at <strong>{profit_pct:.1f}%</strong>. Focus on reducing refund rate to improve bottom line.",
            "info"
        )
    
    with col2:
        # Margin by Category
        cat_breakdown = kpi_calc.compute_breakdown(filtered_df, 'category')
        if len(cat_breakdown) > 0:
            fig = px.bar(cat_breakdown, x='category', y='margin_pct', color='margin_pct',
                        color_continuous_scale='RdYlGn', title='Gross Margin % by Category')
            fig.add_hline(y=sim_params['margin_floor'], line_dash="dash", 
                         annotation_text=f"Floor: {sim_params['margin_floor']}%")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            margin_insight, margin_type = insight_gen.margin_insight(cat_breakdown)
            insight_gen.display_insight("Category Margin Analysis", margin_insight, margin_type)
    
    # ==================== ROW 3 ====================
    st.markdown("### 🎯 Pareto & Performance Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Pareto Chart
        fig, pareto_data = analytics.create_pareto_chart(filtered_df, data['products'])
        st.plotly_chart(fig, use_container_width=True)
        
        pareto_insight, pareto_type = insight_gen.pareto_insight(pareto_data)
        insight_gen.display_insight("80/20 Analysis", pareto_insight, pareto_type)
    
    with col2:
        # Dual Axis Chart
        if len(daily_kpis) > 0:
            fig = analytics.create_dual_axis_chart(daily_kpis)
            st.plotly_chart(fig, use_container_width=True)
            
            corr = daily_kpis['revenue'].corr(daily_kpis['orders'])
            corr_text = "strong positive" if corr > 0.7 else "moderate" if corr > 0.4 else "weak"
            insight_gen.display_insight(
                "Revenue-Order Correlation",
                f"Revenue and order volume show a <strong>{corr_text} correlation ({corr:.2f})</strong>. "
                f"{'Higher order volume directly drives revenue growth.' if corr > 0.7 else 'Consider strategies to increase average order value.'}",
                "success" if corr > 0.7 else "info"
            )
    
    # ==================== ROW 4 ====================
    st.markdown("### 📈 Growth & Cumulative Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        # Growth Trends
        if len(daily_kpis) > 7:
            fig = analytics.create_growth_trend_chart(daily_kpis)
            st.plotly_chart(fig, use_container_width=True)
            
            avg_growth = daily_kpis['revenue'].pct_change().mean() * 100
            insight_gen.display_insight(
                "Growth Momentum",
                f"Average daily growth rate is <strong>{avg_growth:.2f}%</strong>. "
                f"{'Positive momentum indicates healthy business expansion.' if avg_growth > 0 else 'Consider promotional activities to stimulate growth.'}",
                "success" if avg_growth > 0 else "warning"
            )
    
    with col2:
        # Cumulative Performance
        if len(daily_kpis) > 0:
            fig = analytics.create_cumulative_performance_chart(daily_kpis)
            st.plotly_chart(fig, use_container_width=True)
            
            insight_gen.display_insight(
                "Performance vs Target",
                f"Cumulative revenue has reached <strong>AED {daily_kpis['revenue'].sum():,.0f}</strong>. "
                f"Track daily to ensure you stay on pace with targets.",
                "info"
            )
    
    # ==================== ROW 5 ====================
    st.markdown("### 🔮 What-If Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # What-If Heatmap
        with st.spinner("Running scenarios..."):
            fig, profit_matrix = analytics.create_whatif_heatmap(simulator, sim_params, filters)
        st.plotly_chart(fig, use_container_width=True)
        
        # Find optimal scenario
        max_profit = max(max(row) for row in profit_matrix)
        insight_gen.display_insight(
            "Optimal Scenario",
            f"Maximum profit potential is <strong>AED {max_profit:,.0f}</strong>. "
            f"Use the heatmap to identify the best discount-budget combination for your goals.",
            "success" if max_profit > 0 else "warning"
        )
    
    with col2:
        # Seasonality Heatmap
        fig = analytics.create_seasonality_heatmap(filtered_df)
        st.plotly_chart(fig, use_container_width=True)
        
        insight_gen.display_insight(
            "Seasonality Patterns",
            "Identify <strong>peak selling hours and days</strong> to optimize promotional timing. "
            "Schedule flash sales during high-traffic periods for maximum impact.",
            "info"
        )
    
    # ==================== ROW 6 ====================
    st.markdown("### 💡 Strategic Recommendations")
    
    recommendations = generate_recommendation(
        kpis, sim_result['results'] if sim_result['results'] else {}, sim_result['violations']
    )
    
    col1, col2 = st.columns(2)
    with col1:
        for rec in recommendations[:len(recommendations)//2 + 1]:
            if rec.startswith("✅"):
                st.success(rec)
            elif rec.startswith("⚠️") or rec.startswith("💡"):
                st.warning(rec)
            elif rec.startswith("🚫") or rec.startswith("❌"):
                st.error(rec)
            else:
                st.info(rec)
    
    with col2:
        for rec in recommendations[len(recommendations)//2 + 1:]:
            if rec.startswith("✅"):
                st.success(rec)
            elif rec.startswith("⚠️") or rec.startswith("💡"):
                st.warning(rec)
            elif rec.startswith("🚫") or rec.startswith("❌"):
                st.error(rec)
            else:
                st.info(rec)


# ============================================================================
# MANAGER VIEW
# ============================================================================

def render_manager_view(data, filters, sim_params):
    """Render Manager Dashboard with operational analytics"""
    
    st.markdown("## 🔧 Operations Dashboard")
    st.markdown("*Operational insights, risk analysis, and customer intelligence*")
    
    # Show cleaning stats
    if st.session_state.cleaning_stats is not None:
        with st.expander("📊 Data Cleaning Statistics", expanded=True):
            display_cleaning_stats(st.session_state.cleaning_stats)
    
    # Initialize
    kpi_calc = KPICalculator(data['sales'], data['products'], data['stores'], data['inventory'])
    simulator = PromoSimulator(data['sales'], data['products'], data['stores'], data['inventory'])
    insight_gen = InsightGenerator()
    analytics = AdvancedAnalytics()
    
    # Apply filters
    start_date = filters.get('date_range', [None, None])[0] if 'date_range' in filters else None
    end_date = filters.get('date_range', [None, None])[1] if 'date_range' in filters and len(filters['date_range']) > 1 else None
    
    filtered_df = kpi_calc.filter_data(
        city=filters.get('city'), channel=filters.get('channel'),
        category=filters.get('category'), start_date=start_date,
        end_date=end_date, brand=filters.get('brand')
    )
    
    kpis = kpi_calc.compute_historical_kpis(filtered_df)
    
    sim_result = simulator.run_simulation(
        discount_pct=sim_params['discount_pct'],
        promo_budget=sim_params['promo_budget'],
        margin_floor=sim_params['margin_floor'],
        simulation_days=sim_params['simulation_days'],
        city=filters.get('city'), channel=filters.get('channel'),
        category=filters.get('category')
    )
    
    # KPI Cards
    st.markdown("### ⚠️ Key Operational Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    stockout_risk = sim_result['results']['stockout_risk_pct'] if sim_result['results'] else 0
    high_risk_skus = sim_result['results']['high_risk_skus'] if sim_result['results'] else 0
    
    col1.metric("Stockout Risk", f"{stockout_risk:.1f}%", delta="High" if stockout_risk > 30 else "Normal", delta_color="inverse")
    col2.metric("Return Rate", f"{kpis['return_rate']:.1f}%", delta="High" if kpis['return_rate'] > 10 else "Normal", delta_color="inverse")
    col3.metric("Payment Failures", f"{kpis['payment_failure_rate']:.1f}%")
    col4.metric("High-Risk SKUs", f"{high_risk_skus}")
    col5.metric("Total Units", f"{kpis['total_units']:,}")
    
    st.markdown("---")
    
    # ==================== ROW 1 ====================
    st.markdown("### 🎯 Risk Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Outlier Detection Plot
        fig, outlier_count = analytics.create_outlier_detection_plot(filtered_df)
        st.plotly_chart(fig, use_container_width=True)
        
        outlier_pct = (outlier_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        insight_gen.display_insight(
            "Outlier Detection",
            f"<strong>{outlier_count:,}</strong> outliers detected ({outlier_pct:.2f}% of transactions). "
            f"{'Investigate these anomalies for potential fraud or data quality issues.' if outlier_pct > 1 else 'Outlier rate is within acceptable limits.'}",
            "warning" if outlier_pct > 1 else "success"
        )
    
    with col2:
        # Scatter Plot
        fig, product_analysis = analytics.create_scatter_plot(filtered_df, data['products'])
        st.plotly_chart(fig, use_container_width=True)
        
        insight_gen.display_insight(
            "Price-Quantity Relationship",
            "Analyze the relationship between pricing and demand across categories. "
            "<strong>Larger bubbles</strong> indicate higher discount levels. Identify sweet spots for pricing optimization.",
            "info"
        )
    
    # ==================== ROW 2 ====================
    st.markdown("### 📦 Margin & Product Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        # Margin Profitability Quadrant
        fig, product_perf = analytics.create_margin_profitability_chart(filtered_df, data['products'])
        st.plotly_chart(fig, use_container_width=True)
        
        stars = len(product_perf[
            (product_perf['revenue'] > product_perf['revenue'].median()) & 
            (product_perf['margin_pct'] > product_perf['margin_pct'].median())
        ])
        insight_gen.display_insight(
            "Product Portfolio Analysis",
            f"<strong>{stars} Star products</strong> (high revenue, high margin) identified. "
            f"Focus marketing on Stars, optimize pricing for Cash Cows, and review Dogs for potential discontinuation.",
            "success" if stars > 0 else "info"
        )
    
    with col2:
        # Performance Matrix
        sales_enriched = filtered_df.merge(
            data['stores'][['store_id', 'city', 'channel']],
            on='store_id', how='left'
        )
        sales_enriched['revenue'] = sales_enriched['qty'] * sales_enriched['selling_price_aed']
        
        matrix_data = sales_enriched.groupby(['city', 'channel']).agg({'revenue': 'sum'}).reset_index()
        fig = analytics.create_performance_matrix(matrix_data, 'city', 'channel', 'revenue')
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            insight_gen.display_insight(
                "City-Channel Matrix",
                "Identify <strong>high-performing combinations</strong> of city and channel. "
                "Darker cells indicate higher revenue - focus expansion efforts on proven combinations.",
                "info"
            )
    
    # ==================== ROW 3 ====================
    st.markdown("### 👥 Customer Intelligence")
    col1, col2 = st.columns(2)
    
    with col1:
        # ROC Curve for HVC
        fig, roc_auc, hvc_count, total_customers, hvc_revenue_pct = analytics.create_roc_curve_hvc(
            filtered_df, data['products'], data['stores']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        hvc_insight, hvc_type = insight_gen.hvc_insight(hvc_count, total_customers, hvc_revenue_pct)
        insight_gen.display_insight("High Value Customer Analysis", hvc_insight, hvc_type)
    
    with col2:
        # Market Basket Analysis
        fig, top_pairs = analytics.apriori_analysis(filtered_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            basket_insight, basket_type = insight_gen.basket_insight(top_pairs)
            insight_gen.display_insight("Market Basket Analysis", basket_insight, basket_type)
        else:
            st.info("Insufficient data for basket analysis")
    
    # ==================== ROW 4 ====================
    st.markdown("### 📊 Inventory & Stockout Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Stockout risk by category
        if sim_result['results'] and 'detail_data' in sim_result:
            detail_data = sim_result['detail_data']
            category_risk = detail_data.groupby('category').agg({
                'stockout_risk': 'mean',
                'sim_total_demand': 'sum',
                'stock_on_hand': 'sum'
            }).reset_index()
            category_risk['stockout_risk'] = category_risk['stockout_risk'] * 100
            
            fig = px.bar(category_risk, x='category', y='stockout_risk', color='stockout_risk',
                        color_continuous_scale='RdYlGn_r', title='Stockout Risk by Category')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            stockout_insight, stockout_type = insight_gen.stockout_insight(stockout_risk, high_risk_skus)
            insight_gen.display_insight("Inventory Risk Alert", stockout_insight, stockout_type)
    
    with col2:
        # Top stockout risk items
        if 'top_risk_items' in sim_result and sim_result['top_risk_items'] is not None:
            st.markdown("#### 🚨 Top 10 Stockout Risk Items")
            st.dataframe(sim_result['top_risk_items'], use_container_width=True, height=350)
            
            insight_gen.display_insight(
                "Action Required",
                "These SKUs have <strong>demand exceeding 80% of available stock</strong>. "
                "Prioritize replenishment to avoid stockouts during the promotion period.",
                "warning"
            )
    
    # ==================== ROW 5 ====================
    st.markdown("### 🔍 Data Quality & Issues")
    col1, col2 = st.columns(2)
    
    with col1:
        # Data Quality Issues
        if 'issues' in data and len(data['issues']) > 0:
            issues_df = data['issues']
            issue_counts = issues_df['issue_type'].value_counts().reset_index()
            issue_counts.columns = ['issue_type', 'count']
            
            fig = px.bar(issue_counts.head(10), x='issue_type', y='count', color='count',
                        color_continuous_scale='Reds', title='Data Quality Issues by Type')
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            top_issue = issue_counts.iloc[0]['issue_type']
            top_count = issue_counts.iloc[0]['count']
            insight_gen.display_insight(
                "Data Quality Alert",
                f"<strong>{top_issue}</strong> is the most common issue ({top_count:,} occurrences). "
                f"Review data collection processes to reduce these errors at source.",
                "warning"
            )
    
    with col2:
        # Donut for payment status
        payment_breakdown = filtered_df.groupby('payment_status').agg({
            'order_id': 'nunique'
        }).reset_index()
        payment_breakdown.columns = ['payment_status', 'orders']
        
        fig = go.Figure(data=[go.Pie(
            labels=payment_breakdown['payment_status'],
            values=payment_breakdown['orders'],
            hole=0.5,
            marker=dict(colors=['#2ecc71', '#e74c3c', '#f39c12'])
        )])
        fig.update_layout(title='Payment Status Distribution', height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        failed_pct = kpis['payment_failure_rate']
        insight_gen.display_insight(
            "Payment Analysis",
            f"<strong>{failed_pct:.1f}%</strong> of payments failed. "
            f"{'Investigate payment gateway issues or customer payment method problems.' if failed_pct > 5 else 'Payment success rate is healthy.'}",
            "warning" if failed_pct > 5 else "success"
        )
    
    # ==================== DOWNLOAD SECTION ====================
    st.markdown("---")
    st.markdown("### 📥 Download Reports")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        csv = data['sales'].to_csv(index=False)
        st.download_button("📄 Cleaned Sales", csv, "cleaned_sales.csv", "text/csv")
    
    with col2:
        if 'issues' in data and len(data['issues']) > 0:
            csv = data['issues'].to_csv(index=False)
            st.download_button("📄 Issues Log", csv, "issues.csv", "text/csv")
    
    with col3:
        if sim_result['results']:
            sim_df = pd.DataFrame([sim_result['results']])
            csv = sim_df.to_csv(index=False)
            st.download_button("📄 Simulation", csv, "simulation.csv", "text/csv")
    
    with col4:
        if 'top_risk_items' in sim_result and sim_result['top_risk_items'] is not None:
            csv = sim_result['top_risk_items'].to_csv(index=False)
            st.download_button("📄 Risk Items", csv, "risk_items.csv", "text/csv")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main application entry point"""
    
    st.markdown('<p class="main-header">🛒 UAE Promo Pulse Simulator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Analytics Dashboard | What-If Promotion Analysis</p>', unsafe_allow_html=True)
    
    view_mode, filters, sim_params = render_sidebar()
    
    if not st.session_state.data_loaded or st.session_state.data is None:
        st.markdown("---")
        
        if st.session_state.raw_data_generated:
            st.success("✅ Raw data generated! Click '🧹 Clean Data' to proceed.")
            
            st.markdown("### 📋 Raw Data Preview")
            if st.session_state.raw_data:
                tabs = st.tabs(["Products", "Stores", "Sales", "Inventory", "Campaigns"])
                with tabs[0]:
                    st.dataframe(st.session_state.raw_data['products'].head(10), use_container_width=True)
                with tabs[1]:
                    st.dataframe(st.session_state.raw_data['stores'].head(10), use_container_width=True)
                with tabs[2]:
                    st.dataframe(st.session_state.raw_data['sales'].head(10), use_container_width=True)
                with tabs[3]:
                    st.dataframe(st.session_state.raw_data['inventory'].head(10), use_container_width=True)
                with tabs[4]:
                    st.dataframe(st.session_state.raw_data['campaigns'], use_container_width=True)
        else:
            st.info("👈 Generate data using sidebar controls to begin.")
            
            with st.expander("📖 Getting Started", expanded=True):
                st.markdown("""
                ### Welcome to UAE Promo Pulse Simulator!
                
                **Step 1:** Click **"🎲 Generate Raw Data"** to create synthetic data
                
                **Step 2:** Click **"🧹 Clean Data"** to validate and clean
                
                **Step 3:** Explore **Executive View** for financial insights
                
                **Step 4:** Explore **Manager View** for operational analytics
                
                #### Chart Types Available:
                - 📈 Historical + Prediction Charts
                - 🍩 Donut & Pie Charts
                - 📊 Pareto Analysis (80/20)
                - 💧 Waterfall Charts
                - 🔥 Heatmaps (What-If, Seasonality)
                - 🎯 Scatter Plots
                - 📉 Dual-Axis Charts
                - 🔍 Outlier Detection
                - 📊 ROC Curves (HVC Analysis)
                - 🛒 Market Basket Analysis
                - 📊 Performance Matrices
                """)
        return
    
    st.markdown("---")
    
    if view_mode == "Executive View":
        render_executive_view(st.session_state.data, filters, sim_params)
    else:
        render_manager_view(st.session_state.data, filters, sim_params)
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #666;'>UAE Promo Pulse Simulator | Advanced Analytics Edition</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

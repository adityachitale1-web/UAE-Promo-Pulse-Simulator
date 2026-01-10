"""
UAE Promo Pulse - KPI Calculator & Promotion Simulator
======================================================
Implements all required KPIs and rule-based demand simulation.

KPI Dictionary (15 KPIs):
-------------------------
Finance/Executive KPIs:
1. Gross Revenue (Paid only)
2. Refund Amount
3. Net Revenue
4. COGS (Cost of Goods Sold)
5. Gross Margin (AED)
6. Gross Margin %
7. Average Discount %
8. Promo Spend (Simulation)
9. Profit Proxy (Simulation)
10. Budget Utilization %

Ops/Manager KPIs:
11. Stockout Risk %
12. Top 10 Stockout Risk Items
13. Return Rate %
14. Payment Failure Rate %
15. High Risk SKU Count

Demand Uplift Logic:
--------------------
Base demand = Average daily sales over last 30 days per product-store
Uplift multiplier = 1 + (discount_pct / 100) × channel_factor × category_factor

Channel Factors:
- Marketplace: 1.8 (highest response to discounts)
- App: 1.5 (medium response)
- Web: 1.2 (lowest response)

Category Factors:
- Electronics: 1.6 (high elasticity)
- Fashion: 1.5
- Beauty: 1.4
- Sports: 1.3
- Home & Garden: 1.2
- Grocery: 1.0 (low elasticity - essentials)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class KPICalculator:
    """
    Comprehensive KPI calculator for retail analytics.
    """
    
    def __init__(self, sales_df, products_df, stores_df, inventory_df):
        self.sales = sales_df.copy()
        self.products = products_df.copy()
        self.stores = stores_df.copy()
        self.inventory = inventory_df.copy()
        
        # Parse datetime
        if 'order_time' in self.sales.columns:
            self.sales['order_time'] = pd.to_datetime(self.sales['order_time'], errors='coerce')
        
        # Merge for enriched analysis
        self.sales = self.sales.merge(
            self.products[['product_id', 'category', 'brand', 'unit_cost_aed', 'base_price_aed']],
            on='product_id', how='left'
        )
        self.sales = self.sales.merge(
            self.stores[['store_id', 'city', 'channel', 'fulfillment_type']],
            on='store_id', how='left'
        )
        
        # Calculate line totals
        self.sales['line_total'] = self.sales['qty'] * self.sales['selling_price_aed']
        self.sales['line_cost'] = self.sales['qty'] * self.sales['unit_cost_aed'].fillna(0)
    
    def filter_data(self, city=None, channel=None, category=None, brand=None, 
                   fulfillment=None, date_from=None, date_to=None):
        """Apply filters to sales data."""
        df = self.sales.copy()
        
        if city and city != 'All':
            df = df[df['city'] == city]
        if channel and channel != 'All':
            df = df[df['channel'] == channel]
        if category and category != 'All':
            df = df[df['category'] == category]
        if brand and brand != 'All':
            df = df[df['brand'] == brand]
        if fulfillment and fulfillment != 'All':
            df = df[df['fulfillment_type'] == fulfillment]
        if date_from:
            df = df[df['order_time'] >= pd.to_datetime(date_from)]
        if date_to:
            df = df[df['order_time'] <= pd.to_datetime(date_to)]
        
        return df
    
    def compute_kpis(self, df):
        """
        Compute all 15 KPIs.
        
        Returns:
            dict: Dictionary containing all KPI values
        """
        if len(df) == 0:
            return {
                'gross_revenue': 0, 'refund_amount': 0, 'net_revenue': 0,
                'cogs': 0, 'gross_margin': 0, 'gross_margin_pct': 0,
                'avg_discount_pct': 0, 'total_orders': 0, 'total_units': 0,
                'aov': 0, 'return_rate': 0, 'payment_failure_rate': 0,
                'unique_products': 0, 'unique_stores': 0
            }
        
        # Filter by payment status
        paid = df[df['payment_status'] == 'Paid']
        refunded = df[df['payment_status'] == 'Refunded']
        failed = df[df['payment_status'] == 'Failed']
        
        # 1. Gross Revenue (Paid only)
        gross_revenue = paid['line_total'].sum()
        
        # 2. Refund Amount
        refund_amount = refunded['line_total'].sum()
        
        # 3. Returns Amount (from paid orders with return_flag)
        returns_amount = paid[paid['return_flag'] == 1]['line_total'].sum()
        
        # 4. Net Revenue
        net_revenue = gross_revenue - refund_amount - returns_amount
        
        # 5. COGS
        cogs = paid['line_cost'].sum()
        
        # 6. Gross Margin (AED)
        gross_margin = net_revenue - cogs
        
        # 7. Gross Margin %
        gross_margin_pct = (gross_margin / net_revenue * 100) if net_revenue > 0 else 0
        
        # 8. Average Discount %
        avg_discount_pct = df['discount_pct'].mean()
        
        # Other metrics
        total_orders = df['order_id'].nunique()
        total_units = paid['qty'].sum()
        aov = net_revenue / total_orders if total_orders > 0 else 0
        
        # 13. Return Rate %
        paid_count = len(paid)
        returns_count = len(paid[paid['return_flag'] == 1])
        return_rate = (returns_count / paid_count * 100) if paid_count > 0 else 0
        
        # 14. Payment Failure Rate %
        total_count = len(df)
        failed_count = len(failed)
        payment_failure_rate = (failed_count / total_count * 100) if total_count > 0 else 0
        
        return {
            'gross_revenue': gross_revenue,
            'refund_amount': refund_amount,
            'returns_amount': returns_amount,
            'net_revenue': net_revenue,
            'cogs': cogs,
            'gross_margin': gross_margin,
            'gross_margin_pct': gross_margin_pct,
            'avg_discount_pct': avg_discount_pct,
            'total_orders': total_orders,
            'total_units': total_units,
            'aov': aov,
            'return_rate': return_rate,
            'payment_failure_rate': payment_failure_rate,
            'unique_products': df['product_id'].nunique(),
            'unique_stores': df['store_id'].nunique()
        }
    
    def compute_daily(self, df):
        """Compute daily aggregated metrics."""
        if len(df) == 0:
            return pd.DataFrame(columns=['date', 'revenue', 'orders', 'units', 'avg_discount'])
        
        df = df.copy()
        df['date'] = df['order_time'].dt.date
        
        daily = df[df['payment_status'] == 'Paid'].groupby('date').agg({
            'line_total': 'sum',
            'order_id': 'nunique',
            'qty': 'sum',
            'discount_pct': 'mean'
        }).reset_index()
        
        daily.columns = ['date', 'revenue', 'orders', 'units', 'avg_discount']
        return daily.sort_values('date')
    
    def compute_breakdown(self, df, dimension):
        """Compute metrics broken down by a dimension."""
        if len(df) == 0 or dimension not in df.columns:
            return pd.DataFrame()
        
        paid = df[df['payment_status'] == 'Paid']
        
        breakdown = paid.groupby(dimension).agg({
            'line_total': 'sum',
            'line_cost': 'sum',
            'order_id': 'nunique',
            'qty': 'sum',
            'discount_pct': 'mean'
        }).reset_index()
        
        breakdown.columns = [dimension, 'revenue', 'cogs', 'orders', 'units', 'avg_discount']
        breakdown['margin'] = breakdown['revenue'] - breakdown['cogs']
        breakdown['margin_pct'] = (breakdown['margin'] / breakdown['revenue'] * 100).fillna(0)
        
        return breakdown.sort_values('revenue', ascending=False)


class PromoSimulator:
    """
    Promotion simulator with rule-based demand uplift and constraint checking.
    
    Uplift Logic:
    - Base demand from last 30 days average
    - Apply channel × category uplift factors
    - Check constraints: budget, margin floor, stock availability
    """
    
    # Channel uplift factors (response to discounts)
    CHANNEL_FACTORS = {
        'Marketplace': 1.8,  # Highest response
        'App': 1.5,          # Medium response
        'Web': 1.2           # Lowest response
    }
    
    # Category uplift factors (price elasticity)
    CATEGORY_FACTORS = {
        'Electronics': 1.6,   # High elasticity
        'Fashion': 1.5,
        'Beauty': 1.4,
        'Sports': 1.3,
        'Home & Garden': 1.2,
        'Grocery': 1.0        # Low elasticity (essentials)
    }
    
    def __init__(self, sales_df, products_df, stores_df, inventory_df):
        self.sales = sales_df.copy()
        self.products = products_df.copy()
        self.stores = stores_df.copy()
        self.inventory = inventory_df.copy()
        
        # Merge data
        self.sales['order_time'] = pd.to_datetime(self.sales['order_time'], errors='coerce')
        self.sales = self.sales.merge(
            self.products[['product_id', 'category', 'unit_cost_aed', 'base_price_aed']],
            on='product_id', how='left'
        )
        self.sales = self.sales.merge(
            self.stores[['store_id', 'city', 'channel']],
            on='store_id', how='left'
        )
    
    def calculate_uplift_factor(self, discount_pct, channel, category):
        """
        Calculate demand uplift factor based on discount, channel, and category.
        
        Formula: uplift = 1 + (discount_pct / 100) × channel_factor × category_factor
        """
        channel_factor = self.CHANNEL_FACTORS.get(channel, 1.3)
        category_factor = self.CATEGORY_FACTORS.get(category, 1.2)
        
        uplift = 1 + (discount_pct / 100) * channel_factor * category_factor
        return uplift
    
    def run_simulation(self, discount_pct, promo_budget, margin_floor, simulation_days,
                      city=None, channel=None, category=None):
        """
        Run promotion simulation with constraint checking.
        
        Returns:
            dict: Simulation results including KPIs, violations, and risk items
        """
        # Filter base data
        df = self.sales.copy()
        if city and city != 'All':
            df = df[df['city'] == city]
        if channel and channel != 'All':
            df = df[df['channel'] == channel]
        if category and category != 'All':
            df = df[df['category'] == category]
        
        if len(df) == 0:
            return {
                'results': None,
                'violations': [{'constraint': 'NO_DATA', 'message': 'No data for selected filters', 'severity': 'HIGH'}],
                'top_risk_items': None,
                'constraint_violators': None
            }
        
        # Calculate baseline demand (last 30 days average per product-store)
        recent = df[df['order_time'] >= (df['order_time'].max() - timedelta(days=30))]
        baseline = recent.groupby(['product_id', 'store_id', 'category', 'channel']).agg({
            'qty': 'sum',
            'selling_price_aed': 'mean',
            'unit_cost_aed': 'first',
            'base_price_aed': 'first'
        }).reset_index()
        
        # Calculate daily baseline
        baseline['daily_demand'] = baseline['qty'] / 30
        
        # Apply uplift factor per row
        baseline['uplift_factor'] = baseline.apply(
            lambda row: self.calculate_uplift_factor(
                discount_pct, 
                row['channel'], 
                row['category']
            ), axis=1
        )
        
        # Simulated demand
        baseline['sim_demand'] = baseline['daily_demand'] * baseline['uplift_factor'] * simulation_days
        
        # Simulated price after discount
        baseline['sim_price'] = baseline['base_price_aed'] * (1 - discount_pct / 100)
        
        # Simulated revenue
        baseline['sim_revenue'] = baseline['sim_demand'] * baseline['sim_price']
        
        # Simulated COGS
        baseline['sim_cogs'] = baseline['sim_demand'] * baseline['unit_cost_aed'].fillna(0)
        
        # Discount cost (subsidy)
        baseline['discount_cost'] = baseline['sim_demand'] * baseline['base_price_aed'] * (discount_pct / 100)
        
        # Aggregate simulation results
        total_sim_revenue = baseline['sim_revenue'].sum()
        total_sim_cogs = baseline['sim_cogs'].sum()
        total_sim_units = baseline['sim_demand'].sum()
        total_discount_cost = baseline['discount_cost'].sum()
        
        # Apply budget cap
        actual_promo_spend = min(total_discount_cost, promo_budget)
        budget_utilization = (actual_promo_spend / promo_budget * 100) if promo_budget > 0 else 0
        
        # Profit proxy
        profit_proxy = total_sim_revenue - total_sim_cogs - actual_promo_spend
        
        # Margin %
        sim_margin_pct = ((total_sim_revenue - total_sim_cogs) / total_sim_revenue * 100) if total_sim_revenue > 0 else 0
        
        # =====================================================================
        # INVENTORY & STOCKOUT ANALYSIS
        # =====================================================================
        inv = self.inventory.copy()
        inv['snapshot_date'] = pd.to_datetime(inv['snapshot_date'])
        
        # Get latest inventory snapshot
        latest_inv = inv[inv['snapshot_date'] == inv['snapshot_date'].max()]
        
        # Filter inventory by selected category if applicable
        if category and category != 'All':
            cat_products = self.products[self.products['category'] == category]['product_id'].tolist()
            latest_inv = latest_inv[latest_inv['product_id'].isin(cat_products)]
        
        # Aggregate stock by product-store
        stock_agg = latest_inv.groupby(['product_id', 'store_id'])['stock_on_hand'].sum().reset_index()
        
        # Merge with simulation demand
        baseline = baseline.merge(stock_agg, on=['product_id', 'store_id'], how='left')
        baseline['stock_on_hand'] = baseline['stock_on_hand'].fillna(0)
        
        # Calculate stockout risk
        baseline['demand_vs_stock'] = baseline['sim_demand'] / baseline['stock_on_hand'].replace(0, 1)
        baseline['stockout_risk'] = (baseline['demand_vs_stock'] > 1).astype(int)
        baseline['excess_demand'] = (baseline['sim_demand'] - baseline['stock_on_hand']).clip(lower=0)
        
        # Risk metrics
        stockout_risk_pct = (baseline['stockout_risk'].sum() / len(baseline) * 100) if len(baseline) > 0 else 0
        high_risk_skus = baseline['stockout_risk'].sum()
        
        # Top 10 risk items
        top_risk = baseline.nlargest(10, 'demand_vs_stock')[
            ['product_id', 'store_id', 'category', 'channel', 'stock_on_hand', 'sim_demand', 'demand_vs_stock']
        ].copy()
        top_risk['demand_vs_stock'] = (top_risk['demand_vs_stock'] * 100).round(1)
        top_risk.columns = ['Product', 'Store', 'Category', 'Channel', 'Stock', 'Sim Demand', 'Risk %']
        
        # =====================================================================
        # CONSTRAINT CHECKING
        # =====================================================================
        violations = []
        constraint_violators = []
        
        # Constraint 1: Margin Floor
        if sim_margin_pct < margin_floor:
            violations.append({
                'constraint': 'MARGIN_FLOOR',
                'message': f'Simulated margin {sim_margin_pct:.1f}% is below floor {margin_floor}%',
                'severity': 'HIGH',
                'gap': margin_floor - sim_margin_pct
            })
            # Find top products causing low margin
            low_margin = baseline[baseline['sim_revenue'] > 0].copy()
            low_margin['item_margin_pct'] = ((low_margin['sim_revenue'] - low_margin['sim_cogs']) / low_margin['sim_revenue'] * 100)
            violators = low_margin[low_margin['item_margin_pct'] < margin_floor].nsmallest(10, 'item_margin_pct')
            if len(violators) > 0:
                constraint_violators.append({
                    'constraint': 'MARGIN_FLOOR',
                    'top_violators': violators[['product_id', 'store_id', 'category', 'item_margin_pct']].to_dict('records')
                })
        
        # Constraint 2: Budget
        if total_discount_cost > promo_budget:
            violations.append({
                'constraint': 'BUDGET_EXCEEDED',
                'message': f'Required spend AED {total_discount_cost:,.0f} exceeds budget AED {promo_budget:,.0f}',
                'severity': 'HIGH',
                'gap': total_discount_cost - promo_budget
            })
            # Find top products consuming budget
            top_spend = baseline.nlargest(10, 'discount_cost')
            constraint_violators.append({
                'constraint': 'BUDGET_EXCEEDED',
                'top_violators': top_spend[['product_id', 'store_id', 'category', 'discount_cost']].to_dict('records')
            })
        
        # Constraint 3: Stock Availability
        stock_violations = baseline[baseline['excess_demand'] > 0]
        if len(stock_violations) > 0:
            total_excess = stock_violations['excess_demand'].sum()
            violations.append({
                'constraint': 'STOCK_INSUFFICIENT',
                'message': f'{len(stock_violations)} product-stores have demand exceeding stock ({total_excess:.0f} units short)',
                'severity': 'MEDIUM',
                'gap': total_excess
            })
            top_stock_violators = stock_violations.nlargest(10, 'excess_demand')
            constraint_violators.append({
                'constraint': 'STOCK_INSUFFICIENT',
                'top_violators': top_stock_violators[['product_id', 'store_id', 'stock_on_hand', 'sim_demand', 'excess_demand']].to_dict('records')
            })
        
        # Constraint 4: Stockout Risk Warning
        if stockout_risk_pct > 30:
            violations.append({
                'constraint': 'HIGH_STOCKOUT_RISK',
                'message': f'{stockout_risk_pct:.1f}% of SKUs at stockout risk',
                'severity': 'MEDIUM',
                'gap': stockout_risk_pct - 30
            })
        
        # =====================================================================
        # COMPILE RESULTS
        # =====================================================================
        results = {
            # Revenue & Profit
            'sim_revenue': total_sim_revenue,
            'sim_cogs': total_sim_cogs,
            'sim_units': total_sim_units,
            'discount_cost': total_discount_cost,
            'actual_promo_spend': actual_promo_spend,
            'profit_proxy': profit_proxy,
            'sim_margin_pct': sim_margin_pct,
            'budget_utilization': budget_utilization,
            
            # Risk metrics
            'stockout_risk_pct': stockout_risk_pct,
            'high_risk_skus': high_risk_skus,
            
            # Uplift info
            'avg_uplift_factor': baseline['uplift_factor'].mean(),
            'demand_increase_pct': (baseline['uplift_factor'].mean() - 1) * 100,
            
            # Simulation params
            'discount_pct': discount_pct,
            'promo_budget': promo_budget,
            'margin_floor': margin_floor,
            'simulation_days': simulation_days
        }
        
        return {
            'results': results,
            'violations': violations,
            'top_risk_items': top_risk,
            'constraint_violators': constraint_violators,
            'simulation_detail': baseline
        }
    
    def get_stockout_by_dimension(self, sim_detail, dimension):
        """Get stockout risk breakdown by city/channel."""
        if sim_detail is None or len(sim_detail) == 0:
            return pd.DataFrame()
        
        # Need to merge back dimension info
        df = sim_detail.copy()
        
        breakdown = df.groupby(dimension).agg({
            'stockout_risk': ['sum', 'count'],
            'excess_demand': 'sum'
        }).reset_index()
        
        breakdown.columns = [dimension, 'at_risk_skus', 'total_skus', 'excess_demand']
        breakdown['risk_pct'] = (breakdown['at_risk_skus'] / breakdown['total_skus'] * 100).round(1)
        
        return breakdown.sort_values('risk_pct', ascending=False)


def generate_recommendation(kpis, sim_results, violations):
    """
    Generate auto recommendations based on computed KPIs and simulation results.
    """
    recs = []
    
    # KPI-based recommendations
    if kpis.get('gross_margin_pct', 0) < 15:
        recs.append("⚠️ **Margin Alert:** Gross margin below 15%. Review pricing strategy and supplier costs.")
    elif kpis.get('gross_margin_pct', 0) > 30:
        recs.append("✅ **Healthy Margins:** Strong gross margin above 30%. Consider competitive pricing to gain market share.")
    
    if kpis.get('return_rate', 0) > 10:
        recs.append("⚠️ **High Returns:** Return rate exceeds 10%. Investigate product quality and description accuracy.")
    
    if kpis.get('payment_failure_rate', 0) > 5:
        recs.append("⚠️ **Payment Issues:** Failure rate above 5%. Review payment gateway and checkout process.")
    
    if kpis.get('avg_discount_pct', 0) > 20:
        recs.append("💡 **Discount Dependency:** Average discount above 20%. Consider value-based selling strategies.")
    
    # Simulation-based recommendations
    if sim_results:
        if sim_results.get('profit_proxy', 0) > 0:
            recs.append(f"✅ **Simulation Viable:** Projected profit AED {sim_results['profit_proxy']:,.0f}. Proceed with promotion.")
        else:
            recs.append(f"🚫 **Simulation Warning:** Projected loss AED {abs(sim_results.get('profit_proxy', 0)):,.0f}. Adjust parameters.")
        
        if sim_results.get('budget_utilization', 0) < 50:
            recs.append("💡 **Budget Underutilized:** Consider increasing discount or expanding promotion scope.")
        elif sim_results.get('budget_utilization', 0) > 100:
            recs.append("⚠️ **Budget Exceeded:** Reduce discount percentage or narrow promotion scope.")
        
        if sim_results.get('stockout_risk_pct', 0) > 30:
            recs.append(f"⚠️ **Stockout Risk:** {sim_results['stockout_risk_pct']:.1f}% SKUs at risk. Increase inventory before promotion.")
        
        if sim_results.get('demand_increase_pct', 0) > 50:
            recs.append(f"📈 **High Uplift Expected:** {sim_results['demand_increase_pct']:.1f}% demand increase. Ensure logistics capacity.")
    
    # Violation-based recommendations
    for v in violations:
        if v['severity'] == 'HIGH':
            recs.append(f"🚫 **{v['constraint']}:** {v['message']}")
        else:
            recs.append(f"💡 **{v['constraint']}:** {v['message']}")
    
    return recs if recs else ["✅ All metrics within healthy ranges. Good to proceed!"]

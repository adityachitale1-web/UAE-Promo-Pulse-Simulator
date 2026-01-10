"""
simulator.py
KPI calculations and promotion simulation
"""

import pandas as pd
import numpy as np

class KPICalculator:
    """Calculate retail KPIs"""
    
    def __init__(self, sales_df, products_df, stores_df, inventory_df):
        self.sales = sales_df.copy()
        self.products = products_df.copy()
        self.stores = stores_df.copy()
        self.inventory = inventory_df.copy()
        
        # Ensure datetime
        if 'order_time' in self.sales.columns:
            self.sales['order_time'] = pd.to_datetime(self.sales['order_time'], errors='coerce')
        
        # Merge data
        self.sales = self.sales.merge(self.products[['product_id', 'category', 'brand', 'unit_cost_aed']], on='product_id', how='left')
        self.sales = self.sales.merge(self.stores[['store_id', 'city', 'channel', 'fulfillment_type']], on='store_id', how='left')
        
        # Calculate line total
        self.sales['line_total'] = self.sales['qty'] * self.sales['selling_price_aed']
        self.sales['line_cost'] = self.sales['qty'] * self.sales['unit_cost_aed'].fillna(0)
    
    def filter_data(self, city=None, channel=None, category=None, brand=None, fulfillment=None, start_date=None, end_date=None):
        """Filter sales data"""
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
        if start_date:
            df = df[df['order_time'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['order_time'] <= pd.to_datetime(end_date)]
        
        return df
    
    def compute_historical_kpis(self, df):
        """Compute historical KPIs"""
        if len(df) == 0:
            return {
                'gross_revenue': 0, 'net_revenue': 0, 'refund_amount': 0,
                'cogs': 0, 'gross_margin': 0, 'gross_margin_pct': 0,
                'total_orders': 0, 'total_units': 0, 'aov': 0,
                'return_rate': 0, 'payment_failure_rate': 0
            }
        
        # Revenue
        paid_sales = df[df['payment_status'] == 'Paid']
        gross_revenue = paid_sales['line_total'].sum()
        
        refunded = df[df['payment_status'] == 'Refunded']
        refund_amount = refunded['line_total'].sum()
        
        returned = paid_sales[paid_sales['return_flag'] == 1]
        return_amount = returned['line_total'].sum()
        
        net_revenue = gross_revenue - refund_amount - return_amount
        
        # Costs
        cogs = paid_sales['line_cost'].sum()
        gross_margin = net_revenue - cogs
        gross_margin_pct = (gross_margin / net_revenue * 100) if net_revenue > 0 else 0
        
        # Volume
        total_orders = df['order_id'].nunique()
        total_units = paid_sales['qty'].sum()
        aov = (net_revenue / total_orders) if total_orders > 0 else 0
        
        # Rates
        return_rate = (len(returned) / len(paid_sales) * 100) if len(paid_sales) > 0 else 0
        failed = df[df['payment_status'] == 'Failed']
        payment_failure_rate = (len(failed) / len(df) * 100) if len(df) > 0 else 0
        
        return {
            'gross_revenue': gross_revenue,
            'net_revenue': net_revenue,
            'refund_amount': refund_amount,
            'cogs': cogs,
            'gross_margin': gross_margin,
            'gross_margin_pct': gross_margin_pct,
            'total_orders': total_orders,
            'total_units': total_units,
            'aov': aov,
            'return_rate': return_rate,
            'payment_failure_rate': payment_failure_rate
        }
    
    def compute_daily_kpis(self, df):
        """Compute daily KPIs"""
        if len(df) == 0:
            return pd.DataFrame(columns=['date', 'revenue', 'orders', 'units'])
        
        df = df.copy()
        df['date'] = df['order_time'].dt.date
        
        daily = df[df['payment_status'] == 'Paid'].groupby('date').agg({
            'line_total': 'sum',
            'order_id': 'nunique',
            'qty': 'sum'
        }).reset_index()
        
        daily.columns = ['date', 'revenue', 'orders', 'units']
        daily = daily.sort_values('date')
        
        return daily
    
    def compute_breakdown(self, df, dimension):
        """Compute KPIs by dimension"""
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
    """Simulate promotional scenarios"""
    
    def __init__(self, sales_df, products_df, stores_df, inventory_df):
        self.sales = sales_df.copy()
        self.products = products_df.copy()
        self.stores = stores_df.copy()
        self.inventory = inventory_df.copy()
        
        # Merge
        self.sales = self.sales.merge(self.products[['product_id', 'category', 'brand', 'unit_cost_aed', 'base_price_aed']], on='product_id', how='left')
        self.sales = self.sales.merge(self.stores[['store_id', 'city', 'channel']], on='store_id', how='left')
    
    def run_simulation(self, discount_pct, promo_budget, margin_floor, simulation_days, city=None, channel=None, category=None):
        """Run promotion simulation"""
        # Filter data
        df = self.sales.copy()
        if city and city != 'All':
            df = df[df['city'] == city]
        if channel and channel != 'All':
            df = df[df['channel'] == channel]
        if category and category != 'All':
            df = df[df['category'] == category]
        
        if len(df) == 0:
            return {'results': None, 'violations': [], 'detail_data': None, 'top_risk_items': None}
        
        # Calculate baseline metrics
        daily_revenue = df['selling_price_aed'].sum() * df['qty'].sum() / 120  # Approx daily
        daily_units = df['qty'].sum() / 120
        
        # Elasticity model (simple)
        elasticity = 1.5
        demand_multiplier = 1 + (discount_pct / 100) * elasticity
        
        # Projected metrics
        sim_daily_units = daily_units * demand_multiplier
        sim_total_units = sim_daily_units * simulation_days
        
        # Revenue after discount
        avg_price = df['selling_price_aed'].mean()
        sim_price = avg_price * (1 - discount_pct / 100)
        sim_revenue = sim_total_units * sim_price
        
        # Costs
        avg_cost = df['unit_cost_aed'].mean()
        sim_cogs = sim_total_units * avg_cost
        
        # Discount cost (budget impact)
        discount_cost = min(sim_total_units * avg_price * (discount_pct / 100), promo_budget)
        
        # Profit
        profit_proxy = sim_revenue - sim_cogs - discount_cost
        
        # Margin check
        margin_pct = ((sim_revenue - sim_cogs) / sim_revenue * 100) if sim_revenue > 0 else 0
        
        # Inventory analysis
        latest_inv = self.inventory.copy()
        latest_inv['snapshot_date'] = pd.to_datetime(latest_inv['snapshot_date'])
        latest_date = latest_inv['snapshot_date'].max()
        latest_inv = latest_inv[latest_inv['snapshot_date'] == latest_date]
        
        # Merge with products for category
        latest_inv = latest_inv.merge(self.products[['product_id', 'category']], on='product_id', how='left')
        
        # Apply category filter to inventory
        if category and category != 'All':
            latest_inv = latest_inv[latest_inv['category'] == category]
        
        # Stockout risk
        product_demand = df.groupby('product_id')['qty'].sum() / 120 * simulation_days * demand_multiplier
        
        inv_risk = latest_inv.groupby(['product_id', 'store_id', 'category']).agg({
            'stock_on_hand': 'sum'
        }).reset_index()
        
        inv_risk = inv_risk.merge(
            product_demand.reset_index().rename(columns={'qty': 'sim_total_demand'}),
            on='product_id', how='left'
        )
        inv_risk['sim_total_demand'] = inv_risk['sim_total_demand'].fillna(0)
        inv_risk['stockout_risk'] = (inv_risk['sim_total_demand'] / inv_risk['stock_on_hand'].replace(0, 1)).clip(0, 1)
        
        stockout_risk_pct = (inv_risk['stockout_risk'] > 0.8).mean() * 100
        high_risk_skus = (inv_risk['stockout_risk'] > 0.8).sum()
        
        # Top risk items
        top_risk = inv_risk.nlargest(10, 'stockout_risk')[['product_id', 'store_id', 'category', 'stock_on_hand', 'sim_total_demand', 'stockout_risk']]
        top_risk['stockout_risk'] = (top_risk['stockout_risk'] * 100).round(1)
        
        # Violations
        violations = []
        if margin_pct < margin_floor:
            violations.append({
                'constraint': 'MARGIN_FLOOR',
                'message': f'Margin {margin_pct:.1f}% is below floor of {margin_floor}%',
                'severity': 'HIGH'
            })
        
        if discount_cost > promo_budget:
            violations.append({
                'constraint': 'BUDGET_EXCEEDED',
                'message': f'Projected cost AED {discount_cost:,.0f} exceeds budget AED {promo_budget:,.0f}',
                'severity': 'HIGH'
            })
        
        if stockout_risk_pct > 30:
            violations.append({
                'constraint': 'STOCKOUT_RISK',
                'message': f'{stockout_risk_pct:.1f}% of SKUs at risk of stockout',
                'severity': 'MEDIUM'
            })
        
        results = {
            'sim_revenue': sim_revenue,
            'sim_cogs': sim_cogs,
            'sim_units': sim_total_units,
            'discount_cost': discount_cost,
            'profit_proxy': profit_proxy,
            'margin_pct': margin_pct,
            'budget_utilization': (discount_cost / promo_budget * 100) if promo_budget > 0 else 0,
            'stockout_risk_pct': stockout_risk_pct,
            'high_risk_skus': high_risk_skus
        }
        
        return {
            'results': results,
            'violations': violations,
            'detail_data': inv_risk,
            'top_risk_items': top_risk
        }
    
    def run_scenario_comparison(self, discount_levels, promo_budget, margin_floor, simulation_days, city=None, channel=None, category=None):
        """Compare multiple scenarios"""
        scenarios = []
        
        for discount in discount_levels:
            result = self.run_simulation(
                discount_pct=discount,
                promo_budget=promo_budget,
                margin_floor=margin_floor,
                simulation_days=simulation_days,
                city=city, channel=channel, category=category
            )
            
            if result['results']:
                scenarios.append({
                    'discount_pct': discount,
                    'revenue': result['results']['sim_revenue'],
                    'profit_proxy': result['results']['profit_proxy'],
                    'margin_pct': result['results']['margin_pct'],
                    'stockout_risk': result['results']['stockout_risk_pct']
                })
        
        return pd.DataFrame(scenarios)


def generate_recommendation(kpis, sim_results, violations):
    """Generate recommendations based on analysis"""
    recommendations = []
    
    # Margin recommendations
    if kpis.get('gross_margin_pct', 0) < 15:
        recommendations.append("⚠️ Current gross margin is below 15%. Review pricing strategy and supplier costs.")
    elif kpis.get('gross_margin_pct', 0) > 30:
        recommendations.append("✅ Healthy gross margin above 30%. Consider strategic discounts to drive volume.")
    
    # Return rate
    if kpis.get('return_rate', 0) > 10:
        recommendations.append("⚠️ Return rate exceeds 10%. Investigate product quality and customer expectations.")
    
    # Payment failures
    if kpis.get('payment_failure_rate', 0) > 8:
        recommendations.append("⚠️ High payment failure rate. Review payment gateway and offer alternative methods.")
    
    # Simulation results
    if sim_results:
        if sim_results.get('profit_proxy', 0) > 0:
            recommendations.append(f"✅ Simulation shows positive profit of AED {sim_results['profit_proxy']:,.0f}. Promotion is viable.")
        else:
            recommendations.append(f"🚫 Simulation shows loss of AED {abs(sim_results.get('profit_proxy', 0)):,.0f}. Adjust parameters.")
        
        if sim_results.get('stockout_risk_pct', 0) > 30:
            recommendations.append(f"⚠️ High stockout risk ({sim_results['stockout_risk_pct']:.1f}%). Increase inventory before promotion.")
    
    # Violations
    for v in violations:
        if v['severity'] == 'HIGH':
            recommendations.append(f"🚫 {v['message']}")
        else:
            recommendations.append(f"💡 {v['message']}")
    
    if not recommendations:
        recommendations.append("✅ All metrics look healthy. Proceed with confidence!")
    
    return recommendations

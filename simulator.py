"""
simulator.py
UAE Promo Pulse Simulator - Simulation and KPI Computation Module
Computes KPIs and runs what-if discount simulations with constraints
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================================
# UPLIFT FACTORS CONFIGURATION
# ============================================================================

# Channel sensitivity to discounts (higher = more responsive)
CHANNEL_UPLIFT_FACTORS = {
    'Marketplace': 1.3,  # Most price-sensitive
    'App': 1.1,
    'Web': 1.0
}

# Category sensitivity to discounts
CATEGORY_UPLIFT_FACTORS = {
    'Electronics': 1.4,  # High-ticket items respond well
    'Fashion': 1.3,
    'Beauty': 1.2,
    'Sports': 1.2,
    'Home & Garden': 1.1,
    'Grocery': 0.9  # Low elasticity
}

# Base uplift per discount percentage point
BASE_UPLIFT_PER_DISCOUNT_PCT = 0.015  # 1.5% demand increase per 1% discount


class KPICalculator:
    """Calculate KPIs from cleaned data"""
    
    def __init__(self, sales_df, products_df, stores_df, inventory_df):
        self.sales = sales_df.copy()
        self.products = products_df.copy()
        self.stores = stores_df.copy()
        self.inventory = inventory_df.copy()
        
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(self.sales['order_time']):
            self.sales['order_time'] = pd.to_datetime(self.sales['order_time'])
        
        # Merge sales with products and stores for enriched analysis
        self.sales_enriched = self.sales.merge(
            self.products[['product_id', 'category', 'brand', 'base_price_aed', 'unit_cost_aed']],
            on='product_id',
            how='left'
        ).merge(
            self.stores[['store_id', 'city', 'channel', 'fulfillment_type']],
            on='store_id',
            how='left'
        )
        
        # Calculate line totals
        self.sales_enriched['line_total'] = self.sales_enriched['qty'] * self.sales_enriched['selling_price_aed']
        self.sales_enriched['line_cost'] = self.sales_enriched['qty'] * self.sales_enriched['unit_cost_aed']
    
    def filter_data(self, city=None, channel=None, category=None, 
                    start_date=None, end_date=None, brand=None, fulfillment=None):
        """Filter sales data based on criteria"""
        df = self.sales_enriched.copy()
        
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
    
    def compute_historical_kpis(self, df=None):
        """Compute historical KPIs from sales data"""
        if df is None:
            df = self.sales_enriched
        
        # Filter for valid calculations
        paid_df = df[df['payment_status'] == 'Paid']
        refund_df = df[df['payment_status'] == 'Refunded']
        failed_df = df[df['payment_status'] == 'Failed']
        
        # 1. Gross Revenue (Paid only)
        gross_revenue = paid_df['line_total'].sum()
        
        # 2. Refund Amount
        refund_amount = refund_df['line_total'].sum()
        
        # 3. Net Revenue
        net_revenue = gross_revenue - refund_amount
        
        # 4. COGS (Cost of Goods Sold)
        cogs = paid_df['line_cost'].sum()
        
        # 5. Gross Margin (AED)
        gross_margin = net_revenue - cogs
        
        # 6. Gross Margin %
        gross_margin_pct = (gross_margin / net_revenue * 100) if net_revenue > 0 else 0
        
        # 7. Average Discount %
        avg_discount_pct = df['discount_pct'].mean() if len(df) > 0 else 0
        
        # 8. Total Orders
        total_orders = df['order_id'].nunique()
        
        # 9. Total Units Sold
        total_units = paid_df['qty'].sum()
        
        # 10. Average Order Value
        aov = gross_revenue / total_orders if total_orders > 0 else 0
        
        # 11. Return Rate %
        returned_orders = df[df['return_flag'] == 1]['order_id'].nunique()
        paid_orders = paid_df['order_id'].nunique()
        return_rate = (returned_orders / paid_orders * 100) if paid_orders > 0 else 0
        
        # 12. Payment Failure Rate %
        total_attempts = len(df)
        failed_count = len(failed_df)
        failure_rate = (failed_count / total_attempts * 100) if total_attempts > 0 else 0
        
        return {
            'gross_revenue': round(gross_revenue, 2),
            'refund_amount': round(refund_amount, 2),
            'net_revenue': round(net_revenue, 2),
            'cogs': round(cogs, 2),
            'gross_margin': round(gross_margin, 2),
            'gross_margin_pct': round(gross_margin_pct, 2),
            'avg_discount_pct': round(avg_discount_pct, 2),
            'total_orders': total_orders,
            'total_units': int(total_units),
            'aov': round(aov, 2),
            'return_rate': round(return_rate, 2),
            'payment_failure_rate': round(failure_rate, 2)
        }
    
    def compute_daily_kpis(self, df=None):
        """Compute daily KPIs for trend analysis"""
        if df is None:
            df = self.sales_enriched
        
        df = df.copy()
        df['date'] = df['order_time'].dt.date
        
        daily = df[df['payment_status'] == 'Paid'].groupby('date').agg({
            'line_total': 'sum',
            'order_id': 'nunique',
            'qty': 'sum',
            'discount_pct': 'mean'
        }).reset_index()
        
        daily.columns = ['date', 'revenue', 'orders', 'units', 'avg_discount']
        daily['date'] = pd.to_datetime(daily['date'])
        
        return daily
    
    def compute_breakdown(self, df=None, group_by='city'):
        """Compute KPIs broken down by dimension"""
        if df is None:
            df = self.sales_enriched
        
        paid_df = df[df['payment_status'] == 'Paid']
        
        breakdown = paid_df.groupby(group_by).agg({
            'line_total': 'sum',
            'line_cost': 'sum',
            'order_id': 'nunique',
            'qty': 'sum',
            'discount_pct': 'mean'
        }).reset_index()
        
        breakdown.columns = [group_by, 'revenue', 'cogs', 'orders', 'units', 'avg_discount']
        breakdown['margin'] = breakdown['revenue'] - breakdown['cogs']
        breakdown['margin_pct'] = (breakdown['margin'] / breakdown['revenue'] * 100).round(2)
        
        return breakdown


class PromoSimulator:
    """Run what-if discount simulations"""
    
    def __init__(self, sales_df, products_df, stores_df, inventory_df):
        self.sales = sales_df.copy()
        self.products = products_df.copy()
        self.stores = stores_df.copy()
        self.inventory = inventory_df.copy()
        
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(self.sales['order_time']):
            self.sales['order_time'] = pd.to_datetime(self.sales['order_time'])
        
        # Get latest inventory snapshot
        if not pd.api.types.is_datetime64_any_dtype(self.inventory['snapshot_date']):
            self.inventory['snapshot_date'] = pd.to_datetime(self.inventory['snapshot_date'])
        
        latest_date = self.inventory['snapshot_date'].max()
        self.current_inventory = self.inventory[
            self.inventory['snapshot_date'] == latest_date
        ].copy()
        
        # Merge data
        self.sales_enriched = self.sales.merge(
            self.products[['product_id', 'category', 'brand', 'base_price_aed', 'unit_cost_aed']],
            on='product_id', how='left'
        ).merge(
            self.stores[['store_id', 'city', 'channel', 'fulfillment_type']],
            on='store_id', how='left'
        )
    
    def compute_baseline_demand(self, days=30, city=None, channel=None, category=None):
        """Compute baseline daily demand per product-store from last N days"""
        
        # Filter to last N days
        end_date = self.sales['order_time'].max()
        start_date = end_date - timedelta(days=days)
        
        df = self.sales_enriched[
            (self.sales_enriched['order_time'] >= start_date) &
            (self.sales_enriched['payment_status'] == 'Paid')
        ].copy()
        
        # Apply filters
        if city and city != 'All':
            df = df[df['city'] == city]
        if channel and channel != 'All':
            df = df[df['channel'] == channel]
        if category and category != 'All':
            df = df[df['category'] == category]
        
        # Compute daily demand per product-store
        baseline = df.groupby(['product_id', 'store_id', 'category', 'channel']).agg({
            'qty': 'sum',
            'selling_price_aed': 'mean',
            'unit_cost_aed': 'first',
            'base_price_aed': 'first'
        }).reset_index()
        
        baseline['daily_demand'] = baseline['qty'] / days
        
        return baseline
    
    def compute_uplift_factor(self, discount_pct, channel, category):
        """Compute demand uplift factor based on discount and characteristics"""
        
        channel_factor = CHANNEL_UPLIFT_FACTORS.get(channel, 1.0)
        category_factor = CATEGORY_UPLIFT_FACTORS.get(category, 1.0)
        
        # Uplift formula: base * channel * category * discount effect
        # Diminishing returns after 20% discount
        if discount_pct <= 20:
            discount_effect = 1 + (discount_pct * BASE_UPLIFT_PER_DISCOUNT_PCT)
        else:
            base_effect = 20 * BASE_UPLIFT_PER_DISCOUNT_PCT
            extra_effect = (discount_pct - 20) * BASE_UPLIFT_PER_DISCOUNT_PCT * 0.5  # Diminishing
            discount_effect = 1 + base_effect + extra_effect
        
        return channel_factor * category_factor * discount_effect
    
    def run_simulation(self, discount_pct, promo_budget, margin_floor, 
                       simulation_days=14, city=None, channel=None, category=None):
        """Run what-if simulation with constraints"""
        
        # Get baseline demand
        baseline = self.compute_baseline_demand(30, city, channel, category)
        
        if len(baseline) == 0:
            return {
                'success': False,
                'message': 'No baseline data available for selected filters',
                'results': None,
                'violations': []
            }
        
        # Merge with current inventory
        sim_data = baseline.merge(
            self.current_inventory[['product_id', 'store_id', 'stock_on_hand']],
            on=['product_id', 'store_id'],
            how='left'
        )
        sim_data['stock_on_hand'] = sim_data['stock_on_hand'].fillna(0)
        
        # Apply filters to simulation data
        if city and city != 'All':
            store_ids = self.stores[self.stores['city'] == city]['store_id']
            sim_data = sim_data[sim_data['store_id'].isin(store_ids)]
        
        # Compute uplift for each row
        sim_data['uplift_factor'] = sim_data.apply(
            lambda row: self.compute_uplift_factor(discount_pct, row['channel'], row['category']),
            axis=1
        )
        
        # Compute simulated demand
        sim_data['sim_daily_demand'] = sim_data['daily_demand'] * sim_data['uplift_factor']
        sim_data['sim_total_demand'] = sim_data['sim_daily_demand'] * simulation_days
        
        # Apply stock constraint
        sim_data['constrained_qty'] = sim_data[['sim_total_demand', 'stock_on_hand']].min(axis=1)
        
        # Compute simulated metrics
        sim_data['sim_selling_price'] = sim_data['base_price_aed'] * (1 - discount_pct / 100)
        sim_data['sim_revenue'] = sim_data['constrained_qty'] * sim_data['sim_selling_price']
        sim_data['sim_cost'] = sim_data['constrained_qty'] * sim_data['unit_cost_aed']
        sim_data['sim_margin'] = sim_data['sim_revenue'] - sim_data['sim_cost']
        sim_data['promo_cost'] = sim_data['constrained_qty'] * sim_data['base_price_aed'] * (discount_pct / 100)
        
        # Aggregate results
        total_revenue = sim_data['sim_revenue'].sum()
        total_cost = sim_data['sim_cost'].sum()
        total_margin = sim_data['sim_margin'].sum()
        total_promo_cost = sim_data['promo_cost'].sum()
        total_qty = sim_data['constrained_qty'].sum()
        
        margin_pct = (total_margin / total_revenue * 100) if total_revenue > 0 else 0
        profit_proxy = total_margin - total_promo_cost
        budget_utilization = (total_promo_cost / promo_budget * 100) if promo_budget > 0 else 0
        
        # Check stockout risk
        sim_data['demand_vs_stock'] = sim_data['sim_total_demand'] / sim_data['stock_on_hand'].replace(0, 0.001)
        sim_data['stockout_risk'] = sim_data['demand_vs_stock'] > 0.8
        stockout_risk_pct = (sim_data['stockout_risk'].sum() / len(sim_data) * 100) if len(sim_data) > 0 else 0
        
        # Check constraint violations
        violations = []
        
        # Budget constraint
        if total_promo_cost > promo_budget:
            violations.append({
                'constraint': 'BUDGET_EXCEEDED',
                'message': f'Promo spend ({total_promo_cost:,.0f} AED) exceeds budget ({promo_budget:,.0f} AED)',
                'severity': 'HIGH',
                'excess': total_promo_cost - promo_budget
            })
        
        # Margin floor constraint
        if margin_pct < margin_floor:
            violations.append({
                'constraint': 'MARGIN_BELOW_FLOOR',
                'message': f'Gross margin ({margin_pct:.1f}%) is below floor ({margin_floor}%)',
                'severity': 'HIGH',
                'shortfall': margin_floor - margin_pct
            })
        
        # Stock constraint (informational)
        stock_limited = sim_data[sim_data['sim_total_demand'] > sim_data['stock_on_hand']]
        if len(stock_limited) > 0:
            violations.append({
                'constraint': 'STOCK_LIMITED',
                'message': f'{len(stock_limited)} product-store combinations are stock-limited',
                'severity': 'MEDIUM',
                'affected_count': len(stock_limited)
            })
        
        # Top risk items
        top_stockout_risk = sim_data.nlargest(10, 'demand_vs_stock')[
            ['product_id', 'store_id', 'category', 'stock_on_hand', 'sim_total_demand', 'demand_vs_stock']
        ].copy()
        top_stockout_risk['demand_vs_stock'] = (top_stockout_risk['demand_vs_stock'] * 100).round(1)
        top_stockout_risk.columns = ['Product', 'Store', 'Category', 'Stock', 'Projected Demand', 'Risk %']
        
        # Results summary
        results = {
            'total_revenue': round(total_revenue, 2),
            'total_cost': round(total_cost, 2),
            'total_margin': round(total_margin, 2),
            'margin_pct': round(margin_pct, 2),
            'promo_spend': round(total_promo_cost, 2),
            'profit_proxy': round(profit_proxy, 2),
            'budget_utilization': round(budget_utilization, 2),
            'total_units': int(total_qty),
            'stockout_risk_pct': round(stockout_risk_pct, 2),
            'high_risk_skus': int(sim_data['stockout_risk'].sum()),
            'simulation_days': simulation_days,
            'discount_pct': discount_pct,
            'promo_budget': promo_budget,
            'margin_floor': margin_floor
        }
        
        return {
            'success': len([v for v in violations if v['severity'] == 'HIGH']) == 0,
            'message': 'Simulation completed' + (' with violations' if violations else ' successfully'),
            'results': results,
            'violations': violations,
            'detail_data': sim_data,
            'top_risk_items': top_stockout_risk
        }
    
    def run_scenario_comparison(self, discount_levels, promo_budget, margin_floor, 
                                 simulation_days=14, city=None, channel=None, category=None):
        """Run multiple scenarios for comparison"""
        
        scenarios = []
        for discount in discount_levels:
            result = self.run_simulation(
                discount, promo_budget, margin_floor, 
                simulation_days, city, channel, category
            )
            if result['results']:
                scenarios.append({
                    'discount_pct': discount,
                    'revenue': result['results']['total_revenue'],
                    'margin': result['results']['total_margin'],
                    'margin_pct': result['results']['margin_pct'],
                    'profit_proxy': result['results']['profit_proxy'],
                    'promo_spend': result['results']['promo_spend'],
                    'stockout_risk': result['results']['stockout_risk_pct'],
                    'feasible': result['success']
                })
        
        return pd.DataFrame(scenarios)


def generate_recommendation(kpis, sim_results, violations):
    """Generate auto-recommendation text based on KPIs and simulation"""
    
    recommendations = []
    
    # Check margin health
    if kpis['gross_margin_pct'] < 20:
        recommendations.append("⚠️ Current gross margin is below 20%. Consider reducing discount depth.")
    elif kpis['gross_margin_pct'] > 35:
        recommendations.append("✅ Healthy gross margin allows room for promotional activity.")
    
    # Check simulation results
    if sim_results:
        if sim_results['margin_pct'] < sim_results.get('margin_floor', 15):
            recommendations.append(f"🚫 Simulated margin ({sim_results['margin_pct']:.1f}%) is below target. Reduce discount or narrow scope.")
        
        if sim_results['budget_utilization'] > 100:
            recommendations.append(f"🚫 Budget exceeded by {sim_results['budget_utilization'] - 100:.1f}%. Scale back promotion.")
        elif sim_results['budget_utilization'] < 50:
            recommendations.append(f"💡 Only {sim_results['budget_utilization']:.1f}% budget utilized. Consider expanding promotion scope.")
        
        if sim_results['stockout_risk_pct'] > 30:
            recommendations.append(f"⚠️ High stockout risk ({sim_results['stockout_risk_pct']:.1f}%). Review inventory for high-demand items.")
        
        if sim_results['profit_proxy'] > 0:
            recommendations.append(f"✅ Positive profit proxy ({sim_results['profit_proxy']:,.0f} AED). Promotion is financially viable.")
        else:
            recommendations.append(f"🚫 Negative profit proxy ({sim_results['profit_proxy']:,.0f} AED). Promotion may not be profitable.")
    
    # Check violations
    high_violations = [v for v in violations if v.get('severity') == 'HIGH']
    if high_violations:
        recommendations.append(f"❌ {len(high_violations)} critical constraint violation(s). Address before proceeding.")
    
    if not recommendations:
        recommendations.append("✅ All metrics within acceptable ranges. Promotion can proceed.")
    
    return recommendations


if __name__ == "__main__":
    # Test with sample data
    import os
    
    if os.path.exists('data/cleaned'):
        sales = pd.read_csv('data/cleaned/sales_cleaned.csv')
        products = pd.read_csv('data/cleaned/products_cleaned.csv')
        stores = pd.read_csv('data/cleaned/stores_cleaned.csv')
        inventory = pd.read_csv('data/cleaned/inventory_cleaned.csv')
        
        # Test KPI Calculator
        kpi_calc = KPICalculator(sales, products, stores, inventory)
        kpis = kpi_calc.compute_historical_kpis()
        print("\nHistorical KPIs:")
        for k, v in kpis.items():
            print(f"  {k}: {v}")
        
        # Test Simulator
        sim = PromoSimulator(sales, products, stores, inventory)
        result = sim.run_simulation(
            discount_pct=15,
            promo_budget=50000,
            margin_floor=15,
            simulation_days=14
        )
        print("\nSimulation Results:")
        print(f"  Success: {result['success']}")
        if result['results']:
            for k, v in result['results'].items():
                print(f"  {k}: {v}")
    else:
        print("Run data_generator.py and cleaner.py first!")

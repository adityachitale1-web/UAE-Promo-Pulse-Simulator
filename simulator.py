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
        budget_utilization = (total_pr

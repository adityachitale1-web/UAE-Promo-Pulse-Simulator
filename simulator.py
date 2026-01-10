import pandas as pd
import numpy as np

class KPICalculator:
    def __init__(self, sales_df, products_df, stores_df, inventory_df):
        self.sales = sales_df.copy()
        self.products = products_df.copy()
        self.stores = stores_df.copy()
        self.inventory = inventory_df.copy()
        
        if 'order_time' in self.sales.columns:
            self.sales['order_time'] = pd.to_datetime(self.sales['order_time'], errors='coerce')
        
        self.sales = self.sales.merge(self.products[['product_id', 'category', 'brand', 'unit_cost_aed', 'base_price_aed']], on='product_id', how='left')
        self.sales = self.sales.merge(self.stores[['store_id', 'city', 'channel', 'fulfillment_type']], on='store_id', how='left')
        self.sales['line_total'] = self.sales['qty'] * self.sales['selling_price_aed']
        self.sales['line_cost'] = self.sales['qty'] * self.sales['unit_cost_aed'].fillna(0)
    
    def filter_data(self, city=None, channel=None, category=None):
        df = self.sales.copy()
        if city and city != 'All': df = df[df['city'] == city]
        if channel and channel != 'All': df = df[df['channel'] == channel]
        if category and category != 'All': df = df[df['category'] == category]
        return df
    
    def compute_kpis(self, df):
        if len(df) == 0:
            return {'gross_revenue': 0, 'net_revenue': 0, 'refund_amount': 0, 'cogs': 0, 'gross_margin': 0,
                    'gross_margin_pct': 0, 'total_orders': 0, 'total_units': 0, 'aov': 0, 'return_rate': 0, 'payment_failure_rate': 0}
        paid = df[df['payment_status'] == 'Paid']
        gross = paid['line_total'].sum()
        refund = df[df['payment_status'] == 'Refunded']['line_total'].sum()
        ret_amt = paid[paid['return_flag'] == 1]['line_total'].sum()
        net = gross - refund - ret_amt
        cogs = paid['line_cost'].sum()
        margin = net - cogs
        margin_pct = (margin / net * 100) if net > 0 else 0
        orders = df['order_id'].nunique()
        units = paid['qty'].sum()
        aov = net / orders if orders > 0 else 0
        ret_rate = (len(paid[paid['return_flag'] == 1]) / len(paid) * 100) if len(paid) > 0 else 0
        fail_rate = (len(df[df['payment_status'] == 'Failed']) / len(df) * 100) if len(df) > 0 else 0
        return {'gross_revenue': gross, 'net_revenue': net, 'refund_amount': refund, 'cogs': cogs, 'gross_margin': margin,
                'gross_margin_pct': margin_pct, 'total_orders': orders, 'total_units': units, 'aov': aov,
                'return_rate': ret_rate, 'payment_failure_rate': fail_rate}
    
    def compute_daily(self, df):
        if len(df) == 0: return pd.DataFrame(columns=['date', 'revenue', 'orders', 'units'])
        df = df.copy()
        df['date'] = df['order_time'].dt.date
        daily = df[df['payment_status'] == 'Paid'].groupby('date').agg({'line_total': 'sum', 'order_id': 'nunique', 'qty': 'sum'}).reset_index()
        daily.columns = ['date', 'revenue', 'orders', 'units']
        return daily.sort_values('date')
    
    def compute_breakdown(self, df, dim):
        if len(df) == 0 or dim not in df.columns: return pd.DataFrame()
        paid = df[df['payment_status'] == 'Paid']
        bd = paid.groupby(dim).agg({'line_total': 'sum', 'line_cost': 'sum', 'order_id': 'nunique', 'qty': 'sum'}).reset_index()
        bd.columns = [dim, 'revenue', 'cogs', 'orders', 'units']
        bd['margin'] = bd['revenue'] - bd['cogs']
        bd['margin_pct'] = (bd['margin'] / bd['revenue'] * 100).fillna(0)
        return bd.sort_values('revenue', ascending=False)


class PromoSimulator:
    def __init__(self, sales_df, products_df, stores_df, inventory_df):
        self.sales = sales_df.copy()
        self.products = products_df.copy()
        self.stores = stores_df.copy()
        self.inventory = inventory_df.copy()
        self.sales = self.sales.merge(self.products[['product_id', 'category', 'unit_cost_aed', 'base_price_aed']], on='product_id', how='left')
        self.sales = self.sales.merge(self.stores[['store_id', 'city', 'channel']], on='store_id', how='left')
    
    def run_simulation(self, discount_pct, promo_budget, margin_floor, simulation_days, city=None, channel=None, category=None):
        df = self.sales.copy()
        if city and city != 'All': df = df[df['city'] == city]
        if channel and channel != 'All': df = df[df['channel'] == channel]
        if category and category != 'All': df = df[df['category'] == category]
        if len(df) == 0: return {'results': None, 'violations': [], 'top_risk_items': None}
        
        daily_units = df['qty'].sum() / 120
        mult = 1 + (discount_pct / 100) * 1.5
        sim_units = daily_units * mult * simulation_days
        avg_price = df['selling_price_aed'].mean()
        sim_price = avg_price * (1 - discount_pct / 100)
        sim_revenue = sim_units * sim_price
        avg_cost = df['unit_cost_aed'].mean()
        sim_cogs = sim_units * avg_cost
        disc_cost = min(sim_units * avg_price * (discount_pct / 100), promo_budget)
        profit = sim_revenue - sim_cogs - disc_cost
        margin_pct = ((sim_revenue - sim_cogs) / sim_revenue * 100) if sim_revenue > 0 else 0
        
        inv = self.inventory.copy()
        inv['snapshot_date'] = pd.to_datetime(inv['snapshot_date'])
        inv = inv[inv['snapshot_date'] == inv['snapshot_date'].max()]
        inv = inv.merge(self.products[['product_id', 'category']], on='product_id', how='left')
        if category and category != 'All': inv = inv[inv['category'] == category]
        
        demand = df.groupby('product_id')['qty'].sum() / 120 * simulation_days * mult
        inv_agg = inv.groupby(['product_id', 'category'])['stock_on_hand'].sum().reset_index()
        inv_agg = inv_agg.merge(demand.reset_index().rename(columns={'qty': 'demand'}), on='product_id', how='left')
        inv_agg['demand'] = inv_agg['demand'].fillna(0)
        inv_agg['risk'] = (inv_agg['demand'] / inv_agg['stock_on_hand'].replace(0, 1)).clip(0, 1)
        risk_pct = (inv_agg['risk'] > 0.8).mean() * 100
        risk_skus = (inv_agg['risk'] > 0.8).sum()
        top_risk = inv_agg.nlargest(10, 'risk')[['product_id', 'category', 'stock_on_hand', 'demand', 'risk']]
        top_risk['risk'] = (top_risk['risk'] * 100).round(1)
        
        violations = []
        if margin_pct < margin_floor: violations.append({'constraint': 'MARGIN', 'message': f'Margin {margin_pct:.1f}% < floor {margin_floor}%', 'severity': 'HIGH'})
        if disc_cost > promo_budget: violations.append({'constraint': 'BUDGET', 'message': 'Cost exceeds budget', 'severity': 'HIGH'})
        if risk_pct > 30: violations.append({'constraint': 'STOCKOUT', 'message': f'{risk_pct:.1f}% SKUs at risk', 'severity': 'MEDIUM'})
        
        return {'results': {'sim_revenue': sim_revenue, 'sim_cogs': sim_cogs, 'sim_units': sim_units, 'discount_cost': disc_cost,
                           'profit_proxy': profit, 'margin_pct': margin_pct, 'stockout_risk_pct': risk_pct, 'high_risk_skus': risk_skus},
                'violations': violations, 'top_risk_items': top_risk}


def generate_recommendation(kpis, sim_results, violations):
    recs = []
    if kpis.get('gross_margin_pct', 0) < 15: recs.append("⚠️ Margin below 15%. Review pricing.")
    if kpis.get('return_rate', 0) > 10: recs.append("⚠️ High return rate. Check quality.")
    if sim_results:
        if sim_results.get('profit_proxy', 0) > 0: recs.append(f"✅ Projected profit: AED {sim_results['profit_proxy']:,.0f}")
        else: recs.append(f"🚫 Projected loss: AED {abs(sim_results.get('profit_proxy', 0)):,.0f}")
        if sim_results.get('stockout_risk_pct', 0) > 30: recs.append("⚠️ High stockout risk. Increase inventory.")
    for v in violations: recs.append(f"{'🚫' if v['severity'] == 'HIGH' else '💡'} {v['message']}")
    return recs if recs else ["✅ All metrics healthy!"]

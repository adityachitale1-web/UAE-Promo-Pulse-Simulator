import pandas as pd
import numpy as np
import os

class DataCleaner:
    def __init__(self):
        self.issues = []
    
    def log_issue(self, rid, itype, detail, action):
        self.issues.append({'record_id': str(rid), 'issue_type': itype, 'issue_detail': detail, 'action_taken': action})
    
    def clean_products(self, df):
        df = df.copy()
        for idx in df[df['unit_cost_aed'].isna()].index:
            cat = df.loc[idx, 'category']
            med = df[df['category'] == cat]['unit_cost_aed'].median()
            if pd.isna(med): med = df['unit_cost_aed'].median()
            df.loc[idx, 'unit_cost_aed'] = med
            self.log_issue(df.loc[idx, 'product_id'], 'MISSING_COST', 'NaN', f'Filled: {med:.2f}')
        return df
    
    def clean_stores(self, df):
        df = df.copy()
        mapping = {'dubai': 'Dubai', 'DUBAI': 'Dubai', 'DXB': 'Dubai',
                   'abu dhabi': 'Abu Dhabi', 'ABU DHABI': 'Abu Dhabi', 'AbuDhabi': 'Abu Dhabi', 'AD': 'Abu Dhabi',
                   'sharjah': 'Sharjah', 'SHARJAH': 'Sharjah', 'Shj': 'Sharjah', 'SHJ': 'Sharjah'}
        for idx in df.index:
            orig = df.loc[idx, 'city']
            if orig in mapping:
                df.loc[idx, 'city'] = mapping[orig]
                self.log_issue(df.loc[idx, 'store_id'], 'CITY_STANDARDIZED', orig, mapping[orig])
        return df
    
    def clean_sales(self, df, products_df):
        df = df.copy()
        dups = df[df.duplicated(subset=['order_id'], keep='first')]
        for _, r in dups.iterrows(): self.log_issue(r['order_id'], 'DUPLICATE', 'Dup', 'Removed')
        df = df.drop_duplicates(subset=['order_id'], keep='first')
        
        df['order_time_orig'] = df['order_time']
        df['order_time'] = pd.to_datetime(df['order_time'], errors='coerce')
        inv = df['order_time'].isna()
        for idx in df[inv].index: self.log_issue(df.loc[idx, 'order_id'], 'INVALID_TIME', str(df.loc[idx, 'order_time_orig']), 'Removed')
        df = df[~inv].drop(columns=['order_time_orig'])
        
        miss = df['discount_pct'].isna()
        for idx in df[miss].index: self.log_issue(df.loc[idx, 'order_id'], 'MISSING_DISCOUNT', 'NaN', 'Set 0')
        df.loc[miss, 'discount_pct'] = 0
        
        out_q = df['qty'] > 20
        for idx in df[out_q].index: self.log_issue(df.loc[idx, 'order_id'], 'OUTLIER_QTY', str(df.loc[idx, 'qty']), 'Capped 20')
        df.loc[out_q, 'qty'] = 20
        
        cap = df['selling_price_aed'].quantile(0.99) * 2
        out_p = df['selling_price_aed'] > cap
        for idx in df[out_p].index: self.log_issue(df.loc[idx, 'order_id'], 'OUTLIER_PRICE', str(df.loc[idx, 'selling_price_aed']), f'Capped {cap:.0f}')
        df.loc[out_p, 'selling_price_aed'] = cap
        return df
    
    def clean_inventory(self, df):
        df = df.copy()
        neg = df['stock_on_hand'] < 0
        for idx in df[neg].index: self.log_issue(f"{df.loc[idx, 'product_id']}_{df.loc[idx, 'store_id']}", 'NEGATIVE_STOCK', str(df.loc[idx, 'stock_on_hand']), 'Set 0')
        df.loc[neg, 'stock_on_hand'] = 0
        df.loc[df['stock_on_hand'] > 1000, 'stock_on_hand'] = 1000
        return df
    
    def clean_all(self, products_df, stores_df, sales_df, inventory_df, output_dir='data/cleaned'):
        self.issues = []
        os.makedirs(output_dir, exist_ok=True)
        products = self.clean_products(products_df)
        stores = self.clean_stores(stores_df)
        sales = self.clean_sales(sales_df, products)
        inventory = self.clean_inventory(inventory_df)
        issues_df = pd.DataFrame(self.issues)
        products.to_csv(f'{output_dir}/products_cleaned.csv', index=False)
        stores.to_csv(f'{output_dir}/stores_cleaned.csv', index=False)
        sales.to_csv(f'{output_dir}/sales_cleaned.csv', index=False)
        inventory.to_csv(f'{output_dir}/inventory_cleaned.csv', index=False)
        issues_df.to_csv(f'{output_dir}/issues.csv', index=False)
        return {'products': products, 'stores': stores, 'sales': sales, 'inventory': inventory, 'issues': issues_df}

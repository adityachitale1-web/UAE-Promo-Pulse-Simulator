"""
cleaner.py
Data cleaning and validation functions
"""

import pandas as pd
import numpy as np
import os

class DataCleaner:
    """Clean and validate retail data"""
    
    def __init__(self):
        self.issues = []
    
    def log_issue(self, record_id, issue_type, issue_detail, action_taken):
        """Log a data quality issue"""
        self.issues.append({
            'record_id': str(record_id),
            'issue_type': issue_type,
            'issue_detail': issue_detail,
            'action_taken': action_taken
        })
    
    def clean_products(self, df):
        """Clean products table"""
        df = df.copy()
        
        # Fill missing unit_cost with median by category
        for idx in df[df['unit_cost_aed'].isna()].index:
            product_id = df.loc[idx, 'product_id']
            category = df.loc[idx, 'category']
            median_cost = df[df['category'] == category]['unit_cost_aed'].median()
            
            if pd.isna(median_cost):
                median_cost = df['unit_cost_aed'].median()
            
            df.loc[idx, 'unit_cost_aed'] = median_cost
            self.log_issue(product_id, 'MISSING_UNIT_COST', 'unit_cost_aed was NaN', f'Filled with category median: {median_cost:.2f}')
        
        return df
    
    def clean_stores(self, df):
        """Clean stores table - standardize city names"""
        df = df.copy()
        
        city_mapping = {
            'dubai': 'Dubai', 'DUBAI': 'Dubai', 'Dubayy': 'Dubai', 'DXB': 'Dubai',
            'abu dhabi': 'Abu Dhabi', 'ABU DHABI': 'Abu Dhabi', 'AbuDhabi': 'Abu Dhabi', 'AD': 'Abu Dhabi',
            'sharjah': 'Sharjah', 'SHARJAH': 'Sharjah', 'Shj': 'Sharjah', 'SHJ': 'Sharjah'
        }
        
        for idx in df.index:
            original_city = df.loc[idx, 'city']
            if original_city in city_mapping:
                df.loc[idx, 'city'] = city_mapping[original_city]
                self.log_issue(df.loc[idx, 'store_id'], 'INCONSISTENT_CITY', f'City was: {original_city}', f'Standardized to: {city_mapping[original_city]}')
        
        return df
    
    def clean_sales(self, df, products_df):
        """Clean sales table"""
        df = df.copy()
        original_len = len(df)
        
        # Remove duplicates
        duplicates = df[df.duplicated(subset=['order_id'], keep='first')]
        for _, row in duplicates.iterrows():
            self.log_issue(row['order_id'], 'DUPLICATE_ORDER_ID', 'Duplicate order_id found', 'Removed duplicate')
        df = df.drop_duplicates(subset=['order_id'], keep='first')
        
        # Fix timestamps
        df['order_time_original'] = df['order_time']
        df['order_time'] = pd.to_datetime(df['order_time'], errors='coerce')
        
        invalid_time_mask = df['order_time'].isna()
        for idx in df[invalid_time_mask].index:
            self.log_issue(df.loc[idx, 'order_id'], 'INVALID_TIMESTAMP', f"Invalid: {df.loc[idx, 'order_time_original']}", 'Record removed')
        
        df = df[~invalid_time_mask].copy()
        df = df.drop(columns=['order_time_original'])
        
        # Fill missing discount_pct
        missing_discount_mask = df['discount_pct'].isna()
        for idx in df[missing_discount_mask].index:
            self.log_issue(df.loc[idx, 'order_id'], 'MISSING_DISCOUNT', 'discount_pct was NaN', 'Filled with 0')
        df.loc[missing_discount_mask, 'discount_pct'] = 0
        
        # Cap outlier quantities
        QTY_MAX = 20
        outlier_qty_mask = df['qty'] > QTY_MAX
        for idx in df[outlier_qty_mask].index:
            old_qty = df.loc[idx, 'qty']
            self.log_issue(df.loc[idx, 'order_id'], 'OUTLIER_QTY', f'Extreme quantity: {old_qty}', f'Capped at {QTY_MAX}')
        df.loc[outlier_qty_mask, 'qty'] = QTY_MAX
        
        # Cap outlier prices (> 99th percentile * 2)
        price_threshold = df['selling_price_aed'].quantile(0.99) * 2
        outlier_price_mask = df['selling_price_aed'] > price_threshold
        for idx in df[outlier_price_mask].index:
            old_price = df.loc[idx, 'selling_price_aed']
            self.log_issue(df.loc[idx, 'order_id'], 'OUTLIER_PRICE', f'Extreme price: {old_price}', f'Capped at {price_threshold:.2f}')
        df.loc[outlier_price_mask, 'selling_price_aed'] = price_threshold
        
        return df
    
    def clean_inventory(self, df):
        """Clean inventory table"""
        df = df.copy()
        
        # Fix negative stock
        negative_mask = df['stock_on_hand'] < 0
        for idx in df[negative_mask].index:
            self.log_issue(f"{df.loc[idx, 'product_id']}_{df.loc[idx, 'store_id']}", 'NEGATIVE_STOCK', f"Stock: {df.loc[idx, 'stock_on_hand']}", 'Set to 0')
        df.loc[negative_mask, 'stock_on_hand'] = 0
        
        # Cap extreme stock
        STOCK_MAX = 1000
        extreme_mask = df['stock_on_hand'] > STOCK_MAX
        for idx in df[extreme_mask].index:
            self.log_issue(f"{df.loc[idx, 'product_id']}_{df.loc[idx, 'store_id']}", 'EXTREME_STOCK', f"Stock: {df.loc[idx, 'stock_on_hand']}", f'Capped at {STOCK_MAX}')
        df.loc[extreme_mask, 'stock_on_hand'] = STOCK_MAX
        
        return df
    
    def clean_all(self, products_df, stores_df, sales_df, inventory_df, output_dir='data/cleaned'):
        """Clean all tables and save"""
        self.issues = []
        
        os.makedirs(output_dir, exist_ok=True)
        
        products_clean = self.clean_products(products_df)
        stores_clean = self.clean_stores(stores_df)
        sales_clean = self.clean_sales(sales_df, products_clean)
        inventory_clean = self.clean_inventory(inventory_df)
        
        issues_df = pd.DataFrame(self.issues)
        
        # Save
        products_clean.to_csv(f'{output_dir}/products_cleaned.csv', index=False)
        stores_clean.to_csv(f'{output_dir}/stores_cleaned.csv', index=False)
        sales_clean.to_csv(f'{output_dir}/sales_cleaned.csv', index=False)
        inventory_clean.to_csv(f'{output_dir}/inventory_cleaned.csv', index=False)
        issues_df.to_csv(f'{output_dir}/issues.csv', index=False)
        
        return {
            'products': products_clean,
            'stores': stores_clean,
            'sales': sales_clean,
            'inventory': inventory_clean,
            'issues': issues_df
        }

"""
cleaner.py
UAE Promo Pulse Simulator - Data Validation and Cleaning Module
Validates, cleans datasets and produces issues log
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

# ============================================================================
# CONFIGURATION - VALID VALUES
# ============================================================================

VALID_CITIES = ['Dubai', 'Abu Dhabi', 'Sharjah']
VALID_CHANNELS = ['App', 'Web', 'Marketplace']
VALID_CATEGORIES = ['Electronics', 'Fashion', 'Grocery', 'Home & Garden', 'Beauty', 'Sports']
VALID_PAYMENT_STATUSES = ['Paid', 'Failed', 'Refunded']
VALID_FULFILLMENT_TYPES = ['Own', '3PL']
VALID_LAUNCH_FLAGS = ['New', 'Regular']

# City name mapping for standardization
CITY_MAPPING = {
    'dubai': 'Dubai', 'DUBAI': 'Dubai', 'dubayy': 'Dubai', 'Dubayy': 'Dubai', 'DXB': 'Dubai', 'dxb': 'Dubai',
    'abu dhabi': 'Abu Dhabi', 'ABU DHABI': 'Abu Dhabi', 'abudhabi': 'Abu Dhabi', 'AbuDhabi': 'Abu Dhabi', 'AD': 'Abu Dhabi', 'ad': 'Abu Dhabi',
    'sharjah': 'Sharjah', 'SHARJAH': 'Sharjah', 'shj': 'Sharjah', 'Shj': 'Sharjah', 'SHJ': 'Sharjah'
}

# Thresholds for outlier detection
QTY_MAX_THRESHOLD = 20
PRICE_MULTIPLIER_THRESHOLD = 5
STOCK_MAX_THRESHOLD = 5000


class DataCleaner:
    """Main class for data validation and cleaning"""
    
    def __init__(self):
        self.issues_log = []
    
    def log_issue(self, record_id, issue_type, issue_detail, action_taken):
        """Add an issue to the log"""
        self.issues_log.append({
            'record_id': str(record_id),
            'issue_type': issue_type,
            'issue_detail': issue_detail,
            'action_taken': action_taken,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    def get_issues_df(self):
        """Return issues log as DataFrame"""
        return pd.DataFrame(self.issues_log)
    
    def save_issues_log(self, filepath='data/cleaned/issues.csv'):
        """Save issues log to CSV"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df = self.get_issues_df()
        df.to_csv(filepath, index=False)
        return df
    
    # ========================================================================
    # PRODUCTS CLEANING
    # ========================================================================
    
    def clean_products(self, df):
        """Clean products table"""
        df = df.copy()
        original_len = len(df)
        
        # 1. Check for missing unit_cost_aed and impute with category median margin
        missing_cost_mask = df['unit_cost_aed'].isna()
        if missing_cost_mask.any():
            for idx in df[missing_cost_mask].index:
                product_id = df.loc[idx, 'product_id']
                category = df.loc[idx, 'category']
                base_price = df.loc[idx, 'base_price_aed']
                
                # Calculate median margin for category
                category_products = df[(df['category'] == category) & (~df['unit_cost_aed'].isna())]
                if len(category_products) > 0:
                    median_margin = (category_products['unit_cost_aed'] / category_products['base_price_aed']).median()
                else:
                    median_margin = 0.55  # Default margin
                
                imputed_cost = round(base_price * median_margin, 2)
                df.loc[idx, 'unit_cost_aed'] = imputed_cost
                
                self.log_issue(
                    product_id, 
                    'MISSING_UNIT_COST',
                    f'unit_cost_aed was NULL for {category} product',
                    f'Imputed with category median margin: {imputed_cost} AED'
                )
        
        # 2. Validate unit_cost <= base_price
        invalid_cost_mask = df['unit_cost_aed'] > df['base_price_aed']
        if invalid_cost_mask.any():
            for idx in df[invalid_cost_mask].index:
                product_id = df.loc[idx, 'product_id']
                old_cost = df.loc[idx, 'unit_cost_aed']
                base_price = df.loc[idx, 'base_price_aed']
                new_cost = round(base_price * 0.6, 2)
                df.loc[idx, 'unit_cost_aed'] = new_cost
                
                self.log_issue(
                    product_id,
                    'INVALID_UNIT_COST',
                    f'unit_cost ({old_cost}) > base_price ({base_price})',
                    f'Corrected to 60% of base_price: {new_cost} AED'
                )
        
        # 3. Validate category values
        invalid_category_mask = ~df['category'].isin(VALID_CATEGORIES)
        if invalid_category_mask.any():
            for idx in df[invalid_category_mask].index:
                product_id = df.loc[idx, 'product_id']
                old_category = df.loc[idx, 'category']
                df.loc[idx, 'category'] = 'Other'
                
                self.log_issue(
                    product_id,
                    'INVALID_CATEGORY',
                    f'Invalid category: {old_category}',
                    'Set to "Other"'
                )
        
        # 4. Validate launch_flag
        invalid_launch_mask = ~df['launch_flag'].isin(VALID_LAUNCH_FLAGS)
        if invalid_launch_mask.any():
            for idx in df[invalid_launch_mask].index:
                product_id = df.loc[idx, 'product_id']
                old_flag = df.loc[idx, 'launch_flag']
                df.loc[idx, 'launch_flag'] = 'Regular'
                
                self.log_issue(
                    product_id,
                    'INVALID_LAUNCH_FLAG',
                    f'Invalid launch_flag: {old_flag}',
                    'Set to "Regular"'
                )
        
        print(f"  Products: {original_len} → {len(df)} rows")
        return df
    
    # ========================================================================
    # STORES CLEANING
    # ========================================================================
    
    def clean_stores(self, df):
        """Clean stores table"""
        df = df.copy()
        original_len = len(df)
        
        # 1. Standardize city names
        for idx in df.index:
            city = df.loc[idx, 'city']
            store_id = df.loc[idx, 'store_id']
            
            if city not in VALID_CITIES:
                if city in CITY_MAPPING:
                    new_city = CITY_MAPPING[city]
                    df.loc[idx, 'city'] = new_city
                    self.log_issue(
                        store_id,
                        'INCONSISTENT_CITY',
                        f'Non-standard city name: "{city}"',
                        f'Corrected to "{new_city}"'
                    )
                else:
                    # Try case-insensitive match
                    city_lower = city.lower().strip()
                    matched = False
                    for valid_city in VALID_CITIES:
                        if city_lower == valid_city.lower():
                            df.loc[idx, 'city'] = valid_city
                            self.log_issue(
                                store_id,
                                'INCONSISTENT_CITY',
                                f'Non-standard city name: "{city}"',
                                f'Corrected to "{valid_city}"'
                            )
                            matched = True
                            break
                    
                    if not matched:
                        df.loc[idx, 'city'] = 'Dubai'  # Default
                        self.log_issue(
                            store_id,
                            'INVALID_CITY',
                            f'Unknown city: "{city}"',
                            'Set to default "Dubai"'
                        )
        
        # 2. Validate channel
        invalid_channel_mask = ~df['channel'].isin(VALID_CHANNELS)
        if invalid_channel_mask.any():
            for idx in df[invalid_channel_mask].index:
                store_id = df.loc[idx, 'store_id']
                old_channel = df.loc[idx, 'channel']
                df.loc[idx, 'channel'] = 'Web'
                
                self.log_issue(
                    store_id,
                    'INVALID_CHANNEL',
                    f'Invalid channel: {old_channel}',
                    'Set to "Web"'
                )
        
        # 3. Validate fulfillment_type
        invalid_fulfillment_mask = ~df['fulfillment_type'].isin(VALID_FULFILLMENT_TYPES)
        if invalid_fulfillment_mask.any():
            for idx in df[invalid_fulfillment_mask].index:
                store_id = df.loc[idx, 'store_id']
                old_type = df.loc[idx, 'fulfillment_type']
                df.loc[idx, 'fulfillment_type'] = 'Own'
                
                self.log_issue(
                    store_id,
                    'INVALID_FULFILLMENT',
                    f'Invalid fulfillment_type: {old_type}',
                    'Set to "Own"'
                )
        
        print(f"  Stores: {original_len} → {len(df)} rows")
        return df
    
    # ========================================================================
    # SALES CLEANING
    # ========================================================================
    
    def clean_sales(self, df, products_df):
        """Clean sales_raw table"""
        df = df.copy()
        original_len = len(df)
        
        # Get product prices for validation
        product_prices = products_df.set_index('product_id')['base_price_aed'].to_dict()
        
        # 1. Handle duplicate order_ids - keep first occurrence
        duplicate_mask = df.duplicated(subset=['order_id'], keep='first')
        if duplicate_mask.any():
            duplicate_ids = df[duplicate_mask]['order_id'].unique()
            for order_id in duplicate_ids:
                self.log_issue(
                    order_id,
                    'DUPLICATE_ORDER_ID',
                    'Duplicate order_id found',
                    'Kept first occurrence, dropped duplicate'
                )
            df = df[~duplicate_mask]
        
        # 2. Parse and validate timestamps
        def parse_timestamp(ts):
            if pd.isna(ts) or ts in ['', 'NULL', 'TBD', 'not_a_time', 'invalid_date']:
                return None
            try:
                # Try standard format first
                return pd.to_datetime(ts, format='%Y-%m-%d %H:%M:%S')
            except:
                try:
                    # Try other formats
                    return pd.to_datetime(ts, infer_datetime_format=True)
                except:
                    return None
        
        df['order_time_parsed'] = df['order_time'].apply(parse_timestamp)
        
        invalid_time_mask = df['order_time_parsed'].isna()
        if invalid_time_mask.any():
            for idx in df[invalid_time_mask].index:
                order_id = df.loc[idx, 'order_id']
                old_time = df.loc[idx, 'order_time']
                self.log_issue(
                    order_id,
                    'INVALID_TIMESTAMP',
                    f'Unparseable timestamp: "{old_time}"',
                    'Record dropped'
                )
            df = df[~invalid_time_mask]
        
        df['order_time'] = df['order_time_parsed']
        df = df.drop(columns=['order_time_parsed'])
        
        # 3. Handle missing discount_pct - impute with 0
        missing_discount_mask = df['discount_pct'].isna()
        if missing_discount_mask.any():
            for idx in df[missing_discount_mask].index:
                order_id = df.loc[idx, 'order_id']
                self.log_issue(
                    order_id,
                    'MISSING_DISCOUNT',
                    'discount_pct was NULL',
                    'Imputed with 0'
                )
            df.loc[missing_discount_mask, 'discount_pct'] = 0
        
        # 4. Handle outlier quantities - cap at threshold
        outlier_qty_mask = df['qty'] > QTY_MAX_THRESHOLD
        if outlier_qty_mask.any():
            for idx in df[outlier_qty_mask].index:
                order_id = df.loc[idx, 'order_id']
                old_qty = df.loc[idx, 'qty']
                self.log_issue(
                    order_id,
                    'OUTLIER_QTY',
                    f'Extreme quantity: {old_qty}',
                    f'Capped at {QTY_MAX_THRESHOLD}'
                )
            df.loc[outlier_qty_mask, 'qty'] = QTY_MAX_THRESHOLD
        
        # 5. Handle outlier prices - cap at threshold multiplier of base price
        for idx in df.index:
            product_id = df.loc[idx, 'product_id']
            selling_price = df.loc[idx, 'selling_price_aed']
            base_price = product_prices.get(product_id, selling_price)
            
            if selling_price > base_price * PRICE_MULTIPLIER_THRESHOLD:
                order_id = df.loc[idx, 'order_id']
                self.log_issue(
                    order_id,
                    'OUTLIER_PRICE',
                    f'Extreme price: {selling_price} AED (base: {base_price})',
                    f'Capped at base_price: {base_price}'
                )
                df.loc[idx, 'selling_price_aed'] = base_price
        
        # 6. Validate payment_status
        invalid_payment_mask = ~df['payment_status'].isin(VALID_PAYMENT_STATUSES)
        if invalid_payment_mask.any():
            for idx in df[invalid_payment_mask].index:
                order_id = df.loc[idx, 'order_id']
                old_status = df.loc[idx, 'payment_status']
                df.loc[idx, 'payment_status'] = 'Paid'
                
                self.log_issue(
                    order_id,
                    'INVALID_PAYMENT_STATUS',
                    f'Invalid payment_status: {old_status}',
                    'Set to "Paid"'
                )
        
        # 7. Validate product_id exists
        valid_products = set(products_df['product_id'])
        invalid_product_mask = ~df['product_id'].isin(valid_products)
        if invalid_product_mask.any():
            for idx in df[invalid_product_mask].index:
                order_id = df.loc[idx, 'order_id']
                product_id = df.loc[idx, 'product_id']
                self.log_issue(
                    order_id,
                    'INVALID_PRODUCT_ID',
                    f'product_id not found: {product_id}',
                    'Record dropped'
                )
            df = df[~invalid_product_mask]
        
        # 8. Ensure qty and prices are positive
        invalid_qty_mask = df['qty'] <= 0
        if invalid_qty_mask.any():
            for idx in df[invalid_qty_mask].index:
                order_id = df.loc[idx, 'order_id']
                old_qty = df.loc[idx, 'qty']
                df.loc[idx, 'qty'] = 1
                self.log_issue(
                    order_id,
                    'INVALID_QTY',
                    f'Non-positive quantity: {old_qty}',
                    'Set to 1'
                )
        
        invalid_price_mask = df['selling_price_aed'] <= 0
        if invalid_price_mask.any():
            for idx in df[invalid_price_mask].index:
                order_id = df.loc[idx, 'order_id']
                product_id = df.loc[idx, 'product_id']
                old_price = df.loc[idx, 'selling_price_aed']
                base_price = product_prices.get(product_id, 100)
                df.loc[idx, 'selling_price_aed'] = base_price
                self.log_issue(
                    order_id,
                    'INVALID_PRICE',
                    f'Non-positive price: {old_price}',
                    f'Set to base_price: {base_price}'
                )
        
        print(f"  Sales: {original_len} → {len(df)} rows")
        return df
    
    # ========================================================================
    # INVENTORY CLEANING
    # ========================================================================
    
    def clean_inventory(self, df, products_df, stores_df):
        """Clean inventory_snapshot table"""
        df = df.copy()
        original_len = len(df)
        
        valid_products = set(products_df['product_id'])
        valid_stores = set(stores_df['store_id'])
        
        # 1. Handle negative stock_on_hand - set to 0
        negative_stock_mask = df['stock_on_hand'] < 0
        if negative_stock_mask.any():
            for idx in df[negative_stock_mask].index:
                record_id = f"{df.loc[idx, 'product_id']}_{df.loc[idx, 'store_id']}_{df.loc[idx, 'snapshot_date']}"
                old_stock = df.loc[idx, 'stock_on_hand']
                self.log_issue(
                    record_id,
                    'NEGATIVE_STOCK',
                    f'Negative stock_on_hand: {old_stock}',
                    'Set to 0'
                )
            df.loc[negative_stock_mask, 'stock_on_hand'] = 0
        
        # 2. Handle extreme stock (9999) - cap at threshold
        extreme_stock_mask = df['stock_on_hand'] > STOCK_MAX_THRESHOLD
        if extreme_stock_mask.any():
            for idx in df[extreme_stock_mask].index:
                record_id = f"{df.loc[idx, 'product_id']}_{df.loc[idx, 'store_id']}_{df.loc[idx, 'snapshot_date']}"
                old_stock = df.loc[idx, 'stock_on_hand']
                self.log_issue(
                    record_id,
                    'EXTREME_STOCK',
                    f'Unrealistic stock_on_hand: {old_stock}',
                    f'Capped at {STOCK_MAX_THRESHOLD}'
                )
            df.loc[extreme_stock_mask, 'stock_on_hand'] = STOCK_MAX_THRESHOLD
        
        # 3. Validate product_id
        invalid_product_mask = ~df['product_id'].isin(valid_products)
        if invalid_product_mask.any():
            count = invalid_product_mask.sum()
            self.log_issue(
                f'BATCH_{count}_records',
                'INVALID_PRODUCT_ID',
                f'{count} inventory records with invalid product_id',
                'Records dropped'
            )
            df = df[~invalid_product_mask]
        
        # 4. Validate store_id
        invalid_store_mask = ~df['store_id'].isin(valid_stores)
        if invalid_store_mask.any():
            count = invalid_store_mask.sum()
            self.log_issue(
                f'BATCH_{count}_records',
                'INVALID_STORE_ID',
                f'{count} inventory records with invalid store_id',
                'Records dropped'
            )
            df = df[~invalid_store_mask]
        
        # 5. Parse snapshot_date
        df['snapshot_date'] = pd.to_datetime(df['snapshot_date'], errors='coerce')
        invalid_date_mask = df['snapshot_date'].isna()
        if invalid_date_mask.any():
            count = invalid_date_mask.sum()
            self.log_issue(
                f'BATCH_{count}_records',
                'INVALID_DATE',
                f'{count} inventory records with invalid snapshot_date',
                'Records dropped'
            )
            df = df[~invalid_date_mask]
        
        print(f"  Inventory: {original_len} → {len(df)} rows")
        return df
    
    # ========================================================================
    # MAIN CLEANING FUNCTION
    # ========================================================================
    
    def clean_all(self, products_df, stores_df, sales_df, inventory_df, output_dir='data/cleaned'):
        """Clean all datasets and save results"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "=" * 60)
        print("DATA CLEANING PROCESS")
        print("=" * 60)
        
        print("\n[1/4] Cleaning Products...")
        products_clean = self.clean_products(products_df)
        
        print("\n[2/4] Cleaning Stores...")
        stores_clean = self.clean_stores(stores_df)
        
        print("\n[3/4] Cleaning Sales...")
        sales_clean = self.clean_sales(sales_df, products_clean)
        
        print("\n[4/4] Cleaning Inventory...")
        inventory_clean = self.clean_inventory(inventory_df, products_clean, stores_clean)
        
        # Save cleaned data
        products_clean.to_csv(f'{output_dir}/products_cleaned.csv', index=False)
        stores_clean.to_csv(f'{output_dir}/stores_cleaned.csv', index=False)
        sales_clean.to_csv(f'{output_dir}/sales_cleaned.csv', index=False)
        inventory_clean.to_csv(f'{output_dir}/inventory_cleaned.csv', index=False)
        
        # Save issues log
        issues_df = self.save_issues_log(f'{output_dir}/issues.csv')
        
        print("\n" + "=" * 60)
        print("CLEANING SUMMARY")
        print("=" * 60)
        print(f"\nTotal issues logged: {len(self.issues_log)}")
        
        if len(issues_df) > 0:
            print("\nIssues by type:")
            print(issues_df['issue_type'].value_counts().to_string())
        
        print(f"\n✓ Cleaned files saved to '{output_dir}/'")
        
        return {
            'products': products_clean,
            'stores': stores_clean,
            'sales': sales_clean,
            'inventory': inventory_clean,
            'issues': issues_df
        }


def load_and_clean_data(raw_dir='data/raw', output_dir='data/cleaned'):
    """Load raw data, clean it, and save results"""
    
    print("Loading raw data...")
    products_df = pd.read_csv(f'{raw_dir}/products_raw.csv')
    stores_df = pd.read_csv(f'{raw_dir}/stores_raw.csv')
    sales_df = pd.read_csv(f'{raw_dir}/sales_raw.csv')
    inventory_df = pd.read_csv(f'{raw_dir}/inventory_snapshot_raw.csv')
    
    cleaner = DataCleaner()
    return cleaner.clean_all(products_df, stores_df, sales_df, inventory_df, output_dir)


if __name__ == "__main__":
    load_and_clean_data()

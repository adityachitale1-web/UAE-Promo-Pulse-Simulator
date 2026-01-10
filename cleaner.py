"""
UAE Promo Pulse - Data Cleaning & Validation Module
====================================================
Implements comprehensive data validation and cleaning with full issue logging.

Validation Rules:
1. Timestamp parsable
2. City/Channel/Category are valid
3. Price and qty in reasonable range
4. unit_cost <= base_price
5. payment_status in allowed set
6. stock_on_hand non-negative

Cleaning Policies:
- Duplicates: Keep first occurrence
- Missing discount: Set to 0 (most conservative)
- Missing unit_cost: Fill with category median
- Outlier qty: Cap at 20 units
- Outlier prices: Cap at 99th percentile × 2
- Invalid timestamps: Drop records
- Invalid categorical values: Standardize to valid values
- Negative inventory: Set to 0
"""

import pandas as pd
import numpy as np
import os

class DataCleaner:
    """
    Comprehensive data cleaner with validation and issue logging.
    """
    
    # Valid values for categorical fields
    VALID_CITIES = ['Dubai', 'Abu Dhabi', 'Sharjah']
    VALID_CHANNELS = ['App', 'Web', 'Marketplace']
    VALID_CATEGORIES = ['Electronics', 'Fashion', 'Grocery', 'Home & Garden', 'Beauty', 'Sports']
    VALID_PAYMENT_STATUS = ['Paid', 'Failed', 'Refunded']
    VALID_FULFILLMENT = ['Own', '3PL']
    
    # City standardization mapping
    CITY_MAPPING = {
        'dubai': 'Dubai', 'DUBAI': 'Dubai', 'DXB': 'Dubai', 'Dubayy': 'Dubai',
        'abu dhabi': 'Abu Dhabi', 'ABU DHABI': 'Abu Dhabi', 'AbuDhabi': 'Abu Dhabi', 
        'AD': 'Abu Dhabi', 'Abu-Dhabi': 'Abu Dhabi', 'abudhabi': 'Abu Dhabi',
        'sharjah': 'Sharjah', 'SHARJAH': 'Sharjah', 'Shj': 'Sharjah', 
        'SHJ': 'Sharjah', 'Sharjha': 'Sharjah'
    }
    
    def __init__(self):
        self.issues = []
        self.stats = {
            'products': {'original': 0, 'cleaned': 0, 'issues': 0},
            'stores': {'original': 0, 'cleaned': 0, 'issues': 0},
            'sales': {'original': 0, 'cleaned': 0, 'issues': 0},
            'inventory': {'original': 0, 'cleaned': 0, 'issues': 0}
        }
    
    def log_issue(self, record_id, issue_type, issue_detail, action_taken, table='sales'):
        """Log a data quality issue."""
        self.issues.append({
            'record_id': str(record_id),
            'table': table,
            'issue_type': issue_type,
            'issue_detail': str(issue_detail)[:200],
            'action_taken': action_taken,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    def clean_products(self, df):
        """
        Clean products table.
        
        Validations:
        - unit_cost_aed not null
        - unit_cost_aed <= base_price_aed
        - category in valid list
        """
        df = df.copy()
        self.stats['products']['original'] = len(df)
        
        # 1. Check for missing unit_cost_aed
        missing_cost = df['unit_cost_aed'].isna()
        for idx in df[missing_cost].index:
            cat = df.loc[idx, 'category']
            # Fill with category median
            cat_median = df[df['category'] == cat]['unit_cost_aed'].median()
            if pd.isna(cat_median):
                cat_median = df['unit_cost_aed'].median()
            df.loc[idx, 'unit_cost_aed'] = cat_median
            self.log_issue(
                df.loc[idx, 'product_id'], 'MISSING_UNIT_COST',
                f'unit_cost_aed was NULL', f'Imputed with category median: {cat_median:.2f}',
                'products'
            )
        
        # 2. Check unit_cost <= base_price
        invalid_cost = df['unit_cost_aed'] > df['base_price_aed']
        for idx in df[invalid_cost].index:
            old_cost = df.loc[idx, 'unit_cost_aed']
            new_cost = df.loc[idx, 'base_price_aed'] * 0.6  # Set to 60% of base
            df.loc[idx, 'unit_cost_aed'] = new_cost
            self.log_issue(
                df.loc[idx, 'product_id'], 'INVALID_UNIT_COST',
                f'unit_cost ({old_cost}) > base_price ({df.loc[idx, "base_price_aed"]})',
                f'Corrected to 60% of base_price: {new_cost:.2f}',
                'products'
            )
        
        # 3. Validate category
        invalid_cat = ~df['category'].isin(self.VALID_CATEGORIES)
        for idx in df[invalid_cat].index:
            self.log_issue(
                df.loc[idx, 'product_id'], 'INVALID_CATEGORY',
                f'Invalid category: {df.loc[idx, "category"]}',
                'Set to most common category',
                'products'
            )
            df.loc[idx, 'category'] = df['category'].mode()[0]
        
        self.stats['products']['cleaned'] = len(df)
        self.stats['products']['issues'] = len([i for i in self.issues if i['table'] == 'products'])
        
        return df
    
    def clean_stores(self, df):
        """
        Clean stores table.
        
        Validations:
        - city standardization
        - channel in valid list
        - fulfillment_type in valid list
        """
        df = df.copy()
        self.stats['stores']['original'] = len(df)
        
        # 1. Standardize city names
        for idx in df.index:
            orig_city = df.loc[idx, 'city']
            if orig_city in self.CITY_MAPPING:
                df.loc[idx, 'city'] = self.CITY_MAPPING[orig_city]
                self.log_issue(
                    df.loc[idx, 'store_id'], 'CITY_STANDARDIZED',
                    f'Non-standard city name: {orig_city}',
                    f'Corrected to: {self.CITY_MAPPING[orig_city]}',
                    'stores'
                )
            elif orig_city not in self.VALID_CITIES:
                # Try case-insensitive match
                matched = False
                for valid_city in self.VALID_CITIES:
                    if orig_city.lower().replace(' ', '').replace('-', '') in valid_city.lower().replace(' ', ''):
                        df.loc[idx, 'city'] = valid_city
                        self.log_issue(
                            df.loc[idx, 'store_id'], 'CITY_STANDARDIZED',
                            f'Fuzzy matched city: {orig_city}',
                            f'Corrected to: {valid_city}',
                            'stores'
                        )
                        matched = True
                        break
                if not matched:
                    df.loc[idx, 'city'] = 'Dubai'  # Default
                    self.log_issue(
                        df.loc[idx, 'store_id'], 'INVALID_CITY',
                        f'Unknown city: {orig_city}',
                        'Defaulted to: Dubai',
                        'stores'
                    )
        
        # 2. Validate channel
        invalid_channel = ~df['channel'].isin(self.VALID_CHANNELS)
        for idx in df[invalid_channel].index:
            self.log_issue(
                df.loc[idx, 'store_id'], 'INVALID_CHANNEL',
                f'Invalid channel: {df.loc[idx, "channel"]}',
                'Kept as-is (may need review)',
                'stores'
            )
        
        # 3. Validate fulfillment_type
        if 'fulfillment_type' in df.columns:
            invalid_fulfill = ~df['fulfillment_type'].isin(self.VALID_FULFILLMENT)
            for idx in df[invalid_fulfill].index:
                df.loc[idx, 'fulfillment_type'] = 'Own'
                self.log_issue(
                    df.loc[idx, 'store_id'], 'INVALID_FULFILLMENT',
                    f'Invalid fulfillment: {df.loc[idx, "fulfillment_type"]}',
                    'Defaulted to: Own',
                    'stores'
                )
        
        self.stats['stores']['cleaned'] = len(df)
        self.stats['stores']['issues'] = len([i for i in self.issues if i['table'] == 'stores'])
        
        return df
    
    def clean_sales(self, df, products_df):
        """
        Clean sales table.
        
        Validations:
        - Duplicate order_id
        - Timestamp parsable
        - qty in reasonable range (1-20)
        - selling_price in reasonable range
        - payment_status in valid set
        - discount_pct not null and in range (0-100)
        """
        df = df.copy()
        self.stats['sales']['original'] = len(df)
        
        # 1. Remove duplicates (keep first)
        duplicates = df[df.duplicated(subset=['order_id'], keep='first')]
        for _, row in duplicates.iterrows():
            self.log_issue(
                row['order_id'], 'DUPLICATE_ORDER_ID',
                f'Duplicate order found',
                'Dropped (kept first occurrence)',
                'sales'
            )
        df = df.drop_duplicates(subset=['order_id'], keep='first')
        
        # 2. Parse and validate timestamps
        df['order_time_orig'] = df['order_time']
        df['order_time'] = pd.to_datetime(df['order_time'], errors='coerce')
        
        invalid_time = df['order_time'].isna()
        for idx in df[invalid_time].index:
            self.log_issue(
                df.loc[idx, 'order_id'], 'INVALID_TIMESTAMP',
                f'Unparseable timestamp: {df.loc[idx, "order_time_orig"]}',
                'Dropped record',
                'sales'
            )
        df = df[~invalid_time].drop(columns=['order_time_orig'])
        
        # 3. Handle missing discount_pct
        missing_disc = df['discount_pct'].isna()
        df.loc[missing_disc, 'discount_pct'] = 0
        for idx in df[missing_disc].index:
            self.log_issue(
                df.loc[idx, 'order_id'], 'MISSING_DISCOUNT',
                'discount_pct was NULL',
                'Imputed with 0 (conservative)',
                'sales'
            )
        
        # 4. Validate discount range (0-100)
        invalid_disc = (df['discount_pct'] < 0) | (df['discount_pct'] > 100)
        for idx in df[invalid_disc].index:
            old_disc = df.loc[idx, 'discount_pct']
            df.loc[idx, 'discount_pct'] = np.clip(old_disc, 0, 100)
            self.log_issue(
                df.loc[idx, 'order_id'], 'INVALID_DISCOUNT',
                f'discount_pct out of range: {old_disc}',
                f'Capped to [0, 100]: {df.loc[idx, "discount_pct"]}',
                'sales'
            )
        
        # 5. Cap outlier qty (max 20)
        outlier_qty = df['qty'] > 20
        for idx in df[outlier_qty].index:
            old_qty = df.loc[idx, 'qty']
            df.loc[idx, 'qty'] = 20
            self.log_issue(
                df.loc[idx, 'order_id'], 'OUTLIER_QTY',
                f'Unusually high qty: {old_qty}',
                'Capped to 20 units',
                'sales'
            )
        
        # Also handle qty < 1
        low_qty = df['qty'] < 1
        for idx in df[low_qty].index:
            old_qty = df.loc[idx, 'qty']
            df.loc[idx, 'qty'] = 1
            self.log_issue(
                df.loc[idx, 'order_id'], 'INVALID_QTY',
                f'Invalid qty: {old_qty}',
                'Set to minimum: 1',
                'sales'
            )
        
        # 6. Cap outlier prices
        price_cap = df['selling_price_aed'].quantile(0.99) * 2
        outlier_price = df['selling_price_aed'] > price_cap
        for idx in df[outlier_price].index:
            old_price = df.loc[idx, 'selling_price_aed']
            df.loc[idx, 'selling_price_aed'] = price_cap
            self.log_issue(
                df.loc[idx, 'order_id'], 'OUTLIER_PRICE',
                f'Unusually high price: {old_price:.2f}',
                f'Capped to 99th percentile × 2: {price_cap:.2f}',
                'sales'
            )
        
        # Handle negative prices
        neg_price = df['selling_price_aed'] <= 0
        for idx in df[neg_price].index:
            # Get base price from products
            pid = df.loc[idx, 'product_id']
            base = products_df[products_df['product_id'] == pid]['base_price_aed'].values
            new_price = base[0] if len(base) > 0 else 100
            df.loc[idx, 'selling_price_aed'] = new_price
            self.log_issue(
                df.loc[idx, 'order_id'], 'INVALID_PRICE',
                f'Non-positive price: {df.loc[idx, "selling_price_aed"]}',
                f'Set to base price: {new_price:.2f}',
                'sales'
            )
        
        # 7. Validate payment_status
        invalid_payment = ~df['payment_status'].isin(self.VALID_PAYMENT_STATUS)
        for idx in df[invalid_payment].index:
            old_status = df.loc[idx, 'payment_status']
            df.loc[idx, 'payment_status'] = 'Paid'  # Default to most common
            self.log_issue(
                df.loc[idx, 'order_id'], 'INVALID_PAYMENT_STATUS',
                f'Invalid payment_status: {old_status}',
                'Defaulted to: Paid',
                'sales'
            )
        
        # 8. Validate product_id exists
        valid_products = set(products_df['product_id'].tolist())
        invalid_product = ~df['product_id'].isin(valid_products)
        if invalid_product.any():
            for idx in df[invalid_product].index:
                self.log_issue(
                    df.loc[idx, 'order_id'], 'INVALID_PRODUCT_ID',
                    f'Product not found: {df.loc[idx, "product_id"]}',
                    'Dropped record',
                    'sales'
                )
            df = df[~invalid_product]
        
        self.stats['sales']['cleaned'] = len(df)
        self.stats['sales']['issues'] = len([i for i in self.issues if i['table'] == 'sales'])
        
        return df
    
    def clean_inventory(self, df):
        """
        Clean inventory table.
        
        Validations:
        - stock_on_hand non-negative
        - stock_on_hand not extreme (< 10000)
        - snapshot_date parsable
        """
        df = df.copy()
        self.stats['inventory']['original'] = len(df)
        
        # 1. Handle negative stock
        neg_stock = df['stock_on_hand'] < 0
        for idx in df[neg_stock].index:
            old_stock = df.loc[idx, 'stock_on_hand']
            df.loc[idx, 'stock_on_hand'] = 0
            self.log_issue(
                f"{df.loc[idx, 'product_id']}_{df.loc[idx, 'store_id']}_{df.loc[idx, 'snapshot_date']}",
                'NEGATIVE_STOCK',
                f'Negative stock_on_hand: {old_stock}',
                'Set to 0',
                'inventory'
            )
        
        # 2. Handle extreme stock (cap at 1000)
        extreme_stock = df['stock_on_hand'] > 1000
        for idx in df[extreme_stock].index:
            old_stock = df.loc[idx, 'stock_on_hand']
            df.loc[idx, 'stock_on_hand'] = 1000
            self.log_issue(
                f"{df.loc[idx, 'product_id']}_{df.loc[idx, 'store_id']}_{df.loc[idx, 'snapshot_date']}",
                'EXTREME_STOCK',
                f'Unusually high stock: {old_stock}',
                'Capped to 1000',
                'inventory'
            )
        
        # 3. Parse snapshot_date
        df['snapshot_date'] = pd.to_datetime(df['snapshot_date'], errors='coerce')
        invalid_date = df['snapshot_date'].isna()
        if invalid_date.any():
            for idx in df[invalid_date].index:
                self.log_issue(
                    f"{df.loc[idx, 'product_id']}_{df.loc[idx, 'store_id']}",
                    'INVALID_SNAPSHOT_DATE',
                    'Unparseable snapshot_date',
                    'Dropped record',
                    'inventory'
                )
            df = df[~invalid_date]
        
        self.stats['inventory']['cleaned'] = len(df)
        self.stats['inventory']['issues'] = len([i for i in self.issues if i['table'] == 'inventory'])
        
        return df
    
    def clean_all(self, products_df, stores_df, sales_df, inventory_df, output_dir='data/cleaned'):
        """
        Clean all datasets and generate issues log.
        
        Returns:
            dict: Dictionary containing cleaned DataFrames and issues log
        """
        self.issues = []
        os.makedirs(output_dir, exist_ok=True)
        
        # Clean in order (products first as it's needed for sales validation)
        products = self.clean_products(products_df)
        stores = self.clean_stores(stores_df)
        sales = self.clean_sales(sales_df, products)
        inventory = self.clean_inventory(inventory_df)
        
        # Create issues DataFrame
        issues_df = pd.DataFrame(self.issues)
        
        # Save cleaned files
        products.to_csv(f'{output_dir}/products_cleaned.csv', index=False)
        stores.to_csv(f'{output_dir}/stores_cleaned.csv', index=False)
        sales.to_csv(f'{output_dir}/sales_cleaned.csv', index=False)
        inventory.to_csv(f'{output_dir}/inventory_cleaned.csv', index=False)
        issues_df.to_csv(f'{output_dir}/issues.csv', index=False)
        
        # Save cleaning summary
        summary = pd.DataFrame([
            {'table': k, **v} for k, v in self.stats.items()
        ])
        summary.to_csv(f'{output_dir}/cleaning_summary.csv', index=False)
        
        return {
            'products': products,
            'stores': stores,
            'sales': sales,
            'inventory': inventory,
            'issues': issues_df,
            'stats': self.stats
        }


if __name__ == '__main__':
    # Test cleaning on generated data
    from data_generator import generate_all_data
    
    raw = generate_all_data('data/raw')
    cleaner = DataCleaner()
    cleaned = cleaner.clean_all(
        raw['products'], raw['stores'], raw['sales'], raw['inventory'],
        'data/cleaned'
    )
    
    print("\nCleaning Summary:")
    for table, stats in cleaned['stats'].items():
        print(f"  {table}: {stats['original']} → {stats['cleaned']} ({stats['issues']} issues)")
    
    print(f"\nTotal issues logged: {len(cleaned['issues'])}")
    if len(cleaned['issues']) > 0:
        print("\nIssue Types:")
        print(cleaned['issues']['issue_type'].value_counts())

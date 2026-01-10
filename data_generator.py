"""
data_generator.py
UAE Promo Pulse Simulator - Synthetic Data Generator
Generates dirty datasets that simulate real-world exports with intentional issues
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Date ranges
TODAY = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
HISTORICAL_START = TODAY - timedelta(days=120)
HISTORICAL_END = TODAY - timedelta(days=1)
INVENTORY_START = TODAY - timedelta(days=30)
SIMULATION_START = TODAY
SIMULATION_END = TODAY + timedelta(days=14)

# Data sizes
NUM_PRODUCTS = 300
NUM_STORES = 18
NUM_SALES = 35000
NUM_CAMPAIGNS = 10

# Categories and Brands
CATEGORIES = ['Electronics', 'Fashion', 'Grocery', 'Home & Garden', 'Beauty', 'Sports']
BRANDS = {
    'Electronics': ['Samsung', 'Apple', 'Sony', 'LG', 'Huawei', 'Lenovo'],
    'Fashion': ['Zara', 'H&M', 'Nike', 'Adidas', 'Puma', 'Levis'],
    'Grocery': ['Almarai', 'Nestle', 'Unilever', 'PepsiCo', 'Mars', 'Kelloggs'],
    'Home & Garden': ['IKEA', 'HomeBox', 'ACE', 'Danube', 'Pottery Barn', 'West Elm'],
    'Beauty': ['Loreal', 'Nivea', 'Maybelline', 'MAC', 'Estee Lauder', 'Clinique'],
    'Sports': ['Nike', 'Adidas', 'Puma', 'Under Armour', 'Reebok', 'Decathlon']
}

# Price ranges by category (base_price in AED)
PRICE_RANGES = {
    'Electronics': (100, 5000),
    'Fashion': (50, 1500),
    'Grocery': (5, 200),
    'Home & Garden': (30, 3000),
    'Beauty': (20, 800),
    'Sports': (40, 2000)
}

# Cities and Channels
CITIES_CLEAN = ['Dubai', 'Abu Dhabi', 'Sharjah']
CHANNELS = ['App', 'Web', 'Marketplace']
FULFILLMENT_TYPES = ['Own', '3PL']

# Dirty city values for injection
DIRTY_CITY_VALUES = {
    'Dubai': ['Dubai', 'DUBAI', 'dubai', 'Dubayy', 'DXB'],
    'Abu Dhabi': ['Abu Dhabi', 'ABU DHABI', 'abu dhabi', 'AbuDhabi', 'AD'],
    'Sharjah': ['Sharjah', 'SHARJAH', 'sharjah', 'Shj', 'SHJ']
}

# Payment statuses
PAYMENT_STATUSES = ['Paid', 'Failed', 'Refunded']

# ============================================================================
# TABLE 1: PRODUCTS
# ============================================================================

def generate_products(num_products=NUM_PRODUCTS):
    """Generate products table with some dirty data (missing unit_cost)"""
    
    products = []
    
    for i in range(1, num_products + 1):
        category = random.choice(CATEGORIES)
        brand = random.choice(BRANDS[category])
        
        price_min, price_max = PRICE_RANGES[category]
        base_price = round(random.uniform(price_min, price_max), 2)
        
        # Unit cost is typically 40-70% of base price
        cost_margin = random.uniform(0.40, 0.70)
        unit_cost = round(base_price * cost_margin, 2)
        
        # Tax rate (UAE VAT is 5%)
        tax_rate = 0.05
        
        # Launch flag
        launch_flag = random.choices(['New', 'Regular'], weights=[0.15, 0.85])[0]
        
        products.append({
            'product_id': f'PROD_{i:04d}',
            'category': category,
            'brand': brand,
            'base_price_aed': base_price,
            'unit_cost_aed': unit_cost,
            'tax_rate': tax_rate,
            'launch_flag': launch_flag
        })
    
    df = pd.DataFrame(products)
    
    # DIRTY DATA: Set unit_cost_aed to NaN for 1-2% of products
    num_missing_cost = int(len(df) * random.uniform(0.01, 0.02))
    missing_cost_idx = np.random.choice(df.index, size=num_missing_cost, replace=False)
    df.loc[missing_cost_idx, 'unit_cost_aed'] = np.nan
    
    print(f"✓ Generated {len(df)} products ({num_missing_cost} with missing unit_cost)")
    return df

# ============================================================================
# TABLE 2: STORES
# ============================================================================

def generate_stores():
    """Generate stores table (clean data)"""
    
    stores = []
    store_id = 1
    
    for city in CITIES_CLEAN:
        for channel in CHANNELS:
            for fulfillment in FULFILLMENT_TYPES:
                stores.append({
                    'store_id': f'STORE_{store_id:02d}',
                    'city': city,
                    'channel': channel,
                    'fulfillment_type': fulfillment
                })
                store_id += 1
    
    df = pd.DataFrame(stores)
    print(f"✓ Generated {len(df)} stores")
    return df

# ============================================================================
# TABLE 3: SALES_RAW (DIRTY HISTORICAL SALES)
# ============================================================================

def generate_sales_raw(products_df, stores_df, num_sales=NUM_SALES):
    """Generate dirty sales data with multiple issues"""
    
    product_ids = products_df['product_id'].tolist()
    store_ids = stores_df['store_id'].tolist()
    
    # Create product price lookup
    product_prices = products_df.set_index('product_id')['base_price_aed'].to_dict()
    
    sales = []
    
    # Generate date range for historical sales
    date_range = pd.date_range(start=HISTORICAL_START, end=HISTORICAL_END, freq='H')
    
    for i in range(1, num_sales + 1):
        product_id = random.choice(product_ids)
        store_id = random.choice(store_ids)
        
        # Random timestamp
        order_time = random.choice(date_range) + timedelta(
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        
        # Quantity (mostly 1-3, occasionally more)
        qty = random.choices([1, 2, 3, 4, 5], weights=[0.5, 0.25, 0.15, 0.07, 0.03])[0]
        
        # Base price with possible discount
        base_price = product_prices.get(product_id, 100)
        discount_pct = random.choices(
            [0, 5, 10, 15, 20, 25, 30],
            weights=[0.4, 0.15, 0.15, 0.12, 0.1, 0.05, 0.03]
        )[0]
        selling_price = round(base_price * (1 - discount_pct / 100), 2)
        
        # Payment status
        payment_status = random.choices(
            PAYMENT_STATUSES,
            weights=[0.85, 0.08, 0.07]
        )[0]
        
        # Return flag (only for Paid orders)
        if payment_status == 'Paid':
            return_flag = random.choices([0, 1], weights=[0.92, 0.08])[0]
        else:
            return_flag = 0
        
        sales.append({
            'order_id': f'ORD_{i:06d}',
            'order_time': order_time.strftime('%Y-%m-%d %H:%M:%S'),
            'product_id': product_id,
            'store_id': store_id,
            'qty': qty,
            'selling_price_aed': selling_price,
            'discount_pct': discount_pct,
            'payment_status': payment_status,
            'return_flag': return_flag
        })
    
    df = pd.DataFrame(sales)
    
    # =========================================================================
    # INJECT DIRTY DATA
    # =========================================================================
    
    dirty_stats = {}
    
    # 1. DUPLICATE ORDER_IDs (0.5-1% of sales)
    num_duplicates = int(len(df) * random.uniform(0.005, 0.01))
    duplicate_idx = np.random.choice(df.index[:-100], size=num_duplicates, replace=False)
    for idx in duplicate_idx:
        # Create a duplicate row with same order_id but slightly different time
        dup_row = df.loc[idx].copy()
        df = pd.concat([df, pd.DataFrame([dup_row])], ignore_index=True)
    dirty_stats['duplicates'] = num_duplicates
    
    # 2. CORRUPTED TIMESTAMPS (1-2% of orders)
    num_corrupted = int(len(df) * random.uniform(0.01, 0.02))
    corrupted_idx = np.random.choice(df.index, size=num_corrupted, replace=False)
    corrupted_values = [
        'not_a_time', 
        'invalid_date', 
        '2024-13-45 99:99:99',
        'NULL',
        '',
        '00-00-0000',
        'TBD',
        '2024/01/01',  # Wrong format
        '01-01-2024 12:00:00'  # Wrong format
    ]
    for idx in corrupted_idx:
        df.loc[idx, 'order_time'] = random.choice(corrupted_values)
    dirty_stats['corrupted_timestamps'] = num_corrupted
    
    # 3. MISSING DISCOUNT_PCT (2-4% of sales)
    num_missing_discount = int(len(df) * random.uniform(0.02, 0.04))
    missing_discount_idx = np.random.choice(df.index, size=num_missing_discount, replace=False)
    df.loc[missing_discount_idx, 'discount_pct'] = np.nan
    dirty_stats['missing_discount'] = num_missing_discount
    
    # 4. OUTLIERS - Extreme qty (0.3-0.5%)
    num_outlier_qty = int(len(df) * random.uniform(0.003, 0.005))
    outlier_qty_idx = np.random.choice(df.index, size=num_outlier_qty, replace=False)
    df.loc[outlier_qty_idx, 'qty'] = random.choices([50, 100, 200], k=num_outlier_qty)
    dirty_stats['outlier_qty'] = num_outlier_qty
    
    # 5. OUTLIERS - Extreme prices (0.3-0.5%)
    num_outlier_price = int(len(df) * random.uniform(0.003, 0.005))
    outlier_price_idx = np.random.choice(df.index, size=num_outlier_price, replace=False)
    for idx in outlier_price_idx:
        df.loc[idx, 'selling_price_aed'] = df.loc[idx, 'selling_price_aed'] * 10
    dirty_stats['outlier_price'] = num_outlier_price
    
    # Shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    
    print(f"✓ Generated {len(df)} sales records with dirty data:")
    for issue, count in dirty_stats.items():
        print(f"  - {issue}: {count} records")
    
    return df

# ============================================================================
# TABLE 4: INVENTORY_SNAPSHOT
# ============================================================================

def generate_inventory_snapshot(products_df, stores_df):
    """Generate inventory snapshots with some dirty data (negative/extreme stock)"""
    
    product_ids = products_df['product_id'].tolist()
    store_ids = stores_df['store_id'].tolist()
    
    # For performance, use a subset of products
    sampled_products = random.sample(product_ids, min(100, len(product_ids)))
    
    inventory = []
    date_range = pd.date_range(start=INVENTORY_START, end=TODAY, freq='D')
    
    for snapshot_date in date_range:
        for product_id in sampled_products:
            for store_id in store_ids:
                # Base stock level
                stock_on_hand = random.randint(0, 500)
                
                # Reorder point
                reorder_point = random.randint(10, 50)
                
                # Lead time
                lead_time_days = random.randint(2, 14)
                
                inventory.append({
                    'snapshot_date': snapshot_date.strftime('%Y-%m-%d'),
                    'product_id': product_id,
                    'store_id': store_id,
                    'stock_on_hand': stock_on_hand,
                    'reorder_point': reorder_point,
                    'lead_time_days': lead_time_days
                })
    
    df = pd.DataFrame(inventory)
    
    # DIRTY DATA: Negative stock for small % 
    num_negative = int(len(df) * 0.005)
    negative_idx = np.random.choice(df.index, size=num_negative, replace=False)
    df.loc[negative_idx, 'stock_on_hand'] = -abs(np.random.randint(1, 50, size=num_negative))
    
    # DIRTY DATA: Extreme stock (9999)
    num_extreme = int(len(df) * 0.003)
    extreme_idx = np.random.choice(df.index, size=num_extreme, replace=False)
    df.loc[extreme_idx, 'stock_on_hand'] = 9999
    
    print(f"✓ Generated {len(df)} inventory snapshots")
    print(f"  - {num_negative} with negative stock")
    print(f"  - {num_extreme} with extreme stock (9999)")
    
    return df

# ============================================================================
# TABLE 5: CAMPAIGN_PLAN
# ============================================================================

def generate_campaign_plan(num_campaigns=NUM_CAMPAIGNS):
    """Generate campaign plan scenarios"""
    
    campaigns = []
    
    for i in range(1, num_campaigns + 1):
        start_date = SIMULATION_START + timedelta(days=random.randint(0, 7))
        end_date = start_date + timedelta(days=random.randint(3, 14))
        
        # City filter (All or specific)
        city = random.choices(['All'] + CITIES_CLEAN, weights=[0.4, 0.2, 0.2, 0.2])[0]
        
        # Channel filter
        channel = random.choices(['All'] + CHANNELS, weights=[0.4, 0.2, 0.2, 0.2])[0]
        
        # Category filter
        category = random.choices(['All'] + CATEGORIES, weights=[0.3] + [0.7/len(CATEGORIES)]*len(CATEGORIES))[0]
        
        # Discount percentage
        discount_pct = random.choice([5, 10, 15, 20, 25, 30])
        
        # Promo budget
        promo_budget_aed = random.choice([10000, 25000, 50000, 75000, 100000, 150000])
        
        campaigns.append({
            'campaign_id': f'CAMP_{i:03d}',
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'city': city,
            'channel': channel,
            'category': category,
            'discount_pct': discount_pct,
            'promo_budget_aed': promo_budget_aed
        })
    
    df = pd.DataFrame(campaigns)
    print(f"✓ Generated {len(df)} campaign scenarios")
    return df

# ============================================================================
# ADDITIONAL: STORES WITH DIRTY CITY VALUES
# ============================================================================

def generate_stores_dirty():
    """Generate stores table with inconsistent city names"""
    
    stores = []
    store_id = 1
    
    for city in CITIES_CLEAN:
        for channel in CHANNELS:
            for fulfillment in FULFILLMENT_TYPES:
                # Use dirty city value randomly
                dirty_city = random.choice(DIRTY_CITY_VALUES[city])
                
                stores.append({
                    'store_id': f'STORE_{store_id:02d}',
                    'city': dirty_city,
                    'channel': channel,
                    'fulfillment_type': fulfillment
                })
                store_id += 1
    
    df = pd.DataFrame(stores)
    
    # Count dirty values
    clean_count = df['city'].isin(CITIES_CLEAN).sum()
    dirty_count = len(df) - clean_count
    
    print(f"✓ Generated {len(df)} stores with inconsistent city names")
    print(f"  - {dirty_count} with non-standard city values")
    
    return df

# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_all_data(output_dir='data/raw'):
    """Generate all datasets and save to CSV files"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("UAE PROMO PULSE SIMULATOR - DATA GENERATOR")
    print("=" * 60)
    print(f"\nGenerating data for:")
    print(f"  Historical period: {HISTORICAL_START.date()} to {HISTORICAL_END.date()}")
    print(f"  Inventory snapshots: {INVENTORY_START.date()} to {TODAY.date()}")
    print(f"  Simulation window: {SIMULATION_START.date()} to {SIMULATION_END.date()}")
    print("\n" + "-" * 60)
    
    # Generate tables
    print("\n[1/5] Generating Products...")
    products_df = generate_products()
    
    print("\n[2/5] Generating Stores (with dirty city values)...")
    stores_df = generate_stores_dirty()
    
    print("\n[3/5] Generating Sales (with multiple issues)...")
    sales_df = generate_sales_raw(products_df, stores_df)
    
    print("\n[4/5] Generating Inventory Snapshots...")
    inventory_df = generate_inventory_snapshot(products_df, stores_df)
    
    print("\n[5/5] Generating Campaign Plans...")
    campaigns_df = generate_campaign_plan()
    
    # Save to CSV
    print("\n" + "-" * 60)
    print("Saving files...")
    
    products_df.to_csv(f'{output_dir}/products_raw.csv', index=False)
    stores_df.to_csv(f'{output_dir}/stores_raw.csv', index=False)
    sales_df.to_csv(f'{output_dir}/sales_raw.csv', index=False)
    inventory_df.to_csv(f'{output_dir}/inventory_snapshot_raw.csv', index=False)
    campaigns_df.to_csv(f'{output_dir}/campaign_plan.csv', index=False)
    
    print(f"\n✓ All files saved to '{output_dir}/' directory")
    
    # Summary
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    print(f"\n{'Table':<30} {'Rows':>10} {'File':<30}")
    print("-" * 70)
    print(f"{'products_raw':<30} {len(products_df):>10} {'products_raw.csv':<30}")
    print(f"{'stores_raw':<30} {len(stores_df):>10} {'stores_raw.csv':<30}")
    print(f"{'sales_raw':<30} {len(sales_df):>10} {'sales_raw.csv':<30}")
    print(f"{'inventory_snapshot_raw':<30} {len(inventory_df):>10} {'inventory_snapshot_raw.csv':<30}")
    print(f"{'campaign_plan':<30} {len(campaigns_df):>10} {'campaign_plan.csv':<30}")
    
    print("\n" + "=" * 60)
    print("DIRTY DATA ISSUES INJECTED")
    print("=" * 60)
    print("""
    1. ✓ Inconsistent city values (Dubai/DUBAI/dubai/Dubayy/DXB etc.)
    2. ✓ Missing unit_cost_aed (1-2% of products)
    3. ✓ Missing discount_pct (2-4% of sales)
    4. ✓ Duplicate order_id (0.5-1% of sales)
    5. ✓ Corrupted timestamps (1-2% of orders)
    6. ✓ Outliers: extreme qty and selling_price (0.3-0.5% each)
    7. ✓ Impossible inventory: negative stock and extreme stock (9999)
    """)
    
    return {
        'products': products_df,
        'stores': stores_df,
        'sales': sales_df,
        'inventory': inventory_df,
        'campaigns': campaigns_df
    }

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    data = generate_all_data()

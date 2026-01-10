"""
data_generator.py
Generates synthetic retail data with intentional data quality issues
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_all_data(output_dir='data/raw'):
    """Generate all synthetic data tables"""
    
    np.random.seed(42)
    random.seed(42)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration
    TODAY = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    HISTORICAL_START = TODAY - timedelta(days=120)
    HISTORICAL_END = TODAY - timedelta(days=1)
    INVENTORY_START = TODAY - timedelta(days=30)
    SIMULATION_START = TODAY
    SIMULATION_END = TODAY + timedelta(days=14)
    
    NUM_PRODUCTS = 300
    NUM_STORES = 18
    NUM_SALES = 35000
    NUM_CAMPAIGNS = 10
    
    CATEGORIES = ['Electronics', 'Fashion', 'Grocery', 'Home & Garden', 'Beauty', 'Sports']
    BRANDS = {
        'Electronics': ['Samsung', 'Apple', 'Sony', 'LG', 'Huawei', 'Lenovo'],
        'Fashion': ['Zara', 'H&M', 'Nike', 'Adidas', 'Puma', 'Levis'],
        'Grocery': ['Almarai', 'Nestle', 'Unilever', 'PepsiCo', 'Mars', 'Kelloggs'],
        'Home & Garden': ['IKEA', 'HomeBox', 'ACE', 'Danube', 'Pottery Barn', 'West Elm'],
        'Beauty': ['Loreal', 'Nivea', 'Maybelline', 'MAC', 'Estee Lauder', 'Clinique'],
        'Sports': ['Nike', 'Adidas', 'Puma', 'Under Armour', 'Reebok', 'Decathlon']
    }
    
    PRICE_RANGES = {
        'Electronics': (100, 5000),
        'Fashion': (50, 1500),
        'Grocery': (5, 200),
        'Home & Garden': (30, 3000),
        'Beauty': (20, 800),
        'Sports': (40, 2000)
    }
    
    CITIES_CLEAN = ['Dubai', 'Abu Dhabi', 'Sharjah']
    CHANNELS = ['App', 'Web', 'Marketplace']
    FULFILLMENT_TYPES = ['Own', '3PL']
    
    DIRTY_CITY_VALUES = {
        'Dubai': ['Dubai', 'DUBAI', 'dubai', 'Dubayy', 'DXB'],
        'Abu Dhabi': ['Abu Dhabi', 'ABU DHABI', 'abu dhabi', 'AbuDhabi', 'AD'],
        'Sharjah': ['Sharjah', 'SHARJAH', 'sharjah', 'Shj', 'SHJ']
    }
    
    PAYMENT_STATUSES = ['Paid', 'Failed', 'Refunded']
    
    # ==================== PRODUCTS ====================
    products = []
    for i in range(1, NUM_PRODUCTS + 1):
        category = random.choice(CATEGORIES)
        brand = random.choice(BRANDS[category])
        price_min, price_max = PRICE_RANGES[category]
        base_price = round(random.uniform(price_min, price_max), 2)
        cost_margin = random.uniform(0.40, 0.70)
        unit_cost = round(base_price * cost_margin, 2)
        
        products.append({
            'product_id': f'PROD_{i:04d}',
            'category': category,
            'brand': brand,
            'base_price_aed': base_price,
            'unit_cost_aed': unit_cost,
            'tax_rate': 0.05,
            'launch_flag': random.choices(['New', 'Regular'], weights=[0.15, 0.85])[0]
        })
    
    products_df = pd.DataFrame(products)
    
    # Inject missing unit_cost
    num_missing = int(len(products_df) * 0.015)
    missing_idx = np.random.choice(products_df.index, size=num_missing, replace=False)
    products_df.loc[missing_idx, 'unit_cost_aed'] = np.nan
    
    # ==================== STORES ====================
    stores = []
    store_id = 1
    for city in CITIES_CLEAN:
        for channel in CHANNELS:
            for fulfillment in FULFILLMENT_TYPES:
                dirty_city = random.choice(DIRTY_CITY_VALUES[city])
                stores.append({
                    'store_id': f'STORE_{store_id:02d}',
                    'city': dirty_city,
                    'channel': channel,
                    'fulfillment_type': fulfillment
                })
                store_id += 1
    
    stores_df = pd.DataFrame(stores)
    
    # ==================== SALES ====================
    product_ids = products_df['product_id'].tolist()
    store_ids = stores_df['store_id'].tolist()
    product_prices = products_df.set_index('product_id')['base_price_aed'].to_dict()
    
    sales = []
    date_range = pd.date_range(start=HISTORICAL_START, end=HISTORICAL_END, freq='H')
    
    for i in range(1, NUM_SALES + 1):
        product_id = random.choice(product_ids)
        store_id = random.choice(store_ids)
        
        order_time = random.choice(date_range) + timedelta(
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        
        qty = random.choices([1, 2, 3, 4, 5], weights=[0.5, 0.25, 0.15, 0.07, 0.03])[0]
        base_price = product_prices.get(product_id, 100)
        discount_pct = random.choices([0, 5, 10, 15, 20, 25, 30], weights=[0.4, 0.15, 0.15, 0.12, 0.1, 0.05, 0.03])[0]
        selling_price = round(base_price * (1 - discount_pct / 100), 2)
        
        payment_status = random.choices(PAYMENT_STATUSES, weights=[0.85, 0.08, 0.07])[0]
        return_flag = random.choices([0, 1], weights=[0.92, 0.08])[0] if payment_status == 'Paid' else 0
        
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
    
    sales_df = pd.DataFrame(sales)
    
    # Inject dirty data
    # Duplicates
    num_duplicates = int(len(sales_df) * 0.007)
    dup_idx = np.random.choice(sales_df.index[:-100], size=num_duplicates, replace=False)
    for idx in dup_idx:
        sales_df = pd.concat([sales_df, pd.DataFrame([sales_df.loc[idx]])], ignore_index=True)
    
    # Corrupted timestamps
    num_corrupted = int(len(sales_df) * 0.015)
    corrupted_idx = np.random.choice(sales_df.index, size=num_corrupted, replace=False)
    corrupted_values = ['not_a_time', 'invalid', '2024-13-45 99:99:99', 'NULL', '', 'TBD']
    for idx in corrupted_idx:
        sales_df.loc[idx, 'order_time'] = random.choice(corrupted_values)
    
    # Missing discount
    num_missing_disc = int(len(sales_df) * 0.03)
    missing_disc_idx = np.random.choice(sales_df.index, size=num_missing_disc, replace=False)
    sales_df.loc[missing_disc_idx, 'discount_pct'] = np.nan
    
    # Outlier qty
    num_outlier_qty = int(len(sales_df) * 0.004)
    outlier_qty_idx = np.random.choice(sales_df.index, size=num_outlier_qty, replace=False)
    sales_df.loc[outlier_qty_idx, 'qty'] = random.choices([50, 100, 200], k=num_outlier_qty)
    
    # Outlier price
    num_outlier_price = int(len(sales_df) * 0.004)
    outlier_price_idx = np.random.choice(sales_df.index, size=num_outlier_price, replace=False)
    for idx in outlier_price_idx:
        sales_df.loc[idx, 'selling_price_aed'] = sales_df.loc[idx, 'selling_price_aed'] * 10
    
    sales_df = sales_df.sample(frac=1).reset_index(drop=True)
    
    # ==================== INVENTORY ====================
    sampled_products = random.sample(product_ids, min(100, len(product_ids)))
    
    inventory = []
    inv_date_range = pd.date_range(start=INVENTORY_START, end=TODAY, freq='D')
    
    for snapshot_date in inv_date_range:
        for product_id in sampled_products:
            for store_id_inv in store_ids:
                inventory.append({
                    'snapshot_date': snapshot_date.strftime('%Y-%m-%d'),
                    'product_id': product_id,
                    'store_id': store_id_inv,
                    'stock_on_hand': random.randint(0, 500),
                    'reorder_point': random.randint(10, 50),
                    'lead_time_days': random.randint(2, 14)
                })
    
    inventory_df = pd.DataFrame(inventory)
    
    # Negative stock
    num_negative = int(len(inventory_df) * 0.005)
    negative_idx = np.random.choice(inventory_df.index, size=num_negative, replace=False)
    inventory_df.loc[negative_idx, 'stock_on_hand'] = -abs(np.random.randint(1, 50, size=num_negative))
    
    # Extreme stock
    num_extreme = int(len(inventory_df) * 0.003)
    extreme_idx = np.random.choice(inventory_df.index, size=num_extreme, replace=False)
    inventory_df.loc[extreme_idx, 'stock_on_hand'] = 9999
    
    # ==================== CAMPAIGNS ====================
    campaigns = []
    for i in range(1, NUM_CAMPAIGNS + 1):
        start_date = SIMULATION_START + timedelta(days=random.randint(0, 7))
        end_date = start_date + timedelta(days=random.randint(3, 14))
        
        campaigns.append({
            'campaign_id': f'CAMP_{i:03d}',
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'city': random.choices(['All'] + CITIES_CLEAN, weights=[0.4, 0.2, 0.2, 0.2])[0],
            'channel': random.choices(['All'] + CHANNELS, weights=[0.4, 0.2, 0.2, 0.2])[0],
            'category': random.choices(['All'] + CATEGORIES, weights=[0.3] + [0.7/len(CATEGORIES)]*len(CATEGORIES))[0],
            'discount_pct': random.choice([5, 10, 15, 20, 25, 30]),
            'promo_budget_aed': random.choice([10000, 25000, 50000, 75000, 100000, 150000])
        })
    
    campaigns_df = pd.DataFrame(campaigns)
    
    # Save files
    products_df.to_csv(f'{output_dir}/products_raw.csv', index=False)
    stores_df.to_csv(f'{output_dir}/stores_raw.csv', index=False)
    sales_df.to_csv(f'{output_dir}/sales_raw.csv', index=False)
    inventory_df.to_csv(f'{output_dir}/inventory_snapshot_raw.csv', index=False)
    campaigns_df.to_csv(f'{output_dir}/campaign_plan.csv', index=False)
    
    return {
        'products': products_df,
        'stores': stores_df,
        'sales': sales_df,
        'inventory': inventory_df,
        'campaigns': campaigns_df
    }

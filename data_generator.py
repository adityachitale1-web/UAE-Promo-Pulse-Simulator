"""
UAE Promo Pulse - Synthetic Data Generator
==========================================
Generates dirty retail data with intentional quality issues for cleaning practice.

Data Specifications:
- Historical sales: Last 120 days
- Inventory snapshots: Last 30 days
- Simulation window: Next 14 days
- Products: 300 (250-350 range)
- Stores: 18 (3 cities × 3 channels × 2 fulfillment types)
- Sales: 35,000 orders (25,000-40,000 range)
- Campaign plans: 10 scenarios

Injected Dirty Data Issues:
1. Inconsistent city values: "Dubai", "DUBAI", "dubai", "DXB"
2. Missing unit_cost_aed: ~1.7% of products
3. Missing discount_pct: ~3% of sales
4. Duplicate order_id: ~0.7% of sales
5. Corrupted timestamps: ~1.4% of orders
6. Outlier qty (50+): ~0.4% of sales
7. Outlier prices (10x): ~0.4% of sales
8. Negative/extreme inventory: ~0.5% of records
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_all_data(output_dir='data/raw'):
    """
    Generate all synthetic datasets with intentional dirty data issues.
    
    Returns:
        dict: Dictionary containing all generated DataFrames
    """
    np.random.seed(42)
    random.seed(42)
    os.makedirs(output_dir, exist_ok=True)
    
    # Date Configuration
    TODAY = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    HISTORICAL_START = TODAY - timedelta(days=120)
    HISTORICAL_END = TODAY - timedelta(days=1)
    INVENTORY_START = TODAY - timedelta(days=30)
    SIMULATION_END = TODAY + timedelta(days=14)
    
    # Master Data Configuration
    CATEGORIES = ['Electronics', 'Fashion', 'Grocery', 'Home & Garden', 'Beauty', 'Sports']
    BRANDS = {
        'Electronics': ['Samsung', 'Apple', 'Sony', 'LG', 'Huawei'],
        'Fashion': ['Zara', 'H&M', 'Nike', 'Adidas', 'Mango'],
        'Grocery': ['Almarai', 'Nestle', 'Unilever', 'PepsiCo', 'KDD'],
        'Home & Garden': ['IKEA', 'HomeBox', 'ACE', 'Danube', 'Home Centre'],
        'Beauty': ['Loreal', 'Nivea', 'Maybelline', 'MAC', 'Sephora'],
        'Sports': ['Nike', 'Adidas', 'Puma', 'Reebok', 'Under Armour']
    }
    PRICE_RANGES = {
        'Electronics': (100, 5000), 'Fashion': (50, 1500), 'Grocery': (5, 200),
        'Home & Garden': (30, 3000), 'Beauty': (20, 800), 'Sports': (40, 2000)
    }
    
    # UAE Cities with dirty variations for injection
    CITIES = ['Dubai', 'Abu Dhabi', 'Sharjah']
    CHANNELS = ['App', 'Web', 'Marketplace']
    FULFILLMENT_TYPES = ['Own', '3PL']
    
    # Dirty city mappings (intentional inconsistencies)
    DIRTY_CITIES = {
        'Dubai': ['Dubai', 'DUBAI', 'dubai', 'DXB', 'Dubayy'],
        'Abu Dhabi': ['Abu Dhabi', 'ABU DHABI', 'AbuDhabi', 'AD', 'Abu-Dhabi'],
        'Sharjah': ['Sharjah', 'SHARJAH', 'Shj', 'SHJ', 'Sharjha']
    }
    
    # ==========================================================================
    # 1. PRODUCTS TABLE (300 products)
    # ==========================================================================
    products = []
    for i in range(1, 301):
        cat = random.choice(CATEGORIES)
        price_min, price_max = PRICE_RANGES[cat]
        base_price = round(random.uniform(price_min, price_max), 2)
        # Ensure unit_cost < base_price (with margin 40-70%)
        unit_cost = round(base_price * random.uniform(0.4, 0.7), 2)
        
        products.append({
            'product_id': f'PROD_{i:04d}',
            'product_name': f'{random.choice(BRANDS[cat])} {cat} Item {i}',
            'category': cat,
            'brand': random.choice(BRANDS[cat]),
            'base_price_aed': base_price,
            'unit_cost_aed': unit_cost,
            'tax_rate': 0.05,
            'launch_flag': random.choices(['New', 'Regular'], weights=[0.15, 0.85])[0]
        })
    
    products_df = pd.DataFrame(products)
    
    # DIRTY DATA: Missing unit_cost_aed for ~1.7% (5 products)
    missing_cost_idx = np.random.choice(products_df.index, size=5, replace=False)
    products_df.loc[missing_cost_idx, 'unit_cost_aed'] = np.nan
    
    # ==========================================================================
    # 2. STORES TABLE (18 stores: 3 cities × 3 channels × 2 fulfillment)
    # ==========================================================================
    stores = []
    store_id = 1
    for city in CITIES:
        for channel in CHANNELS:
            for ftype in FULFILLMENT_TYPES:
                # Inject dirty city names randomly
                dirty_city = random.choice(DIRTY_CITIES[city])
                stores.append({
                    'store_id': f'STORE_{store_id:02d}',
                    'store_name': f'{dirty_city} {channel} {ftype}',
                    'city': dirty_city,
                    'channel': channel,
                    'fulfillment_type': ftype,
                    'opening_date': (TODAY - timedelta(days=random.randint(365, 1000))).strftime('%Y-%m-%d')
                })
                store_id += 1
    
    stores_df = pd.DataFrame(stores)
    
    # ==========================================================================
    # 3. SALES TABLE (35,000 orders)
    # ==========================================================================
    product_ids = products_df['product_id'].tolist()
    store_ids = stores_df['store_id'].tolist()
    prices = products_df.set_index('product_id')['base_price_aed'].to_dict()
    
    sales = []
    dates = pd.date_range(start=HISTORICAL_START, end=HISTORICAL_END, freq='H')
    
    for i in range(1, 35001):
        pid = random.choice(product_ids)
        base = prices.get(pid, 100)
        
        # Discount distribution (weighted towards lower discounts)
        disc = random.choices([0, 5, 10, 15, 20, 25, 30], weights=[0.4, 0.15, 0.15, 0.12, 0.1, 0.05, 0.03])[0]
        selling_price = round(base * (1 - disc / 100), 2)
        
        # Payment status distribution
        pay_status = random.choices(['Paid', 'Failed', 'Refunded'], weights=[0.85, 0.08, 0.07])[0]
        
        # Return flag only for Paid orders
        return_flag = random.choices([0, 1], weights=[0.92, 0.08])[0] if pay_status == 'Paid' else 0
        
        order_time = random.choice(dates) + timedelta(minutes=random.randint(0, 59))
        
        sales.append({
            'order_id': f'ORD_{i:06d}',
            'order_time': order_time.strftime('%Y-%m-%d %H:%M:%S'),
            'product_id': pid,
            'store_id': random.choice(store_ids),
            'qty': random.choices([1, 2, 3, 4, 5], weights=[0.5, 0.25, 0.15, 0.07, 0.03])[0],
            'selling_price_aed': selling_price,
            'discount_pct': disc,
            'payment_status': pay_status,
            'return_flag': return_flag,
            'customer_id': f'CUST_{random.randint(1, 5000):05d}'
        })
    
    sales_df = pd.DataFrame(sales)
    
    # DIRTY DATA INJECTIONS FOR SALES:
    
    # 1. Duplicate order_id (~0.7% = 250 duplicates)
    dup_indices = np.random.choice(len(sales_df) - 100, size=250, replace=False)
    duplicates = sales_df.iloc[dup_indices].copy()
    sales_df = pd.concat([sales_df, duplicates], ignore_index=True)
    
    # 2. Corrupted timestamps (~1.4% = 500 records)
    corrupt_time_idx = np.random.choice(sales_df.index, size=500, replace=False)
    corrupt_values = ['invalid', 'NULL', '', '2024-13-45', 'not_a_time', 'NaT', '00-00-0000']
    for idx in corrupt_time_idx:
        sales_df.loc[idx, 'order_time'] = random.choice(corrupt_values)
    
    # 3. Missing discount_pct (~3% = 1000 records)
    missing_disc_idx = np.random.choice(sales_df.index, size=1000, replace=False)
    sales_df.loc[missing_disc_idx, 'discount_pct'] = np.nan
    
    # 4. Outlier qty (50+) for ~0.4% = 150 records
    outlier_qty_idx = np.random.choice(sales_df.index, size=150, replace=False)
    sales_df.loc[outlier_qty_idx, 'qty'] = np.random.choice([50, 75, 100, 150, 200], size=150)
    
    # 5. Outlier prices (10x) for ~0.4% = 150 records
    outlier_price_idx = np.random.choice(sales_df.index, size=150, replace=False)
    sales_df.loc[outlier_price_idx, 'selling_price_aed'] = sales_df.loc[outlier_price_idx, 'selling_price_aed'] * 10
    
    # 6. Invalid payment status for small %
    invalid_payment_idx = np.random.choice(sales_df.index, size=50, replace=False)
    sales_df.loc[invalid_payment_idx, 'payment_status'] = random.choices(['Unknown', 'Pending', 'ERROR', ''], k=50)
    
    # Shuffle the dataframe
    sales_df = sales_df.sample(frac=1).reset_index(drop=True)
    
    # ==========================================================================
    # 4. INVENTORY SNAPSHOT TABLE (30 days × subset of products × stores)
    # ==========================================================================
    # Use subset of products for reasonable size
    sampled_products = random.sample(product_ids, min(100, len(product_ids)))
    
    inventory = []
    for d in pd.date_range(start=INVENTORY_START, end=TODAY, freq='D'):
        for pid in sampled_products:
            for stid in store_ids:
                stock = random.randint(0, 500)
                reorder = random.randint(10, 50)
                lead_time = random.randint(2, 14)
                
                inventory.append({
                    'snapshot_date': d.strftime('%Y-%m-%d'),
                    'product_id': pid,
                    'store_id': stid,
                    'stock_on_hand': stock,
                    'reorder_point': reorder,
                    'lead_time_days': lead_time
                })
    
    inventory_df = pd.DataFrame(inventory)
    
    # DIRTY DATA: Negative stock (~0.5%)
    neg_stock_idx = np.random.choice(inventory_df.index, size=int(len(inventory_df)*0.005), replace=False)
    inventory_df.loc[neg_stock_idx, 'stock_on_hand'] = -abs(np.random.randint(1, 50, size=len(neg_stock_idx)))
    
    # DIRTY DATA: Extreme stock (9999) for small %
    extreme_stock_idx = np.random.choice(inventory_df.index, size=int(len(inventory_df)*0.002), replace=False)
    inventory_df.loc[extreme_stock_idx, 'stock_on_hand'] = 9999
    
    # ==========================================================================
    # 5. CAMPAIGN PLAN TABLE (10 scenarios)
    # ==========================================================================
    campaigns = []
    for i in range(1, 11):
        start = TODAY + timedelta(days=random.randint(0, 7))
        campaigns.append({
            'campaign_id': f'CAMP_{i:03d}',
            'campaign_name': f'Promo Campaign {i}',
            'start_date': start.strftime('%Y-%m-%d'),
            'end_date': (start + timedelta(days=random.randint(3, 14))).strftime('%Y-%m-%d'),
            'city': random.choice(['All'] + CITIES),
            'channel': random.choice(['All'] + CHANNELS),
            'category': random.choice(['All'] + CATEGORIES),
            'discount_pct': random.choice([5, 10, 15, 20, 25, 30]),
            'promo_budget_aed': random.choice([10000, 25000, 50000, 75000, 100000]),
            'status': 'Planned'
        })
    
    campaigns_df = pd.DataFrame(campaigns)
    
    # ==========================================================================
    # SAVE ALL FILES
    # ==========================================================================
    products_df.to_csv(f'{output_dir}/products_raw.csv', index=False)
    stores_df.to_csv(f'{output_dir}/stores_raw.csv', index=False)
    sales_df.to_csv(f'{output_dir}/sales_raw.csv', index=False)
    inventory_df.to_csv(f'{output_dir}/inventory_snapshot_raw.csv', index=False)
    campaigns_df.to_csv(f'{output_dir}/campaign_plan.csv', index=False)
    
    # Generate metadata summary
    metadata = {
        'generation_date': TODAY.strftime('%Y-%m-%d'),
        'historical_start': HISTORICAL_START.strftime('%Y-%m-%d'),
        'historical_end': HISTORICAL_END.strftime('%Y-%m-%d'),
        'inventory_start': INVENTORY_START.strftime('%Y-%m-%d'),
        'simulation_end': SIMULATION_END.strftime('%Y-%m-%d'),
        'products_count': len(products_df),
        'stores_count': len(stores_df),
        'sales_count': len(sales_df),
        'inventory_records': len(inventory_df),
        'campaigns_count': len(campaigns_df),
        'dirty_data_injected': {
            'missing_unit_cost': 5,
            'duplicate_orders': 250,
            'corrupted_timestamps': 500,
            'missing_discounts': 1000,
            'outlier_qty': 150,
            'outlier_prices': 150,
            'negative_inventory': int(len(inventory_df)*0.005),
            'extreme_inventory': int(len(inventory_df)*0.002),
            'invalid_payment_status': 50
        }
    }
    
    pd.DataFrame([metadata]).to_csv(f'{output_dir}/metadata.csv', index=False)
    
    return {
        'products': products_df,
        'stores': stores_df,
        'sales': sales_df,
        'inventory': inventory_df,
        'campaigns': campaigns_df,
        'metadata': metadata
    }


if __name__ == '__main__':
    data = generate_all_data()
    print("Data generation complete!")
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            print(f"  {k}: {len(v)} records")

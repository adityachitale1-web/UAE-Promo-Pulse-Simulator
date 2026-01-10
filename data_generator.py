import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_all_data(output_dir='data/raw'):
    np.random.seed(42)
    random.seed(42)
    os.makedirs(output_dir, exist_ok=True)
    
    TODAY = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    HISTORICAL_START = TODAY - timedelta(days=120)
    HISTORICAL_END = TODAY - timedelta(days=1)
    INVENTORY_START = TODAY - timedelta(days=30)
    
    CATEGORIES = ['Electronics', 'Fashion', 'Grocery', 'Home & Garden', 'Beauty', 'Sports']
    BRANDS = {
        'Electronics': ['Samsung', 'Apple', 'Sony', 'LG'],
        'Fashion': ['Zara', 'H&M', 'Nike', 'Adidas'],
        'Grocery': ['Almarai', 'Nestle', 'Unilever', 'PepsiCo'],
        'Home & Garden': ['IKEA', 'HomeBox', 'ACE', 'Danube'],
        'Beauty': ['Loreal', 'Nivea', 'Maybelline', 'MAC'],
        'Sports': ['Nike', 'Adidas', 'Puma', 'Reebok']
    }
    PRICE_RANGES = {
        'Electronics': (100, 5000), 'Fashion': (50, 1500), 'Grocery': (5, 200),
        'Home & Garden': (30, 3000), 'Beauty': (20, 800), 'Sports': (40, 2000)
    }
    CITIES = ['Dubai', 'Abu Dhabi', 'Sharjah']
    CHANNELS = ['App', 'Web', 'Marketplace']
    DIRTY_CITIES = {
        'Dubai': ['Dubai', 'DUBAI', 'dubai', 'DXB'],
        'Abu Dhabi': ['Abu Dhabi', 'ABU DHABI', 'AbuDhabi', 'AD'],
        'Sharjah': ['Sharjah', 'SHARJAH', 'Shj', 'SHJ']
    }
    
    products = []
    for i in range(1, 301):
        cat = random.choice(CATEGORIES)
        price_min, price_max = PRICE_RANGES[cat]
        base_price = round(random.uniform(price_min, price_max), 2)
        products.append({
            'product_id': f'PROD_{i:04d}', 'category': cat, 'brand': random.choice(BRANDS[cat]),
            'base_price_aed': base_price, 'unit_cost_aed': round(base_price * random.uniform(0.4, 0.7), 2),
            'tax_rate': 0.05, 'launch_flag': random.choices(['New', 'Regular'], weights=[0.15, 0.85])[0]
        })
    products_df = pd.DataFrame(products)
    products_df.loc[np.random.choice(products_df.index, size=5, replace=False), 'unit_cost_aed'] = np.nan
    
    stores = []
    sid = 1
    for city in CITIES:
        for channel in CHANNELS:
            for ftype in ['Own', '3PL']:
                stores.append({'store_id': f'STORE_{sid:02d}', 'city': random.choice(DIRTY_CITIES[city]),
                              'channel': channel, 'fulfillment_type': ftype})
                sid += 1
    stores_df = pd.DataFrame(stores)
    
    product_ids = products_df['product_id'].tolist()
    store_ids = stores_df['store_id'].tolist()
    prices = products_df.set_index('product_id')['base_price_aed'].to_dict()
    
    sales = []
    dates = pd.date_range(start=HISTORICAL_START, end=HISTORICAL_END, freq='H')
    for i in range(1, 35001):
        pid = random.choice(product_ids)
        disc = random.choices([0, 5, 10, 15, 20, 25, 30], weights=[0.4, 0.15, 0.15, 0.12, 0.1, 0.05, 0.03])[0]
        pay = random.choices(['Paid', 'Failed', 'Refunded'], weights=[0.85, 0.08, 0.07])[0]
        sales.append({
            'order_id': f'ORD_{i:06d}',
            'order_time': (random.choice(dates) + timedelta(minutes=random.randint(0, 59))).strftime('%Y-%m-%d %H:%M:%S'),
            'product_id': pid, 'store_id': random.choice(store_ids),
            'qty': random.choices([1, 2, 3, 4, 5], weights=[0.5, 0.25, 0.15, 0.07, 0.03])[0],
            'selling_price_aed': round(prices.get(pid, 100) * (1 - disc / 100), 2),
            'discount_pct': disc, 'payment_status': pay,
            'return_flag': random.choices([0, 1], weights=[0.92, 0.08])[0] if pay == 'Paid' else 0
        })
    sales_df = pd.DataFrame(sales)
    
    for idx in np.random.choice(len(sales_df) - 100, size=250, replace=False):
        sales_df = pd.concat([sales_df, sales_df.iloc[[idx]]], ignore_index=True)
    for idx in np.random.choice(sales_df.index, size=500, replace=False):
        sales_df.loc[idx, 'order_time'] = random.choice(['invalid', 'NULL', '', '2024-13-45'])
    sales_df.loc[np.random.choice(sales_df.index, size=1000, replace=False), 'discount_pct'] = np.nan
    sales_df.loc[np.random.choice(sales_df.index, size=150, replace=False), 'qty'] = random.choices([50, 100, 200], k=150)
    for idx in np.random.choice(sales_df.index, size=150, replace=False):
        sales_df.loc[idx, 'selling_price_aed'] *= 10
    sales_df = sales_df.sample(frac=1).reset_index(drop=True)
    
    sampled = random.sample(product_ids, 100)
    inventory = []
    for d in pd.date_range(start=INVENTORY_START, end=TODAY, freq='D'):
        for pid in sampled:
            for stid in store_ids:
                inventory.append({'snapshot_date': d.strftime('%Y-%m-%d'), 'product_id': pid, 'store_id': stid,
                                 'stock_on_hand': random.randint(0, 500), 'reorder_point': random.randint(10, 50),
                                 'lead_time_days': random.randint(2, 14)})
    inventory_df = pd.DataFrame(inventory)
    inventory_df.loc[np.random.choice(inventory_df.index, size=int(len(inventory_df)*0.005), replace=False), 'stock_on_hand'] = -abs(np.random.randint(1, 50, size=int(len(inventory_df)*0.005)))
    
    campaigns = []
    for i in range(1, 11):
        start = TODAY + timedelta(days=random.randint(0, 7))
        campaigns.append({'campaign_id': f'CAMP_{i:03d}', 'start_date': start.strftime('%Y-%m-%d'),
                         'end_date': (start + timedelta(days=random.randint(3, 14))).strftime('%Y-%m-%d'),
                         'city': random.choice(['All'] + CITIES), 'channel': random.choice(['All'] + CHANNELS),
                         'category': random.choice(['All'] + CATEGORIES),
                         'discount_pct': random.choice([5, 10, 15, 20, 25, 30]),
                         'promo_budget_aed': random.choice([10000, 25000, 50000, 75000, 100000])})
    campaigns_df = pd.DataFrame(campaigns)
    
    products_df.to_csv(f'{output_dir}/products_raw.csv', index=False)
    stores_df.to_csv(f'{output_dir}/stores_raw.csv', index=False)
    sales_df.to_csv(f'{output_dir}/sales_raw.csv', index=False)
    inventory_df.to_csv(f'{output_dir}/inventory_snapshot_raw.csv', index=False)
    campaigns_df.to_csv(f'{output_dir}/campaign_plan.csv', index=False)
    
    return {'products': products_df, 'stores': stores_df, 'sales': sales_df, 'inventory': inventory_df, 'campaigns': campaigns_df}

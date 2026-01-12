"""
UAE Promo Pulse - Main Dashboard
================================
Streamlit dashboard with Executive and Manager views.
Includes all 12 required chart types + Data Upload Option.

Features:
- Generate Sample Data OR Upload Your Own CSV Files
- SMART FILE DETECTION - Validates files immediately on upload
- 5+ Sidebar Filters
- 15 KPIs with full visualization
- 12 Chart Types
- What-If Simulation
- Download cleaned data, issues log, and ALL DATA as ZIP
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import io
import zipfile

from data_generator import generate_all_data
from cleaner import DataCleaner
from simulator import KPICalculator, PromoSimulator, generate_recommendation

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="UAE Promo Pulse",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# FILE DETECTION AND VALIDATION SYSTEM
# =============================================================================

# Define file signatures - unique and required columns for each file type
FILE_SIGNATURES = {
    'products': {
        'required': ['product_id', 'category', 'base_price_aed'],
        'optional': ['unit_cost_aed', 'brand', 'tax_rate', 'launch_flag', 'product_name'],
        'unique_identifiers': ['product_id', 'base_price_aed'],  # Columns that strongly identify this file
        'description': 'Product master data with pricing'
    },
    'stores': {
        'required': ['store_id', 'city', 'channel'],
        'optional': ['fulfillment_type', 'store_name', 'opening_date'],
        'unique_identifiers': ['store_id', 'channel', 'city'],
        'description': 'Store/location master data'
    },
    'sales': {
        'required': ['order_id', 'order_time', 'product_id', 'store_id', 'qty', 'selling_price_aed', 'payment_status'],
        'optional': ['discount_pct', 'return_flag', 'customer_id'],
        'unique_identifiers': ['order_id', 'order_time', 'payment_status'],
        'description': 'Sales transactions'
    },
    'inventory': {
        'required': ['snapshot_date', 'product_id', 'store_id', 'stock_on_hand'],
        'optional': ['reorder_point', 'lead_time_days'],
        'unique_identifiers': ['snapshot_date', 'stock_on_hand'],
        'description': 'Inventory snapshots'
    }
}


def detect_file_type(df):
    """
    Detect the type of file based on its columns.
    
    Returns:
        tuple: (detected_type, confidence_score, match_details)
               detected_type: 'products', 'stores', 'sales', 'inventory', or 'unknown'
               confidence_score: 0-100 indicating match confidence
               match_details: dict with matching info
    """
    if df is None or len(df.columns) == 0:
        return 'unknown', 0, {'error': 'Empty or invalid file'}
    
    columns = set([col.lower().strip() for col in df.columns])
    original_columns = set(df.columns.tolist())
    
    best_match = 'unknown'
    best_score = 0
    best_details = {}
    
    for file_type, signature in FILE_SIGNATURES.items():
        required = set([col.lower() for col in signature['required']])
        optional = set([col.lower() for col in signature['optional']])
        unique_ids = set([col.lower() for col in signature['unique_identifiers']])
        all_expected = required | optional
        
        # Calculate matches
        required_matches = required & columns
        optional_matches = optional & columns
        unique_matches = unique_ids & columns
        all_matches = all_expected & columns
        
        # Calculate scores
        required_score = len(required_matches) / len(required) * 60 if required else 0
        unique_score = len(unique_matches) / len(unique_ids) * 30 if unique_ids else 0
        optional_score = len(optional_matches) / len(optional) * 10 if optional else 0
        
        total_score = required_score + unique_score + optional_score
        
        # Check for missing required columns
        missing_required = required - columns
        
        details = {
            'required_columns': list(signature['required']),
            'found_required': list(required_matches),
            'missing_required': list(missing_required),
            'optional_found': list(optional_matches),
            'total_columns_in_file': len(df.columns),
            'matched_columns': len(all_matches),
            'description': signature['description']
        }
        
        if total_score > best_score:
            best_score = total_score
            best_match = file_type
            best_details = details
    
    # If score is too low, mark as unknown
    if best_score < 30:
        best_match = 'unknown'
        best_details = {
            'error': 'Could not determine file type',
            'columns_found': list(df.columns)[:10],  # Show first 10 columns
            'suggestion': 'Please check if this is the correct file format'
        }
    
    return best_match, round(best_score, 1), best_details


def validate_file_for_type(df, expected_type):
    """
    Validate if a file matches the expected type.
    
    Returns:
        dict: {
            'is_valid': bool,
            'detected_type': str,
            'confidence': float,
            'is_correct_type': bool,
            'errors': list,
            'warnings': list,
            'details': dict
        }
    """
    result = {
        'is_valid': False,
        'detected_type': 'unknown',
        'confidence': 0,
        'is_correct_type': False,
        'errors': [],
        'warnings': [],
        'details': {}
    }
    
    if df is None:
        result['errors'].append('No file uploaded or file is empty')
        return result
    
    try:
        # Detect what type of file this actually is
        detected_type, confidence, details = detect_file_type(df)
        
        result['detected_type'] = detected_type
        result['confidence'] = confidence
        result['details'] = details
        
        # Check if it matches expected type
        if detected_type == expected_type:
            result['is_correct_type'] = True
            
            # Check for missing required columns
            if details.get('missing_required'):
                result['errors'].append(
                    f"Missing required columns: {details['missing_required']}"
                )
            else:
                result['is_valid'] = True
                
            # Add warnings for missing optional columns
            signature = FILE_SIGNATURES[expected_type]
            optional_cols = set([col.lower() for col in signature['optional']])
            file_cols = set([col.lower().strip() for col in df.columns])
            missing_optional = optional_cols - file_cols
            
            if missing_optional:
                result['warnings'].append(
                    f"Missing optional columns (will use defaults): {list(missing_optional)}"
                )
        
        elif detected_type == 'unknown':
            result['errors'].append(
                f"Cannot identify file type. Expected: {expected_type.upper()}"
            )
            result['errors'].append(
                f"Required columns for {expected_type}: {FILE_SIGNATURES[expected_type]['required']}"
            )
        
        else:
            # Wrong file type detected
            result['errors'].append(
                f"Wrong file type! This looks like a {detected_type.upper()} file, "
                f"but you uploaded it as {expected_type.upper()}"
            )
            result['warnings'].append(
                f"Detected columns suggest this is: {details.get('description', detected_type)}"
            )
        
        # Additional data quality checks
        if result['is_valid']:
            # Check for empty dataframe
            if len(df) == 0:
                result['errors'].append('File contains no data rows')
                result['is_valid'] = False
            
            # Check for mostly empty columns
            empty_cols = [col for col in df.columns if df[col].isna().sum() / len(df) > 0.9]
            if empty_cols:
                result['warnings'].append(
                    f"Columns mostly empty (>90% null): {empty_cols[:5]}"
                )
        
    except Exception as e:
        result['errors'].append(f"Error reading file: {str(e)}")
    
    return result


def render_validation_result(validation_result, file_type):
    """Render validation result with appropriate styling."""
    
    if validation_result['is_valid']:
        st.success(f"✅ Valid {file_type.upper()} file detected!")
        
        # Show details
        details = validation_result['details']
        st.markdown(f"""
        <div style="background: #d4edda; padding: 0.5rem; border-radius: 5px; font-size: 0.8rem; margin-top: 0.3rem;">
            <strong>Columns found:</strong> {len(details.get('found_required', []))} required, {len(details.get('optional_found', []))} optional<br>
            <strong>Total rows:</strong> Will be shown after processing
        </div>
        """, unsafe_allow_html=True)
        
        # Show warnings if any
        for warning in validation_result['warnings']:
            st.warning(f"⚠️ {warning}", icon="⚠️")
    
    elif validation_result['is_correct_type'] and not validation_result['is_valid']:
        # Right type but missing columns
        st.error(f"❌ {file_type.upper()} file has issues")
        for error in validation_result['errors']:
            st.error(f"  • {error}")
        for warning in validation_result['warnings']:
            st.warning(f"  • {warning}")
    
    else:
        # Wrong file type
        detected = validation_result['detected_type']
        if detected != 'unknown':
            st.error(f"🚫 Wrong file! Detected as: **{detected.upper()}**")
            st.markdown(f"""
            <div style="background: #f8d7da; padding: 0.5rem; border-radius: 5px; font-size: 0.8rem;">
                <strong>Expected:</strong> {file_type.upper()} file<br>
                <strong>Got:</strong> {detected.upper()} file<br>
                <strong>Tip:</strong> Please upload this file to the correct uploader
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error(f"❓ Unrecognized file format")
            for error in validation_result['errors']:
                st.markdown(f"<small>• {error}</small>", unsafe_allow_html=True)


def read_and_validate_upload(uploaded_file, expected_type):
    """
    Read uploaded file and validate it.
    
    Returns:
        tuple: (dataframe or None, validation_result)
    """
    if uploaded_file is None:
        return None, None
    
    try:
        df = pd.read_csv(uploaded_file)
        validation = validate_file_for_type(df, expected_type)
        return df, validation
    except Exception as e:
        return None, {
            'is_valid': False,
            'detected_type': 'unknown',
            'confidence': 0,
            'is_correct_type': False,
            'errors': [f"Error reading CSV: {str(e)}"],
            'warnings': [],
            'details': {}
        }


# =============================================================================
# LOGO AND STYLES
# =============================================================================
def render_logo():
    """Render SVG logo for UAE Promo Pulse"""
    logo_html = """
    <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 1rem;">
        <svg width="420" height="100" viewBox="0 0 420 100" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#1E3A5F;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#2E5A8F;stop-opacity:1" />
                </linearGradient>
                <linearGradient id="pulseGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:#00D4AA;stop-opacity:1" />
                    <stop offset="50%" style="stop-color:#00E5BB;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#00F5CC;stop-opacity:1" />
                </linearGradient>
                <linearGradient id="uaeGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" style="stop-color:#00A86B;stop-opacity:1" />
                    <stop offset="33%" style="stop-color:#FFFFFF;stop-opacity:1" />
                    <stop offset="66%" style="stop-color:#000000;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#CE1126;stop-opacity:1" />
                </linearGradient>
                <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
                    <feDropShadow dx="2" dy="2" stdDeviation="3" flood-opacity="0.3"/>
                </filter>
            </defs>
            <rect x="5" y="10" width="410" height="80" rx="15" ry="15" fill="url(#bgGrad)" filter="url(#shadow)"/>
            <rect x="5" y="10" width="8" height="80" rx="4" ry="0" fill="url(#uaeGrad)"/>
            <g transform="translate(25, 25)">
                <circle cx="25" cy="25" r="22" fill="rgba(255,255,255,0.1)" stroke="url(#pulseGrad)" stroke-width="2"/>
                <path d="M15 20 L18 20 L22 35 L35 35 L38 23 L20 23" fill="none" stroke="url(#pulseGrad)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                <circle cx="24" cy="42" r="3" fill="url(#pulseGrad)"/>
                <circle cx="34" cy="42" r="3" fill="url(#pulseGrad)"/>
                <path d="M42 25 Q47 15, 52 25 Q57 35, 62 25" fill="none" stroke="url(#pulseGrad)" stroke-width="2" stroke-linecap="round"/>
            </g>
            <text x="95" y="45" font-family="Arial Black, sans-serif" font-size="22" font-weight="900" fill="#FFFFFF">UAE</text>
            <text x="150" y="45" font-family="Arial Black, sans-serif" font-size="22" font-weight="900" fill="url(#pulseGrad)">PROMO</text>
            <text x="260" y="45" font-family="Arial Black, sans-serif" font-size="22" font-weight="900" fill="#FFFFFF">PULSE</text>
            <text x="95" y="70" font-family="Arial, sans-serif" font-size="11" fill="rgba(255,255,255,0.8)" letter-spacing="1">RETAIL ANALYTICS &amp; PROMOTION SIMULATOR</text>
            <path d="M355 35 L365 35 L370 25 L375 45 L380 30 L385 40 L390 35 L400 35" 
                  fill="none" stroke="url(#pulseGrad)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <animate attributeName="opacity" values="0.5;1;0.5" dur="2s" repeatCount="indefinite"/>
            </path>
        </svg>
    </div>
    """
    st.markdown(logo_html, unsafe_allow_html=True)


def render_sidebar_logo():
    """Render smaller logo for sidebar"""
    sidebar_logo = """
    <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 1rem; padding: 10px;">
        <svg width="200" height="55" viewBox="0 0 200 55" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="sbBgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#1E3A5F;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#2E5A8F;stop-opacity:1" />
                </linearGradient>
                <linearGradient id="sbPulseGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:#00D4AA;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#00F5CC;stop-opacity:1" />
                </linearGradient>
            </defs>
            <rect x="2" y="2" width="196" height="50" rx="10" fill="url(#sbBgGrad)"/>
            <text x="15" y="25" font-family="Arial Black, sans-serif" font-size="11" font-weight="900" fill="#FFFFFF">UAE</text>
            <text x="45" y="25" font-family="Arial Black, sans-serif" font-size="11" font-weight="900" fill="url(#sbPulseGrad)">PROMO PULSE</text>
            <text x="15" y="40" font-family="Arial, sans-serif" font-size="8" fill="rgba(255,255,255,0.7)">Retail Analytics Dashboard</text>
            <path d="M150 20 L160 20 L165 12 L170 28 L175 18 L180 24 L185 20 L195 20" 
                  fill="none" stroke="url(#sbPulseGrad)" stroke-width="1.5" stroke-linecap="round">
                <animate attributeName="opacity" values="0.5;1;0.5" dur="2s" repeatCount="indefinite"/>
            </path>
        </svg>
    </div>
    """
    st.sidebar.markdown(sidebar_logo, unsafe_allow_html=True)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 1.1rem; color: #666; text-align: center; 
        margin-bottom: 1.5rem; margin-top: -0.5rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #1E3A5F 0%, #2E5A8F 100%);
        padding: 1rem; border-radius: 12px; color: white; text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1); height: 100%;
    }
    .kpi-value {font-size: 1.6rem; font-weight: bold; color: #00D4AA;}
    .kpi-label {font-size: 0.8rem; color: rgba(255,255,255,0.8); margin-top: 0.2rem;}
    .kpi-delta {font-size: 0.75rem; margin-top: 0.2rem;}
    .kpi-delta-positive {color: #00D4AA;}
    .kpi-delta-negative {color: #f5576c;}
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.2rem; border-radius: 10px; color: white;
        margin: 0.5rem 0 1rem 0; font-size: 0.9rem; line-height: 1.6;
    }
    .insight-success {background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);}
    .insight-warning {background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);}
    .insight-title {font-weight: bold; font-size: 1rem; margin-bottom: 0.5rem;}
    .section-header {
        background: linear-gradient(90deg, #1E3A5F 0%, transparent 100%);
        padding: 0.7rem 1rem; border-radius: 8px; color: white;
        font-size: 1.1rem; font-weight: 600; margin: 1.2rem 0 0.8rem 0;
    }
    .view-badge {
        display: inline-block; padding: 0.3rem 0.8rem; border-radius: 20px;
        font-size: 0.75rem; font-weight: bold; margin-bottom: 0.5rem;
    }
    .view-executive {background: linear-gradient(90deg, #667eea, #764ba2); color: white;}
    .view-manager {background: linear-gradient(90deg, #11998e, #38ef7d); color: white;}
    
    /* Updated Constraint Cards - Better Readability */
    .constraint-card {
        background: linear-gradient(135deg, #1E3A5F 0%, #2E5A8F 100%);
        border-left: 5px solid #ffc107;
        padding: 1rem 1.2rem; border-radius: 8px; margin: 0.5rem 0;
        color: #ffffff; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    .constraint-card strong {
        color: #ffc107; font-size: 1rem;
    }
    .constraint-card-error {
        background: linear-gradient(135deg, #8B0000 0%, #B22222 100%);
        border-left: 5px solid #ff6b6b;
        padding: 1rem 1.2rem; border-radius: 8px; margin: 0.5rem 0;
        color: #ffffff; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    .constraint-card-error strong {
        color: #ffcccc; font-size: 1rem;
    }
    
    .upload-box {
        background: #f8f9fa; border: 2px dashed #dee2e6; border-radius: 10px;
        padding: 1rem; margin: 0.5rem 0; text-align: center;
    }
    .upload-success {
        background: #d4edda; border: 2px solid #28a745; border-radius: 10px;
        padding: 0.5rem 1rem; margin: 0.3rem 0;
    }
    .file-status-valid {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 4px solid #28a745;
        padding: 0.5rem; border-radius: 5px; margin: 0.3rem 0;
        font-size: 0.8rem;
    }
    .file-status-invalid {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 4px solid #dc3545;
        padding: 0.5rem; border-radius: 5px; margin: 0.3rem 0;
        font-size: 0.8rem;
    }
    .file-status-warning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
        border-left: 4px solid #ffc107;
        padding: 0.5rem; border-radius: 5px; margin: 0.3rem 0;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def kpi_card(label, value, delta=None, delta_type="positive"):
    """Render a styled KPI card"""
    delta_html = ""
    if delta is not None:
        delta_class = f"kpi-delta-{delta_type}"
        delta_icon = "↑" if delta_type == "positive" else "↓"
        delta_html = f'<div class="kpi-delta {delta_class}">{delta_icon} {delta}</div>'
    
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def section_header(title, icon="📊"):
    """Render section header"""
    st.markdown(f'<div class="section-header">{icon} {title}</div>', unsafe_allow_html=True)


def view_badge(view):
    """Render view badge"""
    css = "view-executive" if view == "Executive" else "view-manager"
    emoji = "👔" if view == "Executive" else "🔧"
    st.markdown(f'<span class="view-badge {css}">{emoji} {view} View</span>', unsafe_allow_html=True)


def safe_get_filter(filters, key, default=None):
    """Safely get filter value"""
    val = filters.get(key, default)
    if val == 'All' or val is None:
        return None
    return val


def create_zip_download(data, sim_results, kpis, daily):
    """Create a ZIP file containing all data for download"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add cleaned data files
        if 'products' in data and data['products'] is not None:
            csv_buffer = io.StringIO()
            data['products'].to_csv(csv_buffer, index=False)
            zip_file.writestr('cleaned_products.csv', csv_buffer.getvalue())
        
        if 'stores' in data and data['stores'] is not None:
            csv_buffer = io.StringIO()
            data['stores'].to_csv(csv_buffer, index=False)
            zip_file.writestr('cleaned_stores.csv', csv_buffer.getvalue())
        
        if 'sales' in data and data['sales'] is not None:
            csv_buffer = io.StringIO()
            data['sales'].to_csv(csv_buffer, index=False)
            zip_file.writestr('cleaned_sales.csv', csv_buffer.getvalue())
        
        if 'inventory' in data and data['inventory'] is not None:
            csv_buffer = io.StringIO()
            data['inventory'].to_csv(csv_buffer, index=False)
            zip_file.writestr('cleaned_inventory.csv', csv_buffer.getvalue())
        
        # Add issues log
        if 'issues' in data and data['issues'] is not None and len(data['issues']) > 0:
            csv_buffer = io.StringIO()
            data['issues'].to_csv(csv_buffer, index=False)
            zip_file.writestr('issues_log.csv', csv_buffer.getvalue())
        
        # Add KPIs summary
        if kpis:
            csv_buffer = io.StringIO()
            pd.DataFrame([kpis]).to_csv(csv_buffer, index=False)
            zip_file.writestr('kpis_summary.csv', csv_buffer.getvalue())
        
        # Add daily data
        if daily is not None and len(daily) > 0:
            csv_buffer = io.StringIO()
            daily.to_csv(csv_buffer, index=False)
            zip_file.writestr('daily_metrics.csv', csv_buffer.getvalue())
        
        # Add simulation results
        if sim_results and sim_results.get('results'):
            csv_buffer = io.StringIO()
            pd.DataFrame([sim_results['results']]).to_csv(csv_buffer, index=False)
            zip_file.writestr('simulation_results.csv', csv_buffer.getvalue())
        
        # Add top risk items
        if sim_results and sim_results.get('top_risk_items') is not None:
            top_risk = sim_results.get('top_risk_items')
            if len(top_risk) > 0:
                csv_buffer = io.StringIO()
                top_risk.to_csv(csv_buffer, index=False)
                zip_file.writestr('top_risk_items.csv', csv_buffer.getvalue())
        
        # Add simulation detail
        if sim_results and sim_results.get('simulation_detail') is not None:
            sim_detail = sim_results.get('simulation_detail')
            if len(sim_detail) > 0:
                csv_buffer = io.StringIO()
                sim_detail.to_csv(csv_buffer, index=False)
                zip_file.writestr('simulation_detail.csv', csv_buffer.getvalue())
        
        # Add README file
        readme_content = f"""UAE Promo Pulse - Data Export
==============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Files Included:
---------------
1. cleaned_products.csv - Cleaned product master data
2. cleaned_stores.csv - Cleaned store master data
3. cleaned_sales.csv - Cleaned sales transactions
4. cleaned_inventory.csv - Cleaned inventory snapshots
5. issues_log.csv - Data quality issues detected during cleaning
6. kpis_summary.csv - Key Performance Indicators
7. daily_metrics.csv - Daily aggregated metrics
8. simulation_results.csv - What-If simulation results
9. top_risk_items.csv - Top stockout risk items
10. simulation_detail.csv - Detailed simulation data

Dashboard: UAE Promo Pulse
Version: 2.0
"""
        zip_file.writestr('README.txt', readme_content)
    
    zip_buffer.seek(0)
    return zip_buffer


def validate_uploaded_data(products_df, stores_df, sales_df, inventory_df):
    """Validate uploaded CSV files have required columns"""
    errors = []
    warnings = []
    
    # Required columns for each file
    required_cols = {
        'products': ['product_id', 'category', 'base_price_aed'],
        'stores': ['store_id', 'city', 'channel'],
        'sales': ['order_id', 'order_time', 'product_id', 'store_id', 'qty', 'selling_price_aed', 'payment_status'],
        'inventory': ['snapshot_date', 'product_id', 'store_id', 'stock_on_hand']
    }
    
    # Optional but recommended columns
    optional_cols = {
        'products': ['unit_cost_aed', 'brand', 'tax_rate', 'launch_flag'],
        'stores': ['fulfillment_type', 'store_name', 'opening_date'],
        'sales': ['discount_pct', 'return_flag', 'customer_id'],
        'inventory': ['reorder_point', 'lead_time_days']
    }
    
    # Check products
    if products_df is not None:
        missing = [c for c in required_cols['products'] if c not in products_df.columns]
        if missing:
            errors.append(f"Products CSV missing required columns: {missing}")
        opt_missing = [c for c in optional_cols['products'] if c not in products_df.columns]
        if opt_missing:
            warnings.append(f"Products CSV missing optional columns (will use defaults): {opt_missing}")
    else:
        errors.append("Products CSV is required")
    
    # Check stores
    if stores_df is not None:
        missing = [c for c in required_cols['stores'] if c not in stores_df.columns]
        if missing:
            errors.append(f"Stores CSV missing required columns: {missing}")
    else:
        errors.append("Stores CSV is required")
    
    # Check sales
    if sales_df is not None:
        missing = [c for c in required_cols['sales'] if c not in sales_df.columns]
        if missing:
            errors.append(f"Sales CSV missing required columns: {missing}")
        opt_missing = [c for c in optional_cols['sales'] if c not in sales_df.columns]
        if opt_missing:
            warnings.append(f"Sales CSV missing optional columns (will use defaults): {opt_missing}")
    else:
        errors.append("Sales CSV is required")
    
    # Check inventory
    if inventory_df is not None:
        missing = [c for c in required_cols['inventory'] if c not in inventory_df.columns]
        if missing:
            errors.append(f"Inventory CSV missing required columns: {missing}")
    else:
        errors.append("Inventory CSV is required")
    
    return errors, warnings


def add_default_columns(df, file_type):
    """Add default values for missing optional columns"""
    if df is None:
        return df
    
    df = df.copy()
    
    if file_type == 'products':
        if 'unit_cost_aed' not in df.columns:
            df['unit_cost_aed'] = df['base_price_aed'] * 0.6
        if 'brand' not in df.columns:
            df['brand'] = 'Unknown'
        if 'tax_rate' not in df.columns:
            df['tax_rate'] = 0.05
        if 'launch_flag' not in df.columns:
            df['launch_flag'] = 'Regular'
    
    elif file_type == 'stores':
        if 'fulfillment_type' not in df.columns:
            df['fulfillment_type'] = 'Own'
        if 'store_name' not in df.columns:
            df['store_name'] = df['store_id']
    
    elif file_type == 'sales':
        if 'discount_pct' not in df.columns:
            df['discount_pct'] = 0
        if 'return_flag' not in df.columns:
            df['return_flag'] = 0
        if 'customer_id' not in df.columns:
            df['customer_id'] = 'CUST_00000'
    
    elif file_type == 'inventory':
        if 'reorder_point' not in df.columns:
            df['reorder_point'] = 20
        if 'lead_time_days' not in df.columns:
            df['lead_time_days'] = 7
    
    return df


# =============================================================================
# CHART FUNCTIONS - All 12 Required Chart Types
# =============================================================================

def create_waterfall_chart(kpis):
    """5. Waterfall Chart - Revenue breakdown"""
    fig = go.Figure(go.Waterfall(
        name="Revenue Flow",
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["Gross Revenue", "Refunds", "Returns", "COGS", "Net Profit"],
        textposition="outside",
        text=[f"AED {kpis['gross_revenue']:,.0f}", 
              f"-AED {kpis['refund_amount']:,.0f}",
              f"-AED {kpis.get('returns_amount', 0):,.0f}",
              f"-AED {kpis['cogs']:,.0f}",
              f"AED {kpis['gross_margin']:,.0f}"],
        y=[kpis['gross_revenue'], 
           -kpis['refund_amount'], 
           -kpis.get('returns_amount', 0),
           -kpis['cogs'], 
           kpis['gross_margin']],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#00D4AA"}},
        decreasing={"marker": {"color": "#f5576c"}},
        totals={"marker": {"color": "#1E3A5F"}}
    ))
    fig.update_layout(
        title="Revenue Waterfall Analysis",
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def create_historical_prediction_chart(daily_df):
    """2. Historical Prediction Chart - Revenue with forecast"""
    if len(daily_df) < 7:
        return None
    
    df = daily_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    df['ma7'] = df['revenue'].rolling(7, min_periods=1).mean()
    df['ma14'] = df['revenue'].rolling(14, min_periods=1).mean()
    
    last_ma = df['ma7'].iloc[-1]
    trend = (df['ma7'].iloc[-1] - df['ma7'].iloc[-7]) / 7 if len(df) >= 7 else 0
    
    future_dates = pd.date_range(start=df['date'].max() + timedelta(days=1), periods=7)
    forecast = [last_ma + trend * (i+1) for i in range(7)]
    forecast_upper = [f * 1.15 for f in forecast]
    forecast_lower = [f * 0.85 for f in forecast]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['revenue'],
        mode='lines+markers',
        name='Actual Revenue',
        line=dict(color='#1E3A5F', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['ma7'],
        mode='lines',
        name='7-Day MA',
        line=dict(color='#00D4AA', width=2, dash='dot')
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates, y=forecast,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#667eea', width=2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates)[::-1],
        y=forecast_upper + forecast_lower[::-1],
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval'
    ))
    
    fig.update_layout(
        title="Historical Revenue with Forecast",
        height=400,
        legend=dict(orientation="h", y=-0.15),
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Date",
        yaxis_title="Revenue (AED)"
    )
    return fig


def create_outlier_detection_plot(sales_df):
    """4. Outlier Detection Plot - Price and Qty anomalies"""
    if len(sales_df) == 0:
        return None
    
    df = sales_df.copy()
    
    df['price_zscore'] = (df['selling_price_aed'] - df['selling_price_aed'].mean()) / (df['selling_price_aed'].std() + 0.001)
    df['qty_zscore'] = (df['qty'] - df['qty'].mean()) / (df['qty'].std() + 0.001)
    
    df['is_outlier'] = ((abs(df['price_zscore']) > 2) | (abs(df['qty_zscore']) > 2)).astype(int)
    
    sample = df.sample(min(1000, len(df)))
    
    fig = px.scatter(
        sample,
        x='qty',
        y='selling_price_aed',
        color='is_outlier',
        color_discrete_map={0: '#00D4AA', 1: '#f5576c'},
        opacity=0.6,
        title="Outlier Detection: Price vs Quantity",
        labels={'is_outlier': 'Outlier', 'qty': 'Quantity', 'selling_price_aed': 'Price (AED)'}
    )
    
    price_upper = df['selling_price_aed'].mean() + 2 * df['selling_price_aed'].std()
    qty_upper = df['qty'].mean() + 2 * df['qty'].std()
    
    fig.add_hline(y=price_upper, line_dash="dash", line_color="red", 
                  annotation_text=f"Price threshold: {price_upper:.0f}")
    fig.add_vline(x=qty_upper, line_dash="dash", line_color="red",
                  annotation_text=f"Qty threshold: {qty_upper:.0f}")
    
    fig.update_layout(
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        legend_title="Is Outlier"
    )
    return fig


def create_comparison_matrix(sales_df, products_df, stores_df):
    """6. Comparison Matrix - Channel vs Category Revenue Heatmap"""
    if len(sales_df) == 0:
        return None
    
    df = sales_df.copy()
    
    if 'category' not in df.columns:
        df = df.merge(products_df[['product_id', 'category']], on='product_id', how='left')
    if 'channel' not in df.columns:
        df = df.merge(stores_df[['store_id', 'channel']], on='store_id', how='left')
    
    if 'category' not in df.columns or 'channel' not in df.columns:
        return None
    
    pivot = df.pivot_table(
        values='line_total',
        index='category',
        columns='channel',
        aggfunc='sum',
        fill_value=0
    )
    
    pivot = pivot / 1000
    
    fig = px.imshow(
        pivot,
        text_auto='.0f',
        color_continuous_scale='Blues',
        title="Revenue Matrix: Category vs Channel (in '000 AED)",
        labels=dict(x="Channel", y="Category", color="Revenue (K)")
    )
    
    fig.update_layout(height=400)
    return fig


def create_margin_profitability_chart(breakdown_df):
    """7. Enhanced Margin Profitability Chart"""
    if len(breakdown_df) == 0:
        return None
    
    df = breakdown_df.copy()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Revenue vs Margin', 'Profitability Quadrant'),
        specs=[[{"type": "bar"}, {"type": "scatter"}]]
    )
    
    fig.add_trace(
        go.Bar(name='Revenue', x=df['category'], y=df['revenue']/1000, marker_color='#1E3A5F'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name='Margin', x=df['category'], y=df['margin']/1000, marker_color='#00D4AA'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['revenue']/1000,
            y=df['margin_pct'],
            mode='markers+text',
            text=df['category'],
            textposition='top center',
            marker=dict(size=df['orders']/df['orders'].max()*50 + 10, color='#667eea', opacity=0.7),
            name='Categories'
        ),
        row=1, col=2
    )
    
    avg_revenue = df['revenue'].mean() / 1000
    avg_margin = df['margin_pct'].mean()
    fig.add_hline(y=avg_margin, line_dash="dash", line_color="gray", row=1, col=2)
    fig.add_vline(x=avg_revenue, line_dash="dash", line_color="gray", row=1, col=2)
    
    fig.update_layout(
        height=400,
        title="Margin Profitability Analysis",
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_xaxes(title_text="Revenue (K AED)", row=1, col=2)
    fig.update_yaxes(title_text="Margin %", row=1, col=2)
    
    return fig


def create_growth_trends_chart(daily_df):
    """9. Growth Trends - Week over Week"""
    if len(daily_df) < 14:
        return None
    
    df = daily_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['week'] = df['date'].dt.isocalendar().week
    df['year'] = df['date'].dt.year
    
    weekly = df.groupby(['year', 'week']).agg({
        'revenue': 'sum',
        'orders': 'sum',
        'units': 'sum'
    }).reset_index()
    
    weekly['week_label'] = weekly['year'].astype(str) + '-W' + weekly['week'].astype(str)
    weekly['revenue_growth'] = weekly['revenue'].pct_change() * 100
    weekly['orders_growth'] = weekly['orders'].pct_change() * 100
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=weekly['week_label'],
            y=weekly['revenue']/1000,
            name='Revenue (K)',
            marker_color='#1E3A5F'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=weekly['week_label'],
            y=weekly['revenue_growth'],
            name='WoW Growth %',
            line=dict(color='#00D4AA', width=3),
            mode='lines+markers'
        ),
        secondary_y=True
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", secondary_y=True)
    
    fig.update_layout(
        title="Weekly Growth Trends",
        height=400,
        legend=dict(orientation="h", y=-0.15),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_yaxes(title_text="Revenue (K AED)", secondary_y=False)
    fig.update_yaxes(title_text="Growth %", secondary_y=True)
    
    return fig


def create_cumulative_performance(daily_df):
    """10. Cumulative Performance Tracker"""
    if len(daily_df) == 0:
        return None
    
    df = daily_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    df['cum_revenue'] = df['revenue'].cumsum()
    df['cum_orders'] = df['orders'].cumsum()
    df['cum_units'] = df['units'].cumsum()
    
    total_days = len(df)
    target_daily = df['revenue'].sum() * 1.1 / total_days
    df['target'] = target_daily * (df.index + 1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['cum_revenue'],
        mode='lines',
        name='Actual Cumulative',
        line=dict(color='#1E3A5F', width=3),
        fill='tozeroy',
        fillcolor='rgba(30,58,95,0.1)'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['target'],
        mode='lines',
        name='Target',
        line=dict(color='#f5576c', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Cumulative Revenue Performance vs Target",
        height=400,
        legend=dict(orientation="h", y=-0.15),
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Date",
        yaxis_title="Cumulative Revenue (AED)"
    )
    
    return fig


def create_performance_matrix(kpis_by_dimension):
    """11. Performance Matrix - KPI Heatmap by City/Channel"""
    if len(kpis_by_dimension) == 0:
        return None
    
    df = kpis_by_dimension.copy()
    
    metrics = ['revenue', 'margin_pct', 'orders', 'avg_discount']
    for m in metrics:
        if m in df.columns:
            min_val = df[m].min()
            max_val = df[m].max()
            df[f'{m}_norm'] = (df[m] - min_val) / (max_val - min_val + 0.001) * 100
    
    norm_cols = [c for c in df.columns if '_norm' in c]
    
    if len(norm_cols) == 0:
        return None
    
    dimension_col = df.columns[0]
    heat_data = df.set_index(dimension_col)[norm_cols]
    heat_data.columns = [c.replace('_norm', '').title() for c in heat_data.columns]
    
    fig = px.imshow(
        heat_data,
        text_auto='.0f',
        color_continuous_scale='RdYlGn',
        title="Performance Matrix (Normalized Scores 0-100)",
        labels=dict(x="Metric", y=dimension_col.title(), color="Score")
    )
    
    fig.update_layout(height=350)
    return fig


def create_whatif_heatmap(simulator, sim_params, filters):
    """12. What-If Analysis Heatmap - Discount vs Category"""
    discounts = [5, 10, 15, 20, 25, 30]
    categories = ['Electronics', 'Fashion', 'Grocery', 'Home & Garden', 'Beauty', 'Sports']
    
    profit_matrix = []
    
    for cat in categories:
        row = []
        for disc in discounts:
            try:
                result = simulator.run_simulation(
                    disc,
                    sim_params['promo_budget'],
                    sim_params['margin_floor'],
                    sim_params['simulation_days'],
                    city=safe_get_filter(filters, 'city'),
                    channel=safe_get_filter(filters, 'channel'),
                    category=cat
                )
                profit = result['results'].get('profit_proxy', 0) if result.get('results') else 0
                row.append(profit / 1000)
            except Exception:
                row.append(0)
        profit_matrix.append(row)
    
    fig = px.imshow(
        profit_matrix,
        x=[f'{d}%' for d in discounts],
        y=categories,
        color_continuous_scale='RdYlGn',
        text_auto='.0f',
        title="What-If Analysis: Profit by Category & Discount (K AED)",
        labels=dict(x="Discount Level", y="Category", color="Profit (K)")
    )
    
    fig.update_layout(height=400)
    return fig


def create_donut_chart(breakdown_df, dimension, value_col='revenue'):
    """8. Donut Chart"""
    if len(breakdown_df) == 0:
        return None
    
    fig = px.pie(
        breakdown_df,
        values=value_col,
        names=dimension,
        hole=0.5,
        title=f"Distribution by {dimension.title()}",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=350, showlegend=True)
    
    return fig


def create_dual_axis_growth_target(daily_df, target_multiplier=1.1):
    """3. Dual Axis Chart - Revenue with Growth Rate"""
    if len(daily_df) < 7:
        return None
    
    df = daily_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    df['growth_rate'] = df['revenue'].pct_change() * 100
    df['target'] = df['revenue'].mean() * target_multiplier
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=df['date'], y=df['revenue'],
            name='Revenue',
            fill='tozeroy',
            fillcolor='rgba(30,58,95,0.3)',
            line=dict(color='#1E3A5F', width=2)
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['date'], y=[df['target'].iloc[0]] * len(df),
            name='Target',
            line=dict(color='#f5576c', width=2, dash='dash')
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['date'], y=df['growth_rate'],
            name='Daily Growth %',
            line=dict(color='#00D4AA', width=2),
            mode='lines+markers'
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Revenue Performance with Growth Rate (Dual Axis)",
        height=400,
        legend=dict(orientation="h", y=-0.15),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_yaxes(title_text="Revenue (AED)", secondary_y=False)
    fig.update_yaxes(title_text="Growth Rate %", secondary_y=True)
    
    return fig


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'raw_data_generated' not in st.session_state:
    st.session_state.raw_data_generated = False
if 'cleaning_stats' not in st.session_state:
    st.session_state.cleaning_stats = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = 'generate'

# Validation states for each file
if 'validation_products' not in st.session_state:
    st.session_state.validation_products = None
if 'validation_stores' not in st.session_state:
    st.session_state.validation_stores = None
if 'validation_sales' not in st.session_state:
    st.session_state.validation_sales = None
if 'validation_inventory' not in st.session_state:
    st.session_state.validation_inventory = None

# Uploaded dataframes
if 'uploaded_products' not in st.session_state:
    st.session_state.uploaded_products = None
if 'uploaded_stores' not in st.session_state:
    st.session_state.uploaded_stores = None
if 'uploaded_sales' not in st.session_state:
    st.session_state.uploaded_sales = None
if 'uploaded_inventory' not in st.session_state:
    st.session_state.uploaded_inventory = None


# =============================================================================
# SIDEBAR
# =============================================================================
render_sidebar_logo()

st.sidebar.markdown("## 🎛️ Dashboard Controls")

# View Toggle
view = st.sidebar.radio(
    "📊 Select View",
    ["Executive (CEO/CFO)", "Manager (Ops)"],
    help="Executive: Financial KPIs | Manager: Operational KPIs"
)
view_type = "Executive" if "Executive" in view else "Manager"

st.sidebar.markdown("---")

# =============================================================================
# DATA SOURCE SELECTION
# =============================================================================
st.sidebar.markdown("### 📁 Data Source")

data_source_option = st.sidebar.radio(
    "Choose data source:",
    ["🎲 Generate Sample Data", "📤 Upload Your Own Data"],
    help="Generate synthetic data for demo or upload your own CSV files"
)

if "Generate" in data_source_option:
    st.session_state.data_source = 'generate'
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🎲 Generate", use_container_width=True):
            with st.spinner("Generating synthetic data..."):
                st.session_state.raw_data = generate_all_data('data/raw')
                st.session_state.raw_data_generated = True
                st.session_state.data_loaded = False
            st.rerun()

    with col2:
        if st.button("🧹 Clean", use_container_width=True, 
                    disabled=not st.session_state.raw_data_generated):
            if st.session_state.raw_data:
                with st.spinner("Cleaning data..."):
                    orig = {}
                    for k, v in st.session_state.raw_data.items():
                        if isinstance(v, pd.DataFrame):
                            orig[k] = len(v)
                    
                    cleaner = DataCleaner()
                    cleaned = cleaner.clean_all(
                        st.session_state.raw_data['products'],
                        st.session_state.raw_data['stores'],
                        st.session_state.raw_data['sales'],
                        st.session_state.raw_data['inventory'],
                        'data/cleaned'
                    )
                    st.session_state.data = cleaned
                    
                    cc = {}
                    for k, v in cleaned.items():
                        if isinstance(v, pd.DataFrame) and k != 'issues':
                            cc[k] = len(v)
                    
                    st.session_state.cleaning_stats = {
                        'original': orig,
                        'cleaned': cc,
                        'removed': {k: orig.get(k, 0) - cc.get(k, 0) for k in cc},
                        'total_issues': len(cleaned['issues']),
                        'issues_summary': cleaned['issues']['issue_type'].value_counts().to_dict() if len(cleaned['issues']) > 0 else {}
                    }
                    st.session_state.data_loaded = True
                st.rerun()

    if st.session_state.raw_data_generated and not st.session_state.data_loaded:
        st.sidebar.success("✅ Raw data generated!")
    if st.session_state.data_loaded and st.session_state.data_source == 'generate':
        st.sidebar.success("✅ Data cleaned & ready!")

else:
    # UPLOAD YOUR OWN DATA - Enhanced with immediate validation
    st.session_state.data_source = 'upload'
    
    st.sidebar.markdown("#### 📤 Upload CSV Files")
    st.sidebar.markdown("<small>Files are validated immediately on upload</small>", unsafe_allow_html=True)
    
    # =========================================================================
    # PRODUCTS UPLOAD
    # =========================================================================
    st.sidebar.markdown("---")
    uploaded_products_file = st.sidebar.file_uploader(
        "📦 Products CSV",
        type=['csv'],
        key='upload_products_file',
        help="Required: product_id, category, base_price_aed"
    )
    
    if uploaded_products_file is not None:
        # Read and validate immediately
        df, validation = read_and_validate_upload(uploaded_products_file, 'products')
        st.session_state.uploaded_products = df
        st.session_state.validation_products = validation
        
        if validation:
            if validation['is_valid']:
                st.sidebar.markdown(f"""
                <div class="file-status-valid">
                    ✅ <strong>Products:</strong> Valid ({len(df):,} rows)<br>
                    <small>Found: {', '.join(validation['details'].get('found_required', []))}</small>
                </div>
                """, unsafe_allow_html=True)
            elif validation['is_correct_type']:
                st.sidebar.markdown(f"""
                <div class="file-status-warning">
                    ⚠️ <strong>Products:</strong> Missing columns<br>
                    <small>{', '.join(validation['details'].get('missing_required', []))}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                detected = validation['detected_type']
                st.sidebar.markdown(f"""
                <div class="file-status-invalid">
                    🚫 <strong>Wrong file!</strong> This is a <strong>{detected.upper()}</strong> file<br>
                    <small>Please upload a Products file here</small>
                </div>
                """, unsafe_allow_html=True)
    
    # =========================================================================
    # STORES UPLOAD
    # =========================================================================
    uploaded_stores_file = st.sidebar.file_uploader(
        "🏪 Stores CSV",
        type=['csv'],
        key='upload_stores_file',
        help="Required: store_id, city, channel"
    )
    
    if uploaded_stores_file is not None:
        df, validation = read_and_validate_upload(uploaded_stores_file, 'stores')
        st.session_state.uploaded_stores = df
        st.session_state.validation_stores = validation
        
        if validation:
            if validation['is_valid']:
                st.sidebar.markdown(f"""
                <div class="file-status-valid">
                    ✅ <strong>Stores:</strong> Valid ({len(df):,} rows)<br>
                    <small>Found: {', '.join(validation['details'].get('found_required', []))}</small>
                </div>
                """, unsafe_allow_html=True)
            elif validation['is_correct_type']:
                st.sidebar.markdown(f"""
                <div class="file-status-warning">
                    ⚠️ <strong>Stores:</strong> Missing columns<br>
                    <small>{', '.join(validation['details'].get('missing_required', []))}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                detected = validation['detected_type']
                st.sidebar.markdown(f"""
                <div class="file-status-invalid">
                    🚫 <strong>Wrong file!</strong> This is a <strong>{detected.upper()}</strong> file<br>
                    <small>Please upload a Stores file here</small>
                </div>
                """, unsafe_allow_html=True)
    
    # =========================================================================
    # SALES UPLOAD
    # =========================================================================
    uploaded_sales_file = st.sidebar.file_uploader(
        "💰 Sales CSV",
        type=['csv'],
        key='upload_sales_file',
        help="Required: order_id, order_time, product_id, store_id, qty, selling_price_aed, payment_status"
    )
    
    if uploaded_sales_file is not None:
        df, validation = read_and_validate_upload(uploaded_sales_file, 'sales')
        st.session_state.uploaded_sales = df
        st.session_state.validation_sales = validation
        
        if validation:
            if validation['is_valid']:
                st.sidebar.markdown(f"""
                <div class="file-status-valid">
                    ✅ <strong>Sales:</strong> Valid ({len(df):,} rows)<br>
                    <small>Found: order_id, order_time, qty, +{len(validation['details'].get('found_required', []))-3} more</small>
                </div>
                """, unsafe_allow_html=True)
            elif validation['is_correct_type']:
                st.sidebar.markdown(f"""
                <div class="file-status-warning">
                    ⚠️ <strong>Sales:</strong> Missing columns<br>
                    <small>{', '.join(validation['details'].get('missing_required', []))}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                detected = validation['detected_type']
                st.sidebar.markdown(f"""
                <div class="file-status-invalid">
                    🚫 <strong>Wrong file!</strong> This is a <strong>{detected.upper()}</strong> file<br>
                    <small>Please upload a Sales file here</small>
                </div>
                """, unsafe_allow_html=True)
    
    # =========================================================================
    # INVENTORY UPLOAD
    # =========================================================================
    uploaded_inventory_file = st.sidebar.file_uploader(
        "📊 Inventory CSV",
        type=['csv'],
        key='upload_inventory_file',
        help="Required: snapshot_date, product_id, store_id, stock_on_hand"
    )
    
    if uploaded_inventory_file is not None:
        df, validation = read_and_validate_upload(uploaded_inventory_file, 'inventory')
        st.session_state.uploaded_inventory = df
        st.session_state.validation_inventory = validation
        
        if validation:
            if validation['is_valid']:
                st.sidebar.markdown(f"""
                <div class="file-status-valid">
                    ✅ <strong>Inventory:</strong> Valid ({len(df):,} rows)<br>
                    <small>Found: {', '.join(validation['details'].get('found_required', []))}</small>
                </div>
                """, unsafe_allow_html=True)
            elif validation['is_correct_type']:
                st.sidebar.markdown(f"""
                <div class="file-status-warning">
                    ⚠️ <strong>Inventory:</strong> Missing columns<br>
                    <small>{', '.join(validation['details'].get('missing_required', []))}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                detected = validation['detected_type']
                st.sidebar.markdown(f"""
                <div class="file-status-invalid">
                    🚫 <strong>Wrong file!</strong> This is a <strong>{detected.upper()}</strong> file<br>
                    <small>Please upload an Inventory file here</small>
                </div>
                """, unsafe_allow_html=True)
    
    # =========================================================================
    # CHECK ALL FILES VALID AND PROCESS
    # =========================================================================
    st.sidebar.markdown("---")
    
    # Check validation status
    all_valid = (
        st.session_state.validation_products is not None and 
        st.session_state.validation_products.get('is_valid', False) and
        st.session_state.validation_stores is not None and 
        st.session_state.validation_stores.get('is_valid', False) and
        st.session_state.validation_sales is not None and 
        st.session_state.validation_sales.get('is_valid', False) and
        st.session_state.validation_inventory is not None and 
        st.session_state.validation_inventory.get('is_valid', False)
    )
    
    # Check if any wrong file type was detected
    wrong_files = []
    for file_type, validation_key in [
        ('Products', 'validation_products'),
        ('Stores', 'validation_stores'),
        ('Sales', 'validation_sales'),
        ('Inventory', 'validation_inventory')
    ]:
        val = getattr(st.session_state, validation_key, None)
        if val and not val.get('is_correct_type', True) and val.get('detected_type', 'unknown') != 'unknown':
            wrong_files.append(f"{file_type} → detected as {val['detected_type'].upper()}")
    
    if wrong_files:
        st.sidebar.error("🚫 File mismatch detected!")
        for wf in wrong_files:
            st.sidebar.markdown(f"<small>• {wf}</small>", unsafe_allow_html=True)
    
    if all_valid:
        st.sidebar.success("✅ All files validated!")
        
        if st.sidebar.button("🔄 Process & Clean Data", use_container_width=True, type="primary"):
            with st.spinner("Processing uploaded data..."):
                try:
                    products_df = st.session_state.uploaded_products
                    stores_df = st.session_state.uploaded_stores
                    sales_df = st.session_state.uploaded_sales
                    inventory_df = st.session_state.uploaded_inventory
                    
                    # Add default columns
                    products_df = add_default_columns(products_df, 'products')
                    stores_df = add_default_columns(stores_df, 'stores')
                    sales_df = add_default_columns(sales_df, 'sales')
                    inventory_df = add_default_columns(inventory_df, 'inventory')
                    
                    # Store original counts
                    orig = {
                        'products': len(products_df),
                        'stores': len(stores_df),
                        'sales': len(sales_df),
                        'inventory': len(inventory_df)
                    }
                    
                    # Clean data
                    cleaner = DataCleaner()
                    cleaned = cleaner.clean_all(
                        products_df, stores_df, sales_df, inventory_df,
                        'data/cleaned'
                    )
                    
                    st.session_state.data = cleaned
                    
                    cc = {}
                    for k, v in cleaned.items():
                        if isinstance(v, pd.DataFrame) and k != 'issues':
                            cc[k] = len(v)
                    
                    st.session_state.cleaning_stats = {
                        'original': orig,
                        'cleaned': cc,
                        'removed': {k: orig.get(k, 0) - cc.get(k, 0) for k in cc},
                        'total_issues': len(cleaned['issues']),
                        'issues_summary': cleaned['issues']['issue_type'].value_counts().to_dict() if len(cleaned['issues']) > 0 else {}
                    }
                    st.session_state.data_loaded = True
                    st.session_state.raw_data_generated = False
                    
                    st.sidebar.success("✅ Data processed successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.sidebar.error(f"Error processing files: {str(e)}")
    else:
        # Show which files are missing or invalid
        missing = []
        if not st.session_state.validation_products or not st.session_state.validation_products.get('is_valid'):
            missing.append("Products")
        if not st.session_state.validation_stores or not st.session_state.validation_stores.get('is_valid'):
            missing.append("Stores")
        if not st.session_state.validation_sales or not st.session_state.validation_sales.get('is_valid'):
            missing.append("Sales")
        if not st.session_state.validation_inventory or not st.session_state.validation_inventory.get('is_valid'):
            missing.append("Inventory")
        
        if missing:
            st.sidebar.info(f"📋 Still need valid: {', '.join(missing)}")
        
        st.sidebar.button("🔄 Process & Clean Data", use_container_width=True, disabled=True)


# =============================================================================
# FILTERS
# =============================================================================
st.sidebar.markdown("---")
filters = {}

if st.session_state.data_loaded and st.session_state.data:
    st.sidebar.markdown("### 🔍 Filters")
    data = st.session_state.data
    
    # Date Range
    if 'sales' in data and data['sales'] is not None:
        try:
            sales_dates = pd.to_datetime(data['sales']['order_time'], errors='coerce')
            valid_dates = sales_dates.dropna()
            if len(valid_dates) > 0:
                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()
                date_range = st.sidebar.date_input(
                    "📅 Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    filters['date_from'] = date_range[0]
                    filters['date_to'] = date_range[1]
        except Exception:
            pass
    
    # City
    if 'stores' in data and 'city' in data['stores'].columns:
        cities = ['All'] + sorted(data['stores']['city'].dropna().unique().tolist())
        filters['city'] = st.sidebar.selectbox("🏙️ City", cities)
    
    # Channel
    if 'stores' in data and 'channel' in data['stores'].columns:
        channels = ['All'] + sorted(data['stores']['channel'].dropna().unique().tolist())
        filters['channel'] = st.sidebar.selectbox("📱 Channel", channels)
    
    # Category
    if 'products' in data and 'category' in data['products'].columns:
        categories = ['All'] + sorted(data['products']['category'].dropna().unique().tolist())
        filters['category'] = st.sidebar.selectbox("📦 Category", categories)
    
    # Brand
    if 'products' in data and 'brand' in data['products'].columns:
        brands = ['All'] + sorted(data['products']['brand'].dropna().unique().tolist())
        filters['brand'] = st.sidebar.selectbox("🏷️ Brand", brands)

# Simulation Parameters
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎮 Simulation Parameters")

sim_params = {
    'discount_pct': st.sidebar.slider("💸 Discount %", 0, 50, 15),
    'promo_budget': st.sidebar.number_input("💰 Promo Budget (AED)", 10000, 500000, 50000, step=5000),
    'margin_floor': st.sidebar.slider("📉 Margin Floor %", 5, 30, 15),
    'simulation_days': st.sidebar.selectbox("📆 Simulation Days", [7, 14], index=1)
}


# =============================================================================
# MAIN CONTENT
# =============================================================================
render_logo()
st.markdown('<p class="main-header">Advanced Retail Analytics & Promotion Simulator for UAE Market</p>', unsafe_allow_html=True)

view_badge(view_type)

if not st.session_state.data_loaded:
    # Welcome Screen
    st.markdown("---")
    
    if st.session_state.data_source == 'generate' and st.session_state.raw_data_generated:
        st.success("✅ Raw data generated! Click 'Clean' in the sidebar to process.")
        
        if st.session_state.raw_data:
            meta = st.session_state.raw_data.get('metadata', {})
            if meta:
                st.markdown("#### 📊 Generated Data Summary")
                cols = st.columns(4)
                cols[0].metric("Products", f"{meta.get('products_count', 0):,}")
                cols[1].metric("Stores", f"{meta.get('stores_count', 0):,}")
                cols[2].metric("Sales", f"{meta.get('sales_count', 0):,}")
                cols[3].metric("Inventory", f"{meta.get('inventory_records', 0):,}")
    
    elif st.session_state.data_source == 'upload':
        # Upload instructions with enhanced validation info
        st.info("👈 Upload your CSV files in the sidebar. Each file is validated immediately!")
        
        # Show validation summary if any files uploaded
        any_uploaded = any([
            st.session_state.validation_products,
            st.session_state.validation_stores,
            st.session_state.validation_sales,
            st.session_state.validation_inventory
        ])
        
        if any_uploaded:
            st.markdown("### 📋 Upload Status")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                val = st.session_state.validation_products
                if val:
                    if val['is_valid']:
                        st.success(f"✅ Products\n{len(st.session_state.uploaded_products):,} rows")
                    elif val['is_correct_type']:
                        st.warning(f"⚠️ Products\nMissing columns")
                    else:
                        st.error(f"🚫 Wrong file\nGot: {val['detected_type']}")
                else:
                    st.info("📦 Products\nNot uploaded")
            
            with col2:
                val = st.session_state.validation_stores
                if val:
                    if val['is_valid']:
                        st.success(f"✅ Stores\n{len(st.session_state.uploaded_stores):,} rows")
                    elif val['is_correct_type']:
                        st.warning(f"⚠️ Stores\nMissing columns")
                    else:
                        st.error(f"🚫 Wrong file\nGot: {val['detected_type']}")
                else:
                    st.info("🏪 Stores\nNot uploaded")
            
            with col3:
                val = st.session_state.validation_sales
                if val:
                    if val['is_valid']:
                        st.success(f"✅ Sales\n{len(st.session_state.uploaded_sales):,} rows")
                    elif val['is_correct_type']:
                        st.warning(f"⚠️ Sales\nMissing columns")
                    else:
                        st.error(f"🚫 Wrong file\nGot: {val['detected_type']}")
                else:
                    st.info("💰 Sales\nNot uploaded")
            
            with col4:
                val = st.session_state.validation_inventory
                if val:
                    if val['is_valid']:
                        st.success(f"✅ Inventory\n{len(st.session_state.uploaded_inventory):,} rows")
                    elif val['is_correct_type']:
                        st.warning(f"⚠️ Inventory\nMissing columns")
                    else:
                        st.error(f"🚫 Wrong file\nGot: {val['detected_type']}")
                else:
                    st.info("📊 Inventory\nNot uploaded")
            
            st.markdown("---")
        
        st.markdown("### 📋 Required Data Format")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 📦 Products CSV
            **Required columns:**
            - `product_id` - Unique product identifier
            - `category` - Product category
            - `base_price_aed` - Base price in AED
            
            **Optional columns:**
            - `unit_cost_aed` - Unit cost (defaults to 60% of base price)
            - `brand` - Brand name
            - `tax_rate` - Tax rate (default: 0.05)
            - `launch_flag` - New/Regular
            """)
            
            st.markdown("""
            #### 🏪 Stores CSV
            **Required columns:**
            - `store_id` - Unique store identifier
            - `city` - City name (e.g., Dubai, Abu Dhabi, Sharjah)
            - `channel` - Sales channel (App, Web, Marketplace)
            
            **Optional columns:**
            - `fulfillment_type` - Own/3PL
            - `store_name` - Store name
            """)
        
        with col2:
            st.markdown("""
            #### 💰 Sales CSV
            **Required columns:**
            - `order_id` - Unique order identifier
            - `order_time` - Order timestamp (YYYY-MM-DD HH:MM:SS)
            - `product_id` - Product ID (must match Products)
            - `store_id` - Store ID (must match Stores)
            - `qty` - Quantity sold
            - `selling_price_aed` - Selling price in AED
            - `payment_status` - Paid/Failed/Refunded
            
            **Optional columns:**
            - `discount_pct` - Discount percentage (default: 0)
            - `return_flag` - 0/1 for returns
            - `customer_id` - Customer identifier
            """)
            
            st.markdown("""
            #### 📊 Inventory CSV
            **Required columns:**
            - `snapshot_date` - Date of inventory snapshot (YYYY-MM-DD)
            - `product_id` - Product ID (must match Products)
            - `store_id` - Store ID (must match Stores)
            - `stock_on_hand` - Current stock quantity
            
            **Optional columns:**
            - `reorder_point` - Reorder threshold
            - `lead_time_days` - Lead time in days
            """)
        
        # Download templates
        st.markdown("---")
        st.markdown("### 📥 Download Sample Templates")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            sample_products = pd.DataFrame({
                'product_id': ['PROD_0001', 'PROD_0002', 'PROD_0003'],
                'category': ['Electronics', 'Fashion', 'Grocery'],
                'brand': ['Samsung', 'Nike', 'Almarai'],
                'base_price_aed': [1500, 350, 25],
                'unit_cost_aed': [900, 200, 15]
            })
            st.download_button(
                "📦 Products Template",
                sample_products.to_csv(index=False),
                "products_template.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            sample_stores = pd.DataFrame({
                'store_id': ['STORE_01', 'STORE_02', 'STORE_03'],
                'city': ['Dubai', 'Abu Dhabi', 'Sharjah'],
                'channel': ['App', 'Web', 'Marketplace'],
                'fulfillment_type': ['Own', '3PL', 'Own']
            })
            st.download_button(
                "🏪 Stores Template",
                sample_stores.to_csv(index=False),
                "stores_template.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col3:
            sample_sales = pd.DataFrame({
                'order_id': ['ORD_000001', 'ORD_000002', 'ORD_000003'],
                'order_time': ['2024-01-15 10:30:00', '2024-01-15 11:45:00', '2024-01-16 09:00:00'],
                'product_id': ['PROD_0001', 'PROD_0002', 'PROD_0003'],
                'store_id': ['STORE_01', 'STORE_02', 'STORE_03'],
                'qty': [1, 2, 5],
                'selling_price_aed': [1350, 315, 22.50],
                'discount_pct': [10, 10, 10],
                'payment_status': ['Paid', 'Paid', 'Paid'],
                'return_flag': [0, 0, 0]
            })
            st.download_button(
                "💰 Sales Template",
                sample_sales.to_csv(index=False),
                "sales_template.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col4:
            sample_inventory = pd.DataFrame({
                'snapshot_date': ['2024-01-15', '2024-01-15', '2024-01-15'],
                'product_id': ['PROD_0001', 'PROD_0002', 'PROD_0003'],
                'store_id': ['STORE_01', 'STORE_02', 'STORE_03'],
                'stock_on_hand': [50, 120, 500],
                'reorder_point': [10, 20, 50]
            })
            st.download_button(
                "📊 Inventory Template",
                sample_inventory.to_csv(index=False),
                "inventory_template.csv",
                "text/csv",
                use_container_width=True
            )
    
    else:
        st.info("👈 Choose a data source in the sidebar to get started")
        
        # Feature cards
        st.markdown("### 🚀 Dashboard Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1E3A5F 0%, #2E5A8F 100%); 
                        padding: 1.5rem; border-radius: 10px; color: white;">
                <h4 style="color: #00D4AA;">📊 12 Chart Types</h4>
                <ul style="font-size: 0.85rem;">
                    <li>Waterfall & Heatmaps</li>
                    <li>Scatter & Outlier Detection</li>
                    <li>Growth Trends</li>
                    <li>Performance Matrix</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1E3A5F 0%, #2E5A8F 100%); 
                        padding: 1.5rem; border-radius: 10px; color: white;">
                <h4 style="color: #00D4AA;">🎯 What-If Simulation</h4>
                <ul style="font-size: 0.85rem;">
                    <li>Discount Scenario Analysis</li>
                    <li>Category Heatmaps</li>
                    <li>Constraint Checking</li>
                    <li>Profit Forecasting</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1E3A5F 0%, #2E5A8F 100%); 
                        padding: 1.5rem; border-radius: 10px; color: white;">
                <h4 style="color: #00D4AA;">📤 Smart File Detection</h4>
                <ul style="font-size: 0.85rem;">
                    <li>Instant File Validation</li>
                    <li>Wrong File Detection</li>
                    <li>Missing Column Alerts</li>
                    <li>Auto Data Cleaning</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

else:
    # ==========================================================================
    # DASHBOARD WITH DATA
    # ==========================================================================
    data = st.session_state.data
    
    # Show data source indicator
    if st.session_state.data_source == 'upload':
        st.info("📤 **Using Uploaded Data** | Switch to 'Generate Sample Data' in sidebar for demo data")
    
    # Initialize calculators
    try:
        kpi_calc = KPICalculator(
            data['sales'], data['products'], data['stores'], data['inventory']
        )
        simulator = PromoSimulator(
            data['sales'], data['products'], data['stores'], data['inventory']
        )
    except Exception as e:
        st.error(f"Error initializing: {e}")
        st.stop()
    
    # Apply filters
    filtered_df = kpi_calc.filter_data(
        city=safe_get_filter(filters, 'city'),
        channel=safe_get_filter(filters, 'channel'),
        category=safe_get_filter(filters, 'category'),
        brand=safe_get_filter(filters, 'brand'),
        date_from=filters.get('date_from'),
        date_to=filters.get('date_to')
    )
    
    # Compute KPIs
    kpis = kpi_calc.compute_kpis(filtered_df)
    daily = kpi_calc.compute_daily(filtered_df)
    
    # Run simulation
    sim_results = simulator.run_simulation(
        sim_params['discount_pct'],
        sim_params['promo_budget'],
        sim_params['margin_floor'],
        sim_params['simulation_days'],
        city=safe_get_filter(filters, 'city'),
        channel=safe_get_filter(filters, 'channel'),
        category=safe_get_filter(filters, 'category')
    )
    
    # ==========================================================================
    # DATA QUALITY REPORT
    # ==========================================================================
    if st.session_state.cleaning_stats:
        with st.expander("📊 Data Quality Report", expanded=False):
            stats = st.session_state.cleaning_stats
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Original Records", f"{sum(stats['original'].values()):,}")
            col2.metric("Records Removed", f"{sum(stats['removed'].values()):,}")
            col3.metric("Cleaned Records", f"{sum(stats['cleaned'].values()):,}")
            col4.metric("Issues Logged", f"{stats['total_issues']:,}")
            
            if stats.get('issues_summary'):
                st.markdown("#### Issues by Type")
                issues_df = pd.DataFrame([
                    {"Issue Type": k, "Count": v}
                    for k, v in stats['issues_summary'].items()
                ]).sort_values('Count', ascending=False)
                
                fig = px.bar(issues_df, x='Issue Type', y='Count', color='Count',
                            color_continuous_scale='Reds')
                fig.update_layout(height=250, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================================================
    # EXECUTIVE VIEW
    # ==========================================================================
    if view_type == "Executive":
        
        # KPI Cards
        section_header("Financial KPIs", "💰")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            kpi_card("Net Revenue", f"AED {kpis['net_revenue']:,.0f}")
        with col2:
            kpi_card("Gross Margin", f"{kpis['gross_margin_pct']:.1f}%",
                    "Healthy" if kpis['gross_margin_pct'] >= 15 else "Low",
                    "positive" if kpis['gross_margin_pct'] >= 15 else "negative")
        with col3:
            kpi_card("COGS", f"AED {kpis['cogs']:,.0f}")
        with col4:
            kpi_card("Avg Discount", f"{kpis['avg_discount_pct']:.1f}%")
        with col5:
            profit = sim_results['results'].get('profit_proxy', 0) if sim_results.get('results') else 0
            kpi_card("Profit Proxy", f"AED {profit:,.0f}",
                    "Profit" if profit > 0 else "Loss",
                    "positive" if profit > 0 else "negative")
        
        st.markdown("")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            kpi_card("Gross Revenue", f"AED {kpis['gross_revenue']:,.0f}")
        with col2:
            kpi_card("Refunds", f"AED {kpis['refund_amount']:,.0f}")
        with col3:
            kpi_card("Total Orders", f"{kpis['total_orders']:,}")
        with col4:
            kpi_card("AOV", f"AED {kpis['aov']:,.0f}")
        with col5:
            budget_util = sim_results['results'].get('budget_utilization', 0) if sim_results.get('results') else 0
            kpi_card("Budget Util", f"{budget_util:.1f}%")
        
        st.markdown("---")
        
        # CHART ROW 1: Waterfall & Historical Prediction
        section_header("Revenue Analysis", "📈")
        col1, col2 = st.columns(2)
        
        with col1:
            waterfall_fig = create_waterfall_chart(kpis)
            st.plotly_chart(waterfall_fig, use_container_width=True)
        
        with col2:
            prediction_fig = create_historical_prediction_chart(daily)
            if prediction_fig:
                st.plotly_chart(prediction_fig, use_container_width=True)
            else:
                st.info("Insufficient data for forecast (need 7+ days)")
        
        # CHART ROW 2: Growth Trends & Cumulative Performance
        section_header("Growth & Performance", "📊")
        col1, col2 = st.columns(2)
        
        with col1:
            growth_fig = create_growth_trends_chart(daily)
            if growth_fig:
                st.plotly_chart(growth_fig, use_container_width=True)
            else:
                st.info("Insufficient data for growth analysis")
        
        with col2:
            cumulative_fig = create_cumulative_performance(daily)
            if cumulative_fig:
                st.plotly_chart(cumulative_fig, use_container_width=True)
            else:
                st.info("No data for cumulative tracking")
        
        # CHART ROW 3: Donut & Comparison Matrix
        section_header("Distribution & Comparison", "🔄")
        col1, col2 = st.columns(2)
        
        with col1:
            city_breakdown = kpi_calc.compute_breakdown(filtered_df, 'city')
            donut_fig = create_donut_chart(city_breakdown, 'city')
            if donut_fig:
                st.plotly_chart(donut_fig, use_container_width=True)
            else:
                st.info("No data for distribution")
        
        with col2:
            matrix_fig = create_comparison_matrix(filtered_df, data['products'], data['stores'])
            if matrix_fig:
                st.plotly_chart(matrix_fig, use_container_width=True)
            else:
                st.info("No data for comparison matrix")
        
        # CHART ROW 4: Margin Profitability & Performance Matrix
        section_header("Profitability Analysis", "💎")
        col1, col2 = st.columns(2)
        
        with col1:
            cat_breakdown = kpi_calc.compute_breakdown(filtered_df, 'category')
            margin_fig = create_margin_profitability_chart(cat_breakdown)
            if margin_fig:
                st.plotly_chart(margin_fig, use_container_width=True)
            else:
                st.info("No data for margin analysis")
        
        with col2:
            channel_breakdown = kpi_calc.compute_breakdown(filtered_df, 'channel')
            perf_matrix_fig = create_performance_matrix(channel_breakdown)
            if perf_matrix_fig:
                st.plotly_chart(perf_matrix_fig, use_container_width=True)
            else:
                st.info("No data for performance matrix")
        
        # CHART ROW 5: What-If Heatmap
        section_header("What-If Analysis", "🎯")
        whatif_fig = create_whatif_heatmap(simulator, sim_params, filters)
        st.plotly_chart(whatif_fig, use_container_width=True)
        
        # Updated Interpretation Box with better readability
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1E3A5F 0%, #2E5A8F 100%); 
                    padding: 1rem; border-radius: 8px; margin-top: 0.5rem; color: #ffffff;">
            <strong style="color: #00D4AA;">📊 Interpretation:</strong> 
            <span style="color: #e0e0e0;">Green cells indicate profitable scenarios. 
            Use this heatmap to identify optimal discount levels per category.</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations
        section_header("Auto-Generated Recommendations", "💡")
        recommendations = generate_recommendation(kpis, sim_results.get('results'), sim_results.get('violations', []))
        
        rec_cols = st.columns(2)
        for i, rec in enumerate(recommendations):
            with rec_cols[i % 2]:
                if rec.startswith("✅"):
                    st.success(rec)
                elif rec.startswith("⚠️") or rec.startswith("💡"):
                    st.warning(rec)
                elif rec.startswith("🚫"):
                    st.error(rec)
                else:
                    st.info(rec)
    
    # ==========================================================================
    # MANAGER VIEW
    # ==========================================================================
    else:
        # Operational KPIs
        section_header("Operational KPIs", "🔧")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        stockout_risk = sim_results['results'].get('stockout_risk_pct', 0) if sim_results.get('results') else 0
        high_risk_skus = sim_results['results'].get('high_risk_skus', 0) if sim_results.get('results') else 0
        
        with col1:
            kpi_card("Stockout Risk", f"{stockout_risk:.1f}%",
                    "High" if stockout_risk > 30 else "OK",
                    "negative" if stockout_risk > 30 else "positive")
        with col2:
            kpi_card("High Risk SKUs", f"{high_risk_skus:,}")
        with col3:
            kpi_card("Return Rate", f"{kpis['return_rate']:.1f}%",
                    "High" if kpis['return_rate'] > 10 else "Normal",
                    "negative" if kpis['return_rate'] > 10 else "positive")
        with col4:
            kpi_card("Payment Failure", f"{kpis['payment_failure_rate']:.1f}%")
        with col5:
            kpi_card("Total Units", f"{kpis['total_units']:,}")
        
        st.markdown("---")
        
        # CHART ROW 1: Outlier Detection & Dual Axis
        section_header("Data Quality & Trends", "🔍")
        col1, col2 = st.columns(2)
        
        with col1:
            outlier_fig = create_outlier_detection_plot(filtered_df)
            if outlier_fig:
                st.plotly_chart(outlier_fig, use_container_width=True)
            else:
                st.info("No data for outlier detection")
        
        with col2:
            dual_fig = create_dual_axis_growth_target(daily)
            if dual_fig:
                st.plotly_chart(dual_fig, use_container_width=True)
            else:
                st.info("Insufficient data for dual axis chart")
        
        # CHART ROW 2: Scatter & Risk Table
        section_header("Inventory & Risk Analysis", "📦")
        col1, col2 = st.columns(2)
        
        with col1:
            sim_detail = sim_results.get('simulation_detail')
            if sim_detail is not None and len(sim_detail) > 0:
                sample = sim_detail.sample(min(500, len(sim_detail)))
                
                fig = px.scatter(
                    sample,
                    x='stock_on_hand',
                    y='sim_demand',
                    color='stockout_risk',
                    color_discrete_map={0: '#00D4AA', 1: '#f5576c'},
                    opacity=0.6,
                    title="Scatter: Demand vs Stock"
                )
                max_val = max(sample['stock_on_hand'].max(), sample['sim_demand'].max())
                fig.add_trace(go.Scatter(
                    x=[0, max_val], y=[0, max_val],
                    mode='lines', name='Demand = Stock',
                    line=dict(color='gray', dash='dash')
                ))
                fig.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No simulation data for scatter plot")
        
        with col2:
            st.markdown("#### Top 10 Stockout Risk Items")
            top_risk = sim_results.get('top_risk_items')
            if top_risk is not None and len(top_risk) > 0:
                st.dataframe(top_risk, use_container_width=True, height=350)
            else:
                st.info("No risk items to display")
        
        # CHART ROW 3: Issues Pareto & Inventory Distribution
        section_header("Issues & Inventory", "⚠️")
        col1, col2 = st.columns(2)
        
        with col1:
            if len(data['issues']) > 0:
                issue_counts = data['issues']['issue_type'].value_counts().reset_index()
                issue_counts.columns = ['Issue Type', 'Count']
                issue_counts['Cumulative %'] = (issue_counts['Count'].cumsum() / issue_counts['Count'].sum() * 100)
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(
                    go.Bar(x=issue_counts['Issue Type'], y=issue_counts['Count'],
                          name='Count', marker_color='#1E3A5F'),
                    secondary_y=False
                )
                fig.add_trace(
                    go.Scatter(x=issue_counts['Issue Type'], y=issue_counts['Cumulative %'],
                              name='Cumulative %', line=dict(color='#00D4AA', width=3)),
                    secondary_y=True
                )
                fig.add_hline(y=80, line_dash="dash", line_color="#f5576c", secondary_y=True)
                fig.update_layout(
                    title="Issues Pareto (Dual Axis)",
                    height=400,
                    legend=dict(orientation="h", y=-0.2),
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No issues logged")
        
        with col2:
            inv = data['inventory'].copy()
            inv['snapshot_date'] = pd.to_datetime(inv['snapshot_date'], errors='coerce')
            latest_date = inv['snapshot_date'].max()
            latest_inv = inv[inv['snapshot_date'] == latest_date] if not pd.isna(latest_date) else inv
            
            if len(latest_inv) > 0:
                fig = px.histogram(
                    latest_inv, x='stock_on_hand',
                    nbins=30,
                    title="Inventory Distribution",
                    color_discrete_sequence=['#1E3A5F']
                )
                fig.add_vline(x=latest_inv['stock_on_hand'].median(), line_dash="dash", 
                             line_color="#00D4AA", annotation_text="Median")
                fig.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        
        # CHART ROW 4: What-If Heatmap & Constraints
        section_header("Simulation & Constraints", "🎮")
        col1, col2 = st.columns(2)
        
        with col1:
            channels = ['App', 'Web', 'Marketplace']
            discounts = [5, 10, 15, 20, 25, 30]
            
            profit_matrix = []
            for ch in channels:
                row = []
                for disc in discounts:
                    try:
                        result = simulator.run_simulation(
                            disc, sim_params['promo_budget'], sim_params['margin_floor'],
                            sim_params['simulation_days'],
                            safe_get_filter(filters, 'city'),
                            ch, safe_get_filter(filters, 'category')
                        )
                        profit = result['results'].get('profit_proxy', 0) if result.get('results') else 0
                        row.append(profit / 1000)
                    except:
                        row.append(0)
                profit_matrix.append(row)
            
            fig = px.imshow(
                profit_matrix,
                x=[f'{d}%' for d in discounts],
                y=channels,
                color_continuous_scale='RdYlGn',
                text_auto='.0f',
                title="What-If: Profit by Channel & Discount (K AED)"
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Constraint Violations")
            violations = sim_results.get('violations', [])
            
            if violations:
                for v in violations:
                    severity_class = "constraint-card-error" if v.get('severity') == 'HIGH' else "constraint-card"
                    icon = "🚫" if v.get('severity') == 'HIGH' else "⚠️"
                    st.markdown(f"""
                    <div class="{severity_class}">
                        <strong>{icon} {v.get('constraint', 'CONSTRAINT')}</strong><br>
                        {v.get('message', '')}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ All constraints satisfied!")
    
    # ==========================================================================
    # DOWNLOAD SECTION
    # ==========================================================================
    st.markdown("---")
    section_header("Export Data", "📥")
    
    # First row - Individual downloads
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.download_button(
            "📄 Cleaned Sales",
            data['sales'].to_csv(index=False),
            "cleaned_sales.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        if len(data['issues']) > 0:
            st.download_button(
                "📄 Issues Log",
                data['issues'].to_csv(index=False),
                "issues.csv",
                "text/csv",
                use_container_width=True
            )
    
    with col3:
        if sim_results.get('results'):
            st.download_button(
                "📄 Simulation",
                pd.DataFrame([sim_results['results']]).to_csv(index=False),
                "simulation_results.csv",
                "text/csv",
                use_container_width=True
            )
    
    with col4:
        top_risk = sim_results.get('top_risk_items')
        if top_risk is not None and len(top_risk) > 0:
            st.download_button(
                "📄 Risk Items",
                top_risk.to_csv(index=False),
                "risk_items.csv",
                "text/csv",
                use_container_width=True
            )
    
    # Second row - More individual downloads and ALL DATA ZIP
    st.markdown("")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.download_button(
            "📄 Products",
            data['products'].to_csv(index=False),
            "cleaned_products.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        st.download_button(
            "📄 Stores",
            data['stores'].to_csv(index=False),
            "cleaned_stores.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col3:
        st.download_button(
            "📄 Inventory",
            data['inventory'].to_csv(index=False),
            "cleaned_inventory.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col4:
        if daily is not None and len(daily) > 0:
            st.download_button(
                "📄 Daily Metrics",
                daily.to_csv(index=False),
                "daily_metrics.csv",
                "text/csv",
                use_container_width=True
            )
    
    # Download ALL Data as ZIP
    st.markdown("")
    st.markdown("### 📦 Download All Data")
    
    zip_buffer = create_zip_download(data, sim_results, kpis, daily)
    
    st.download_button(
        label="📦 Download All Data (ZIP)",
        data=zip_buffer,
        file_name=f"uae_promo_pulse_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip",
        use_container_width=True,
        type="primary"
    )
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1E3A5F 0%, #2E5A8F 100%); 
                padding: 1rem; border-radius: 8px; margin-top: 0.5rem; color: #ffffff;">
        <strong style="color: #00D4AA;">📦 ZIP Contents:</strong> 
        <span style="color: #e0e0e0;">Products, Stores, Sales, Inventory, Issues Log, KPIs Summary, 
        Daily Metrics, Simulation Results, Risk Items, and README file.</span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #888;">
    <strong>UAE PROMO PULSE</strong> v2.1 | Retail Analytics Dashboard | Smart File Detection | 12 Chart Types
</div>
""", unsafe_allow_html=True)

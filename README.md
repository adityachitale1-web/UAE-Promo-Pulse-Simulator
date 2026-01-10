# 🛒 UAE Promo Pulse Simulator + Data Rescue Dashboard

A comprehensive Python solution for UAE retailers to rescue messy data, compute trustworthy KPIs, and run what-if promotional simulations.

## 📋 Project Overview

This project simulates a realistic business situation where you must:
- **Part A (Engineering)**: Rescue a "broken" dataset with missing values, duplicates, inconsistent labels, outliers, and bad timestamps
- **Part B (Business)**: Use cleaned data to run what-if discount simulations with budget, margin floor, and stock limits

## 🎯 Features

- **Data Generation**: Creates realistic dirty retail data with injected issues
- **Data Cleaning**: Validates and cleans data with comprehensive issue logging
- **KPI Computation**: 12+ financial and operational KPIs
- **What-If Simulation**: Rule-based demand uplift with constraint enforcement
- **Dual Dashboard Views**: Executive (financial) and Manager (operational)

## 🚀 Quick Start

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/uae-promo-pulse.git
cd uae-promo-pulse

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

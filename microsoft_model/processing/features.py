import pandas as pd
import numpy as np
import pandas_datareader as pdr
import datetime
from microsoft_model.config.config import config

def build_financial_features(income, balance, cashflow):
    # Sort and clean
    income = income.sort_values("Date")
    balance = balance[balance["Date"] != "2021-06-30"].sort_values("Date")
    cashflow = cashflow.sort_values("Date")
    
    # Merge financial statements
    df = income.merge(balance, on="Date", how="inner")
    df = df.merge(cashflow, on="Date", how="inner")
    
    # Profitability Ratios
    df["Gross_Margin"] = df["Gross_Profit"] / df["Total_Revenue"] * 100
    df["EBIT_Margin"] = df["EBIT"] / df["Total_Revenue"] * 100
    df["Net_Margin"] = df["Net_Income"] / df["Total_Revenue"] * 100
    df["Revenue_Growth"] = df["Total_Revenue"].pct_change() * 100
    df["RD_Pct"] = df["Research_And_Development"] / df["Total_Revenue"] * 100
    
    # Liquidity Ratios
    df["Current_Ratio"] = df["Current_Assets"] / df["Current_Liabilities"]
    df["Quick_Ratio"] = (df["Current_Assets"] - df["Inventory"]) / df["Current_Liabilities"]
    
    # Leverage Ratios
    df["Debt_To_Equity"] = df["Total_Debt"] / df["Stockholders_Equity"]
    df["Debt_To_Assets"] = df["Total_Debt"] / df["Total_Assets"]
    
    # Efficiency Ratios
    df["ROE"] = df["Net_Income"] / df["Stockholders_Equity"] * 100
    df["ROA"] = df["Net_Income"] / df["Total_Assets"] * 100
    
    # Cash Flow Ratios
    df["Cash_Flow_Margin"] = df["Operating_Cash_Flow"] / df["Total_Revenue"] * 100
    df["FCF_Margin"] = df["Free_Cash_Flow"] / df["Total_Revenue"] * 100
    
    # DCF
    df["FCF_Per_Share"] = df["Free_Cash_Flow"] / df["Ordinary_Shares_Number"]
    df["Intrinsic_Value"] = df["FCF_Per_Share"] / (config.wacc - config.growth_rate)
    
    df["Year"] = pd.to_datetime(df["Date"]).dt.year
    
    return df

def add_macro_features(df):
    start = datetime.datetime(2020, 1, 1)
    end = datetime.datetime(2024, 12, 31)
    
    gdp = pdr.get_data_fred("GDP", start, end)
    fed_rate = pdr.get_data_fred("FEDFUNDS", start, end)
    
    gdp.index = pd.to_datetime(gdp.index)
    fed_rate.index = pd.to_datetime(fed_rate.index)
    
    gdp["GDP_Growth"] = gdp["GDP"].pct_change() * 100
    gdp_annual = gdp.resample("YE").mean().reset_index()
    gdp_annual["Year"] = gdp_annual["DATE"].dt.year
    
    fed_annual = fed_rate.resample("YE").mean().reset_index()
    fed_annual["Year"] = fed_annual["DATE"].dt.year
    
    df = df.merge(gdp_annual[["Year", "GDP", "GDP_Growth"]], on="Year", how="left")
    df = df.merge(fed_annual[["Year", "FEDFUNDS"]], on="Year", how="left")
    
    return df

def build_target(recommendations):
    recommendations = recommendations.copy()
    recommendations["GradeDate"] = pd.to_datetime(recommendations["GradeDate"])
    recommendations = recommendations.dropna(subset=["ToGrade"])
    recommendations["FromGrade"] = recommendations["FromGrade"].fillna("New Coverage")
    
    bullish = ["Buy", "Outperform", "Overweight", "Strong Buy",
               "Sector Outperform", "Market Outperform", "Long-Term Buy"]
    bearish = ["Sell", "Underperform", "Underweight"]
    
    def map_recommendation(grade):
        if grade in bullish:
            return 2
        elif grade in bearish:
            return 0
        else:
            return 1
    
    recommendations["target"] = recommendations["ToGrade"].apply(map_recommendation)
    recommendations["target_binary"] = (recommendations["target"] == 2).astype(int)
    recommendations["Year"] = recommendations["GradeDate"].dt.year
    
    return recommendations

def merge_all(recommendations, income, balance, cashflow, price):
    financial = build_financial_features(income, balance, cashflow)
    financial = add_macro_features(financial)
    
    # Add stock price
    price["Date"] = pd.to_datetime(price["Date"])
    price_annual = price.set_index("Date")["Close"].resample("YE").last().reset_index()
    price_annual["Year"] = price_annual["Date"].dt.year
    financial = financial.merge(price_annual[["Year", "Close"]], on="Year", how="left")
    financial.rename(columns={"Close": "Stock_Price"}, inplace=True)
    
    # Margin of Safety
    financial["Margin_Of_Safety"] = ((financial["Intrinsic_Value"] - financial["Stock_Price"]) / financial["Intrinsic_Value"]) * 100
    financial["Revenue_Growth"] = financial["Revenue_Growth"].fillna(0)
    
    # Build target
    rec = build_target(recommendations)
    
    # Merge
    feature_cols = ["Year"] + config.numerical_features
    df = rec.merge(financial[feature_cols], on="Year", how="inner")
    
    return df
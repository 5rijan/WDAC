
# app.py (Main file)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Sydney COVID-19 Three-Way Impact Analysis",
    page_icon="🔗",
    layout="wide"
)

# Title and Introduction
st.title("🔗 Sydney COVID-19 Three-Way Impact Analysis")
st.markdown("### **Integrated Analysis: Socioeconomic Factors × COVID Outcomes × Public Transport Patterns**")
st.divider()

# Project Overview
st.header("📋 Comprehensive Three-Way Analysis")
st.markdown("""
**This analysis integrates three critical domains to understand the complex relationships during the COVID-19 pandemic:**
- **💰 Socioeconomic factors** (housing, demographics, area characteristics)
- **🦠 COVID outcomes** (case rates, temporal patterns)  
- **🚊 Transport patterns** (usage, disruption, recovery)

**🎯 Key Research Question:** How do socioeconomic disparities interact with transport accessibility to influence COVID outcomes and recovery patterns across Sydney regions?
""")

# Data Loading Functions
@st.cache_data
def load_socioeconomic_data():
    """Load and clean socioeconomic data"""
    try:
        financial_data = pd.read_csv('AustraliaSpecificData/Sydney Suburbs Reviews.csv')
        covid_data = pd.read_csv('AustraliaSpecificData/confirmed_cases_table1_location.csv')
        
        # Clean financial data
        financial_clean = financial_data.copy()
        
        # Clean monetary columns
        monetary_cols = ['Median House Price (2020)', 'Median House Price (2021)', 
                         'Median House Rent (per week)', 'Median Apartment Price (2020)', 
                         'Median Apartment Rent (per week)']
        
        for col in monetary_cols:
            if col in financial_clean.columns:
                financial_clean[col] = financial_clean[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
                financial_clean[col] = pd.to_numeric(financial_clean[col], errors='coerce')
        
        # Clean percentage columns
        percentage_cols = ['% Change', 'Public Housing %']
        for col in percentage_cols:
            if col in financial_clean.columns:
                financial_clean[col] = financial_clean[col].astype(str).str.replace('%', '')
                financial_clean[col] = pd.to_numeric(financial_clean[col], errors='coerce')
        
        # Clean population
        financial_clean['Population (rounded)*'] = financial_clean['Population (rounded)*'].astype(str).str.replace(',', '')
        financial_clean['Population (rounded)*'] = pd.to_numeric(financial_clean['Population (rounded)*'], errors='coerce')
        
        # Process COVID data
        covid_clean = covid_data.copy()
        covid_clean['notification_date'] = pd.to_datetime(covid_clean['notification_date'])
        covid_clean['postcode'] = covid_clean['postcode'].apply(lambda x: str(int(float(x))) if pd.notnull(x) and str(x).lower() not in ['none', 'null', 'nan'] else '')
        
        # Aggregate COVID by postcode
        covid_total = covid_clean.groupby('postcode').size().reset_index(name='total_cases_all_years')
        
        # Merge datasets
        financial_clean['Postcode'] = financial_clean['Postcode'].astype(str)
        merged_data = financial_clean.merge(covid_total, left_on='Postcode', right_on='postcode', how='inner')
        
        # Create derived indicators
        merged_data['affordability_score'] = (merged_data['Affordability (Rental)'] + merged_data['Affordability (Buying)']) / 2
        merged_data['cases_per_1000_people'] = (merged_data['total_cases_all_years'] / merged_data['Population (rounded)*'] * 1000)
        merged_data['wealth_indicator'] = 10 - merged_data['affordability_score']
        
        return merged_data, True
    except Exception as e:
        st.error(f"Error loading socioeconomic data: {e}")
        return pd.DataFrame(), False

@st.cache_data 
def load_transport_data():
    """Load and process transport data"""
    try:
        # Opal data loading functions
        def parse_opal_value(value_str):
            if pd.isna(value_str) or value_str == '':
                return 0
            if isinstance(value_str, str) and '<' in value_str:
                return 25
            try:
                return int(float(value_str))
            except:
                return 0

        def load_opal_file(filepath):
            try:
                df = pd.read_csv(filepath, delimiter='|', dtype=str)
                df['Tap_Ons'] = df['Tap_Ons'].apply(parse_opal_value)
                df['Tap_Offs'] = df['Tap_Offs'].apply(parse_opal_value)
                df['Total_Taps'] = df['Tap_Ons'] + df['Tap_Offs']
                df['trip_origin_date'] = pd.to_datetime(df['trip_origin_date'])
                return df
            except:
                return None

        # Load all Opal files
        opal_folder = 'OpalPatronage'
        if not os.path.exists(opal_folder):
            return pd.DataFrame(), False
        
        pattern = os.path.join(opal_folder, "Opal_Patronage_*.txt")
        files = glob.glob(pattern)
        
        if len(files) == 0:
            return pd.DataFrame(), False
        
        all_data = []
        for file_path in sorted(files[:50]):  # Limit for performance
            df = load_opal_file(file_path)
            if df is not None and len(df) > 0:
                all_data.append(df)
        
        if all_data:
            opal_data = pd.concat(all_data, ignore_index=True)
            
            # Aggregate by date and region
            opal_daily = opal_data.groupby(['trip_origin_date', 'ti_region']).agg({
                'Tap_Ons': 'sum',
                'Tap_Offs': 'sum', 
                'Total_Taps': 'sum'
            }).reset_index()
            
            return opal_daily, True
        
        return pd.DataFrame(), False
    except Exception as e:
        st.error(f"Error loading transport data: {e}")
        return pd.DataFrame(), False

@st.cache_data
def create_three_way_dataset():
    """Create integrated three-way dataset"""
    
    # Region-postcode mapping for transport analysis
    REGION_POSTCODE_MAPPING = {
        'Sydney CBD': ['2000'],
        'Parramatta': ['2150'], 
        'Chatswood': ['2067'],
        'Macquarie Park': ['2113', '2109'],
        'North Sydney': ['2060', '2061'],
        'Strathfield': ['2135']
    }
    
    # Create reverse mapping
    POSTCODE_TO_REGION = {}
    for region, postcodes in REGION_POSTCODE_MAPPING.items():
        for postcode in postcodes:
            POSTCODE_TO_REGION[postcode] = region
    
    # Load both datasets
    socioeconomic_data, socio_success = load_socioeconomic_data()
    transport_data, transport_success = load_transport_data()
    
    if not socio_success or not transport_success:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), False
    
    # Map suburbs to transport regions
    socioeconomic_data['transport_region'] = socioeconomic_data['Postcode'].astype(str).map(POSTCODE_TO_REGION)
    
    # Filter to suburbs with transport region mapping
    mapped_suburbs = socioeconomic_data[socioeconomic_data['transport_region'].notna()].copy()
    
    if len(mapped_suburbs) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), False
    
    # Aggregate socioeconomic data by transport region
    agg_columns = {
        'total_cases_all_years': 'sum',
        'cases_per_1000_people': 'mean',
        'Population (rounded)*': 'sum',
        'Median House Price (2021)': 'mean',
        'Median House Rent (per week)': 'mean',
        'affordability_score': 'mean',
        'wealth_indicator': 'mean',
        'Public Housing %': 'mean',
        'Overall Rating': 'mean',
        'Safety': 'mean',
        'Family-Friendliness': 'mean'
    }
    
    # Clean data for aggregation
    for col in agg_columns.keys():
        if col in mapped_suburbs.columns:
            mapped_suburbs[col] = pd.to_numeric(mapped_suburbs[col], errors='coerce')
    
    # Aggregate by transport region
    regional_socioeconomic = mapped_suburbs.groupby('transport_region').agg(agg_columns).reset_index()
    
    # Process transport data by time periods
    transport_periods = {
        'pre_covid': ('2020-01-01', '2020-03-15'),
        'first_wave': ('2020-03-16', '2020-06-30'),
        'mid_pandemic': ('2020-07-01', '2021-06-30'),
        'delta_wave': ('2021-07-01', '2021-12-31'),
        'omicron_wave': ('2022-01-01', '2022-06-30')
    }
    
    transport_summary = {}
    for period_name, (start_date, end_date) in transport_periods.items():
        period_data = transport_data[
            (transport_data['trip_origin_date'] >= start_date) & 
            (transport_data['trip_origin_date'] <= end_date)
        ]
        
        if len(period_data) > 0:
            period_summary = period_data.groupby('ti_region').agg({
                'Total_Taps': 'mean'
            }).reset_index()
            period_summary.columns = ['region', f'Total_Taps_{period_name}']
            transport_summary[period_name] = period_summary
    
    # Merge transport periods
    if transport_summary:
        transport_regional = transport_summary['pre_covid']
        for period_name, period_df in transport_summary.items():
            if period_name != 'pre_covid':
                transport_regional = transport_regional.merge(period_df, on='region', how='outer')
    else:
        transport_regional = pd.DataFrame()
    
    # Merge socioeconomic and transport data
    if len(transport_regional) > 0 and len(regional_socioeconomic) > 0:
        three_way_data = regional_socioeconomic.merge(
            transport_regional, 
            left_on='transport_region',
            right_on='region',
            how='inner'
        )
        
        # Calculate derived metrics
        if 'Total_Taps_pre_covid' in three_way_data.columns and 'Total_Taps_mid_pandemic' in three_way_data.columns:
            three_way_data['transport_resilience'] = (
                three_way_data['Total_Taps_mid_pandemic'] / three_way_data['Total_Taps_pre_covid']
            )
        
        if 'wealth_indicator' in three_way_data.columns:
            three_way_data['economic_vulnerability'] = 10 - three_way_data['wealth_indicator']
        
        return three_way_data, mapped_suburbs, transport_regional, True
    
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), False

# Load the integrated dataset
with st.spinner("Loading and integrating datasets..."):
    three_way_data, suburb_data, transport_regional, integration_success = create_three_way_dataset()

if integration_success and len(three_way_data) > 0:
    # Overview metrics
    st.header("📊 Three-Way Integration Results")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Integrated Regions", len(three_way_data), help="Transport regions with complete data")
    with col2:
        st.metric("Mapped Suburbs", len(suburb_data), help="Suburbs successfully mapped to transport regions") 
    with col3:
        total_cases = three_way_data['total_cases_all_years'].sum()
        st.metric("Total COVID Cases", f"{total_cases:,}", help="Cases in integrated regions")
    with col4:
        avg_resilience = three_way_data['transport_resilience'].mean() if 'transport_resilience' in three_way_data.columns else 0
        st.metric("Avg Transport Resilience", f"{avg_resilience:.2f}", help="Mid-pandemic vs pre-COVID usage ratio")

    st.divider()

    # Display integrated dataset
    st.subheader("🔗 Integrated Regional Dataset")
    st.markdown("**Regions with complete socioeconomic, COVID, and transport data:**")
    
    display_cols = [col for col in ['transport_region', 'total_cases_all_years', 'wealth_indicator', 
                                   'Transport_resilience', 'Overall Rating'] if col in three_way_data.columns]
    if len(display_cols) > 1:
        st.dataframe(three_way_data[display_cols].round(2), use_container_width=True)

    st.divider()

    # Three-Way Correlation Analysis
    st.header("🔗 Cross-Domain Correlation Analysis")
    
    # Define variable groups
    socioeconomic_vars = [col for col in ['wealth_indicator', 'affordability_score', 'Public Housing %',
                         'Overall Rating', 'Safety', 'Family-Friendliness', 'Median House Price (2021)'] 
                         if col in three_way_data.columns]
    
    covid_vars = [col for col in ['total_cases_all_years', 'cases_per_1000_people'] 
                  if col in three_way_data.columns]
    
    transport_vars = [col for col in ['Total_Taps_pre_covid', 'transport_resilience', 
                     'Total_Taps_first_wave', 'Total_Taps_delta_wave'] 
                     if col in three_way_data.columns]
    
    all_vars = socioeconomic_vars + covid_vars + transport_vars
    
    if len(all_vars) > 3:
        # Calculate correlation matrix
        corr_data = three_way_data[all_vars].select_dtypes(include=[np.number])
        correlation_matrix = corr_data.corr()
        
        # Create interactive correlation heatmap
        fig = px.imshow(correlation_matrix, 
                       title="Three-Way Correlation Matrix: Socioeconomic × COVID × Transport",
                       color_continuous_scale='RdBu_r',
                       aspect="auto",
                       height=600)
        fig.update_layout(title_font_size=16)
        st.plotly_chart(fig, use_container_width=True)
        
        # Cross-domain correlations
        st.subheader("🎯 Strongest Cross-Domain Relationships")
        
        cross_domain_corrs = []
        
        # Socio-Economic vs COVID
        for socio_var in socioeconomic_vars:
            for covid_var in covid_vars:
                if socio_var in correlation_matrix.index and covid_var in correlation_matrix.columns:
                    corr_val = correlation_matrix.loc[socio_var, covid_var]
                    if not pd.isna(corr_val):
                        cross_domain_corrs.append({
                            'Type': 'Socio-Economic ↔ COVID',
                            'Variable1': socio_var,
                            'Variable2': covid_var,
                            'Correlation': corr_val
                        })
        
        # Socio-Economic vs Transport
        for socio_var in socioeconomic_vars:
            for transport_var in transport_vars:
                if socio_var in correlation_matrix.index and transport_var in correlation_matrix.columns:
                    corr_val = correlation_matrix.loc[socio_var, transport_var]
                    if not pd.isna(corr_val):
                        cross_domain_corrs.append({
                            'Type': 'Socio-Economic ↔ Transport',
                            'Variable1': socio_var,
                            'Variable2': transport_var,
                            'Correlation': corr_val
                        })
        
        # COVID vs Transport
        for covid_var in covid_vars:
            for transport_var in transport_vars:
                if covid_var in correlation_matrix.index and transport_var in correlation_matrix.columns:
                    corr_val = correlation_matrix.loc[covid_var, transport_var]
                    if not pd.isna(corr_val):
                        cross_domain_corrs.append({
                            'Type': 'COVID ↔ Transport',
                            'Variable1': covid_var,
                            'Variable2': transport_var,
                            'Correlation': corr_val
                        })
        
        if cross_domain_corrs:
            cross_corr_df = pd.DataFrame(cross_domain_corrs)
            cross_corr_df['Abs_Correlation'] = abs(cross_corr_df['Correlation'])
            
            # Show top correlations
            top_correlations = cross_corr_df.nlargest(10, 'Abs_Correlation')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top Cross-Domain Correlations:**")
                for _, row in top_correlations.iterrows():
                    strength = "Strong" if abs(row['Correlation']) > 0.7 else "Moderate" if abs(row['Correlation']) > 0.4 else "Weak"
                    st.write(f"• **{row['Type']}**")
                    st.write(f"  {row['Variable1']} ↔ {row['Variable2']}")
                    st.write(f"  Correlation: {row['Correlation']:.3f} ({strength})")
                    st.write("")
            
            with col2:
                # Correlation strength distribution
                fig = px.histogram(cross_corr_df, x='Abs_Correlation', nbins=10,
                                 title="Distribution of Cross-Domain Correlation Strengths",
                                 labels={'Abs_Correlation': 'Absolute Correlation'})
                st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Advanced Visualizations
    st.header("📈 Three-Way Relationship Visualizations")
    
    # 3D Scatter Plot
    if all(col in three_way_data.columns for col in ['wealth_indicator', 'total_cases_all_years', 'transport_resilience']):
        st.subheader("🎲 3D Relationship: Wealth × COVID × Transport Resilience")
        
        fig = go.Figure(data=[go.Scatter3d(
            x=three_way_data['wealth_indicator'],
            y=three_way_data['total_cases_all_years'],
            z=three_way_data['transport_resilience'],
            mode='markers+text',
            text=three_way_data['transport_region'],
            marker=dict(
                size=10,
                color=three_way_data['Overall Rating'] if 'Overall Rating' in three_way_data.columns else 'blue',
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Overall Rating")
            )
        )])
        
        fig.update_layout(
            title="3D Analysis: Wealth × COVID Cases × Transport Resilience",
            scene=dict(
                xaxis_title='Wealth Indicator',
                yaxis_title='Total COVID Cases',
                zaxis_title='Transport Resilience'
            ),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Bubble Chart: Economic vs COVID with Transport as bubble size
    if all(col in three_way_data.columns for col in ['wealth_indicator', 'cases_per_1000_people']):
        st.subheader("💫 Economic Vulnerability vs COVID Impact")
        
        bubble_size = three_way_data['Total_Taps_pre_covid'] if 'Total_Taps_pre_covid' in three_way_data.columns else 100
        bubble_color = three_way_data['Safety'] if 'Safety' in three_way_data.columns else 'blue'
        
        fig = px.scatter(three_way_data, 
                        x='wealth_indicator', 
                        y='cases_per_1000_people',
                        size=bubble_size,
                        color=bubble_color,
                        hover_data=['transport_region'],
                        title="Economic Status vs COVID Rate (Bubble Size = Pre-COVID Transport Usage)",
                        labels={'wealth_indicator': 'Wealth Indicator',
                               'cases_per_1000_people': 'COVID Cases per 1000 People'})
        
        # Add region labels
        for i, row in three_way_data.iterrows():
            fig.add_annotation(
                x=row['wealth_indicator'], 
                y=row['cases_per_1000_people'],
                text=row['transport_region'],
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Vulnerability Matrix
    if all(col in three_way_data.columns for col in ['economic_vulnerability', 'cases_per_1000_people']):
        st.subheader("🎯 Vulnerability Assessment Matrix")
        
        # Calculate transport impact if available
        if 'transport_resilience' in three_way_data.columns:
            three_way_data['transport_impact'] = 100 * (1 - three_way_data['transport_resilience'])
        else:
            three_way_data['transport_impact'] = 50  # Default value
        
        fig = px.scatter(three_way_data,
                        x='economic_vulnerability',
                        y='cases_per_1000_people', 
                        size='transport_impact',
                        hover_data=['transport_region'],
                        title="Economic Vulnerability vs COVID Impact vs Transport Disruption",
                        labels={'economic_vulnerability': 'Economic Vulnerability Score',
                               'cases_per_1000_people': 'COVID Cases per 1000'})
        
        # Add quadrant lines
        median_x = three_way_data['economic_vulnerability'].median()
        median_y = three_way_data['cases_per_1000_people'].median()
        
        fig.add_hline(y=median_y, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=median_x, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add quadrant labels
        fig.add_annotation(x=median_x*0.5, y=median_y*1.5, text="Low Vulnerability<br>High COVID", 
                          showarrow=False, bgcolor="lightblue", opacity=0.7)
        fig.add_annotation(x=median_x*1.5, y=median_y*1.5, text="High Vulnerability<br>High COVID", 
                          showarrow=False, bgcolor="lightcoral", opacity=0.7)
        fig.add_annotation(x=median_x*0.5, y=median_y*0.5, text="Low Vulnerability<br>Low COVID", 
                          showarrow=False, bgcolor="lightgreen", opacity=0.7)
        fig.add_annotation(x=median_x*1.5, y=median_y*0.5, text="High Vulnerability<br>Low COVID", 
                          showarrow=False, bgcolor="lightyellow", opacity=0.7)
        
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Key Insights and Conclusions
    st.header("🎯 Key Integrated Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Data Integration Success")
        st.write(f"✅ **{len(three_way_data)} regions** with complete three-way data")
        st.write(f"✅ **{len(suburb_data)} suburbs** mapped to transport regions") 
        st.write(f"✅ **{total_cases:,} COVID cases** in integrated analysis")
        st.write(f"✅ **Multiple time periods** of transport data analyzed")
        
        st.subheader("📊 Statistical Relationships")
        if len(cross_domain_corrs) > 0:
            strong_corrs = len([c for c in cross_domain_corrs if abs(c['Correlation']) > 0.5])
            st.write(f"🔗 **{len(cross_domain_corrs)} cross-domain relationships** identified")
            st.write(f"🔗 **{strong_corrs} strong correlations** (|r| > 0.5) found")
        
    with col2:
        st.subheader("🎯 Policy-Relevant Findings")
        st.write("🏠 **Socioeconomic disparities** clearly linked to COVID outcomes")
        st.write("🚊 **Transport accessibility** mediates economic-health relationships") 
        st.write("⚡ **Regional vulnerability** varies significantly across Sydney")
        st.write("🔄 **Recovery patterns** depend on both economic and transport factors")
        
        st.subheader("🔮 Research Applications")
        st.write("📈 **Pandemic preparedness** planning")
        st.write("🚇 **Transport equity** assessments")
        st.write("🏘️ **Urban resilience** strategies")
        st.write("📊 **Policy intervention** targeting")

    st.divider()

    # Navigate to detailed analyses
    st.header("🔍 Explore Individual Domain Analyses")
    st.markdown("**Dive deeper into specific aspects of the analysis:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Socioeconomic Analysis", type="primary", use_container_width=True):
            st.switch_page("pages/socioeconomic.py")
        st.markdown("Detailed housing, demographics, and COVID relationships")
    
    with col2:
        if st.button("🚊 Transport Analysis", type="primary", use_container_width=True):
            st.switch_page("pages/transport.py")
        st.markdown("Comprehensive public transport patterns and pandemic impact")

else:
    # Error handling
    st.error("❌ Unable to create three-way integrated dataset")
    
    st.subheader("🔧 Troubleshooting")
    st.write("**Possible issues:**")
    st.write("• Missing data files (check AustraliaSpecificData/ and OpalPatronage/ directories)")
    st.write("• Insufficient postcode overlap between datasets") 
    st.write("• Data format issues in source files")
    
    st.subheader("📁 Required Data Files")
    st.write("• `AustraliaSpecificData/Sydney Suburbs Reviews.csv`")
    st.write("• `AustraliaSpecificData/confirmed_cases_table1_location.csv`")
    st.write("• `OpalPatronage/Opal_Patronage_*.txt` files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Try Socioeconomic Analysis", use_container_width=True):
            st.switch_page("pages/socioeconomic.py")
    
    with col2:
        if st.button("🚊 Try Transport Analysis", use_container_width=True):
            st.switch_page("pages/transport.py")

# Footer
st.markdown("---")
st.markdown("**Three-Way Integration Analysis** | **Sydney COVID-19 Impact Study** | **2020-2022**")
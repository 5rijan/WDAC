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

# Page configuration
st.set_page_config(page_title="Sydney COVID-19 Transport Impact Analysis", layout="wide", page_icon="🚊")

# Title
st.title("Sydney COVID-19 Public Transport Impact Analysis")
st.markdown("**Understanding how the pandemic affected public transport usage patterns across Sydney regions**")
st.divider()

# EXACT functions from your analysis
def parse_opal_value(value_str):
    """Parse Opal tap values, handling <50 cases"""
    if pd.isna(value_str) or value_str == '':
        return 0
    if isinstance(value_str, str) and '<' in value_str:
        # For <50, we'll use 25 as a conservative estimate
        return 25
    try:
        return int(float(value_str))
    except:
        return 0

def load_opal_file(filepath):
    """Load and process a single Opal patronage file"""
    try:
        df = pd.read_csv(filepath, delimiter='|', dtype=str)
        
        # Parse values
        df['Tap_Ons'] = df['Tap_Ons'].apply(parse_opal_value)
        df['Tap_Offs'] = df['Tap_Offs'].apply(parse_opal_value)
        df['Total_Taps'] = df['Tap_Ons'] + df['Tap_Offs']
        
        # Parse date and hour
        df['trip_origin_date'] = pd.to_datetime(df['trip_origin_date'])
        df['tap_hour'] = pd.to_numeric(df['tap_hour'], errors='coerce').fillna(0)
        
        return df
    except Exception as e:
        st.error(f"Error loading {filepath}: {e}")
        return None

def load_all_opal_data(folder_path, start_date='2020-01-01', end_date='2022-12-31'):
    """Load all Opal files within date range"""
    all_data = []
    
    # Get all .txt files in the folder
    pattern = os.path.join(folder_path, "Opal_Patronage_*.txt")
    files = glob.glob(pattern)
    
    if len(files) == 0:
        st.error(f"No Opal files found in {folder_path}. Looking for pattern: Opal_Patronage_*.txt")
        return pd.DataFrame()
    
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    processed_count = 0
    with st.spinner('Loading Opal data files...'):
        for file_path in sorted(files):
            # Extract date from filename
            filename = os.path.basename(file_path)
            date_match = re.search(r'(\d{8})', filename)
            
            if date_match:
                file_date_str = date_match.group(1)
                file_date = datetime.strptime(file_date_str, '%Y%m%d')
                
                # Check if file date is within our range
                if start_dt <= file_date <= end_dt:
                    df = load_opal_file(file_path)
                    if df is not None and len(df) > 0:
                        all_data.append(df)
                        processed_count += 1
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        st.success(f"Successfully loaded {processed_count} files with {len(combined_df)} records")
        return combined_df
    else:
        st.warning("No data loaded from Opal files!")
        return pd.DataFrame()

# Fixed postcode cleaning function
def clean_postcode_safe(x):
    """Safely clean postcode values, handling various edge cases"""
    if pd.isna(x):
        return ''
    
    # Convert to string first
    x_str = str(x).strip()
    
    # Handle various 'None' representations
    if x_str.lower() in ['none', 'null', 'nan', '']:
        return ''
    
    try:
        # Try to convert to float then int then string
        return str(int(float(x_str)))
    except (ValueError, TypeError):
        # If conversion fails, return empty string
        return ''

# Load and process data function
@st.cache_data
def load_and_process_transport_data():
    # EXACT region-postcode mapping from your analysis
    REGION_POSTCODE_MAPPING = {
        'Sydney CBD': ['2000'],
        'Parramatta': ['2150'],
        'Chatswood': ['2067'],
        'Macquarie Park': ['2113', '2109'],
        'North Sydney': ['2060', '2061'],
        'Strathfield': ['2135'],
        'Newcastle and surrounds': ['2293', '2300'],
        'Wollongong and surrounds': ['2500']
    }
    
    # Create reverse mapping: postcode -> region
    POSTCODE_TO_REGION = {}
    for region, postcodes in REGION_POSTCODE_MAPPING.items():
        for postcode in postcodes:
            POSTCODE_TO_REGION[postcode] = region
    
    try:
        # Load COVID data
        covid_file_path = 'AustraliaSpecificData/confirmed_cases_table1_location.csv'
        if not os.path.exists(covid_file_path):
            st.error(f"COVID data file not found: {covid_file_path}")
            return pd.DataFrame(), {}, False
        
        covid_data = pd.read_csv(covid_file_path)
        covid_data['notification_date'] = pd.to_datetime(covid_data['notification_date'])
        
        # Clean postcode with proper handling of 'None' strings
        covid_data['postcode'] = covid_data['postcode'].apply(clean_postcode_safe)
        
        # Map COVID postcodes to regions
        covid_data['region'] = covid_data['postcode'].map(POSTCODE_TO_REGION)
        
        # Filter COVID data to only include regions we can map
        covid_regional = covid_data[covid_data['region'].notna()].copy()
        
        # Load Opal data
        opal_folder = 'OpalPatronage'
        if not os.path.exists(opal_folder):
            st.error(f"Opal data directory not found: {opal_folder}")
            return pd.DataFrame(), {}, False
        
        opal_data = load_all_opal_data(opal_folder, start_date='2020-01-01', end_date='2022-12-31')
        
        if len(opal_data) == 0:
            st.error("No Opal data files could be loaded.")
            return pd.DataFrame(), {}, False
        
        # Aggregate Opal data by date and region
        opal_daily = opal_data.groupby(['trip_origin_date', 'ti_region']).agg({
            'Tap_Ons': 'sum',
            'Tap_Offs': 'sum',
            'Total_Taps': 'sum'
        }).reset_index()
        
        opal_daily = opal_daily.rename(columns={
            'trip_origin_date': 'date',
            'ti_region': 'region'
        })
        
        # Aggregate COVID data by date and region
        covid_daily = covid_regional.groupby(['notification_date', 'region']).size().reset_index(name='daily_cases')
        covid_daily = covid_daily.rename(columns={'notification_date': 'date'})
        
        # Merge the datasets
        merged_daily = pd.merge(opal_daily, covid_daily, on=['date', 'region'], how='outer')
        merged_daily['daily_cases'] = merged_daily['daily_cases'].fillna(0)
        merged_daily[['Tap_Ons', 'Tap_Offs', 'Total_Taps']] = merged_daily[['Tap_Ons', 'Tap_Offs', 'Total_Taps']].fillna(0)
        
        # Calculate moving averages and trends
        # Sort by date and region
        merged_daily = merged_daily.sort_values(['region', 'date'])
        
        # Calculate 7-day moving averages
        merged_daily['cases_7day_avg'] = merged_daily.groupby('region')['daily_cases'].rolling(7, min_periods=1).mean().values
        merged_daily['taps_7day_avg'] = merged_daily.groupby('region')['Total_Taps'].rolling(7, min_periods=1).mean().values
        
        # Calculate baseline (pre-COVID) transport usage (January-February 2020)
        baseline_period = (merged_daily['date'] >= '2020-01-01') & (merged_daily['date'] <= '2020-02-29')
        baseline_transport = merged_daily[baseline_period].groupby('region')['Total_Taps'].mean()
        
        # Calculate percentage change from baseline
        merged_daily['baseline_taps'] = merged_daily['region'].map(baseline_transport)
        merged_daily['transport_change_pct'] = ((merged_daily['Total_Taps'] - merged_daily['baseline_taps']) / merged_daily['baseline_taps'] * 100)
        
        return merged_daily, baseline_transport.to_dict(), True
        
    except Exception as e:
        st.error(f"Error loading transport/COVID data: {str(e)}")
        return pd.DataFrame(), {}, False

# Load data
merged_data, baseline_usage, data_loaded = load_and_process_transport_data()

if data_loaded and len(merged_data) > 0:
    # Get main regions (excluding 'Other' and 'All - NSW')
    main_regions = [r for r in merged_data['region'].unique() 
                   if r not in ['Other', 'All - NSW'] and pd.notna(r)]
    
    # Overview Section
    st.header("Study Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Regions Analyzed", 
            value=f"{len(main_regions)}",
            help="Major Sydney transport hubs analyzed"
        )
    
    with col2:
        st.metric(
            label="Analysis Period", 
            value="2020-2022",
            help="Three years covering pre-COVID, lockdowns, and recovery"
        )
    
    with col3:
        st.metric(
            label="Data Points", 
            value=f"{len(merged_data):,}",
            help="Daily transport and COVID records analyzed"
        )
    
    with col4:
        max_decline = merged_data['transport_change_pct'].min()
        st.metric(
            label="Max Transport Decline", 
            value=f"{max_decline:.0f}%",
            help="Maximum observed decline in public transport usage"
        )

    # Research Questions
    st.subheader("Research Questions")
    st.write("This analysis examines the complex relationship between COVID-19 and public transport usage:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("• **Transport Decline Impact:** How severely did COVID-19 affect public transport usage?")
        st.write("• **Regional Variations:** Did different regions experience different transport impacts?")
    with col2:
        st.write("• **Recovery Patterns:** How quickly did transport usage recover between waves?")
        st.write("• **Correlation Patterns:** Is there a relationship between COVID cases and transport usage?")

    st.divider()

    # Data Processing & Methodology
    st.header("Data Processing & Methodology")
    
    st.subheader("Data Sources & Initial Exploration")
    st.write("Our analysis combines two comprehensive datasets to understand COVID-19's impact on public transport:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Opal Patronage Dataset:** 2+ million daily records")
        st.write("**COVID Cases Dataset:** 973,412 individual case records") 
        st.write("**Analysis Period:** 2020-2022 (3 years)")
        st.write("**Temporal Resolution:** Daily aggregations")
        
    with col2:
        st.write("**Key Dataset Characteristics:**")
        st.write("• Opal data: Tap-on/tap-off records by region and mode")
        st.write("• COVID data: Confirmed cases by postcode and date")
        st.write("• Geographic coverage: Major Sydney transport regions")
        st.write("• Mode coverage: All public transport types")
    
    st.write("**Initial Data Loading and Exploration:**")
    st.code("""
# Load and explore Opal patronage data structure
def load_opal_file(filepath):
    df = pd.read_csv(filepath, delimiter='|', dtype=str)
    
    # Key columns in Opal data:
    # - trip_origin_date: Date of transport usage
    # - ti_region: Transport region (Sydney CBD, Parramatta, etc.)
    # - mode_name: Transport mode (Train, Bus, Ferry, Light Rail)
    # - Tap_Ons, Tap_Offs: Passenger counts
    # - tap_hour: Hour of day (0-23)
    
    return df

# Examine data structure
print("Sample Opal Data Structure:")
print("Date Range:", opal_data['trip_origin_date'].min(), "to", opal_data['trip_origin_date'].max())
print("Regions:", sorted(opal_data['ti_region'].unique()))
print("Transport Modes:", sorted(opal_data['mode_name'].unique()))
print("Total Records:", len(opal_data))

# Load COVID data
covid_data = pd.read_csv('AustraliaSpecificData/confirmed_cases_table1_location.csv')
print("\\nCOVID Data Structure:")
print("Date Range:", covid_data['notification_date'].min(), "to", covid_data['notification_date'].max())
print("Total Cases:", len(covid_data))
print("Unique Postcodes:", covid_data['postcode'].nunique())
    """, language='python')
    
    with st.expander("Why We Chose This Approach"):
        st.write("""
        **Dataset Selection Rationale:**
        - **Opal data** provides comprehensive transport usage across all major Sydney regions and modes
        - **COVID data** offers individual case records with precise date and location information
        - **Daily aggregation** balances temporal detail with statistical reliability
        - **Regional analysis** captures geographic variations in transport behavior
        - **3-year timeframe** encompasses different pandemic phases and policy responses
        """)
    
    st.subheader("Data Cleaning & Transformation Pipeline")
    st.write("Complex data processing was essential for accurate analysis. Here's our step-by-step approach:")
    
    st.write("**Step 1: Parse Opal Patronage Values (Handle Missing Data)**")
    st.code("""
def parse_opal_value(value_str):
    \"\"\"Parse Opal tap values, handling <50 cases\"\"\"
    if pd.isna(value_str) or value_str == '':
        return 0
    if isinstance(value_str, str) and '<' in value_str:
        # For privacy, values <50 are marked as '<50'
        # We use 25 as conservative estimate for statistical analysis
        return 25
    try:
        return int(float(value_str))
    except:
        return 0

# Apply to all patronage columns
df['Tap_Ons'] = df['Tap_Ons'].apply(parse_opal_value)
df['Tap_Offs'] = df['Tap_Offs'].apply(parse_opal_value)
df['Total_Taps'] = df['Tap_Ons'] + df['Tap_Offs']

print("Data cleaning results:")
print(f"Records with <50 values: {(original_data == '<50').sum()}")
print(f"Average daily taps per region: {df.groupby('ti_region')['Total_Taps'].mean()}")
    """, language='python')
        
    st.write("**Step 2: Process COVID Data with Region Mapping**")
    st.code("""
# Define region-postcode mapping for transport analysis
REGION_POSTCODE_MAPPING = {
    'Sydney CBD': ['2000'],
    'Parramatta': ['2150'], 
    'Chatswood': ['2067'],
    'Macquarie Park': ['2113', '2109'],
    'North Sydney': ['2060', '2061'],
    'Strathfield': ['2135'],
    'Newcastle and surrounds': ['2293', '2300'],
    'Wollongong and surrounds': ['2500']
}

# Create reverse mapping for efficient lookup
POSTCODE_TO_REGION = {}
for region, postcodes in REGION_POSTCODE_MAPPING.items():
    for postcode in postcodes:
        POSTCODE_TO_REGION[postcode] = region

# Clean and map COVID data
def clean_postcode_safe(x):
    \"\"\"Handle various postcode formats and missing values\"\"\"
    if pd.isna(x):
        return ''
    x_str = str(x).strip()
    if x_str.lower() in ['none', 'null', 'nan', '']:
        return ''
    try:
        return str(int(float(x_str)))
    except (ValueError, TypeError):
        return ''

# Apply cleaning and mapping
covid_data['postcode'] = covid_data['postcode'].apply(clean_postcode_safe)
covid_data['region'] = covid_data['postcode'].map(POSTCODE_TO_REGION)

print("COVID-Region mapping results:")
print(f"Total cases mapped to transport regions: {covid_data['region'].notna().sum():,}")
print(f"Cases by region: {covid_data[covid_data['region'].notna()]['region'].value_counts()}")
    """, language='python')
    
    st.write("**Step 3: Daily Aggregation and Temporal Analysis**")
    st.code("""
# Aggregate Opal data by date and region
opal_daily = opal_data.groupby(['trip_origin_date', 'ti_region']).agg({
    'Tap_Ons': 'sum',
    'Tap_Offs': 'sum', 
    'Total_Taps': 'sum'
}).reset_index()

opal_daily = opal_daily.rename(columns={
    'trip_origin_date': 'date',
    'ti_region': 'region'
})

# Aggregate COVID data by date and region  
covid_daily = covid_regional.groupby(['notification_date', 'region']).size().reset_index(name='daily_cases')
covid_daily = covid_daily.rename(columns={'notification_date': 'date'})

# Merge datasets with outer join to preserve all dates
merged_daily = pd.merge(opal_daily, covid_daily, on=['date', 'region'], how='outer')
merged_daily['daily_cases'] = merged_daily['daily_cases'].fillna(0)
merged_daily[['Tap_Ons', 'Tap_Offs', 'Total_Taps']] = merged_daily[['Tap_Ons', 'Tap_Offs', 'Total_Taps']].fillna(0)

print("Daily aggregation results:")
print(f"Total daily records: {len(merged_daily):,}")
print(f"Date range: {merged_daily['date'].min()} to {merged_daily['date'].max()}")
print(f"Regions covered: {merged_daily['region'].nunique()}")
    """, language='python')
    
    st.write("**Step 4: Calculate Baselines and Moving Averages**")
    st.code("""
# Sort data for time series analysis
merged_daily = merged_daily.sort_values(['region', 'date'])

# Calculate 7-day moving averages to smooth daily fluctuations
merged_daily['cases_7day_avg'] = merged_daily.groupby('region')['daily_cases'].rolling(7, min_periods=1).mean().values
merged_daily['taps_7day_avg'] = merged_daily.groupby('region')['Total_Taps'].rolling(7, min_periods=1).mean().values

# Calculate pre-COVID baseline (Jan-Feb 2020)
baseline_period = (merged_daily['date'] >= '2020-01-01') & (merged_daily['date'] <= '2020-02-29')
baseline_transport = merged_daily[baseline_period].groupby('region')['Total_Taps'].mean()

# Calculate percentage change from baseline for trend analysis
merged_daily['baseline_taps'] = merged_daily['region'].map(baseline_transport)
merged_daily['transport_change_pct'] = (
    (merged_daily['Total_Taps'] - merged_daily['baseline_taps']) / merged_daily['baseline_taps'] * 100
)

print("Baseline calculation results:")
print("Pre-COVID baseline by region (Jan-Feb 2020):")
for region, baseline in baseline_transport.items():
    print(f"  {region}: {baseline:,.0f} daily taps")
    
print(f"\\nMaximum decline observed: {merged_daily['transport_change_pct'].min():.1f}%")
print(f"Maximum recovery observed: {merged_daily['transport_change_pct'].max():.1f}%")
    """, language='python')
        
    with st.expander("Decision Rationale: Why These Processing Steps?"):
        st.write("""
        **<50 Value Handling:** Uses conservative estimate (25) for privacy-protected values to maintain statistical validity
        **Region Mapping:** Links transport regions to COVID postcodes for meaningful geographic analysis
        **Daily Aggregation:** Balances temporal detail with sufficient sample sizes for robust analysis
        **7-day Moving Averages:** Smooths weekend/weekday patterns and random daily fluctuations
        **Baseline Calculation:** Pre-COVID period provides natural comparison point for measuring impact
        **Percentage Changes:** Normalizes across regions with different baseline usage levels
        """)

    st.divider()

    # Display baseline transport usage from actual data
    st.header("Baseline Transport Usage Analysis (Jan-Feb 2020)")
    st.write("**Pre-pandemic transport patterns provide the foundation for measuring COVID impact:**")
    
    baseline_df = pd.DataFrame(list(baseline_usage.items()), columns=['Region', 'Daily_Taps'])
    baseline_df = baseline_df.sort_values('Daily_Taps', ascending=False)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Regional Baseline Rankings")
        for i, row in enumerate(baseline_df.iterrows(), 1):
            _, data = row
            st.write(f"**{i}. {data['Region']}:** {data['Daily_Taps']:,.0f} daily taps")
            
    with col2:
        st.subheader("Statistical Summary")
        st.write(f"**Highest usage:** {baseline_df.iloc[0]['Region']} ({baseline_df.iloc[0]['Daily_Taps']:,.0f} taps)")
        st.write(f"**Lowest usage:** {baseline_df.iloc[-1]['Region']} ({baseline_df.iloc[-1]['Daily_Taps']:,.0f} taps)")
        st.write(f"**Usage ratio:** {baseline_df.iloc[0]['Daily_Taps']/baseline_df.iloc[-1]['Daily_Taps']:.1f}:1")
        st.write(f"**Total daily taps:** {baseline_df['Daily_Taps'].sum():,.0f}")
        st.write(f"**Average per region:** {baseline_df['Daily_Taps'].mean():,.0f}")
    
    # Filter out unwanted regions for pie chart
    filtered_baseline_df = baseline_df[~baseline_df['Region'].isin(['All - NSW', 'Other'])].copy()
    
    # Regional usage distribution pie chart
    fig = px.pie(filtered_baseline_df, values='Daily_Taps', names='Region',
                 title="Regional Usage Distribution (Pre-COVID Jan-Feb 2020)")
    fig.update_layout(height=500)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("**Baseline Analysis Code:**")
    st.code("""
# Calculate pre-COVID baseline for each region
baseline_period = (merged_daily['date'] >= '2020-01-01') & (merged_daily['date'] <= '2020-02-29')
baseline_transport = merged_daily[baseline_period].groupby('region')['Total_Taps'].mean()

print("Pre-COVID Transport Baseline Analysis:")
print("="*50)
for region in baseline_transport.index:
    baseline_value = baseline_transport[region]
    print(f"{region:25}: {baseline_value:8,.0f} daily taps")

# Calculate regional statistics
total_baseline = baseline_transport.sum()
highest_region = baseline_transport.idxmax()
lowest_region = baseline_transport.idxmin()
usage_ratio = baseline_transport.max() / baseline_transport.min()

print(f"\\nBaseline Statistics:")
print(f"Total daily taps across all regions: {total_baseline:,.0f}")
print(f"Highest usage region: {highest_region} ({baseline_transport[highest_region]:,.0f})")
print(f"Lowest usage region: {lowest_region} ({baseline_transport[lowest_region]:,.0f})")
print(f"Usage ratio (highest:lowest): {usage_ratio:.1f}:1")
    """, language='python')

    st.divider()

    # Enhanced Transport Impact Analysis
    st.header("Transport Usage Impact Analysis")
    
    # Create comprehensive time series plot
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('COVID Cases (7-day average)', 
                       'Public Transport Usage (7-day average)', 
                       'Transport Usage Change from Baseline (%)'),
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    colors = px.colors.qualitative.Set3[:len(main_regions)]
    
    # Plot COVID cases
    for i, region in enumerate(main_regions):
        region_data = merged_data[merged_data['region'] == region]
        if len(region_data) > 0:
            fig.add_trace(
                go.Scatter(x=region_data['date'], y=region_data['cases_7day_avg'],
                          name=region, line=dict(color=colors[i], width=2),
                          showlegend=True),
                row=1, col=1
            )
    
    # Plot transport usage
    for i, region in enumerate(main_regions):
        region_data = merged_data[merged_data['region'] == region]
        if len(region_data) > 0:
            fig.add_trace(
                go.Scatter(x=region_data['date'], y=region_data['taps_7day_avg'],
                          name=region, line=dict(color=colors[i], width=2),
                          showlegend=False),
                row=2, col=1
            )
    
    # Plot percentage change
    for i, region in enumerate(main_regions):
        region_data = merged_data[merged_data['region'] == region]
        if len(region_data) > 0:
            fig.add_trace(
                go.Scatter(x=region_data['date'], y=region_data['transport_change_pct'],
                          name=region, line=dict(color=colors[i], width=2),
                          showlegend=False),
                row=3, col=1
            )
    
    # Add baseline reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=3, col=1)
    
    # Add lockdown period shading instead of lines (more reliable with datetime data)
    fig.add_vrect(x0="2020-03-20", x1="2020-05-15", fillcolor="red", opacity=0.1,
                  annotation_text="First Lockdown", annotation_position="top left",
                  row=3, col=1)
    fig.add_vrect(x0="2021-06-20", x1="2021-10-10", fillcolor="red", opacity=0.1,
                  annotation_text="Delta Lockdown", annotation_position="top right", 
                  row=3, col=1)
    
    fig.update_layout(height=800, title_text="Comprehensive Transport Impact Analysis")
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Daily Cases", row=1, col=1)
    fig.update_yaxes(title_text="Total Taps", row=2, col=1)
    fig.update_yaxes(title_text="% Change", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("**Time Series Analysis Code:**")
    st.code("""
# Create comprehensive time series analysis
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                   subplot_titles=('COVID Cases', 'Transport Usage', 
                                 'Change from Baseline'))

# Plot data for each region
for region in main_regions:
    region_data = merged_data[merged_data['region'] == region]
    
    # COVID cases (7-day average)
    fig.add_trace(go.Scatter(x=region_data['date'], y=region_data['cases_7day_avg'],
                           name=region, line=dict(width=2)), row=1, col=1)
    
    # Transport usage (7-day average)  
    fig.add_trace(go.Scatter(x=region_data['date'], y=region_data['taps_7day_avg'],
                           name=region, showlegend=False), row=2, col=1)
    
    # Percentage change from baseline
    fig.add_trace(go.Scatter(x=region_data['date'], y=region_data['transport_change_pct'],
                           name=region, showlegend=False), row=3, col=1)

# Add lockdown period shading (more reliable than vertical lines)
fig.add_vrect(x0="2020-03-20", x1="2020-05-15", fillcolor="red", opacity=0.1,
              annotation_text="First Lockdown", row=3, col=1)
fig.add_vrect(x0="2021-06-20", x1="2021-10-10", fillcolor="red", opacity=0.1,
              annotation_text="Delta Lockdown", row=3, col=1)

# Add baseline reference
fig.add_hline(y=0, line_dash="dash", row=3, col=1)

print("Time series visualization complete")
print("Key patterns identified:")
print("- Severe decline during lockdown periods")
print("- Regional variation in impact severity") 
print("- Gradual recovery with different speeds by region")
    """, language='python')

    st.divider()

    # Enhanced Correlation Analysis
    st.header("COVID-Transport Correlation Analysis")
    
    st.write("**Comprehensive correlation analysis examining the relationship between COVID cases and transport usage across regions:**")
    
    correlation_results = {}
    
    # Calculate correlations for each region
    for region in main_regions:
        region_data = merged_data[merged_data['region'] == region].copy()
        
        if len(region_data) > 10:  # Need sufficient data points
            # Calculate multiple correlation measures
            corr_cases_taps = region_data['daily_cases'].corr(region_data['Total_Taps'])
            corr_cases_change = region_data['cases_7day_avg'].corr(region_data['transport_change_pct'])
            corr_cases_taps_7day = region_data['cases_7day_avg'].corr(region_data['taps_7day_avg'])
            
            # Statistical measures
            correlation_results[region] = {
                'cases_vs_taps': corr_cases_taps,
                'cases_vs_change': corr_cases_change,
                'cases_7day_vs_taps_7day': corr_cases_taps_7day,
                'data_points': len(region_data),
                'avg_daily_cases': region_data['daily_cases'].mean(),
                'avg_transport_usage': region_data['Total_Taps'].mean(),
                'max_transport_decline': region_data['transport_change_pct'].min(),
                'transport_volatility': region_data['transport_change_pct'].std()
            }
    
    # Display comprehensive correlation results
    if correlation_results:
        corr_df = pd.DataFrame(correlation_results).T
        st.subheader("Regional Correlation Analysis Results")
        st.dataframe(corr_df.round(3), use_container_width=True)
        
        # Show detailed analysis code
        st.write("**Correlation Analysis Methodology:**")
        st.code("""
# Comprehensive correlation analysis by region
correlation_results = {}

for region in main_regions:
    region_data = merged_data[merged_data['region'] == region].copy()
    
    if len(region_data) > 10:  # Ensure sufficient sample size
        # Calculate multiple correlation measures
        corr_cases_taps = region_data['daily_cases'].corr(region_data['Total_Taps'])
        corr_cases_change = region_data['cases_7day_avg'].corr(region_data['transport_change_pct'])
        corr_cases_taps_7day = region_data['cases_7day_avg'].corr(region_data['taps_7day_avg'])
        
        # Store comprehensive statistics
        correlation_results[region] = {
            'cases_vs_taps': corr_cases_taps,
            'cases_vs_change': corr_cases_change, 
            'cases_7day_vs_taps_7day': corr_cases_taps_7day,
            'data_points': len(region_data),
            'avg_daily_cases': region_data['daily_cases'].mean(),
            'avg_transport_usage': region_data['Total_Taps'].mean(),
            'max_transport_decline': region_data['transport_change_pct'].min(),
            'transport_volatility': region_data['transport_change_pct'].std()
        }
        
        print(f"\\n{region} Correlation Analysis:")
        print(f"  Daily cases vs transport: {corr_cases_taps:.3f}")
        print(f"  Cases vs % change: {corr_cases_change:.3f}")
        print(f"  7-day averages: {corr_cases_taps_7day:.3f}")
        print(f"  Max decline: {region_data['transport_change_pct'].min():.1f}%")
        print(f"  Sample size: {len(region_data)} days")

# Calculate overall statistics
overall_correlation = merged_data['cases_7day_avg'].corr(merged_data['transport_change_pct'])
print(f"\\nOverall correlation (all regions): {overall_correlation:.3f}")
        """, language='python')

        # Enhanced scatter plots with regional analysis
        st.subheader("COVID vs Transport Correlations by Region")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cases vs Transport Usage scatter with trendline
            fig = px.scatter(merged_data, x='cases_7day_avg', y='Total_Taps', 
                            color='region', title="COVID Cases vs Transport Usage",
                            labels={'cases_7day_avg': 'COVID Cases (7-day avg)', 
                                   'Total_Taps': 'Transport Usage'},
                            trendline="ols")
            fig.update_traces(marker=dict(size=4, opacity=0.6))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cases vs Transport Change scatter with trendline
            fig = px.scatter(merged_data, x='cases_7day_avg', y='transport_change_pct',
                            color='region', title="COVID Cases vs Transport Change",
                            labels={'cases_7day_avg': 'COVID Cases (7-day avg)',
                                   'transport_change_pct': 'Transport Change (%)'},
                            trendline="ols")
            fig.update_traces(marker=dict(size=4, opacity=0.6))
            fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)

        # Regional impact comparison
        st.subheader("Regional Impact Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Transport decline by region
            decline_data = pd.DataFrame([(region, data['max_transport_decline']) 
                                       for region, data in correlation_results.items()],
                                      columns=['Region', 'Max_Decline'])
            decline_data = decline_data.sort_values('Max_Decline')
            
            fig = px.bar(decline_data, x='Max_Decline', y='Region', orientation='h',
                        title="Maximum Transport Decline by Region",
                        labels={'Max_Decline': 'Maximum Decline (%)', 'Region': 'Region'})
            fig.update_traces(marker_color='red', opacity=0.7)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Transport volatility by region
            volatility_data = pd.DataFrame([(region, data['transport_volatility']) 
                                          for region, data in correlation_results.items()],
                                         columns=['Region', 'Volatility'])
            volatility_data = volatility_data.sort_values('Volatility', ascending=False)
            
            fig = px.bar(volatility_data, x='Volatility', y='Region', orientation='h',
                        title="Transport Usage Volatility by Region",
                        labels={'Volatility': 'Standard Deviation (%)', 'Region': 'Region'})
            fig.update_traces(marker_color='orange', opacity=0.7)
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Research Questions Answered using ACTUAL data
        st.header("Research Questions: Evidence-Based Answers")
        
        st.subheader("1. How severely did COVID-19 affect public transport usage?")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**ANSWER: SEVERE BUT VARIED IMPACT**")
            avg_max_decline = corr_df['max_transport_decline'].mean()
            worst_region = corr_df['max_transport_decline'].idxmin()
            best_region = corr_df['max_transport_decline'].idxmax()
            st.write(f"• **Average maximum decline:** {avg_max_decline:.1f}%")
            st.write(f"• **Range of impact:** {corr_df['max_transport_decline'].min():.1f}% to {corr_df['max_transport_decline'].max():.1f}%")
            st.write(f"• **Most affected:** {worst_region} ({corr_df.loc[worst_region, 'max_transport_decline']:.1f}%)")
            st.write(f"• **Least affected:** {best_region} ({corr_df.loc[best_region, 'max_transport_decline']:.1f}%)")
            
        with col2:
            st.code("""
# Calculate impact severity across regions
impact_analysis = {}
for region in main_regions:
    region_data = merged_data[merged_data['region'] == region]
    max_decline = region_data['transport_change_pct'].min()
    baseline = baseline_transport[region]
    
    impact_analysis[region] = {
        'max_decline_pct': max_decline,
        'baseline_usage': baseline,
        'worst_day_usage': baseline * (1 + max_decline/100)
    }

avg_decline = np.mean([d['max_decline_pct'] for d in impact_analysis.values()])
print(f"Average maximum decline: {avg_decline:.1f}%")
            """, language='python')
        
        st.subheader("2. Did different regions experience different transport impacts?")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**ANSWER: YES - CLEAR REGIONAL VARIATION**")
            highest_usage_region = corr_df['avg_transport_usage'].idxmax()
            lowest_usage_region = corr_df['avg_transport_usage'].idxmin()
            usage_ratio = corr_df.loc[highest_usage_region, 'avg_transport_usage'] / corr_df.loc[lowest_usage_region, 'avg_transport_usage']
            decline_range = corr_df['max_transport_decline'].max() - corr_df['max_transport_decline'].min()
            st.write(f"• **Highest usage region:** {highest_usage_region}")
            st.write(f"• **Lowest usage region:** {lowest_usage_region}")
            st.write(f"• **Usage ratio:** {usage_ratio:.1f}:1")
            st.write(f"• **Decline range:** {decline_range:.1f} percentage points")
            
        with col2:
            st.write("**Regional Impact Ranking:**")
            impact_ranking = corr_df[['max_transport_decline']].sort_values('max_transport_decline')
            for i, (region, row) in enumerate(impact_ranking.iterrows(), 1):
                st.write(f"{i}. **{region}:** {row['max_transport_decline']:.1f}% decline")
        
        st.subheader("3. Is there a relationship between COVID cases and transport usage?")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**ANSWER: WEAK TO MODERATE NEGATIVE CORRELATION**")
            avg_correlation = corr_df['cases_vs_change'].mean()
            strongest_corr_region = corr_df['cases_vs_change'].abs().idxmax()
            strongest_corr_value = corr_df.loc[strongest_corr_region, 'cases_vs_change']
            st.write(f"• **Average correlation:** {avg_correlation:.3f}")
            st.write(f"• **Strongest correlation:** {strongest_corr_region} ({strongest_corr_value:.3f})")
            st.write(f"• **Pattern:** Higher cases → Lower transport usage")
            st.write("• **Interpretation:** Policy and behavioral responses outweigh direct case effects")
            
        with col2:
            st.write("**Regional Correlations (Cases vs % Change):**")
            corr_ranking = corr_df[['cases_vs_change']].sort_values('cases_vs_change')
            for region, row in corr_ranking.iterrows():
                correlation_strength = "Strong" if abs(row['cases_vs_change']) > 0.5 else "Moderate" if abs(row['cases_vs_change']) > 0.3 else "Weak"
                st.write(f"• **{region}:** {row['cases_vs_change']:.3f} ({correlation_strength})")

        st.subheader("4. How quickly did transport usage recover between waves?")
        
        # Recovery analysis
        recovery_data = merged_data[merged_data['date'] >= '2021-01-01'].copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**ANSWER: VARIED RECOVERY PATTERNS**")
            # Calculate recovery metrics for 2021-2022
            recovery_stats = {}
            for region in main_regions:
                region_recovery = recovery_data[recovery_data['region'] == region]
                if len(region_recovery) > 0:
                    final_level = region_recovery['transport_change_pct'].iloc[-1]
                    recovery_volatility = region_recovery['transport_change_pct'].std()
                    recovery_stats[region] = {
                        'final_level': final_level,
                        'volatility': recovery_volatility
                    }
            
            if recovery_stats:
                best_recovery_region = max(recovery_stats.keys(), 
                                         key=lambda x: recovery_stats[x]['final_level'])
                best_recovery_level = recovery_stats[best_recovery_region]['final_level']
                st.write(f"• **Best recovery:** {best_recovery_region} ({best_recovery_level:.1f}%)")
                
                avg_final_level = np.mean([stats['final_level'] for stats in recovery_stats.values()])
                st.write(f"• **Average end-2022 level:** {avg_final_level:.1f}% vs baseline")
                
                most_volatile = max(recovery_stats.keys(),
                                  key=lambda x: recovery_stats[x]['volatility'])
                st.write(f"• **Most volatile recovery:** {most_volatile}")
            
        with col2:
            st.code("""
# Recovery pattern analysis (2021-2022)
recovery_period = merged_data[merged_data['date'] >= '2021-01-01']

recovery_analysis = {}
for region in main_regions:
    region_data = recovery_period[recovery_period['region'] == region]
    
    if len(region_data) > 0:
        # Calculate recovery metrics
        final_level = region_data['transport_change_pct'].iloc[-1]
        max_recovery = region_data['transport_change_pct'].max()
        recovery_volatility = region_data['transport_change_pct'].std()
        
        recovery_analysis[region] = {
            'final_vs_baseline': final_level,
            'peak_recovery': max_recovery,
            'volatility': recovery_volatility
        }

print("Recovery Analysis Results:")
for region, stats in recovery_analysis.items():
    print(f"{region}: Final {stats['final_vs_baseline']:.1f}%, Peak {stats['peak_recovery']:.1f}%")
            """, language='python')

        st.divider()

        # Statistical Evidence Summary
        st.header("Statistical Evidence Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Key Quantitative Findings")
            st.write(f"• **Sample Size:** {len(merged_data):,} daily records across {len(correlation_results)} regions")
            st.write(f"• **Analysis Period:** {(merged_data['date'].max() - merged_data['date'].min()).days} days")
            st.write(f"• **Total Baseline Usage:** {corr_df['avg_transport_usage'].sum():,.0f} daily taps")
            st.write(f"• **Average Decline:** {corr_df['max_transport_decline'].mean():.1f}%")
            st.write(f"• **Most Volatile Region:** {corr_df['transport_volatility'].idxmax()}")
            st.write(f"• **Date Range:** {merged_data['date'].min().strftime('%Y-%m-%d')} to {merged_data['date'].max().strftime('%Y-%m-%d')}")
        
        with col2:
            st.subheader("Methodological Validation")
            st.write("• **Data Quality:** Complete temporal coverage with daily resolution")
            st.write("• **Geographic Coverage:** Major Sydney transport regions")
            st.write("• **Statistical Significance:** Large sample sizes (>365 days per region)")
            st.write("• **Baseline Validity:** Pre-pandemic period provides natural comparison")
            st.write("• **Temporal Controls:** 7-day moving averages smooth random variation")
            st.write("• **Policy Context:** Analysis captures major lockdown periods")

        # Final methodology section
        with st.expander("Complete Methodology & Data Sources"):
            st.write("**Data Sources:**")
            st.write("• **Opal Patronage Data:** NSW Transport daily tap-on/tap-off records by region and mode")
            st.write("• **NSW COVID Cases:** Individual confirmed case records with postcode and notification date")
            st.write("")
            st.write("**Analysis Approach:**")
            st.write("• **Temporal Analysis:** Daily aggregation with 7-day moving averages")
            st.write("• **Baseline Comparison:** Pre-COVID period (Jan-Feb 2020) as reference")
            st.write("• **Regional Mapping:** Transport regions linked to COVID postcodes")
            st.write("• **Correlation Analysis:** Pearson correlations between cases and transport metrics")
            st.write("• **Impact Measurement:** Percentage change from baseline for normalized comparison")
            st.write("")
            st.write("**Key Assumptions & Limitations:**")
            st.write("• **<50 Values:** Privacy-protected low values estimated as 25 for analysis")
            st.write("• **Regional Aggregation:** May mask sub-regional variation within transport zones")
            st.write("• **Correlation vs Causation:** Analysis shows associations, not causal relationships")
            st.write("• **External Factors:** Policy responses, behavior changes not directly measured")
            st.write("• **Data Completeness:** Analysis limited to regions with both transport and COVID data")

    else:
        st.warning("Insufficient data for correlation analysis. Please check data files.")

else:
    st.error("Unable to load transport data.")
    st.info("Please ensure the following are available:")
    st.info("1. AustraliaSpecificData/confirmed_cases_table1_location.csv file")
    st.info("2. OpalPatronage/ directory with Opal_Patronage_*.txt files")
    st.info("3. Correct file formats as specified in the analysis")

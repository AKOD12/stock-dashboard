import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import io

def parse_uploaded_csv(uploaded_file):
    """Parse the uploaded CSV file, skipping header rows."""
    # Read the file content
    content = uploaded_file.getvalue().decode('utf-8')
    
    # Skip the first three lines (header rows)
    data_content = '\n'.join(content.split('\n')[3:])
    
    # Read the CSV content into a DataFrame
    df = pd.read_csv(io.StringIO(data_content))
    
    # Extract ticker symbols from the Symbol column
    return df[df['Symbol'].notna()]['Symbol'].tolist()

def get_earnings_estimates(ticker):
    try:
        stock = yf.Ticker(ticker)
        earnings = stock.earnings_estimate
        
        # First try to get estimates using column names
        if '+1Y' in earnings.columns and '0Y' in earnings.columns:
            current_year = earnings.loc[earnings.index == 'Average Estimate', '0Y'].iloc[0]
            next_year = earnings.loc[earnings.index == 'Average Estimate', '+1Y'].iloc[0]
        else:
            # Fallback to positional indexing if column names don't match
            current_year = earnings.iloc[0, 2]  # Current year average estimate
            next_year = earnings.iloc[0, 3]     # Next year average estimate
            
        return pd.Series([current_year, next_year])
    except Exception as e:
        print(f"Error getting earnings estimates for {ticker}: {str(e)}")
        return pd.Series([None, None])

def get_revenue_estimates(ticker):
    try:
        stock = yf.Ticker(ticker)
        revenue = stock.revenue_estimate
        
        # First try to get estimates using column names
        if '+1Y' in revenue.columns and '0Y' in revenue.columns:
            current_year = revenue.loc[revenue.index == 'Average Estimate', '0Y'].iloc[0]
            next_year = revenue.loc[revenue.index == 'Average Estimate', '+1Y'].iloc[0]
        else:
            # Fallback to positional indexing if column names don't match
            current_year = revenue.iloc[0, 2]  # Current year average estimate
            next_year = revenue.iloc[0, 3]     # Next year average estimate
            
        return pd.Series([current_year, next_year])
    except Exception as e:
        print(f"Error getting revenue estimates for {ticker}: {str(e)}")
        return pd.Series([None, None])

def get_earnings_surprise(ticker):
    try:
        earnings_history = yf.Ticker(ticker).earnings_history
        if not earnings_history.empty:
            return earnings_history['surprisePercent'].iloc[3] * 100 # Most recent surprise percentage
    except:
        return None

def scan_for_highs(tickers_df=None, uploaded_file=None, threshold_pct=2.0):
    # Get tickers either from uploaded file or exchange file
    if uploaded_file is not None:
        tickers = parse_uploaded_csv(uploaded_file)
    else:
        tickers = tickers_df['Ticker']  # Use the original exchange file format
    
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Fetch data and check for 1-year high
    one_year_highs = {}
    for i, ticker in enumerate(tickers):
        try:
            status_text.text(f"Scanning {ticker}...")
            progress_bar.progress((i + 1) / len(tickers))
            
            stock = yf.Ticker(ticker)
            
            # Get current day's data
            today_data = stock.history(period='1d')
            if today_data.empty:
                continue
            today_price = today_data['Close'].iloc[0]
            today_high = today_data['High'].iloc[0]
            
            # Get historical data for 52-week high
            hist = stock.history(start=start_date, end=end_date)
            if hist.empty:
                continue
                
            # Calculate the max price in the past year including today's high
            max_price = max(hist['High'].max(), today_high)
            
            if today_price is None or max_price is None:
                continue
            
            # Calculate percentage from high
            pct_from_high = ((max_price - today_price) / max_price) * 100
            
            # Consider it a "hit" if within threshold_pct of high
            is_near_high = True
            
            # Get company info including short interest data
            info = stock.info
            if 'longBusinessSummary' not in info:
                continue
            
            # Get earnings and revenue estimates first
            eps_estimates = get_earnings_estimates(ticker)
            eps_current, eps_next = eps_estimates.iloc[0], eps_estimates.iloc[1]
            
            rev_estimates = get_revenue_estimates(ticker)
            rev_current, rev_next = rev_estimates.iloc[0], rev_estimates.iloc[1]
            
            surprise_pct = get_earnings_surprise(ticker)

            one_year_highs[ticker] = {
                "Today Price": today_price,
                "1-Year High": max_price,
                "Percent From High": pct_from_high,
                "Near High": is_near_high,
                "Name": info.get('shortName', 'Name not available'),
                "Description": info.get('longBusinessSummary', 'Description not available'),
                "Industry": info.get('industry', 'N/A'),
                "Sector": info.get('sector', 'N/A'),
                "Short Ratio": info.get('shortRatio', None),
                "Short Percent of Float": info.get('shortPercentOfFloat', None),
                "Current Year EPS": eps_current,
                "Next Year EPS": eps_next,
                "Current Year Revenue": rev_current,
                "Next Year Revenue": rev_next,
                "Earnings Surprise": surprise_pct
            }
                    
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    # Convert to DataFrame
    output_df = pd.DataFrame.from_dict(one_year_highs, orient='index')
    
    # Filter for only those near highs and with descriptions
    hits_df = output_df[
        (output_df['Near High'] == True) & 
        (output_df['Description'].notna()) & 
        (output_df['Description'] != 'Description not available')
    ].copy()
    
    # Sort by percentage from high
    hits_df = hits_df.sort_values('Percent From High')
    
    return hits_df

def display_interactive_table(df):
    # Create the DataFrame with formatted values
    summary_df = pd.DataFrame(index=df.index)
    
    # Add all columns with proper formatting
    summary_df['% From High'] = df['Percent From High'].apply(lambda x: f"{x:.2f}%")
    summary_df['Name'] = df['Name'].fillna('N/A')
    summary_df['Industry'] = df['Industry'].fillna('N/A')
    summary_df['Sector'] = df['Sector'].fillna('N/A')
    summary_df['Recent Earnings Surprise'] = df['Earnings Surprise'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
    summary_df['Short Ratio'] = df['Short Ratio'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
    summary_df['Short % of Float'] = df['Short Percent of Float'].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
    summary_df['Current Year EPS'] = df['Current Year EPS'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
    summary_df['Next Year EPS'] = df['Next Year EPS'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
    
    # Calculate and format EPS Growth %
    summary_df['EPS Growth %'] = df.apply(
        lambda row: f"{((row['Next Year EPS'] - row['Current Year EPS']) / row['Current Year EPS'] * 100):.1f}%" 
        if pd.notnull(row['Next Year EPS']) and pd.notnull(row['Current Year EPS']) and row['Current Year EPS'] != 0 
        else "N/A", axis=1
    )
    
    summary_df['Current Year Revenue'] = df['Current Year Revenue'].apply(lambda x: f"${x/1_000_000:.1f}M" if pd.notnull(x) else "N/A")
    summary_df['Next Year Revenue'] = df['Next Year Revenue'].apply(lambda x: f"${x/1_000_000:.1f}M" if pd.notnull(x) else "N/A")
    
    # Calculate and format Revenue Growth %
    summary_df['Revenue Growth %'] = df.apply(
        lambda row: f"{((row['Next Year Revenue'] - row['Current Year Revenue']) / row['Current Year Revenue'] * 100):.1f}%" 
        if pd.notnull(row['Next Year Revenue']) and pd.notnull(row['Current Year Revenue']) and row['Current Year Revenue'] != 0 
        else "N/A", axis=1
    )

    # Convert index to column
    summary_df = summary_df.reset_index()
    summary_df = summary_df.rename(columns={'index': 'Ticker'})

    # Store the original index to DataFrame mapping for looking up descriptions
    ticker_to_description = df['Description'].to_dict()
    
    # Create an interactive dataframe
    selection = st.dataframe(
        summary_df,
        use_container_width=True,
        height=400,
        column_config={
            "Ticker": st.column_config.TextColumn(
                "Ticker",
                help="Click ticker to view company details",
                width="medium"
            ),
            "Company": st.column_config.TextColumn("Company", width="medium"),
            "Short Ratio": st.column_config.TextColumn("Short Ratio", width="medium"),
            "Short % of Float": st.column_config.TextColumn("Short % of Float", width="medium"),
            "Current Year Revenue": st.column_config.TextColumn("Current Year Revenue", width="medium"),
            "Next Year Revenue": st.column_config.TextColumn("Next Year Revenue", width="medium"),
            "EPS Growth %": st.column_config.TextColumn("EPS Growth %", width="medium"),
            "Revenue Growth %": st.column_config.TextColumn("Revenue Growth %", width="medium"),
            "Recent Earnings Surprise": st.column_config.TextColumn("Recent Earnings Surprise", width="medium"),
        },
        hide_index=True
    )
    
    # Add a separate selectbox for choosing tickers
    selected_ticker = st.selectbox(
        "Select a stock to view details:",
        options=df.index.tolist(),
        key="ticker_selector"
    )

    # Display company description if a ticker is selected
    if selected_ticker:
        st.write("---")
        st.write(f"### {selected_ticker} - Company Details")
        st.write(ticker_to_description[selected_ticker])

    return selected_ticker

def display_stock_details(stock_data):
    # Add Company Description
    if pd.notnull(stock_data['Description']):
        st.write("#### Company Description")
        st.write(stock_data['Description'])

# Set page config
st.set_page_config(page_title="Stock Scanner", layout="wide")

# Title
st.title("52 Week High Scanner")

# Add exchange selection and file upload option
col1, col2 = st.columns(2)

with col1:
    exchange = st.radio(
        "Select stock exchange or upload custom list:",
        options=["NYSE", "NASDAQ", "Custom Upload"],
        horizontal=True
    )

with col2:
    if exchange == "Custom Upload":
        uploaded_file = st.file_uploader("Upload your watchlist CSV file", type=['csv'])
    else:
        uploaded_file = None

# Map exchange selection to file name
if exchange != "Custom Upload":
    exchange_file = "nyse_tickers.csv" if exchange == "NYSE" else "nasdaq_tickers.csv"

# Add threshold slider
#threshold = st.slider("Select maximum percentage from 52-week high:", 0.0, 5.0, 2.0, 0.1)

# Add scan button
if st.button("Scan for Stocks"):
    # Run the scan with selected exchange file or uploaded file
    with st.spinner(f'Scanning stocks...'):
        if exchange == "Custom Upload" and uploaded_file is not None:
            results_df = scan_for_highs(uploaded_file=uploaded_file, threshold_pct= 0)
        else:
            tickers_df = pd.read_csv(exchange_file) # Keep the original slicing
            results_df = scan_for_highs(tickers_df=tickers_df, threshold_pct= 0)
        
        # Store results in session state
        st.session_state['scan_results'] = results_df
        
        # Display number of stocks found
        st.success(f"Found {len(results_df)} stocks at their 52-week highs!")

# Display results if available
if 'scan_results' in st.session_state and not st.session_state['scan_results'].empty:
    selected_ticker = display_interactive_table(st.session_state['scan_results'])
    
    # If a ticker is selected, display the details
    if selected_ticker:
        st.write("---")
        st.write(f"### {selected_ticker} - Company Details")
        display_stock_details(st.session_state['scan_results'].loc[selected_ticker])
    
    # Option to download results
    csv = st.session_state['scan_results'].to_csv()
    st.download_button(
        label="Download scan results as CSV",
        data=csv,
        file_name=f"{exchange.lower()}_scanner_results.csv",
        mime="text/csv"
    )
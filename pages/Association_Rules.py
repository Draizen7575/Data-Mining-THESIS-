import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Navigation buttons at the top
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🏠 Home"):
        st.switch_page("Home.py")

with col2:
    if st.button("Kmeans Clustering"):
        st.switch_page("pages/KMEANS.py")

with col3:
    if st.button("Association Rules"):
        st.switch_page("pages/Association_Rules.py")

with col4:
    if st.button("Sentiment Analysis"):
        st.switch_page("pages/Sentiment_Analysis.py")

st.markdown("<h1 style='text-align: center;'>Association Rules Visualization</h1>", unsafe_allow_html=True)

def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    else:
        return None

st.markdown("Upload Your Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

df = load_data(uploaded_file)
# df = pd.read_csv("modified_dataset.csv")

def SuperMain():
    grouped = df.groupby('orderid')['purchased_item'].apply(list).reset_index()
    filtered = grouped[grouped['purchased_item'].apply(len) > 1]
    transactions = filtered['purchased_item'].tolist()

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df_encoded, min_support=0.005, use_colnames=True)
    frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: frozenset(x))
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
    #rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2, num_itemsets=10)


    frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

    frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)

    def plot_bar(data, x_col, y_col, title, xlabel, ylabel, color='skyblue'):
        plt.figure(figsize=(10, 6))
        sns.barplot(x=x_col, y=y_col, data=data, color=color)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

    @st.fragment
    def Top10():
        # Top 10 Frequent Item Sets
        st.markdown("<h3 style='text-align: center;'>Top 10 Frequent Itemsets</h3>", unsafe_allow_html=True)
        # st.markdown("### Top 10 Frequent Itemsets")
        plot_bar(frequent_itemsets.head(10), 'support', 'itemsets', 'Top 10 Frequent Itemsets', 'Support', 'Itemsets')

        
    Top10()

    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M').astype(str)

    @st.fragment
    def Top5forMonth(df):
        st.markdown("<h3 style='text-align: center;'>Monthly Top Items</h3>", unsafe_allow_html=True)
        # st.markdown("### Monthly Top Items")
        
        # Convert date column to datetime and then extract month period
        df['month'] = df['date'].dt.to_period('M').astype(str)
        
        # Create a dictionary to map 'YYYY-MM' to 'Month Year' and sort by the original date
        month_map = {m: pd.to_datetime(m).strftime('%B %Y') for m in sorted(df['month'].unique())}
        
        # Select month from dropdown (displaying sorted 'Month Year')
        selected_month_readable = st.selectbox("Select Month", list(month_map.values()))
        
        # Get the original 'YYYY-MM' format for filtering
        selected_month = [k for k, v in month_map.items() if v == selected_month_readable][0]
        
        st.markdown(f"<h3 style='text-align: center;'>Top Items for {selected_month_readable}</h3>", unsafe_allow_html=True)
        # st.markdown(f"### Top Items for {selected_month_readable}")
        
        # Create basket for the selected month
        basket = pd.pivot_table(df, index='orderid', columns='purchased_item', aggfunc='size', fill_value=0)
        basket = basket > 0  # Convert counts to binary values (1 if purchased, 0 otherwise)
        
        # Filter the data for the selected month
        monthly_data = basket.loc[df[df['month'] == selected_month]['orderid'].unique()]
        
        # Run the apriori algorithm
        frequent_items = apriori(monthly_data, min_support=0.01, use_colnames=True)
        top_items = frequent_items.sort_values(by='support', ascending=False).head(5)
        
        # Plot the top 5 items
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_items['support'], y=top_items['itemsets'].apply(lambda x: ', '.join(list(x))), color='skyblue')
        plt.title(f'Top Items with Highest Support - {selected_month_readable}', fontsize=16)
        plt.xlabel('Support', fontsize=14)
        plt.ylabel('Itemsets', fontsize=14)
        plt.tight_layout()
        
        # Display the plot in Streamlit
        st.pyplot(plt.gcf())
        plt.close()

    # Call the function
    Top5forMonth(df)

    # Display the default table
    st.markdown("<h3 style='text-align: center;'>Itemsets Support and Confidence</h3>", unsafe_allow_html=True)
    # st.markdown("### Itemsets Support and Confidence")
    #table = rules[['antecedents', 'consequents', 'support', 'confidence']]
    #st.write(table)

    @st.fragment
    def SupportAndConfidence():
        # Add a selectbox for users to choose sorting preference
        sort_option = st.selectbox(
            "Sort by:",
            ["Itemsets Support and Confidence", "Highest Confidence", "Highest Support"]
        )

        # Function to format float or convert and format string
        def format_value(x):
            try:
                return f"{float(x):.4f}"
            except ValueError:
                return x  # Return as is if it's not a valid float

        # Format support and confidence to 4 decimal places
        rules['support'] = rules['support'].apply(format_value)
        rules['confidence'] = rules['confidence'].apply(format_value)

        if sort_option == "Highest Confidence":
            confidence_threshold = 0.54
            high_confidence_rules = rules[rules['confidence'].apply(lambda x: float(x) > confidence_threshold)]
            high_confidence_rules = high_confidence_rules.sort_values(by='confidence', ascending=False, key=lambda x: x.astype(float))
            table = high_confidence_rules[['antecedents', 'consequents', 'confidence']]
            st.markdown("<h3 style='text-align: center;'>Itemsets with High Confidence</h3>", unsafe_allow_html=True)
            # st.markdown("### Itemsets with High Confidence")
            st.write(table)

        elif sort_option == "Highest Support":
            support_threshold = 0.02
            high_support_rules = rules[rules['support'].apply(lambda x: float(x) > support_threshold)]
            high_support_rules = high_support_rules.sort_values(by='support', ascending=False, key=lambda x: x.astype(float))
            table = high_support_rules[['antecedents', 'consequents', 'support']]
            st.markdown("<h3 style='text-align: center;'>Itemsets with High Support</h3>", unsafe_allow_html=True)
            # st.markdown("### Itemsets with High Support")
            st.write(table)

        elif sort_option == "Itemsets Support and Confidence":
            st.markdown("<h3 style='text-align: center;'>Itemsets Support and Confidence</h3>", unsafe_allow_html=True)
            # st.markdown("### Itemsets Support and Confidence")
            table = rules[['antecedents', 'consequents', 'support', 'confidence']]
            st.write(table)

    SupportAndConfidence()

    def Top2Itemsets():
        st.markdown("<h3 style='text-align: center;'>Top 10 2-Itemsets with Highest Support</h3>", unsafe_allow_html=True)
        # st.markdown("### Top 10 2-Itemsets with Highest Support")
        
        itemsets_2 = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x.split(', ')) == 2)]
        itemsets_2.loc[:, 'itemsets'] = itemsets_2['itemsets'].apply(lambda x: ' → '.join(x.split(', ')))

        # Sort by support and get top 10
        top_10_itemsets = itemsets_2.sort_values('support', ascending=False).head(10)

        # Create the horizontal bar chart
        fig, ax = plt.subplots(figsize=(17, 8))
        bars = ax.barh(top_10_itemsets['itemsets'], top_10_itemsets['support'], color='teal')

        # Customize the plot
        ax.set_title('Top 10 2-Itemsets with Highest Support', fontsize=16)
        ax.set_xlabel('Support', fontsize=12)
        ax.set_ylabel('Itemsets', fontsize=12)

        # Adjust y-axis labels
        ax.tick_params(axis='y', labelsize=12)

        # Add value labels to the end of each bar
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                    ha='left', va='center', fontsize=8)

        # Invert y-axis to match the image order
        ax.invert_yaxis()

        # Adjust layout and display
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    Top2Itemsets()

if df is not None:
    st.write("Data loaded successfully!")
    SuperMain()
else:
    st.write("Please upload a CSV file.")
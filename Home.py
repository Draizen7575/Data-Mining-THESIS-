import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



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

st.title("DATA VISUALIZATION AND ANALYSIS")

st.write(
    """
    ## ANALYZING CUSTOMER PREFERENCES OF BICYCLE PARTS FROM SHOPEE USING DATA MINING
    
    Welcome to our study on analyzing customer preferences of bicycle parts from Shopee using data mining techniques. This project aims to uncover insights into customer behavior and preferences in the bicycle parts market on Shopee.
    
    Our analysis employs various data mining methods including K-means clustering, association rule mining, and sentiment analysis to understand customer preferences and trends in the bicycle parts and accessories market.
    
    The dataset used in this study, containing Shopee bicycle parts and accessories reviews, can be found [here](https://huggingface.co/datasets/lllaurenceee/Shopee_Bicycle_Reviews).
    """
)

st.title("EXPLORATORY DATA ANALYSIS")

@st.cache_data
def load_data():
    #return pd.read_csv("hf://datasets/lllaurenceee/Shopee_Bicycle_Reviews/Dataset_D_Duplicate.csv")
   return pd.read_csv("Dataset.csv")
df = load_data()

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
    Top10Sel = st.selectbox(
        "Sort by:",
        ["Most Purchased Items", "Shops by Sales Volume", "Products by Average Rating", "Product Variations"]
    )
    if Top10Sel == "Most Purchased Items":
        st.markdown("<h3 style='text-align: center;'>Top 10 Most Purchased</h3>", unsafe_allow_html=True)
        # st.markdown("### Top 10 Most Purchased Items")
        item_counts = df['purchased_item'].value_counts().head(10)
        plot_bar(pd.DataFrame({'itemsets': item_counts.index, 'support': item_counts.values}), 'support', 'itemsets', 'Top 10 Most Purchased Items', 'Number of Purchases', 'Purchased Item')

    elif Top10Sel == "Shops by Sales Volume":
        st.markdown("<h3 style='text-align: center;'>Top 10 Shops by Sales Volume</h3>", unsafe_allow_html=True)
        # st.markdown("### Top 10 Shops by Sales Volume")
        shop_sales = df['shop'].value_counts().head(10)
        plot_bar(pd.DataFrame({'shop': shop_sales.index, 'sales': shop_sales.values}), 'sales', 'shop', 'Top 10 Shops by Sales Volume', 'Number of Orders', 'Shop')

    elif Top10Sel == "Products by Average Rating":
        st.markdown("<h3 style='text-align: center;'>Top 10 Products by Average Rating</h3>", unsafe_allow_html=True)
        # st.markdown("### Top 10 Products by Average Rating")
        avg_rating = df.groupby('purchased_item')['rating'].mean().sort_values(ascending=False).head(10)
        plot_bar(pd.DataFrame({'product': avg_rating.index, 'rating': avg_rating.values}), 'rating', 'product', 'Top 10 Products by Average Rating', 'Average Rating', 'Product Name')

    elif Top10Sel == "Product Variations":
        st.markdown("<h3 style='text-align: center;'>Top 10 Product Variations</h3>", unsafe_allow_html=True)
        # st.markdown("### Top 10 Product Variations")
        variation_counts = df['variation'].value_counts().head(10)
        plot_bar(pd.DataFrame({'variation': variation_counts.index, 'count': variation_counts.values}), 'count', 'variation', 'Top 10 Product Variations', 'Number of Purchases', 'Variation')

Top10()
@st.fragment
def PriceDist():
    st.markdown("<h3 style='text-align: center;'>Price Distribution of Products</h3>", unsafe_allow_html=True)
    # st.markdown("### Price Distribution of Products")
    price_range = st.slider('Select Price Range', 0, 8000, (0, 8000))
    filtered_df = df[(df['price'] >= price_range[0]) & (df['price'] <= price_range[1])]

    plt.figure(figsize=(8, 6))
    sns.histplot(filtered_df['price'], bins=20, kde=True, color='blue')
    plt.title('Price Distribution of Products', fontsize=14)
    plt.xlabel('Price', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

PriceDist()

st.markdown("<h3 style='text-align: center;'>Monthly Sales Trends</h3>", unsafe_allow_html=True)
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.to_period('M').astype(str)
monthly_sales = df['month'].value_counts().sort_index()
plt.figure(figsize=(12, 6))
sns.lineplot(x=monthly_sales.index, y=monthly_sales.values, marker='o', color='teal')
plt.title('Monthly Sales Trends', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Number of Orders', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(plt.gcf())
plt.close()

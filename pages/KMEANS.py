import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

st.markdown("<h1 style='text-align: center;'>KMeans Clustering Visualization</h1>", unsafe_allow_html=True)
# st.title('KMeans Clustering Visualization')


#@st.cache_data
#def load_data():
#     return pd.read_csv("hf://datasets/lllaurenceee/Shopee_Bicycle_Reviews/Dataset_D_Duplicate.csv")
#    # return pd.read_csv("modified_dataset.csv")
#df = load_data()


@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    else:
        return None

st.markdown("Upload Your Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

df = load_data(uploaded_file)

@st.fragment
def KMeans_SuperMain():
    # Convert DataFrame to a list of lists (excluding the header)
    data = [row.tolist() for index, row in df.iterrows()]

    # Extract headers
    headers = df.columns.tolist()

    # ALL ENCODERS
    shop_encoder = LabelEncoder()
    shop_column = [row[1] for row in data]
    encoded_shop = shop_encoder.fit_transform(shop_column)

    brand_encoder = LabelEncoder()
    brand_column = [row[4] for row in data]
    encoded_brand = brand_encoder.fit_transform(brand_column)

    date_encoder = LabelEncoder()
    date_column = [row[9] for row in data]
    encoded_date = date_encoder.fit_transform(date_column)

    purchased_item_encoder = LabelEncoder()
    purchased_item_column = [row[6] for row in data]
    encoded_purchased_item = purchased_item_encoder.fit_transform(purchased_item_column)

    color_encoder = LabelEncoder()
    color_column = [row[8] for row in data]
    encoded_color = color_encoder.fit_transform(color_column)

    # Update data with encoded values Continuation of Encders
    for idx, row in enumerate(data):
        row[4] = encoded_brand[idx]
        row[9] = encoded_date[idx]
        row[6] = encoded_purchased_item[idx]
        row[1] = encoded_shop[idx]
        row[8] = encoded_color[idx]

    # KMEANS APPLICATION FUNCTION
    def apply_kmeans_one_column(data, column_idx, encoder=None, n_clusters=0):
        X = np.array([[row[column_idx]] for row in data])

        # Scale the data if it's continuous and not in special columns
        if isinstance(X[0][0], (int, float)) and column_idx not in [5, 6]:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_scaled)
        labels = kmeans.labels_

        # Calculate and print the silhouette score
        silhouette_avg = silhouette_score(X_scaled, labels)
        st.write(f"Silhouette Score for column {headers[column_idx]}: {silhouette_avg:.2f}")

        # Count the number of points in each cluster
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique, counts))

        # Create a mapping from cluster label to name (adjusting for +1)
        if encoder:
            cluster_names = [encoder.inverse_transform([int(center)])[0] for center in np.round(kmeans.cluster_centers_).flatten()]
        else:
            cluster_names = [f"Cluster {i + 1}" for i in unique]  # Fallback with 1-based indexing

        # Map cluster labels to names, adjusting indices
        cluster_name_mapping = {i: cluster_names[i] for i in unique}

        # Sort clusters from highest to lowest
        sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)

        # Print the sorted clusters with names
        st.write("Cluster sizes:")
        for cluster, size in sorted_clusters:
            st.write(f"{cluster_name_mapping[cluster]}: {size} points")

        # Visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(X, [i for i in range(len(X))], c=kmeans.labels_, cmap='rainbow')

        # Plot centroids
        centroid_colors = [plt.cm.rainbow(label/n_clusters) for label in range(n_clusters)]
        plt.scatter(kmeans.cluster_centers_, [i for i in range(n_clusters)], s=200, c=centroid_colors, marker='X')

        # Add legend based on encoder
        if encoder:
            original_values = [encoder.inverse_transform([int(center)])[0] for center in np.round(kmeans.cluster_centers_).flatten()]
            legend_handles = [plt.Line2D([0], [0], marker='X', color='w', label=value, markersize=10, markerfacecolor=centroid_colors[i]) for i, value in enumerate(original_values)]
            plt.legend(handles=legend_handles, title=headers[column_idx])
        else:
            legend_handles = [plt.Line2D([0], [0], marker='X', color='w', label=f"Centroid {i+1}", markersize=10, markerfacecolor=centroid_colors[i]) for i in range(n_clusters)]
            plt.legend(handles=legend_handles, title=headers[column_idx])

        if column_idx == 9:  # Date column
            plt.xlabel("Date")
            flattened_X = X.flatten()
            encoded_dates = np.linspace(min(flattened_X), max(flattened_X), n_clusters)
            decoded_dates = [encoder.inverse_transform([int(date)])[0] for date in encoded_dates]
            formatted_dates = [pd.to_datetime(date).strftime('%b-%Y') for date in decoded_dates]
            plt.xticks(ticks=encoded_dates, labels=formatted_dates, rotation=45)
        else:
            plt.xlabel(headers[column_idx])

        plt.ylabel('Data Points Index')
        plt.title(f'KMeans Clustering on {headers[column_idx]}')
        #plt.show()
        st.pyplot(plt.gcf())

    def kmeans_two_columns(data, columns, encoders=None, n_clusters=3):
        col_indices = [headers.index(col) for col in columns]
        
        # Extract columns of interest
        X = np.array([[row[i] for i in col_indices] for row in data])
        
        # If encoders are provided, encode categorical columns
        if encoders:
            for i, col in enumerate(columns):
                if col in encoders:
                    le = encoders[headers.index(col)]
                    X[:, i] = le.transform(X[:, i])
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_

        # Compute silhouette score
        silhouette_avg = silhouette_score(X, labels)
        st.write(f"Silhouette Score for columns {columns[0]} vs {columns[1]}: {silhouette_avg:.2f}")

        # Visualize the clustering result if only 2 columns
        if len(columns) == 2:
            plt.figure(figsize=(10, 10))
            plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', marker='o')
            plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=200, c='black', marker='X')

            # Adjust labels for BRAND vs DATE
            if columns == ["brand", "date"]:
                unique_brands = sorted(list(set(encoders[0].inverse_transform(X[:, 0].astype(int)))))
                dates = sorted(list(set(encoders[1].inverse_transform(X[:, 1].astype(int)))))
                min_date = dates[0]
                max_date = dates[-1]
                mid_date = dates[len(dates)//2]
                plt.xticks(ticks=range(len(unique_brands)), labels=unique_brands)
                plt.yticks(ticks=[0, len(dates)//2, len(dates)-1], labels=[min_date, mid_date, max_date])
            elif columns == ["brand", "price"]:
                brand_price = sorted(list(set(encoders[0].inverse_transform(X[:, 0].astype(int)))))
                plt.xticks(ticks=range(len(brand_price)), labels=brand_price)
            elif columns == ["date", "price"]:
                dates = sorted(list(set(encoders[0].inverse_transform(X[:, 0].astype(int)))))
                min_date = dates[0]
                max_date = dates[-1]
                mid_date = dates[len(dates)//2]
                plt.xticks(ticks=[0, len(dates)//2, len(dates)-1], labels=[min_date, mid_date, max_date])
            elif columns == ["date", "orderid"]:
                dates = sorted(list(set(encoders[0].inverse_transform(X[:, 0].astype(int)))))
                min_date = dates[0]
                max_date = dates[-1]
                mid_date = dates[len(dates)//2]
                plt.xticks(ticks=[0, len(dates)//2, len(dates)-1], labels=[min_date, mid_date, max_date])
            elif columns == ["purchased_item", "price"]:
                p_item = sorted(list(set(encoders[0].inverse_transform(X[:, 0].astype(int)))))
                plt.xticks(ticks=range(len(p_item)), labels=p_item, rotation=90)
            else:
                x1 = sorted(list(set(encoders[0].inverse_transform(X[:, 0].astype(int)))))
                y1 = sorted(list(set(encoders[1].inverse_transform(X[:, 1].astype(int)))))
                plt.xticks(ticks=range(len(x1)), labels=x1, rotation=45)
                plt.yticks(ticks=range(len(y1)), labels=y1)
            
            plt.xlabel(columns[0])
            plt.ylabel(columns[1])
            
            # Create a legend for cluster centers
            legend_handles = [plt.Line2D([0], [0], marker='X', color='w', label=f"Customer: {i+1}", markersize=10, markerfacecolor='black') for i in range(n_clusters)]
            plt.legend(handles=legend_handles, title='Group of Customer')
            
            plt.title(f'KMeans Clustering: {columns[0]} vs {columns[1]}')
            #plt.show()
            st.pyplot(plt.gcf())

    option = st.multiselect(
        "Select columns to cluster:",
        ["SHOP", "PRICE", "BRAND", "DATE", "BRAND VS PRICE", "BRAND VS DATE", "BRAND VS PURCHASED ITEM", "DATE VS PRICE","RATING", "PURCHASED ITEM", "COLOR", "DATE VS ORDER ID", "PURCHASED ITEM VS PRICE"],
        default=["SHOP", "BRAND", "PRICE", "DATE", "BRAND VS PRICE", "BRAND VS DATE", "BRAND VS PURCHASED ITEM", "DATE VS PRICE"],
    )

    with st.expander("KMEANS SINGLE COLUMN", expanded=True):
        # APPLICATION OF SINGLE COLUMN KMEANS CLUSTERING FUNCTION
        if "SHOP" in option:
            st.write(""" ## KMEANS FOR SHOP """)
            apply_kmeans_one_column(data, column_idx=1, encoder=shop_encoder, n_clusters=7)
            st.write("Higher cluster points indicates the popularity of the Shop")
        if "PRICE" in option:
            st.write(""" ## KMEANS FOR PRICE """)
            apply_kmeans_one_column(data, column_idx=5, n_clusters=4)
            st.write("Higher cluster points means costumer loves it")
        if "BRAND" in option:
            st.write(""" ## KMEANS FOR BRAND """)
            apply_kmeans_one_column(data, column_idx=4, encoder=brand_encoder, n_clusters=8)
            st.write("Higher cluster points indicates the popularity of the Brand")
        if "PURCHASED ITEM" in option:
            st.write(""" ## KMEANS FOR PURCHASED ITEM """) #
            apply_kmeans_one_column(data, column_idx=6, encoder=purchased_item_encoder, n_clusters=4) #
        if "COLOR" in option:
            st.write(""" ## KMEANS FOR COLOR """) #
            apply_kmeans_one_column(data, column_idx=8, encoder=color_encoder, n_clusters=3) #
        if "DATE" in option:
            st.write(""" ## KMEANS FOR DATE """)
            apply_kmeans_one_column(data, column_idx=9, encoder=date_encoder, n_clusters=4)
        if "RATING" in option:
            st.write(""" ## KMEANS FOR RATING """) #
            apply_kmeans_one_column(data, column_idx=10, n_clusters=3) #

    # APPLICATION OF TWO COLUMNS KMEANS CLUSTERING FUNCTION
    with st.expander("KMEANS TWO COLUMN", expanded=True):
        if "BRAND VS PRICE" in option:
            st.write(""" ## KMEANS FOR BRAND VS PRICE """)
            kmeans_two_columns(data, ["brand", "price"], encoders=[brand_encoder])
        if "BRAND VS DATE" in option:
            st.write(""" ## KMEANS FOR BRAND VS DATE """)
            kmeans_two_columns(data, ["brand", "date"], encoders=[brand_encoder, date_encoder])
        if "BRAND VS PURCHASED ITEM" in option:
            st.write(""" ## KMEANS FOR BRAND VS PURCHASED ITEM """)
            kmeans_two_columns(data, ["brand", "purchased_item"], encoders=[brand_encoder, purchased_item_encoder])
        if "DATE VS PRICE" in option:
            st.write(""" ## KMEANS FOR DATE VS PRICE """)
            kmeans_two_columns(data, ["date", "price"], encoders=[date_encoder])
        if "DATE VS ORDER ID" in option:
            st.write(""" ## KMEANS FOR DATE VS ORDER ID """) # 
            kmeans_two_columns(data, ["date", "orderid"], encoders=[date_encoder]) # 
        if "PURCHASED ITEM VS PRICE" in option:
            st.write(""" ## KMEANS FOR PURCHASED ITEM VS PRICE """) #
            kmeans_two_columns(data, ["purchased_item", "price"], encoders=[purchased_item_encoder]) #

if df is not None:
    st.write("Data loaded successfully!")
    KMeans_SuperMain()
else:
    st.write("Please upload a CSV file.")
import pandas as pd

# Download the dataset from
# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# Import dataset
print('Importing dataset')
df = pd.read_excel("online_retail_II.xlsx", sheet_name=None)
df_all = df["Year 2009-2010"].append(df["Year 2010-2011"], ignore_index=True)

# Make a copy to avoid modifying the original
retail = df_all.copy()

# Extract date
retail["InvoiceDate"] = pd.to_datetime(
    retail["InvoiceDate"], format="%d-%m-%Y %H:%M"
).dt.date

# Remove transactions without CustomerID and duplicated transactions
print('Removing transactions without CustomerID, then deduplicate')
retail = retail[~retail["Customer ID"].isnull()].drop_duplicates()

print('Split into testing and training dataset')
# Reserve one quarter of transactions from June to August 2011
# to calculate probability of purchase and the expected revenue
df_test = retail[
    retail["InvoiceDate"].between(
        pd.Timestamp("2011-06-01"), pd.Timestamp("2011-08-31")
    )
].copy()

# Take 1 year of transactions from June 2010 to May 2011
# to calculate RFM segments
df_train = retail[
    retail["InvoiceDate"].between(
        pd.Timestamp("2010-06-01"), pd.Timestamp("2011-05-31")
    )
].copy()

# Check the shape of training and testing dataset
print(df_test.shape, df_train.shape)

print('Calculating R, F, M values')
# Calculate revenue per transaction
df_train["MonetaryValue"] = df_train["Quantity"] * df_train["Price"]

# Calculate recency
max_date = max(df_train["InvoiceDate"])
df_train["Recency"] = max_date - df_train["InvoiceDate"]

# Prepare the Recency, Frequency, MonetaryValue columns
df_train_rfm = df_train.groupby(["Customer ID"]).agg(
    {"Recency": min, "Invoice": "count", "MonetaryValue": sum}
)
df_train_rfm.columns = ["Recency", "Frequency", "MonetaryValue"]

# create labels and assign them to quantile membership
print('Calculating R, F, M segments')
r_labels = range(4, 0, -1)
r_groups = pd.qcut(df_train_rfm.Recency, q=4, labels=r_labels)
f_labels = range(1, 5)
f_groups = pd.qcut(df_train_rfm.Frequency, q=4, labels=f_labels)
m_labels = range(1, 5)
m_groups = pd.qcut(df_train_rfm.MonetaryValue, q=4, labels=m_labels)

# make a new column for group labels
df_train_rfm["R"] = r_groups.values
df_train_rfm["F"] = f_groups.values
df_train_rfm["M"] = m_groups.values

print('Calculating expected revenue and purchase probability')
# Number of customers and expected revenue
# in each segment who have made new purchases (i.e., in df_test)
df_test["ExpectedRevenue"] = df_test["Quantity"] * df_test["Price"]
df_temp = df_train_rfm.join(
    df_test.groupby("Customer ID")["ExpectedRevenue"].sum(), how="inner"
)
df_test_rfm = df_temp.groupby(["R", "F", "M"]).agg({"ExpectedRevenue": "count"})
df_test_rfm.columns = ["ReturningCustomerCount"]

# Number of customers in each segment
df_left = df_train_rfm.groupby(["R", "F", "M"]).agg({"R": "count"})
df_left.columns = ["CustomerCount"]

# Calculate expected revenue for each (F, M) segment in the "next" quarter
# We want customers to come back so (R) should be ignored
df_right = df_temp.groupby(["F", "M"])["ExpectedRevenue"].mean()

# Left join (RFM <- FM)
model_data = df_left.join(df_right).reset_index().set_index(["R", "F", "M"])
model_data.columns = ["CustomerCount", "ExpectedRevenue"]

# Calculate the purchase probability
model_data["Probability"] = (
    df_test_rfm["ReturningCustomerCount"] / model_data["CustomerCount"]
)

print('Exporting files')
# Output
model_data.fillna(0, inplace=True)
model_data.to_csv("model_data.csv")
df_temp.to_csv("df_temp.csv")

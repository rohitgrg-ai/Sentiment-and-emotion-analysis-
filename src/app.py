import streamlit as st

st.title("My Invoice App")

# Customer Details
st.header("Customer Details")

customer_name = st.text_input("Customer Name")
customer_email = st.text_input("Customer Email")
customer_phone = st.text_input("Customer Phone")

# Product Details
st.header("Product Details")

product_name = st.text_input("Product Name")
quantity = st.number_input("Quantity", min_value=1, step=1)
price = st.number_input("Price per Item", min_value=0.0, step=0.01)

# Calculate total
total = quantity * price

st.write("Total Amount: ₹", total)

# Generate Invoice
if st.button("Generate Invoice"):

    if customer_name and product_name:
        st.subheader("Invoice")

        st.write("**Customer Name:**", customer_name)
        st.write("**Email:**", customer_email)
        st.write("**Phone:**", customer_phone)

        st.write("---")

        st.write("**Product:**", product_name)
        st.write("**Quantity:**", quantity)
        st.write("**Price per Item:** ₹", price)

        st.write("---")

        st.subheader(f"Total Amount: ₹{total:.2f}")

        st.success("Invoice generated successfully!")

    else:
        st.error("Please enter customer name and product name.")
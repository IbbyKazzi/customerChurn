import streamlit as st
from streamlit_option_menu import option_menu

# Inject custom CSS to control width
st.markdown("""
    <style>
    .option-menu-container {
        max-width: 100%;
        margin: auto;
    }
    </style>
    """, unsafe_allow_html=True)

# Wrap the menu in a container
with st.container():
    selected = option_menu(
        menu_title=None,
        options=["Home", "Analysis", "Cust Service", "Retention"],
        icons=["house", "bar-chart", "person-lines-fill", "shield-check"],
        orientation="horizontal"
    )

if selected == "Home":
    st.sidebar.header("🏠 Home")
    import home_page
    home_page.run()
elif selected == "Analysis":
    st.sidebar.header("📊 Analysis")
    import churn_analysis
    churn_analysis.run()
elif selected == "Cust Service":
    st.sidebar.header("👤 Customer Service")
    import customerService
    customerService.run()
    #run_customerService()
elif selected == "Retention":
    st.sidebar.header("🛡️ Retention")
    import retention
    retention.run()









































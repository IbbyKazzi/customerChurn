import streamlit as st
from streamlit_option_menu import option_menu

selected = option_menu(
        menu_title=None,
        options=["Home", "Analysis", "Service", "Retention", "Model History"],
        icons=["house", "bar-chart", "person-lines-fill", "shield-check", "clock-history"],
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
elif selected == "Service":
    st.sidebar.header("👤 Customer Service")
    import customerService
    customerService.run()
    #run_customerService()
elif selected == "Retention":
    st.sidebar.header("🛡️ Retention")
    import retention
    retention.run()
elif selected == "Model History":
    st.sidebar.header("🧠 Model History")
    from model_history import show_model_history
    show_model_history()

















































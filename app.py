from streamlit_option_menu import option_menu
import streamlit as st

# Set page title and icon
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="https://cdn-icons-png.flaticon.com/512/11264/11264700.png",
    layout="wide"
)

# Inject CSS to constrain content width
st.markdown("""
    <style>
        .centered-container {
            max-width: 900px;
            margin: auto;
            padding-top: 2rem;
        }
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Horizontal menu
selected = option_menu(
    menu_title=None,
    options=["Home", "Analysis", "Service", "Retention", "History"],
    icons=["house", "bar-chart", "person-lines-fill", "shield-check", "clock-history"],
    orientation="horizontal"
)

# Wrap routed content in a centered container
with st.container():
    st.markdown('<div class="centered-container">', unsafe_allow_html=True)

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

    elif selected == "Retention":
        st.sidebar.header("🛡️ Retention")
        import retention
        retention.run()

    elif selected == "History":
        st.sidebar.header("🧠 Model History")
        from model_history import show_model_history
        show_model_history()

    st.markdown('</div>', unsafe_allow_html=True)

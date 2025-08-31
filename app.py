from streamlit_option_menu import option_menu
import streamlit as st

# Set page title and icon
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="https://cdn-icons-png.flaticon.com/512/11264/11264700.png",
    layout="centered"
)

# Inject CSS to constrain content width
st.markdown("""
    <style>
        .centered-container {
            max-width: 1120px;  
            margin: auto;
            padding-top: 2rem;
        }

        div[data-testid="option-menu"] .nav-pills {
            flex-wrap: nowrap !important;
            overflow-x: auto;
            justify-content: center;
        }
        @media (max-width: 768px) {
            div[data-testid="option-menu"] .nav-link {
                font-size: 13px;
                padding: 0.3rem 0.5rem;
            }
        }
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)



# Wrap routed content in a centered container
with st.container():
    st.markdown('<div class="centered-container">', unsafe_allow_html=True)

    # Horizontal menu
    selected = option_menu(
        menu_title=None,
        options=["Home", "Analysis", "Service", "Retention", "Model Pipeline"],
        icons=["house", "bar-chart", "person-lines-fill", "shield-check", "diagram-3"],
        orientation="horizontal"
    )


    if selected == "Home":
        st.sidebar.header("🏠 Home")
        #import home_page
        #home_page.run()
        import executive_dash as ed
        ed.run()

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

    elif selected == "Model Pipeline":
        st.sidebar.header("🧩 Model Pipeline")
        from model_history import run
        run();

    st.markdown('</div>', unsafe_allow_html=True)
























from streamlit_option_menu import option_menu
import streamlit as st
import base64


# Set page title and icon
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="https://cdn-icons-png.flaticon.com/512/11264/11264700.png",
    layout="centered"
)

st.markdown("""
    <style>
        .help-icon {
            position: fixed;
            top: 10px;
            left: 10px;
            z-index: 1000;
            font-size: 24px;
            cursor: pointer;
        }
        .help-icon:hover {
            color: #1f77b4;
        }
    </style>
    <a href="https://your-docs-url.com" target="_blank">
        <div class="help-icon">❓</div>
    </a>
""", unsafe_allow_html=True)

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
        @media (max-width: 800px) {
            div[data-testid="option-menu"] .nav-link {
                font-size: 12px;
                padding: 0.3rem 0.5rem;
            }
        }
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)



# Set the logo
def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = get_base64_image("assets/logo2.png")
st.sidebar.markdown(
    f"<img src='data:image/png;base64,{logo_base64}' width='120'>",
    unsafe_allow_html=True
)

# Wrap routed content in a centered container
with st.container():
    st.markdown('<div class="centered-container">', unsafe_allow_html=True)

    # Horizontal menu
    selected = option_menu(
        menu_title=None,
        options=["Home", "Analysis", "Service", "Retention", "Pipeline"],
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

    elif selected == "Pipeline":
        st.sidebar.header("🧩 Model Pipeline")
        from model_history import run
        run();

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")  # Horizontal line
st.markdown(
    "<div style='text-align: center; font-size: 0.9em;'>"
    "© 2025 — Developed by EdgeTel UTS Group 1 <br>"
    "Members: Murray Atkins, Ibrahim Kassem, Bradley Moore, Preeti Sowrab <br>"
    "<a href='https://github.com/IbbyKazzi/customerChurn' target='_blank'>GitHub</a> | "  
    "<a href='mailto:info@EdgeTel.com.au'>Contact</a>"
    "</div>",
    unsafe_allow_html=True
)












































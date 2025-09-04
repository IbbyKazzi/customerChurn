from streamlit_option_menu import option_menu
import streamlit as st
import base64

# Set page title and icon
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="https://cdn-icons-png.flaticon.com/512/11264/11264700.png",
    layout="centered"
)

# Inject CSS to constrain content width
st.markdown("""
    <style>
        /* Lock the option menu at the top */
        div[data-testid="option_menu"] {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            z-index: 999;
            background-color: white;
            padding: 0.5rem 1rem;
            border-bottom: 1px solid #eee;
        }

        /* Push content down to avoid overlap */
        .block-container {
            padding-top: 5rem;
            max-width: 950px;
            margin: auto;
        }
    </style>
""", unsafe_allow_html=True)

# Horizontal menu
selected = option_menu(
    menu_title=None,
    options=["Home", "Analysis", "Service", "Retention", "Pipeline", "How To"],
    icons=["house", "bar-chart", "person-lines-fill", "shield-check", "diagram-3", "question-circle"],
    orientation="horizontal"
)

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
#with st.container():
st.markdown('<div class="fixed-menu">', unsafe_allow_html=True)




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
























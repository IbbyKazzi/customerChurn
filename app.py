from streamlit_option_menu import option_menu
import streamlit as st
import base64

# Set page title and icon
st.set_page_config(
    page_title="EdgeTel by UTS Dashboard",
    page_icon="https://cdn-icons-png.flaticon.com/512/11264/11264700.png",
    layout="centered"
)
st.markdown("""
    <style>
        div[data-testid="option-menu"] {
            position: fixed;
            top: 0;
            z-index: 999;
            background-color: white;
            padding: 0.5rem 1rem;
            border-bottom: 1px solid #eee;
        }
        .block-container {
            padding-top: 3rem;
            max-width: 1000px;
            margin: auto;
        }
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

col1, col2, col3 = st.columns([0.1, 9.8, 0.1])  # Wider center column
with col2:
    selected = option_menu(
        menu_title=None,
        options=["Executive Dashboard", "Sales Dashboard", "Customer Service Dashboard", "Retention Tactics", "Technical information", "How to guide"],
        icons=["speedometer","bar-chart","person-lines-fill","shield-check", "diagram-3", "question-circle"],
        orientation="horizontal"
    )    

   if selected == "Executive Dashboard":
    st.sidebar.header("🧑‍💼 Executive Dashboard")
        #import home_page
        #home_page.run()
        import executive_dash as ed
        ed.run()       
    
    elif selected == "Sales Dashboard":
        st.sidebar.header("📊 Sales Dashboard")
        import churn_analysis
        churn_analysis.run()
    
    elif selected == "Customer Service Dashboard":
        st.sidebar.header("👤 Customer Service Dashboard")
        import customerService
        customerService.run()
        st.markdown("""
            <br><br><br><br><br><br><br><br><br><br>
            <br><br><br><br><br><br><br><br><br><br>
        """, unsafe_allow_html=True)
    
    elif selected == "Retention Tactics":
        st.sidebar.header("🛡️ Retention Tactics")
        import retention
        retention.run()
    
    elif selected == "Technical Dashboard":
        st.sidebar.header("🧩 Technical Dashboard")
        from model_history import run
        run();

    elif selected == "How to guide":
        st.sidebar.header("❓ How to guide")
        from howTo import run
        run();
    
    
    st.markdown("---")  # Horizontal line
    st.markdown(
        "<div style='text-align: center; font-size: 0.9em;'>"
        "© 2025 — Developed by EdgeTel UTS Group 1 <br>"
        "Members: Murray Atkin, Ibrahim Kassem, Bradley Moore, Preeti Sowrab <br>"
        "<a href='https://github.com/IbbyKazzi/customerChurn' target='_blank'>GitHub</a> | "  
        "<a href='mailto:info@EdgeTel.com.au'>Contact</a>"
        "</div>",
        unsafe_allow_html=True
    )
















































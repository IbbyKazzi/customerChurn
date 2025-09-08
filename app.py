from streamlit_option_menu import option_menu
import streamlit as st
import base64

# Set page title and icon
st.set_page_config(
    page_title="EdgeTel by UTS Dashboard",
    page_icon="https://cdn-icons-png.flaticon.com/512/11264/11264700.png",
    layout="wide"
)
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Comic+Neue:wght@400;700&display=swap');
        html, body, [class*="css"]  {
            font-family: 'Comic Sans MS', 'Comic Neue', cursive, sans-serif !important;
            background-color: #d6eaf8 !important;
        }
        .navbar {
            font-family: 'Comic Sans MS', 'Comic Neue', cursive, sans-serif;
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            background: #b3d8fd;
            border-bottom: 1.5px solid #7ab8f5;
            box-shadow: 0 2px 8px rgba(0,0,0,0.03);
            z-index: 1000;
            display: flex;
            justify-content: center;
            
            margin: 0;
            padding: 0;
            border-radius: 0 !important; /* Ensures no rounded corners */

        }
        .navbar ul {
            display: flex;
            gap: 2.5rem;
            margin: 0;
            padding: 0.75rem 0;
            list-style: none;
            
            margin: 0;
            padding: 0.75rem 0;
            width: 100%;
            justify-content: center;

        }
        .navbar li {
            font-size: 1.08rem;
            font-weight: 600;
            color: #22577a;
            letter-spacing: 0.01em;
            transition: color 0.2s, border-bottom 0.2s;
            border-bottom: 2.5px solid transparent;
            cursor: pointer;

            margin: 0;
            padding: 0;

        }
        .navbar li.selected {
            color: #1565c0;
            border-bottom: 2.5px solid #1565c0;
            background: none;
        }
        .navbar li:hover {
            color: #1565c0;
        }
        .block-container {
            padding-top: 4.5rem;
            max-width: 1500px;
            margin: auto;
            font-family: 'Comic Sans MS', 'Comic Neue', cursive, sans-serif;
        }
        /* Option menu customizations */
        .st-emotion-cache-1n76uvr, .st-emotion-cache-1n76uvr * {
            font-family: 'Comic Sans MS', 'Comic Neue', cursive, sans-serif !important;
        }
        div[data-testid="stHorizontalBlock"] {
            background: #b3d8fd !important;
            border-radius: 12px;
            box-shadow: 0 2px 12px rgba(21,101,192,0.08);
            padding: 0.5rem 0.5rem 0.5rem 0.5rem;
        }
        .st-emotion-cache-1n76uvr button, .st-emotion-cache-1n76uvr button span {
            color: #22577a !important;
            background: #e6f2ff !important;
            border-radius: 8px 8px 0 0 !important;
            border-bottom: 3px solid transparent !important;
            transition: color 0.18s, border-bottom 0.18s, background 0.18s;
        }
        .st-emotion-cache-1n76uvr button[aria-selected="true"], .st-emotion-cache-1n76uvr button[aria-selected="true"] span {
            color: #1565c0 !important;
            background: #e3f0fb !important;
            border-bottom: 3px solid #1565c0 !important;
        }
        .st-emotion-cache-1n76uvr button:hover, .st-emotion-cache-1n76uvr button:hover span {
            color: #1565c0 !important;
            background: #d0e7fa !important;
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

col1, col2, col3 = st.columns([0.1, 50, 0.1])  # Wider center column
with col2:
    
    selected = option_menu(
        menu_title=None,
        options=["Executive Pulse", "Revenue Intelligence", "CX Insights", "Retention Strategies", "Technical Information", "Navigation Guide"],
        icons=["speedometer","bar-chart","person-lines-fill","shield-check", "diagram-3", "question-circle"],
        orientation="horizontal",
       styles={
            "container": {"background-color": "#ADD8E6", "padding": "5px"}, 
            "icon": {"color": "black", "font-size": "20px"},  
            "nav-link": {
                "background-color": "#ADD8E6", 
                "color": "black",  
                "font-size": "16px",  
                "padding": "10px",  
                "border-radius": "7px",  
                
            },
            "nav-link-selected": {
                "background-color": "#1A2A3A",  
                "color": "white",  
            },
       },     
    )

    
    if selected == "Executive Pulse":
        st.sidebar.header("🧑‍💼 Executive Pulse")
        st.sidebar.subheader("High-level summary of churn metrics and business impact")
        st.sidebar.markdown("This tab provides a consolidated view of month-on-month customer churn trends, key performance indicators, and strategic signals. Designed for C-suite executives, it highlights business health, customer retention rates, and financial implications at a glance.")
        #import home_page
        #home_page.run()
        import executive_dash as ed
        ed.run()       
    
    elif selected == "Revenue Intelligence":
        st.sidebar.header("📊 Revenue Intelligence")
        st.sidebar.subheader("Sales-driven churn insights and performance breakdown")
        st.sidebar.markdown("Dive into churn patterns segmented by sales channels, regions, and customer segments. This tab helps Sales leaders identify at-risk accounts, understand conversion bottlenecks, and align sales strategy with retention goals.")
        import churn_analysis
        churn_analysis.run()
    
    elif selected == "CX Insights":
        st.sidebar.header("👤 CX Insights")
        st.sidebar.subheader("Customer experience and service quality analysis")
        st.sidebar.markdown("Explore how customer service interactions, support ticket trends, and satisfaction scores correlate with churn. This tab equips Customer Experience teams with actionable insights to enhance service quality and improve customer loyalty.")
        import customerService
        customerService.run()
       
    
    elif selected == "Retention Strategies":
        st.sidebar.header("🛡️ Retention Strategies")
        st.sidebar.subheader("Proactive strategies to reduce churn and boost loyalty")
        st.sidebar.markdown("Focus on retention initiatives by analyzing the effectiveness of loyalty programs, promotional offers, and customer engagement tactics. This tab supports Marketing and Retention teams in crafting targeted campaigns to retain high-value customers.")
        import retention
        retention.run()
    
    elif selected == "Technical Information":
        st.sidebar.header("🧩 Technical Dashboard")
        st.sidebar.subheader("Model performance and technical documentation")
        st.sidebar.markdown("Access detailed information on the churn prediction model, including performance metrics, feature importance, and version history. This tab is tailored for Data Science and IT teams to monitor model health and ensure alignment with business objectives.")
        from model_history import run
        run();

    elif selected == "Navigation Guide":
        st.sidebar.header("❓ Navigation Guide")
        st.sidebar.subheader("How to use the dashboard effectively")
        st.sidebar.markdown("This section provides a comprehensive guide on navigating the dashboard, understanding each tab's purpose, and leveraging the insights provided. Ideal for new users and stakeholders unfamiliar with the platform.")  
        from howTo import run
        run();
    
    
    st.markdown(
    """
    <style>
    .divider {
        border-top: 10px solid #1A2A3A;  
        margin: 20px 0;  
        width: 100%; 
        

    }
    </style>
    <div class="divider"></div>
    """,
    unsafe_allow_html=True
  )
    
    st.markdown(
        "<div style='text-align: center; font-size: 0.9em;'>"
        "© 2025 — Developed by EdgeTel UTS Group 1 <br>"
        "Members: Murray Atkin, Ibrahim Kassem, Bradley Moore, Preeti Sowrab <br>"
        "<a href='https://github.com/IbbyKazzi/customerChurn' target='_blank'>GitHub</a> | "  
        "<a href='mailto:info@EdgeTel.com.au'>Contact</a>"
        "</div>",
        unsafe_allow_html=True
    )

























































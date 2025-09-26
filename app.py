from streamlit_option_menu import option_menu
import streamlit as st
import base64

# Set page title and icon
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #f4f9fd !important;
    color: #2b3e51;
    padding: 0;
    margin: 0;
}

.navbar {
    position: fixed;
    top: 1rem;
    left: 50%;
    transform: translateX(-50%);
    width: calc(100% - 2rem);
    max-width: 1200px;
    background: #ffffff;
    border: 1px solid #dce5ee;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
    border-radius: 14px;
    z-index: 1000;
    display: flex;
    justify-content: center;
    padding: 0.5rem 1.5rem;
    transition: all 0.3s ease-in-out;
}

.navbar ul {
    display: flex;
    gap: 2rem;
    margin: 0;
    padding: 0;
    list-style: none;
    align-items: center;
}

.nav-item .icon {
    color: black;
}

.nav-item.selected .icon {
    color: white;
}

.navbar li {
    font-size: 0.95rem;
    font-weight: 600;
    color: #3a4b5c;
    padding: 0.4rem 1rem;
    border-radius: 8px;
    transition: all 0.2s ease;
    cursor: pointer;
    user-select: none;
}

.navbar li:hover {
    background: #f0f6fc;
    color: #1565c0;
}

.navbar li.selected {
    background: #e3f0fb;
    color: #1565c0;
    box-shadow: inset 0 -2px 0 #1565c0;
}

.block-container {
    padding-top: 7rem;
    max-width: 1200px;
    margin: auto;
    background: #ffffff;
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
    padding: 2rem;
}

div[data-testid="stHorizontalBlock"] {
    background: #ffffff !important;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
    padding: 1rem;
}

.st-emotion-cache-1n76uvr,
.st-emotion-cache-1n76uvr * {
    font-family: 'Inter', sans-serif !important;
}

.st-emotion-cache-1n76uvr button,
.st-emotion-cache-1n76uvr button span {
    font-size: 0.9rem;
    font-weight: 600;
    color: #22577a !important;
    background: #f4f9fd !important;
    border-radius: 8px !important;
    border: 1px solid transparent !important;
    transition: all 0.2s ease;
}

.st-emotion-cache-1n76uvr button[aria-selected="true"],
.st-emotion-cache-1n76uvr button[aria-selected="true"] span {
    color: #1565c0 !important;
    background: #e3f0fb !important;
    border: 1px solid #1565c0 !important;
}

.st-emotion-cache-1n76uvr button:hover,
.st-emotion-cache-1n76uvr button:hover span {
    color: #1565c0 !important;
    background: #d0e7fa !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #f4f9fd;
}

::-webkit-scrollbar-thumb {
    background-color: #c2d4e5;
    border-radius: 10px;
    border: 2px solid #f4f9fd;
}

/* Sidebar layout */
section[data-testid="stSidebar"] {
    /*min-width: 400px !important;*/
    max-width: 400px !important;
    height: 100vh !important;
    overflow-y: auto;
    padding-top: 2rem;
    background-color: #ffffff;
    box-shadow: 2px 0 6px rgba(0,0,0,0.05);
}
            
@media screen and (max-width: 768px) {
    .block-container {
        padding: 1rem !important;
    }
    
    section[data-testid="stSidebar"] {
        display: none !important;
        max-width: 100% !important;
        min-width: 100% !important;
        height: auto !important;
        padding: 1rem !important;
        box-shadow: none !important;
    }

    .navbar ul {
        flex-direction: column;
        gap: 1rem;
    }

    .navbar {
        flex-direction: column;
        align-items: center;
        padding: 1rem;
    }

    .st-emotion-cache-1n76uvr button {
        font-size: 14px !important;
        padding: 8px 12px !important;
    }
}


</style>
""", unsafe_allow_html=True)


# Set the logo
def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = get_base64_image("assets/logo2.png")


st.sidebar.markdown(
    f"""
    <div style='text-align: left;'>
        <img src='data:image/png;base64,{logo_base64}' width='120'><br>
        <h3 style='margin-top: 5px; font-size: 18px;'>Churn Reduction App</h3>
        <p style='font-size:14px; color:gray;'>Predict. Prevent. Retain.</p>
    </div>
    """,
    unsafe_allow_html=True
)



col2 = st.container()
with col2:
    
    selected = option_menu(
    menu_title=None,
    options=[
        "Executive Pulse", "Revenue Intelligence", "Customer Insights",
        "Retention Strategies", "Technical Information", "Navigation Guide"
    ],
    icons=[
        "speedometer", "bar-chart", "person-lines-fill",
        "shield-check", "diagram-3", "question-circle"
    ],
    orientation="horizontal",
    styles={
        "container": {
            "background-color": "#ADD8E6",
            "padding": "10px",
            "margin": "0px",
            "width": "100vw",
            "border-radius": "0px",
        },
        "icon": {
            "color": "blue",
            "font-size": "20px"
        },
        "nav-link": {
            "background-color": "#ADD8E6",
            "color": "black",
            "font-size": "16px",
            "padding": "10px 16px",
            "border-radius": "0px",  # Avoid inner rounding too
            "border": "2px solid transparent"
        },
        "nav-link-selected": {
            "background-color": "#1A2A3A",
            "color": "white",
            "border": "2px solid #1A2A3A",
            "padding": "10px 16px",
            "border-radius": "0px"
        }
    }
)

    
    if selected == "Executive Pulse":
        st.sidebar.header("🧑‍💼 Executive Pulse")
        st.sidebar.subheader("High-level summary of churn metrics and business impact")
        st.markdown("""
        <style>
        .justified-text {
            text-align: justify;
        }
        </style>
        """, unsafe_allow_html=True)
        st.sidebar.markdown("This tab provides a consolidated view of month-on-month customer churn trends, key performance indicators, and strategic signals. Designed for C-suite executives, it highlights business health, customer retention rates, and financial implications at a glance.")
        #import home_page
        #home_page.run()
        import executive_dash as ed
        ed.run()       
    
    elif selected == "Revenue Intelligence":
        st.sidebar.header("📊 Revenue Intelligence")
        st.sidebar.subheader("Sales-driven churn insights and performance breakdown")
        st.markdown("""
        <style>
        .justified-text {
            text-align: justify;
        }
        </style>
        """, unsafe_allow_html=True)
        st.sidebar.markdown("Dive into churn patterns segmented by sales channels, regions, and customer segments. This tab helps Sales leaders identify at-risk accounts, understand conversion bottlenecks, and align sales strategy with retention goals.")
        import churn_analysis
        churn_analysis.run()
    
    elif selected == "Customer Insights":
        st.sidebar.header("👤 Customer Insights")
        st.sidebar.subheader("Customer experience and service quality analysis")
        st.markdown("""
        <style>
        .justified-text {
            text-align: justify;
        }
        </style>
        """, unsafe_allow_html=True)
        st.sidebar.markdown("Explore how customer service interactions, support ticket trends, and satisfaction scores correlate with churn. This tab equips Customer Experience teams with actionable insights to enhance service quality and improve customer loyalty.")
        import customerService
        customerService.run()
       
    
    elif selected == "Retention Strategies":
        st.sidebar.header("🛡️ Retention Strategies")
        st.sidebar.subheader("Proactive strategies to reduce churn and boost loyalty")
        st.markdown("""
        <style>
        .justified-text {
            text-align: justify;
        }
        </style>
        """, unsafe_allow_html=True)
        st.sidebar.markdown("Focus on retention initiatives by analyzing the effectiveness of loyalty programs, promotional offers, and customer engagement tactics. This tab supports Marketing and Retention teams in crafting targeted campaigns to retain high-value customers.")
        import retention
        retention.run()
    
    elif selected == "Technical Information":
        st.sidebar.header("🧩 Technical Dashboard")
        st.sidebar.subheader("Model performance and technical documentation")
        st.markdown("""
        <style>
        .justified-text {
            text-align: justify;
        }
        </style>
        """, unsafe_allow_html=True)
        st.sidebar.markdown("Access detailed information on the churn prediction model, including performance metrics, feature importance, and version history. This tab is tailored for Data Science and IT teams to monitor model health and ensure alignment with business objectives.")
        from model_history import run
        run();

    elif selected == "Navigation Guide":
        st.sidebar.header("❓ Navigation Guide")
        st.sidebar.subheader("How to use the dashboard effectively")
        st.markdown("""
        <style>
        .justified-text {
            text-align: justify;
        }
        </style>
        """, unsafe_allow_html=True)
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









































































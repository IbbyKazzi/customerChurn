import streamlit as st

def run():
    st.markdown("<h3 style='text-align: center;'> EdgeTel by UTS Dashboard - Navigation Guide</h3>", unsafe_allow_html=True)

    st.markdown("""<h4> Welcome to the <strong>EdgeTel Dashboard</strong> — your all-in-one hub for understanding customer churn, revenue dynamics, customer experience (CX), and retention strategies. Whether you're part of executive leadership, marketing, customer service, or data science, this guide will help you make the most of the platform.
</h4> <hr>""", unsafe_allow_html=True)

    st.markdown("""<h3>📂 Overview of Dashboard tabs</h3><hr>""", unsafe_allow_html=True)


    st.markdown("""
    | **Tab Name**             | **Purpose**                                                                |
    |--------------------------|----------------------------------------------------------------------------|
    | **Executive Pulse**      | High-level summary of churn metrics and business health                    |
    | **Revenue Intelligence** | Deep dive into revenue, channels, trends, and financial impact             |
    | **CX Insights**          | Analysis of customer service, satisfaction, and churn                      |
    | **Retention Strategies** | Interventions to reduce churn and improve loyalty                          |
    | **Technical Information**| Model performance, feature importance, version control                     |
    | **Navigation Guide**     | This guide — how to navigate and use the dashboard                         |
    ---
    """, unsafe_allow_html=True)

   
    st.markdown("### 1️⃣ Executive Pulse  \n📊 *For Executives / Business Leaders*")
    st.markdown("""
- **Churn Rate Trends**: Monthly trends in churn  
- **KPIs**: Total customers, retained/lost users, MRR  
- **Business Impact**: Understand implications of churn  
- **Strategic Signals**: Detect churn accelerators  

**How to use:**  
• Get an overview for leadership reports  
• Spot red flags and explore deeper via other tabs  
    """)

    st.markdown("### 2️⃣ Revenue Intelligence  \n💡 *For Sales, Growth, and Finance Teams*")
    st.markdown("""
- **Revenue by Channel/Segment**: Identify lucrative or loss-heavy segments  
- **Revenue vs Churn**: Analyze income vs loss  
- **Forecasts**: View projected revenue trends  

**How to use:**  
• Refine pricing or bundling strategies  
• Focus retention on highest-value channels  
    """)

    st.markdown("### 3️⃣ CX (Customer Experience) Insights  \n🧑 *For CX, Product, and Support Teams*")
    st.markdown("""
- **Support Tickets**: Complaint trends  
- **Satisfaction Scores (CSAT/NPS)**  
- **Response Time**: Resolution efficiency  
- **Feedback Sentiment**: Identify churn signals from free text  

**How to use:**  
• Find root causes of dissatisfaction  
• Align CX efforts to reduce churn  
    """)

    st.markdown("### 4️⃣ Retention Strategies  \n🛡️ *For Marketing & Retention Teams*")
    st.markdown("""
- **Programs & Offers**: Loyalty and promotion performance  
- **Customer Segments at Risk**  
- **Churn Predictors**: Usage drops, feedback issues  
- **Suggested Actions**: Targeted offers, UX fixes  

**How to use:**  
• Focus on high-risk, high-value customers  
• Coordinate cross-functional actions  
    """)

    st.markdown("### 5️⃣ Technical Information  \n🔧 *For Data Science & IT Teams*")
    st.markdown("""
- **Model Performance**: Accuracy, recall, AUC  
- **Feature Importance**  
- **Version History & Retraining Info**  
- **Data Assumptions & Notes**  

**How to use:**  
• Monitor model health & drift  
• Share important features with business users  
    """)

    st.markdown("### 6️⃣ Navigation Guide  \n *You’re Here!*")
    st.markdown("""
Use this tab as a reference whenever you're unsure how to interpret visuals or use specific features.
    """)

    st.markdown("---")
    st.markdown("""<h3>🛠 Common navigation tips</h3><hr>""", unsafe_allow_html=True)

    st.markdown("""
- **Sidebar**: Shows helpful info based on the selected tab  
- **Top Menu**: Use it to switch between tabs quickly  
- **Tooltips & Descriptions**: Hover for extra explanations  
- **Filters**: Available in some tabs to zoom into date ranges or customer segments  
    """)
    st.markdown("""<h3>✅ Best practices</h3><hr>""", unsafe_allow_html=True)
    

    st.markdown("""
1. Start with **Executive Pulse**  
2. Use **Revenue Intelligence** and **CX Insights** to diagnose causes  
3. Check **Technical Information** to validate model reliability  
4. Plan campaigns in **Retention Strategies**  
5. Return to **Navigation Guide** for clarification anytime  
    """)
    
    st.markdown("""<h3>📘 Glossary of key terms</h3><hr>""", unsafe_allow_html=True)


    st.markdown("""
| **Term**      | **Meaning**                                                                 |
|---------------|------------------------------------------------------------------------------|
| **Churn**     | % of customers who leave within a time period                               |
| **Retention** | % of customers retained over time                                           |
| **KPI**       | Key Performance Indicator — metric used to evaluate success                 |
| **LTV**       | Lifetime Value — estimated total value a customer brings                    |
| **CSAT**      | Customer Satisfaction Score                                                 |
| **NPS**       | Net Promoter Score — measures customer loyalty                              |
| **Feature Importance** | Variables that most impact model predictions                       |
| **Model Drift**| When model performance degrades over time due to changing data patterns    |
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.info("Need help? Contact us at info@EdgeTel.com.au or visit the [GitHub Repo](https://github.com/IbbyKazzi/customerChurn)", icon="💬")

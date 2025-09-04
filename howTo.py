import streamlit as st
def run():        
        st.subheader("📘 How to Use This Dashboard")
        st.markdown("""
                **Home Tab**\n
                ***Choose Your Segmentation Level:*** Use the slider to select how many customer segments (clusters) you'd like to generate.
                More clusters = more detailed insights.\n
                ***Explore Cluster Distribution:*** View how customers are grouped based on churn risk, tenure, billing, and contract type.
                This helps you identify patterns and target specific segments.\n
                ***Review Cluster Summaries:*** Each cluster shows average churn rate, tenure, and charges. Use this to understand
                customer behavior and prioritize retention strategies.\n 
                ***Generate Segment Descriptions:*** Click the “Generate GPT Segment Descriptions” button to get AI-powered summaries
                of each cluster - perfect for presentations or strategy planning.\n
                ***Tip:*** Hover over ℹ️ icons for extra guidance
        """)

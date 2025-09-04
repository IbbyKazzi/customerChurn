import streamlit as st
def run():        
        st.subheader("📘 How to Use This Dashboard")
        st.markdown("""
            <p><strong>Home Tab</strong></p>
            <p><em><strong>Choose Your Segmentation Level:</em></strong> Use the slider to select how many customer segments (clusters) you'd like to generate.
            More clusters = more detailed insights.</p>
            <p><em>Explore Cluster Distribution:</em> View how customers are grouped based on churn risk, tenure, billing, and contract type.
            This helps you identify patterns and target specific segments.</p>
            <p><em>Review Cluster Summaries:</em> Each cluster shows average churn rate, tenure, and charges. Use this to understand
            customer behavior and prioritize retention strategies.</p>
            <p><em>Generate Segment Descriptions:</em> Click the “Generate GPT Segment Descriptions” button to get AI-powered summaries
            of each cluster — perfect for presentations or strategy planning.</p>
        
            <br><br><br><br><br><br><br><br><br><br>
        
            <p><em>Tip:</em> Hover over ℹ️ icons for extra guidance</p>
            <br><br><br><br><br><br><br><br><br><br>
        """, unsafe_allow_html=True)

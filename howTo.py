import streamlit as st
def run():        
        st.subheader("📘 How to Use This Dashboard")
        st.markdown("""
            <h4><strong>Home Tab</strong></h4>
            <p><em><strong>Choose Your Segmentation Level:</em></strong> Use the slider to select how many customer segments (clusters) you'd like to generate.
            More clusters = more detailed insights.</p>
            <p><em><strong>Explore Cluster Distribution:</em></strong> View how customers are grouped based on churn risk, tenure, billing, and contract type.
            This helps you identify patterns and target specific segments.</p>
            <p><em><strong>Review Cluster Summaries:</em></strong> Each cluster shows average churn rate, tenure, and charges. Use this to understand
            customer behavior and prioritize retention strategies.</p>
            <p><em><strong>Generate Segment Descriptions:</em></strong> Click the “Generate GPT Segment Descriptions” button to get AI-powered summaries
            of each cluster — perfect for presentations or strategy planning.</p>
        
            <br><br><br><br><br><br><br><br><br><br>
        
            <p><em><strong>Tip:</em></strong> Hover over ℹ️ icons for extra guidance</p>
            <br><br><br><br><br><br><br><br><br><br>
        """, unsafe_allow_html=True)

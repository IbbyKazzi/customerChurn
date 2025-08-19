# 1) Build customer clusters
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
from settings import MODEL_PATH_T3, MODEL_PATH_T21, DATA_PATH

def run():

    features_for_clustering = [
        'Month', 'MonthlyCharges', 'TotalCharges',
        'Contract', 'InternetService', 'TechSupport',
        'PaymentMethod'
    ]
    
    df = pd.read_csv(DATA_PATH)
    
    X_cluster = df[features_for_clustering]
    X_scaled  = StandardScaler().fit_transform(X_cluster)
    
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # 2) Create summary per cluster
    cluster_summary = df.groupby('cluster').agg({
        'Churn':'mean',
        'Month':'mean',
        'MonthlyCharges':'mean',
        'TotalCharges':'mean',
        'Contract':'mean',
        'InternetService':'mean',
        'TechSupport':'mean',
        'PaymentMethod':'mean'
    }).reset_index()
    
    # 3) Describe each cluster using GPT
    import openai
    
    def llm_cluster_description(row):
        prompt = (
            "You are an expert in telecommunications customer analytics. "
            "Based on the following cluster characteristics from a telco churn dataset, "
            "assign one short business-friendly segment name (2–4 words) and a brief description. "
            "Here are the characteristics:\n"
            f"- Average churn rate: {row['Churn']:.2f}\n"
            f"- Average tenure: {row['Month']:.1f} months\n"
            f"- Average monthly charges: {row['MonthlyCharges']:.2f}\n"
            #f"- % Month-to-month contract: {row['Contract_Month-to-month']*100:.1f}%\n"
            f"- % using Fiber: {row['InternetService']*100:.1f}%\n"
            f"- % without Tech Support: {row['TechSupport']*100:.1f}%\n"
            f"- % using Electronic check: {row['PaymentMethod']*100:.1f}%\n\n"
            "Use one of the following label styles if it applies: "
            "'New Cost-Sensitive Users', 'Tech-Avoidant Casuals', 'Bundled Value Seekers', "
            "'Contract Expiry Risks', 'Electronic-Check Churners', 'Loyal Seniors', 'Streaming-Heavy Customers'. "
            "If none of these apply, generate a similar style. "
            "Return output in the format: <Segment Name>: <Description>"
        )
        
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role":"user","content":prompt}]
        )
        return response.choices[0].message.content.strip()
    
    # Apply LLM to each cluster
    cluster_summary['Segment_Profile'] = cluster_summary.apply(llm_cluster_description, axis=1)
    
    cluster_summary

from fpdf import FPDF
import pandas as pd
import numpy as np

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'E1 Analytics | Intelligent Data Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255) # Light Blue
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 7, body)
        self.ln()

def generate_business_insight(col_name, stats):
    """
    The 'Brain' that turns numbers into English sentences.
    """
    mean = stats['mean']
    median = stats['median']
    std = stats['std']
    
    narrative = []
    
    # 1. Analyze Central Tendency (The "Skew")
    diff = ((mean - median) / mean) * 100
    if abs(diff) < 5:
        narrative.append(f"The data for {col_name} is very balanced. The average ({mean:.2f}) is close to the typical value ({median:.2f}), indicating a normal distribution without major outliers.")
    elif mean > median:
        narrative.append(f"We detected a 'Positive Skew' in {col_name}. The average ({mean:.2f}) is significantly higher than the median ({median:.2f}). This usually means a few high-value outliers (whales) are pulling the numbers up.")
    else:
        narrative.append(f"We detected a 'Negative Skew' in {col_name}. The average ({mean:.2f}) is lower than the median ({median:.2f}), suggesting a cluster of low-value performance dragging the metric down.")

    # 2. Analyze Volatility (Stability)
    cv = (std / mean) * 100 if mean != 0 else 0
    if cv < 10:
        narrative.append(f"Stability Alert: {col_name} is highly consistent (Volatility: Low). Predictability is high.")
    elif cv > 50:
        narrative.append(f"Volatility Warning: {col_name} fluctuates wildly (Volatility: High). Expect significant swings in performance.")
    else:
        narrative.append(f"{col_name} shows moderate fluctuation. It is relatively stable but requires monitoring.")

    return " ".join(narrative)

def generate_correlation_insight(df, numeric_cols):
    """
    Finds relationships (e.g., Marketing Spend vs Revenue).
    """
    if len(numeric_cols) < 2:
        return "Not enough numeric data to detect relationships."
        
    corr_matrix = df[numeric_cols].corr()
    insights = []
    
    # Iterate through unique pairs
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            col1 = numeric_cols[i]
            col2 = numeric_cols[j]
            val = corr_matrix.iloc[i, j]
            
            if val > 0.7:
                insights.append(f"• Strong Positive Driver: As '{col1}' increases, '{col2}' almost always increases (Correlation: {val:.2f}). Focus on '{col1}' to drive growth.")
            elif val < -0.7:
                insights.append(f"• Negative Inverse: Higher '{col1}' leads to lower '{col2}' (Correlation: {val:.2f}).")
    
    if not insights:
        return "No strong statistical relationships were detected between the metrics."
    
    return "\n".join(insights)
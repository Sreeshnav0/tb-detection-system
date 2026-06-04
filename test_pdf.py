import hashlib
import os
import json
original_md5 = hashlib.md5
def patched_md5(*args, **kwargs):
    kwargs.pop('usedforsecurity', None)
    return original_md5(*args, **kwargs)
hashlib.md5 = patched_md5

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from io import BytesIO

def generate_pdf_report(report_data, username):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Blue Header Bar
    p.setFillColorRGB(0.168, 0.271, 0.616) # #2b459d
    p.rect(0, height - 120, width, 120, fill=1, stroke=0)
    
    p.setFillColor(colors.white)
    p.setFont("Helvetica-Bold", 40)
    p.drawCentredString(width/2, height - 70, "TB Detection AI")
    
    p.setFont("Helvetica", 16)
    p.drawCentredString(width/2, height - 95, "AI-Powered Clinical Decision Support System")
    
    # Title
    p.setFillColor(colors.black)
    p.setFont("Helvetica-Bold", 24)
    p.drawCentredString(width/2, height - 170, "Clinical Analysis Report")
    
    p.setStrokeColor(colors.lightgrey)
    p.line(50, height - 185, width - 50, height - 185)

    curr_y = height - 220

    def draw_section_header(canvas, y, text):
        canvas.setFillColorRGB(0.95, 0.95, 0.95)
        canvas.rect(50, y - 5, width - 100, 25, fill=1, stroke=0)
        canvas.setFillColor(colors.black)
        canvas.setFont("Helvetica-Bold", 14)
        canvas.drawString(60, y, text)
        return y - 40

    # 1. Patient & Report Information
    curr_y = draw_section_header(p, curr_y, "1. Patient & Report Information")
    p.setFont("Helvetica", 12)
    p.drawString(70, curr_y, f"Report ID: {report_data.get('id', 'N/A')}")
    curr_y -= 20
    p.drawString(70, curr_y, f"Date: {report_data.get('date', 'N/A')}")
    curr_y -= 20
    p.drawString(70, curr_y, f"Physician/User: {username}")
    curr_y -= 40

    # 2. Scan Information
    curr_y = draw_section_header(p, curr_y, "2. Scan Information")
    p.setFont("Helvetica", 12)
    p.drawString(70, curr_y, f"Filename: {report_data.get('filename', 'N/A')}")
    curr_y -= 20
    p.drawString(70, curr_y, "Modality: Chest X-Ray")
    curr_y -= 40

    # 3. Diagnostic Analysis Results
    curr_y = draw_section_header(p, curr_y, "3. Diagnostic Analysis Results")
    p.setFont("Helvetica", 12)
    p.drawString(70, curr_y, "Prediction: ")
    
    pred = report_data.get('prediction', 'Analyzing...')
    if pred == 'Tuberculosis':
        p.setFillColor(colors.red)
    else:
        p.setFillColor(colors.green)
    p.setFont("Helvetica-Bold", 12)
    p.drawString(135, curr_y, pred)
    
    p.setFillColor(colors.black)
    p.setFont("Helvetica", 12)
    curr_y -= 20
    p.drawString(70, curr_y, f"Confidence Score: {report_data.get('confidence', '0%')}")
    curr_y -= 20
    p.drawString(70, curr_y, f"Risk Level: {report_data.get('risk_level', 'Low')}")
    curr_y -= 40

    # 4. Visual Evidence (Grad-CAM Analysis)
    curr_y = draw_section_header(p, curr_y, "4. Visual Evidence (Grad-CAM Analysis)")
    
    img_y = curr_y - 180
    orig_path = report_data.get('original_image')
    grad_path = report_data.get('gradcam_image')
    
    if orig_path and os.path.exists(orig_path):
        p.drawImage(orig_path, 80, img_y, width=200, height=180)
        p.setFont("Helvetica-Oblique", 10)
        p.drawCentredString(180, img_y - 15, "Original X-Ray")
        
    if grad_path and os.path.exists(grad_path):
        p.drawImage(grad_path, 320, img_y, width=200, height=180)
        p.setFont("Helvetica-Oblique", 10)
        p.drawCentredString(420, img_y - 15, "AI Heatmap Focus")
    
    curr_y = img_y - 60

    # 5. Clinical Interpretation
    curr_y = draw_section_header(p, curr_y, "5. Clinical Interpretation")
    p.setFont("Helvetica", 11)
    if pred == 'Tuberculosis':
        interpretation = "The AI model detected patterns consistent with Tuberculosis. The heatmap indicates high-activations in specific lung regions which should be clinically correlated."
    else:
        interpretation = "The AI model detected no significant patterns of Tuberculosis. The lung regions show normal radiologic characteristics within the scope of this analysis."
    
    p.drawString(70, curr_y, interpretation)
    curr_y -= 40

    # 6. Recommended Clinical Actions
    curr_y = draw_section_header(p, curr_y, "6. Recommended Clinical Actions")
    p.setFont("Helvetica", 11)
    if pred == 'Tuberculosis':
        actions = [
            "- Immediate sputum smear microscopy or GeneXpert test.",
            "- Chest CT scan to evaluate extent of infection.",
            "- Patient isolation and consultation with a pulmonologist."
        ]
    else:
        actions = [
            "- Routine annual screening.",
            "- Maintain healthy lifestyle.",
            "- Clinically correlate with other symptoms if present."
        ]
    
    for action in actions:
        p.drawString(80, curr_y, action)
        curr_y -= 18

    p.showPage()
    p.save()
    
    buffer.seek(0)
    return buffer

test_data = {
    "prediction": "Tuberculosis",
    "confidence": "99.98%",
    "risk_level": "High",
    "tb_probability": "99.98%",
    "id": "#3",
    "date": "2026-03-04 10:47:51"
}

import traceback
try:
    buf = generate_pdf_report(test_data, "test_user")
    with open("test_report.pdf", "wb") as f:
        f.write(buf.read())
    print("Success: test_report.pdf generated")
except Exception as e:
    traceback.print_exc()

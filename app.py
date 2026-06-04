import hashlib
original_md5 = hashlib.md5
def patched_md5(*args, **kwargs):
    kwargs.pop('usedforsecurity', None)
    return original_md5(*args, **kwargs)
hashlib.md5 = patched_md5

import os
import json
from flask import Flask, request, jsonify, send_from_directory, render_template, redirect, url_for, session
from flask_cors import CORS
from functools import wraps
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.utils import simpleSplit
from io import BytesIO
import utils
import datetime

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.urandom(24)
CORS(app)

# Persistent user storage
basedir = os.path.abspath(os.path.dirname(__file__))
USERS_FILE = os.path.join(basedir, 'users.json')

def load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading users: {e}")
    return {"admin": "password123"}

def save_users(users_dict):
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump(users_dict, f, indent=4)
        print(f"Users saved to {USERS_FILE}")
    except Exception as e:
        print(f"Error saving users: {e}")

users = load_users()
# Force save on startup to verify write permissions
save_users(users)
print(f"Current users in system: {list(users.keys())}", flush=True)

# Persistent reports storage
REPORTS_FILE = 'reports.json'

def load_reports():
    print(f"Loading reports from: {REPORTS_FILE}", flush=True)
    if os.path.exists(REPORTS_FILE):
        try:
            with open(REPORTS_FILE, 'r') as f:
                content = f.read().strip()
                if not content:
                    return {}
                return json.loads(content)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error loading reports: {e}")
            return {}
    return {}

def save_report(username, report_data):
    all_reports = load_reports()
    if username not in all_reports:
        all_reports[username] = []
    
    # Add unique ID and timestamp
    report_id = f"#{len(all_reports[username]) + 1}"
    import datetime
    report_data['id'] = report_id
    report_data['date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    all_reports[username].insert(0, report_data) # Newest first
    
    try:
        with open(REPORTS_FILE, 'w') as f:
            json.dump(all_reports, f, indent=4)
    except Exception as e:
        print(f"Error saving report: {e}")

def generate_pdf_report(report_data, username):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Blue Header Bar
    p.setFillColorRGB(0.168, 0.271, 0.616) # #2b459d
    p.rect(0, height - 100, width, 100, fill=1, stroke=0)
    
    p.setFillColor(colors.white)
    p.setFont("Helvetica-Bold", 32)
    p.drawCentredString(width/2, height - 60, "TB Detection AI")
    
    p.setFont("Helvetica", 12)
    p.drawCentredString(width/2, height - 80, "AI-Powered Clinical Decision Support System")
    
    # Title
    p.setFillColor(colors.black)
    p.setFont("Helvetica-Bold", 18)
    p.drawCentredString(width/2, height - 140, "Clinical Analysis Report")
    
    p.setStrokeColor(colors.lightgrey)
    p.line(50, height - 155, width - 50, height - 155)

    curr_y = height - 190

    def draw_section_header(canvas, y, text):
        canvas.setFillColorRGB(0.95, 0.95, 0.95)
        canvas.rect(50, y - 5, width - 100, 20, fill=1, stroke=0)
        canvas.setFillColor(colors.black)
        canvas.setFont("Helvetica-Bold", 11)
        canvas.drawString(60, y, text)
        return y - 30

    # 1. Patient & Report Information
    curr_y = draw_section_header(p, curr_y, "1. Patient & Report Information")
    p.setFont("Helvetica", 10)
    p.drawString(70, curr_y, f"Report ID: {report_data.get('id', 'N/A')}")
    curr_y -= 15
    p.drawString(70, curr_y, f"Date: {report_data.get('date', 'N/A')}")
    curr_y -= 15
    p.drawString(70, curr_y, f"Physician/User: {username}")
    curr_y -= 30

    # 2. Scan Information
    curr_y = draw_section_header(p, curr_y, "2. Scan Information")
    p.setFont("Helvetica", 10)
    p.drawString(70, curr_y, f"Filename: {report_data.get('filename', 'N/A')}")
    curr_y -= 15
    p.drawString(70, curr_y, "Modality: Chest X-Ray")
    curr_y -= 30

    # 3. Diagnostic Analysis Results
    curr_y = draw_section_header(p, curr_y, "3. Diagnostic Analysis Results")
    p.setFont("Helvetica", 10)
    p.drawString(70, curr_y, "Prediction: ")
    
    pred = report_data.get('prediction', 'Analyzing...')
    if pred == 'Tuberculosis':
        p.setFillColor(colors.red)
    else:
        p.setFillColor(colors.green)
    p.setFont("Helvetica-Bold", 10)
    p.drawString(125, curr_y, pred)
    
    p.setFillColor(colors.black)
    p.setFont("Helvetica", 10)
    curr_y -= 15
    p.drawString(70, curr_y, f"Confidence Score: {report_data.get('confidence', '0%')}")
    curr_y -= 15
    p.drawString(70, curr_y, f"Risk Level: {report_data.get('risk_level', 'Low')}")
    curr_y -= 30

    # 4. Visual Evidence (Grad-CAM Analysis)
    curr_y = draw_section_header(p, curr_y, "4. Visual Evidence (Grad-CAM Analysis)")
    
    img_y = curr_y - 120
    orig_path = report_data.get('original_image')
    grad_path = report_data.get('gradcam_image')
    
    if orig_path and os.path.exists(orig_path):
        p.drawImage(orig_path, 80, img_y, width=150, height=130)
        p.setFont("Helvetica-Oblique", 8)
        p.drawCentredString(155, img_y - 10, "Original X-Ray")
        
    if grad_path and os.path.exists(grad_path):
        p.drawImage(grad_path, 350, img_y, width=150, height=130)
        p.setFont("Helvetica-Oblique", 8)
        p.drawCentredString(425, img_y - 10, "AI Heatmap Focus")
    
    curr_y = img_y - 40

    # 5. Clinical Interpretation
    curr_y = draw_section_header(p, curr_y, "5. Clinical Interpretation")
    p.setFont("Helvetica", 9)
    if pred == 'Tuberculosis':
        interpretation = "The AI model detected patterns consistent with Tuberculosis. The heatmap indicates high-activations in specific lung regions which should be clinically correlated."
    else:
        interpretation = "The AI model detected no significant patterns of Tuberculosis. The lung regions show normal radiologic characteristics within the scope of this analysis."
    
    # Text Wrapping for Interpretation
    lines = simpleSplit(interpretation, "Helvetica", 9, width - 140)
    for line in lines:
        p.drawString(70, curr_y, line)
        curr_y -= 12
    curr_y -= 18

    # 6. Recommended Clinical Actions
    curr_y = draw_section_header(p, curr_y, "6. Recommended Clinical Actions")
    p.setFont("Helvetica", 9)
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
        # Wrap actions as well just in case
        a_lines = simpleSplit(action, "Helvetica", 9, width - 150)
        for i, a_line in enumerate(a_lines):
            indent = 80 if i == 0 else 90
            p.drawString(indent, curr_y, a_line)
            curr_y -= 12
        curr_y -= 3

    p.showPage()
    p.save()
    
    buffer.seek(0)
    return buffer

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            print(f"Access denied to {request.path}: User not logged in", flush=True)
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Configuration
UPLOAD_FOLDER = 'uploads'
GRADCAM_FOLDER = os.path.join('static', 'gradcam')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

# Load model on startup
try:
    utils.load_model('models/best_model.pth')
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        print(f"Login attempt for: '{username}'", flush=True)
        
        if users.get(username) == password:
            print(f"Login success: {username}", flush=True)
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('index'))
        else:
            print(f"Login failed: '{username}' not matches. Registered users: {list(users.keys())}", flush=True)
            return render_template('login.html', error="Invalid credentials")
            
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        fullname = request.form.get('fullname', '').strip()
        print(f"Signup attempt: '{username}'", flush=True)
        
        if username in users:
            print(f"Signup failed: {username} exists", flush=True)
            return render_template('signup.html', error="User already exists")
            
        users[username] = password
        save_users(users)
        print(f"Account created: {username}", flush=True)
        return redirect(url_for('login'))
        
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Preprocess
        input_tensor = utils.preprocess_image(filepath)

        # Generate TB-specific activation map (target_class_idx=1 for TB)
        gradcam_filename = f"tb_map_{filename}"
        gradcam_path = os.path.join(GRADCAM_FOLDER, gradcam_filename)
        
        class_idx, probs = utils.generate_gradcam(input_tensor, filepath, gradcam_path, target_class_idx=1)
        
        normal_prob = probs[0].item()
        tb_prob = probs[1].item()
        
        classes = ['Normal', 'Tuberculosis']
        prediction = classes[class_idx]
        confidence = probs[class_idx].item()
        
        # Risk level strictly depends on TB probability
        risk_level = utils.get_risk_level(tb_prob)

        # Save to history
        report_data = {
            "prediction": str(prediction),
            "confidence": f"{float(confidence):.2%}",
            "risk_level": str(risk_level),
            "tb_probability": f"{float(tb_prob):.2%}",
            "original_image": filepath,
            "gradcam_image": gradcam_path,
            "filename": filename
        }
        print(f"DEBUG: Saving report data for {session.get('username')}: {report_data}", flush=True)
        print(f"DEBUG: Session state: {dict(session)}", flush=True)
        if session.get('username'):
            print(f"DEBUG: Saving report for {session['username']}", flush=True)
            save_report(session['username'], report_data)
        else:
            print("WARNING: No username in session, report NOT saved", flush=True)

        # Build absolute URL for gradcam
        return jsonify({
            "prediction": str(prediction),
            "confidence": str(confidence),
            "tb_probability": str(tb_prob),
            "risk_level": str(risk_level),
            "gradcam_image": f"/static/gradcam/{gradcam_filename}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/static/gradcam/<filename>')
def serve_gradcam(filename):
    return send_from_directory(GRADCAM_FOLDER, filename)

@app.route('/reports')
@login_required
def reports():
    try:
        username = session.get('username')
        print(f"Serving reports page for user: {username}", flush=True)
        return render_template('reports.html')
    except Exception as e:
        print(f"CRITICAL ERROR rendering reports.html: {e}", flush=True)
        return f"Internal Error: {str(e)}", 500

@app.route('/download_report/<report_id>')
@login_required
def download_report(report_id):
    username = session.get('username')
    all_reports = load_reports()
    user_reports = all_reports.get(username, [])
    
    # Find the specific report
    report = next((r for r in user_reports if r['id'] == report_id), None)
    
    if not report:
        return "Report not found", 404
        
    pdf_buffer = generate_pdf_report(report, username)
    filename = f"TB_Report_{report_id.replace('#', '')}.pdf"
    
    from flask import send_file
    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name=filename,
        mimetype='application/pdf'
    )

@app.route('/api/reports')
@login_required
def get_user_reports():
    print(f"API: Fetching reports for: {session.get('username')}", flush=True)
    all_reports = load_reports()
    username = session.get('username')
    user_reports = all_reports.get(username, [])
    return jsonify(user_reports)

@app.route('/api/export_reports')
@login_required
def export_reports():
    username = session.get('username')
    all_reports = load_reports()
    user_reports = all_reports.get(username, [])
    
    if not user_reports:
        return "No reports found", 404
        
    import io
    import csv
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header: matches screenshot requirements
    writer.writerow(['Report ID', 'Date', 'Image Name', 'Prediction', 'Confidence', 'Risk Level'])
    
    for report in user_reports:
        writer.writerow([
            report.get('id', 'N/A'),
            report.get('date', 'N/A'),
            report.get('filename', 'N/A'),
            report.get('prediction', 'N/A'),
            report.get('confidence', 'N/A'),
            report.get('risk_level', 'N/A')
        ])
    
    output.seek(0)
    
    from flask import Response
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=patient_reports.csv"}
    )

@app.route('/recommendations')
@login_required
def recommendations():
    try:
        username = session.get('username')
        all_reports = load_reports()
        user_reports = all_reports.get(username, [])
        
        # Get latest report if available
        latest_report = user_reports[0] if user_reports else None
        
        print(f"Serving recommendations for user: {username}", flush=True)
        return render_template('recommendations.html', report=latest_report)
    except Exception as e:
        print(f"CRITICAL ERROR serving recommendations: {e}", flush=True)
        return f"Internal Error: {str(e)}", 500

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5005))
    print(f"TB Detection API running at http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)

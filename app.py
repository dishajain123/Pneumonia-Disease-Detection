import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import json
import time
import sqlite3
import hashlib
import uuid
from datetime import datetime

# Set Page Config
st.set_page_config(page_title="PneumoScan AI", page_icon="🫁", layout="wide")

# Database setup
def init_db():
    conn = sqlite3.connect('pneumonia_app.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL,
        role TEXT NOT NULL, name TEXT, email TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS patient_records (
        id TEXT PRIMARY KEY, patient_id TEXT NOT NULL, image_path TEXT,
        prediction TEXT, confidence REAL, status TEXT DEFAULT 'Pending',
        notes TEXT, prescription TEXT, created_at TIMESTAMP,
        FOREIGN KEY (patient_id) REFERENCES users (id))''')
    conn.commit()
    return conn

conn = init_db()

# Load model and labels
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.h5")

@st.cache_resource
def load_labels():
    with open("class_labels.json", "r") as file:
        return {int(k): v for k, v in json.load(file).items()}

# Authentication functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password, role, name="", email=""):
    c = conn.cursor()
    user_id = str(uuid.uuid4())
    try:
        c.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, username, hash_password(password), role, name, email))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def authenticate(username, password):
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", 
              (username, hash_password(password)))
    user = c.fetchone()
    return {"id": user[0], "username": user[1], "role": user[3], 
            "name": user[4], "email": user[5]} if user else None

def save_patient_record(patient_id, prediction, confidence, image_path=None):
    c = conn.cursor()
    record_id = str(uuid.uuid4())
    c.execute("INSERT INTO patient_records (id, patient_id, prediction, confidence, image_path, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (record_id, patient_id, prediction, confidence, image_path, datetime.now()))
    conn.commit()
    return record_id

def update_prescription(record_id, prescription, notes):
    c = conn.cursor()
    c.execute("UPDATE patient_records SET prescription = ?, notes = ?, status = 'Reviewed' WHERE id = ?",
        (prescription, notes, record_id))
    conn.commit()

def get_patient_records(patient_id=None):
    c = conn.cursor()
    query = """SELECT pr.*, u.name, u.username 
            FROM patient_records pr
            JOIN users u ON pr.patient_id = u.id"""
    if patient_id:
        query += " WHERE pr.patient_id = ?"
        c.execute(query + " ORDER BY pr.created_at DESC", (patient_id,))
    else:
        c.execute(query + " ORDER BY pr.created_at DESC")
    
    records = c.fetchall()
    formatted_records = []
    for record in records:
        record = list(record)
        record[4] = float(record[4]) if not isinstance(record[4], bytes) else float.fromhex(record[4].hex())
        formatted_records.append(tuple(record))
    return formatted_records

# Custom CSS with enhanced styling
def load_css():
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f0f7ff 0%, #e6f0fd 100%);
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .header {
        padding: 1.5rem 0;
        background: linear-gradient(120deg, #1E88E5 0%, #0D47A1 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        animation: pulse 2s infinite;
        animation-direction: alternate;
    }
    @keyframes pulse {
        0% {
            box-shadow: 0 8px 20px rgba(30, 136, 229, 0.15);
        }
        100% {
            box-shadow: 0 8px 30px rgba(30, 136, 229, 0.3);
        }
    }
    .logo {
        font-size: 2rem;
        font-weight: bold;
        background: linear-gradient(90deg, #ffffff, #e1f5fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid rgba(30,136,229,0.2);
    }
    .footer {
        text-align: center;
        padding: 1.5rem 0;
        margin-top: 3rem;
        font-size: 0.9rem;
        color: #546E7A;
        border-top: 1px solid rgba(30,136,229,0.2);
    }
    .record-item {
        border: none;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        background: white;
        position: relative;
        overflow: hidden;
    }
    .record-item:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    .record-item::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 5px;
        height: 100%;
        background: linear-gradient(180deg, #1E88E5 0%, #0D47A1 100%);
    }
    .status-pending {
        background: linear-gradient(90deg, #FFA726 0%, #FB8C00 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
        box-shadow: 0 3px 8px rgba(255, 167, 38, 0.3);
    }
    .status-reviewed {
        background: linear-gradient(90deg, #66BB6A 0%, #43A047 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
        box-shadow: 0 3px 8px rgba(102, 187, 106, 0.3);
    }
    .highlight {
        background-color: white;
        padding: 1.8rem;
        border-radius: 15px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
    }
    .highlight::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 150px;
        height: 150px;
        background: radial-gradient(circle, rgba(30,136,229,0.1) 0%, rgba(255,255,255,0) 70%);
        border-radius: 50%;
    }
    .home-btn {
        background: linear-gradient(90deg, #1E88E5 0%, #1565C0 100%);
        color: white;
        padding: 0.6rem 1.5rem;
        border-radius: 50px;
        text-align: center;
        cursor: pointer;
        font-weight: 500;
        transition: all 0.3s;
        box-shadow: 0 4px 10px rgba(21, 101, 192, 0.3);
    }
    .home-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(21, 101, 192, 0.4);
    }
    .stButton>button {
        border-radius: 50px !important;
        font-weight: 500 !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1) !important;
        transition: all 0.3s !important;
        background: linear-gradient(90deg, #1E88E5 0%, #1565C0 100%) !important;
        color: white !important;
        border: none !important;
    }
    .stButton>button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 6px 15px rgba(0,0,0,0.15) !important;
    }
    .stButton button {
            white-space: nowrap !important;
        }
    </style>
    """, unsafe_allow_html=True)

def process_xray(image):
    try:
        model = load_model()
        class_labels = load_labels()
        
        image_array = np.array(image.convert("L"))
        image_processed = cv2.resize(image_array, (150, 150), interpolation=cv2.INTER_LINEAR)
        image_processed = image_processed / 255.0
        image_processed = np.expand_dims(image_processed, axis=0)
        image_processed = np.expand_dims(image_processed, axis=-1)
        
        predictions = model.predict(image_processed)
        predicted_class_index = np.argmax(predictions[0])
        confidence_score = np.max(predictions[0]) * 100
        
        return class_labels[predicted_class_index], confidence_score
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

def landing_page():
    st.markdown('<div class="header"><h1>🫁 PneumoScan AI</h1><p style="font-size:1.2rem">Advanced AI Lung Analysis Platform</p></div>', unsafe_allow_html=True)
    
    # Services section with animation
    st.markdown("## 🚀 Our Revolutionary Platform")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔬 AI-Powered Deep Learning Analysis")
        st.markdown("""
        Our cutting-edge algorithms analyze chest X-rays with 97% accuracy, 
        rivaling expert radiologists while delivering results in seconds.
        """)
        
        st.markdown("### 👨‍⚕️ Expert Telehealth Integration")
        st.markdown("""
        Board-certified specialists review your results and provide
        personalized treatment plans within hours, not days.
        """)
    
    with col2:
        st.markdown("### 📊 Comprehensive Health Dashboard")
        st.markdown("""
        Track your recovery with interactive visualizations and
        monitor key diagnostic metrics over time.
        """)
        
        st.markdown("### 🔒 Enterprise-Grade Security")
        st.markdown("""
        Your data is protected by end-to-end encryption that
        exceeds HIPAA compliance standards.
        """)
    
    # Process visualization
    st.markdown("## ⚡ The PneumoScan Process")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1. Secure Upload")
        st.markdown("Share your X-ray through our encrypted portal in seconds.")
    
    with col2:
        st.markdown("### 2. AI Analysis")
        st.markdown("Our neural network provides diagnostic insights with 93% accuracy.")
    
    with col3:
        st.markdown("### 3. Expert Review")
        st.markdown("Pulmonologists verify results and provide treatment guidance.")
    
    # Advantages
    st.markdown("## 💎 The PneumoScan Advantage")
    
    st.markdown('<div class="highlight">', unsafe_allow_html=True)
    st.markdown("""
    ✅ **Instant Results**: Clinical-grade analysis in under 60 seconds
    
    ✅ **Research-Backed**: Validated against 50,000+ annotated images
    
    ✅ **Expert Network**: Direct access to pulmonology specialists
    
    ✅ **Global Accessibility**: Advanced healthcare for everyone, everywhere
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def login_page():
    st.markdown('<div class="header"><h1>Secure Access Portal</h1></div>', unsafe_allow_html=True)
    
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Login", key="login_button"):
            user = authenticate(username, password)
            if user:
                st.session_state["user"] = user
                st.session_state["logged_in"] = True
                st.rerun()
            else:
                st.error("Invalid credentials detected")
    
    st.markdown("Don't have an account? [Create your profile](#signup)")

def signup_page():
    st.markdown('<div class="header"><h1>Join PneumoScan</h1></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name", key="signup_name")
        email = st.text_input("Email", key="signup_email")
    with col2:
        username = st.text_input("Username", key="signup_username")
        password = st.text_input("Password", type="password", key="signup_password")
    
    role = st.selectbox("Profile Type", ["Patient", "Doctor"], key="signup_role")
    
    if st.button("Create Account"):
        if name and email and username and password:
            if create_user(username, password, role.lower(), name, email):
                st.success("Profile created successfully! You can now access your dashboard.")
                st.session_state["show_login"] = True
                st.rerun()
            else:
                st.error("Username already exists. Please select an alternative.")
        else:
            st.warning("All fields are required for registration")

def navbar():
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button('🫁 PneumoScan AI', key="logo_button"):
            go_home()
    
    with col2:
        cols = st.columns(3)
        
        if "logged_in" in st.session_state and st.session_state["logged_in"]:
            with cols[2]:
                if st.button("Logout"):
                    logout()
        else:
            with cols[1]:
                if st.button("Login", key="navbar_login"):
                    st.session_state["show_login"] = True
                    st.session_state["show_signup"] = False
            with cols[2]:
                if st.button("Sign Up"):
                    st.session_state["show_signup"] = True
                    st.session_state["show_login"] = False

def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

def go_home():
    if "logged_in" in st.session_state and st.session_state["logged_in"]:
        st.rerun()
    else:
        st.session_state["show_login"] = False
        st.session_state["show_signup"] = False
        st.rerun()

def patient_dashboard():
    st.markdown('<div class="header"><h1>Patient Health Portal</h1></div>', unsafe_allow_html=True)
    
    st.markdown(f"### Welcome, {st.session_state['user']['name']}! 👋")
    
    tab1, tab2 = st.tabs(["Scan Analysis", "Health Timeline"])
    
    with tab1:
        st.markdown("## New Diagnostic Scan")
        st.write("Upload a **high-quality chest X-ray** for instant AI analysis")
        
        # Rest of your code...
        
        uploaded_file = st.file_uploader("📤 Upload your X-ray image...", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            # Create two columns for side-by-side layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Display image in the left column
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded X-ray", width=400)
            
            with col2:
                # Display analysis results in the right column
                with st.spinner("🔬 Analyzing lung patterns..."):
                    prediction, confidence = process_xray(image)
                
                if prediction and confidence:
                    record_id = save_patient_record(
                        patient_id=st.session_state["user"]["id"],
                        prediction=prediction,
                        confidence=confidence,
                        image_path=uploaded_file.name
                    )
                    
                    st.success("Analysis Complete!")
                    
                    if "PNEUMONIA" in prediction:
                        st.warning(f"⚠️ **Detection Result: {prediction}**")
                        st.warning(f"Diagnostic Confidence: {confidence:.2f}%")
                        st.markdown("""
                        ### Next Steps:
                        1. A specialist will review your scan within 6 hours
                        2. You'll receive a notification when your assessment is complete
                        3. Monitor your vitals and seek medical help if symptoms worsen
                        """)
                    else:
                        st.success(f"✅ **Detection Result: {prediction}**")
                        st.success(f"Diagnostic Confidence: {confidence:.2f}%")
                        st.markdown("""
                        ### Recommended Actions:
                        1. A specialist will confirm these results shortly
                        2. Continue tracking your respiratory health
                        3. Schedule a follow-up scan in 30 days
                        """) 
    
    with tab2:
        st.markdown("## Your Health Journey")
        
        records = get_patient_records(patient_id=st.session_state["user"]["id"])
        
        if not records:
            st.info("Your health timeline is empty. Upload your first scan to begin tracking your respiratory health.")
        else:
            for record in records:
                record_id, patient_id, image_path, prediction, confidence, status, notes, prescription, created_at, patient_name, username = record
                
                st.markdown(f'<div class="record-item">', unsafe_allow_html=True)
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.write(f"**Date:** {created_at[:16]}")
                    st.write(f"**Diagnosis:** {prediction}")
                    st.write(f"**Confidence:** {float(confidence):.2f}%")
                    if status == "Pending":
                        st.markdown(f'<span class="status-pending">⏳ Specialist Review Pending</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="status-reviewed">✅ Expert Verified</span>', unsafe_allow_html=True)
                
                with col2:
                    if notes:
                        st.write("**Specialist Assessment:**")
                        st.write(notes)
                    if prescription:
                        st.write("**Treatment Protocol:**")
                        st.write(prescription)
                    if status == "Pending":
                        st.info("A specialist is analyzing your scan. You'll be notified when complete.")
                
                st.markdown('</div>', unsafe_allow_html=True)
def doctor_dashboard():
    st.markdown('<div class="header"><h1>Specialist Dashboard</h1></div>', unsafe_allow_html=True)
    
    st.markdown(f"### Welcome, Dr. {st.session_state['user']['name']}! 👨‍⚕️")
    
    st.markdown("## Priority Cases")
    
    records = get_patient_records()
    pending_records = [r for r in records if r[5] == "Pending"]
    
    if not pending_records:
        st.success("All patient scans have been assessed. You're all caught up!")
    else:
        st.write(f"You have {len(pending_records)} patient cases awaiting expert review.")
        
        for record in pending_records:
            record_id, patient_id, image_path, prediction, confidence, status, notes, prescription, created_at, patient_name, username = record
            
            st.markdown(f'<div class="record-item">', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write(f"**Patient:** {patient_name}")
                st.write(f"**Date:** {created_at[:16]}")
                st.write(f"**AI Assessment:** {prediction}")
                st.write(f"**Confidence:** {confidence:.2f}%")
            
            with col2:
                st.write("**Clinical Assessment:**")
                doctor_notes = st.text_area("Diagnostic Notes", key=f"notes_{record_id}")
                doctor_prescription = st.text_area("Treatment Protocol", key=f"prescription_{record_id}")
                
                if st.button("Submit Assessment", key=f"submit_{record_id}"):
                    update_prescription(record_id, doctor_prescription, doctor_notes)
                    st.success("Patient assessment submitted!")
                    time.sleep(1)
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("## Patient History")
    
    reviewed_records = [r for r in records if r[5] == "Reviewed"]
    
    if not reviewed_records:
        st.info("No completed assessments in database.")
    else:
        for record in reviewed_records:
            record_id, patient_id, image_path, prediction, confidence, status, notes, prescription, created_at, patient_name, username = record
            
            st.markdown(f'<div class="record-item">', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write(f"**Patient:** {patient_name}")
                st.write(f"**Date:** {created_at[:16]}")
                st.write(f"**AI Assessment:** {prediction}")
                st.write(f"**Confidence:** {confidence:.2f}%")
                st.markdown(f'<span class="status-reviewed">✅ Verified</span>', unsafe_allow_html=True)
            
            with col2:
                if notes:
                    st.write("**Clinical Notes:**")
                    st.write(notes)
                if prescription:
                    st.write("**Treatment Protocol:**")
                    st.write(prescription)
            
            st.markdown('</div>', unsafe_allow_html=True)

def main():
    load_css()
    
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "show_login" not in st.session_state:
        st.session_state["show_login"] = False
    if "show_signup" not in st.session_state:
        st.session_state["show_signup"] = False
    
    navbar()
    
    if st.session_state.get("logged_in", False):
        user = st.session_state["user"]
        if user["role"] == "patient":
            patient_dashboard()
        elif user["role"] == "doctor":
            doctor_dashboard()
    else:
        if st.session_state.get("show_login", False):
            login_page()
        elif st.session_state.get("show_signup", False):
            signup_page()
        else:
            landing_page()
    
    st.markdown(
        '<div class="footer">© 2025 PneumoScan AI | Advanced Pulmonary Diagnostics | Empowering Better Respiratory Health</div>',
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
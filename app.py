import streamlit as st
from ultralytics import YOLO
import tempfile
from PIL import Image
import os

st.set_page_config(page_title="Traffic Sign Detection", page_icon="🚦", layout="centered")
st.title("🚦 Traffic Sign Detection (YOLOv11)")

# -------------------------------
# 🔧 ตัวเลือกโมเดล
# -------------------------------
MODEL_OPTIONS = {
    "Raw Model (ไม่ Augmented)": "weights/raw.pt",
    "Augmented Model": "weights/aug.pt"
}
selected_model = st.radio("🧠 เลือกโมเดลที่ต้องการใช้", list(MODEL_OPTIONS.keys()))

# โหลดโมเดล (cache เพื่อไม่โหลดซ้ำ)
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model

model = load_model(MODEL_OPTIONS[selected_model])

# -------------------------------
# 📁 Upload image
# -------------------------------
uploaded_file = st.file_uploader("📸 อัปโหลดภาพที่มีป้ายจราจร...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # แสดงภาพต้นฉบับ
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # สร้างไฟล์ชั่วคราว
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        img_path = tmp.name

    # ตรวจจับด้วย YOLO
    with st.spinner("🔍 กำลังตรวจจับป้ายจราจร..."):
        results = model(img_path)

    # แสดงภาพที่มี bounding boxes
    result_img = results[0].plot()  # numpy array
    st.image(result_img, caption="✅ Detection Result", use_column_width=True)

    # แสดงตารางผลลัพธ์
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        data = []
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            data.append({
                "Class": model.names[cls],
                "Confidence": round(conf, 3)
            })
        st.dataframe(data, use_container_width=True)
    else:
        st.warning("⚠️ ไม่พบป้ายจราจรในภาพนี้")

    # ลบไฟล์ temp
    os.remove(img_path)
else:
    st.info("⬆️ กรุณาอัปโหลดภาพเพื่อเริ่มตรวจจับ")

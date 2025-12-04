import warnings
warnings.filterwarnings("ignore")

from PIL import Image
import gradio as gr
import json

from ocr_and_classifycontent_qwenmodel import ocr_and_classify

# -------------------------------
# Helper: parse JSON output
# -------------------------------
def process_ui(img):
    try:
        response = ocr_and_classify(img)
        clean_txt = response.replace("```", "").replace("json", "").strip()
        data = json.loads(clean_txt)

        # Format phone
        if "เบอร์โทร" in data:
            if isinstance(data["เบอร์โทร"], str):
                data["เบอร์โทร"] = data["เบอร์โทร"].replace(" ", ", ")

        return (
            data.get("ชื่อ Account",""),
            data.get("ที่อยู่",""),
            data.get("ข้อมูลที่ขอความช่วยเหลือ",""),
            data.get("ประเภทความช่วยเหลือ",""),
            data.get("เบอร์โทร","")
        )

    except Exception as e:
        print("OCR Error:", e)
        return ("error","error","error","error","error")

# -------------------------------
# Gradio UI
# -------------------------------
with gr.Blocks(title="OCR + Classification Demo") as demo:

    gr.Markdown("## 📄 OCR + Form Extraction (Qwen Model)")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload image")
            with gr.Row():
                gr.ClearButton([input_image])
                btn = gr.Button("Submit", variant="primary")

        with gr.Column():
            name = gr.Textbox(label="ชื่อ Account")
            addr = gr.Textbox(label="ที่อยู่")
            details = gr.TextArea(label="ข้อมูลที่ขอความช่วยเหลือ")
            type_help = gr.Textbox(label="ประเภทความช่วยเหลือ")
            calls = gr.Textbox(label="เบอร์โทร")

    btn.click(
        fn=process_ui,
        inputs=input_image,
        outputs=[name, addr, details, type_help, calls]
    )

demo.queue()
demo.launch(server_name="0.0.0.0", server_port=7860)

from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import numpy as np

# -------------------------------
# Device setup
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Loading model on {device.upper()} (dtype={torch_dtype})")
if device == "cpu":
    print("⚠️ Running on CPU — may be very slow")

# -------------------------------
# Schema for prompt
# -------------------------------
def content_requests_schema():
    return {
        "description": "โปรดตอบกลับเป็น JSON ตาม schema นี้เท่านั้น (ภาษาไทย)",
        "example": {
            "ชื่อ Account": "นางสาว พิมพ์พิชา ใจดี",
            "ที่อยู่": "บ้านเลขที่ 45/7 ถ.พัฒนา ซอยจงใจ ต.ในเมือง อ.เมือง จ.เชียงใหม่ 50000",
            "ข้อมูลที่ขอความช่วยเหลือ": "ต้องการอาหารและน้ำดื่มสำหรับครอบครัว 4 คน เนื่องจากบ้านถูกน้ำท่วม",
            "ประเภทความช่วยเหลือ": "อาหาร",
            "เบอร์โทร": ["0812345678", "+66891234567"],
            "เวลาที่สะดวกติดต่อ": "09:00-18:00",
            "ฉุกเฉิน": True
        },
        "constraints": {
            "ชื่อ Account": "string",
            "ที่อยู่": "string",
            "ข้อมูลที่ขอความช่วยเหลือ": "string",
            "ประเภทความช่วยเหลือ": "enum: ['การแพทย์','การเงิน','ที่พักอาศัย','อาหาร','กฎหมาย/ปรึกษา','อื่นๆ']",
            "เบอร์โทร": "array of string, regex ^(?:\\+66|0)\\d{8,9}$",
            "เวลาที่สะดวกติดต่อ": "string, optional",
            "ฉุกเฉิน": "boolean"
        }
    }

# -------------------------------
# Load model & processor
# -------------------------------
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch_dtype,
    device_map="auto" if device == "cuda" else None,
    low_cpu_mem_usage=True
)

if device == "cpu":
    model = model.to(device)

model.eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
print("✓ Model and processor loaded successfully!")

# -------------------------------
# Convert input to RGB Tensor
# -------------------------------
def prepare_image(img_input):
    # Convert input to PIL
    if isinstance(img_input, np.ndarray):
        img = Image.fromarray(img_input.astype("uint8"))
    elif isinstance(img_input, Image.Image):
        img = img_input
    elif isinstance(img_input, str):
        img = Image.open(img_input)
    else:
        raise ValueError("Invalid image input type")

    # Convert to RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Convert to numpy array
    img_np = np.array(img).astype("uint8")

    # Convert to tensor (C,H,W) and add batch dim
    img_tensor = torch.tensor(img_np).permute(2,0,1).unsqueeze(0)
    return img_tensor

# -------------------------------
# Main OCR + classification
# -------------------------------
@torch.inference_mode()
def ocr_and_classify(img_input):
    schema_info = content_requests_schema()
    img_tensor = prepare_image(img_input)

    prompt = f"""
You are an OCR extraction assistant.

Task:
Extract text from the image (OCR) and classify it into the following fields.

Output rules:
1. Return ONLY JSON according to the schema.
2. If a field is missing in the image, return "-".
3. Do NOT add extra explanation.
4. Field names must remain exactly the same.

Schema Description:
{schema_info['description']}

Example JSON:
{schema_info['example']}

Constraints:
{schema_info['constraints']}
"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    text_prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    inputs = processor(
        text=[text_prompt],
        images=[img_tensor],  # send tensor batch
        padding=True,
        return_tensors="pt"
    )

    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    output_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False
    )

    # Remove prompt tokens
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]

    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]

    return response

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

#VoiceBot UI with Gradio
import os
import json
import gradio as gr
from datetime import datetime, timezone

from brain_of_the_doctor import encode_image, analyze_image_with_query

system_prompt="""You have to act as a professional doctor, i know you are not but this is for learning purpose. 
            What's in this image?. Do you find anything wrong with it medically? 
            If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in 
            your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
            Donot say 'In the image I see' but say 'With what I see, I think you have ....'
            Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot, 
            Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""


def _predict_disease_category_from_text(text: str) -> str:
    t=(text or "").lower()
    keyword_to_category={
        "acne": "acne",
        "pimple": "acne",
        "blackhead": "acne",
        "whitehead": "acne",
        "dandruff": "dandruff",
        "seborrheic": "dandruff",
        "rash": "rash",
        "eczema": "eczema",
        "dermatitis": "eczema",
        "psoriasis": "psoriasis",
        "fungal": "fungal_infection",
        "ringworm": "fungal_infection",
        "allergy": "allergic_reaction",
        "allergic": "allergic_reaction",
        "bacterial": "bacterial_infection",
        "infection": "infection",
        "rosacea": "rosacea",
        "hives": "hives",
    }
    for k, v in keyword_to_category.items():
        if k in t:
            return v
    return "unknown"


def _derive_disease_type(category: str) -> str:
    mapping = {
        "dandruff": "scalp_condition",
        "acne": "dermatologic",
        "eczema": "dermatologic",
        "dermatitis": "dermatologic",
        "psoriasis": "dermatologic",
        "fungal_infection": "dermatologic",
        "rosacea": "dermatologic",
        "hives": "immunologic",
        "allergic_reaction": "immunologic",
        "bacterial_infection": "infectious",
        "infection": "infectious",
        "rash": "dermatologic",
    }
    return mapping.get((category or "").lower(), "unknown")


def _suggest_solutions(category: str) -> list:
    c = (category or "").lower()
    if c == "dandruff":
        return [
            "Use medicated anti-dandruff shampoo (ketoconazole, zinc pyrithione, or selenium sulfide)",
            "Avoid harsh hair products with alcohol",
            "Moisturize scalp with light oil (e.g., tea tree oil or diluted coconut oil)",
            "Maintain a balanced diet with zinc, omega-3, and B vitamins",
        ]
    if c == "acne":
        return [
            "Use gentle cleanser; avoid over-washing",
            "Apply benzoyl peroxide or salicylic acid spot treatments",
            "Non-comedogenic moisturizer and sunscreen daily",
            "See a dermatologist if nodular/cystic or scarring",
        ]
    if c in ("eczema", "dermatitis"):
        return [
            "Fragrance-free emollients multiple times daily",
            "Short lukewarm showers; avoid hot water",
            "Low-potency topical steroid for flares (consult clinician)",
            "Identify and avoid triggers (soaps, wool, allergens)",
        ]
    if c == "psoriasis":
        return [
            "Use coal tar or salicylic acid shampoos for scalp",
            "Topical corticosteroids/vitamin D analogs (per clinician)",
            "Manage stress; avoid skin trauma",
        ]
    if c == "fungal_infection":
        return [
            "Topical antifungal (clotrimazole/ketoconazole) as directed",
            "Keep area clean, dry, and cool",
            "Avoid sharing towels/clothing",
        ]
    if c in ("allergic_reaction", "hives"):
        return [
            "Oral antihistamine (non-drowsy) as needed",
            "Cold compresses to reduce itching",
            "Identify and avoid suspected allergen",
        ]
    if c in ("bacterial_infection", "infection"):
        return [
            "Keep affected area clean",
            "Seek evaluation for possible antibiotics",
            "Monitor fever, spreading redness, or pain",
        ]
    return [
        "Monitor symptoms and avoid irritants",
        "Over-the-counter relief where appropriate",
        "Consult a clinician if symptoms persist or worsen",
    ]


def _recommended_doctor(category: str) -> str:
    c = (category or "").lower()
    if c in ("acne", "eczema", "dermatitis", "psoriasis", "fungal_infection", "rosacea", "dandruff", "rash"):
        return "Dermatologist"
    if c in ("allergic_reaction", "hives"):
        return "Allergist/Immunologist"
    if c in ("bacterial_infection", "infection"):
        return "Primary care physician"
    return "Primary care physician"


def _urgency_level(category: str) -> str:
    c = (category or "").lower()
    if c in ("bacterial_infection", "infection"):
        return "urgent"
    return "non_emergency"


def process_inputs(image_filepath, symptom_text):
    # Build prompt from typed symptoms only
    symptom_text_clean=(symptom_text or "").strip()
    combined_text = "Typed symptoms: " + symptom_text_clean if symptom_text_clean else "No symptoms provided"

    # Handle the image input with LLM vision if provided
    if image_filepath:
        doctor_response = analyze_image_with_query(
            query=system_prompt + "\n" + combined_text,
            encoded_image=encode_image(image_filepath),
            model="meta-llama/llama-4-scout-17b-16e-instruct"
        )
    else:
        doctor_response = "No image provided for me to analyze"

    # Build JSON response
    disease_category = _predict_disease_category_from_text(doctor_response + "\n" + symptom_text_clean)
    disease_type = _derive_disease_type(disease_category)
    solutions = _suggest_solutions(disease_category)
    doctor = _recommended_doctor(disease_category)
    urgency = _urgency_level(disease_category)

    response_json = {
        "user_prompt": symptom_text_clean,
        "disease_category": disease_category,
        "disease_type": disease_type,
        "solutions": solutions,
        "recommended_doctor": doctor,
        "urgency": urgency,
        "uploaded_image": image_filepath or "",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    return response_json


def flag_response(json_data: dict):
    try:
        ts = datetime.now(timezone.utc).isoformat()
        with open("flags.log", "a", encoding="utf-8") as f:
            f.write(ts + " " + json.dumps(json_data, ensure_ascii=False) + "\n")
        gr.Info("Flagged. Thank you.")
    except Exception as e:
        gr.Warning(f"Could not flag: {e}")


APP_CSS = """
:root { --orange: #f57c00; --gray: #6b7280; }

#app-title { text-align: center; font-weight: 700; font-size: 28px; margin: 8px 0 16px; }

.card { background: white; border-radius: 12px; padding: 16px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); }
.controls-row { gap: 12px; }

.submit-btn > button { background: var(--orange) !important; border-color: var(--orange) !important; color: #fff !important; }
.clear-btn > button { background: var(--gray) !important; border-color: var(--gray) !important; color: #fff !important; }

.console { background: #0b1021; color: #e3e7ff; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; border-radius: 10px; padding: 12px; height: 260px; overflow: auto; border: 1px solid #1c2340; }
.flag-row { display: flex; justify-content: flex-end; }
"""


def build_ui():
    with gr.Blocks(css=APP_CSS, theme=gr.themes.Default()) as demo:
        gr.Markdown("# AI Doctor with Vision", elem_id="app-title")

        # Input card
        with gr.Group(elem_classes=["card"]):
            with gr.Row():
                image_input = gr.Image(type="filepath", label="Upload Image", height=260)
                symptom_input = gr.Textbox(label="Enter Symptoms", placeholder="Type your symptoms here...", lines=8)
            with gr.Row(elem_classes=["controls-row"]):
                clear_btn = gr.Button("Clear", elem_classes=["clear-btn"]) 
                submit_btn = gr.Button("Submit", elem_classes=["submit-btn"]) 

        # Output card
        with gr.Group(elem_classes=["card"]):
            json_output = gr.JSON(label="Doctor's JSON Response", elem_classes=["console"]) 
            with gr.Row(elem_classes=["flag-row"]):
                flag_btn = gr.Button("Flag")

        # Wire events
        submit_btn.click(
            fn=process_inputs,
            inputs=[image_input, symptom_input],
            outputs=[json_output]
        )
        clear_btn.click(
            fn=lambda: (None, "", {}),
            inputs=None,
            outputs=[image_input, symptom_input, json_output]
        )
        flag_btn.click(
            fn=flag_response,
            inputs=[json_output],
            outputs=[]
        )

    return demo


demo = build_ui()

demo.launch(debug=True)

#http://127.0.0.1:7860
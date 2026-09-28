"""
RetinaFlow — Dual-Model Retinal Disease Diagnostic System
Author: Mohamed Mostafa Elbasyouni (markegyptian55-cloud)
Repository: https://github.com/markegyptian55-cloud/RetinaFlow
Space: https://huggingface.co/spaces/egyx/RetinaFlow
License: MIT
"""

import os
import sys
import time
import hashlib
from typing import Dict, Tuple, Optional, Any
import numpy as np
from PIL import Image

try:
    import spaces
except ImportError:
    class _MockSpaces:
        @staticmethod
        def GPU(fn=None, **kwargs):
            if fn is not None and callable(fn):
                return fn
            return lambda f: f
    spaces = _MockSpaces()

import onnxruntime as ort
import gradio as gr

# ============================================================================
# CONFIGURATION & METADATA
# ============================================================================
HF_REPO_ID = "egyx/RetinaFlow"

CLASS_NAMES = [
    'AMD', 'Cataract', 'Diabetic Retinopathy', 'Glaucoma',
    'Hypertension', 'Myopia', 'Normal', 'Others'
]

# Clinical data dictionary with full English & Arabic definitions
CLINICAL_DATA = {
    'Normal': {
        'en': {
            'name': 'Normal Retinal Fundus',
            'badge': '🟢 ROUTINE / LOW RISK',
            'desc': 'Optic nerve head, macula lutea, and retinal microvasculature exhibit physiological anatomical features with no visible signs of retinopathy or optic neuropathy.'
        },
        'ar': {
            'name': 'شبكية سليمة وطبيعية',
            'badge': '🟢 روتيني / طبيعي (منخفض الخطورة)',
            'desc': 'المظهر العام لقاع العين سليم. القرص البصري، البقعة الصفراء، والتروية الشبكية ضمن المعايير التشريحية السليمة تماماً.'
        },
        'color': '#10b981',
        'icon': '✅',
    },
    'AMD': {
        'en': {
            'name': 'Age-Related Macular Degeneration (AMD)',
            'badge': '🟡 SPECIALIST REVIEW (MODERATE RISK)',
            'desc': 'Degeneration of the central macular region impacting fine visual acuity. Prompt vitreoretinal specialist referral recommended to evaluate dry vs. wet progression.'
        },
        'ar': {
            'name': 'ضمور البقعة الصفراء المرتبط بالعمر (AMD)',
            'badge': '🟡 متابعة تخصصية (متوسط الخطورة)',
            'desc': 'تنكس في منطقة اللطاخة الصفراء المسؤولة عن حدة الرؤية المركزية. يستدعي فحصاً دورياً مع أخصائي الشبكية لتفادي تطور الحالة.'
        },
        'color': '#f59e0b',
        'icon': '👁️',
    },
    'Cataract': {
        'en': {
            'name': 'Cataract',
            'badge': '🟡 OPHTHALMIC EVALUATION (MODERATE RISK)',
            'desc': 'Progressive opacification of the ocular crystalline lens leading to visual degradation. Evaluated for surgical phacoemulsification and intraocular lens implantation.'
        },
        'ar': {
            'name': 'المياه البيضاء (عتامة العدسة)',
            'badge': '🟡 تقييم بصري وجراحي (متوسط الخطورة)',
            'desc': 'عتامة في عدسة العين الطبيعية تؤدي لضبابية الرؤية وصعوبة فحص القاع بوضوح. تُعالج بنجاح عبر استبدال العدسة جراحياً.'
        },
        'color': '#f59e0b',
        'icon': '🔍',
    },
    'Diabetic Retinopathy': {
        'en': {
            'name': 'Diabetic Retinopathy',
            'badge': '🔴 URGENT CLINICAL REVIEW (HIGH RISK)',
            'desc': 'Retinal microvascular pathology secondary to diabetes mellitus, with signs of microaneurysms, hemorrhages, hard exudates, or neovascularization. Dilated ophthalmic exam indicated.'
        },
        'ar': {
            'name': 'اعتلال الشبكية السكري (DR)',
            'badge': '🔴 تقييم سريري عاجل (عالي الخطورة)',
            'desc': 'أضرار وعائية دقيقة ناجمة عن السكري (نزيف شبكي، ارتشاحات دهنية، أو نمو أوعية دموية جديدة). يتطلب فحصاً وتدخلاً عاجلاً.'
        },
        'color': '#ef4444',
        'icon': '🩸',
    },
    'Glaucoma': {
        'en': {
            'name': 'Glaucoma (Optic Neuropathy)',
            'badge': '🔴 URGENT CLINICAL REVIEW (HIGH RISK)',
            'desc': 'Progressive optic neuropathy with neuroretinal rim thinning and increased optic disc cupping. Urgent intraocular pressure (IOP) tonometry and visual field testing required.'
        },
        'ar': {
            'name': 'المياه الزرقاء (اعتلال العصب البصري)',
            'badge': '🔴 تقييم سريري عاجل (عالي الخطورة)',
            'desc': 'اعتلال تدريجي في ألياف العصب البصري وزيادة تقعر القرص البصري. يتطلب قياس ضغط العين وفحص المجال البصري لتفادي فقدان الرؤية.'
        },
        'color': '#ef4444',
        'icon': '⚠️',
    },
    'Hypertension': {
        'en': {
            'name': 'Hypertensive Retinopathy',
            'badge': '🚨 RARE CRITICAL FINDING (URGENT)',
            'desc': 'Retinal vascular alterations caused by systemic arterial hypertension (arteriolar narrowing, AV nicking, or flame hemorrhages). Immediate cardiovascular and BP triage required.'
        },
        'ar': {
            'name': 'اعتلال الشبكية الناتج عن ضغط الدم (HT)',
            'badge': '🚨 حالة حرجة ونادرة (أولوية عاجلة)',
            'desc': 'تغيرات وعائية شريانية ناتجة عن ارتفاع ضغط الدم (تصلب شرياني، نزف لهبي). يستدعي ضبطاً فورياً لضغط الدم ومتابعة قلبية وبصرية.'
        },
        'color': '#ff7849',
        'icon': '❤️‍🩹',
    },
    'Myopia': {
        'en': {
            'name': 'Pathological Myopia',
            'badge': '🟡 PERIODIC MONITORING (MODERATE RISK)',
            'desc': 'High axial myopia associated with posterior staphyloma and chorioretinal thinning. Periodic peripheral fundus examination advised to monitor retinal detachment risk.'
        },
        'ar': {
            'name': 'قصر النظر المرضي / الشديد',
            'badge': '🟡 فحص شبكية دوري (متوسط الخطورة)',
            'desc': 'استطالة محورية في مقلة العين مع ترقق الغشاء الشبكي وتغيرات في المشيمية. يستلزم مراقبة دورية لتجنب مخاطر تمزق أو انفصال الشبكية.'
        },
        'color': '#38bdf8',
        'icon': '👓',
    },
    'Others': {
        'en': {
            'name': 'Other Retinal Lesions',
            'badge': '🟡 CLINICAL CORRELATION (MODERATE RISK)',
            'desc': 'Features suggestive of secondary or alternative fundus anomalies (e.g., epiretinal membrane, retinal vein occlusion, drusen). Direct ophthalmological examination recommended.'
        },
        'ar': {
            'name': 'آفات / اعتلالات شبكية أخرى',
            'badge': '🟡 فحص سريري إضافي (مراجعة مطلوبة)',
            'desc': 'تم رصد شذوذ أو علامات لآفات شبكية أخرى (مثل الأغشية التليفية، أو انسداد وريدي، أو دروزن). يُوصى بالمطابقة السريرية المباشرة.'
        },
        'color': '#a855f7',
        'icon': '🔬',
    }
}

MODEL_CONFIGS = {
    "General-Purpose (ResNet-50)": {
        "rel_path": "models/resnet50_m5a.onnx",
        "input_size": 224,
        "name": "ResNet-50 (M5a)",
        "badge": "🏆 Highest Overall Accuracy",
        "badge_ar": "🏆 أعلى دقة عامة",
        "blurb_en": "Optimized for overall classification across all 8 classes (71.7% Accuracy, Macro-F1 0.60).",
        "blurb_ar": "الأفضل للدقة الإجمالية الشاملة عبر كافة الفئات الـ 8 (دقة 71.7%، Macro-F1 0.60).",
    },
    "High-Sensitivity (EfficientNet-B5)": {
        "rel_path": "models/efficientnet_b5_m6.onnx",
        "input_size": 300,
        "name": "EfficientNet-B5 (M6)",
        "badge": "🎯 High Hypertension Recall",
        "badge_ar": "🎯 حساسية فائقة للضغط",
        "blurb_en": "Specialized to screen rare, high-risk conditions — 38.5% Hypertension recall (vs 15.4% for general model).",
        "blurb_ar": "مخصص لالتقاط الحالات الحرجة النادرة مثل اعتلال ضغط الدم (38.5% Recall مقابل 15.4%).",
    },
}

UI_TEXTS = {
    "en": {
        "lang_btn": "🌐 العربية",
        "top_status": "● Models ready · ONNX CPU",
        "title": "Retina<span style='color: #ff7849;'>Flow</span>",
        "subtitle": "Dual-Model Deep Learning Diagnostic System for Retinal Fundus Photography",
        "acc_val": "71.7%",
        "acc_lbl": "TEST ACCURACY",
        "ht_val": "38.5%",
        "ht_lbl": "HT RECALL (2.5×)",
        "accordion_title": "ℹ️ Model Architecture & Clinical Trade-Off Guide",
        "tradeoff_md": """### ⚖️ Why RetinaFlow Uses a Dual-Champion Architecture
Our clinical dataset aggregates **11,839 fundus images** with extreme class skew:
* **Hypertension** is ultra-rare (only 88 total images, a 53.4× imbalance vs Normal).
* **ResNet-50 (M5a):** General-purpose champion with highest overall accuracy (**71.7%**) and balanced Macro-F1 (0.60), but 15.4% recall on Hypertension.
* **EfficientNet-B5 (M6):** High-sensitivity champion achieving **38.5% recall on Hypertension** (2.5× higher), catching critical rare cases at the expense of lower overall accuracy (62.6%).
* **Dual Consensus Mode:** Evaluates both engines concurrently and alerts the practitioner if a rare condition is detected.

*Disclaimer: For research and educational decision-support only. Not an FDA/CE-certified diagnostic device.*""",
        "card1_num": "01",
        "card1_title": "Fundus image",
        "upload_label": "Upload Retinal Fundus Photograph",
        "mode_label": "Diagnostic Engine Mode",
        "mode_choices": [
            "⚡ Dual Consensus (Both Models)",
            "General-Purpose (ResNet-50)",
            "High-Sensitivity (EfficientNet-B5)",
        ],
        "mode_blurb": "**Dual Consensus Mode:** Runs both ResNet-50 and EfficientNet-B5 in parallel. ResNet-50 provides overall diagnostic precision, while EfficientNet-B5 screens for rare, high-risk pathologies.",
        "btn_analyze": "⛶ Analyze image",
        "btn_reset": "🔄 Reset",
        "privacy_note": "🔒 Runs locally on device via ONNX. The image never leaves your machine.",
        "examples_label": "Select a sample fundus image to test",
        "card2_num": "02",
        "card2_title": "Model assessment",
        "placeholder_html": """<div style="text-align: center; padding: 48px 20px; color: #808593;">
            <div style="font-size: 2.8rem; margin-bottom: 12px; opacity: 0.6;">👁️</div>
            <h3 style="color: #ffffff; font-size: 1.15rem; margin-bottom: 6px; font-weight: 600;">Awaiting Fundus Image</h3>
            <p style="font-size: 0.88rem; max-width: 320px; margin: 0 auto; color: #5a5e6b;">
                Upload a fundus photo or pick one of the sample images, then click <strong>Analyze image</strong>.
            </p>
        </div>""",
        "label_m1": "Primary Probabilities (ResNet-50)",
        "label_m2": "High-Sensitivity Probabilities (EfficientNet-B5)",
        "footer_text": "RetinaFlow Research Platform • Built by Mohamed Mostafa Elbasyouni • GitHub: <a href='https://github.com/markegyptian55-cloud/RetinaFlow' target='_blank' style='color:#ff7849;text-decoration:none;'>markegyptian55-cloud/RetinaFlow</a> • Research & Educational Demonstration Only."
    },
    "ar": {
        "lang_btn": "🌐 English",
        "top_status": "● النماذج جاهزة · ONNX CPU",
        "title": "Retina<span style='color: #ff7849;'>Flow</span>",
        "subtitle": "نظام الذكاء الاصطناعي السريري لتشخيص أمراض شبكية العين عبر صور قاع العين",
        "acc_val": "71.7%",
        "acc_lbl": "دقة الاختبار الإجمالية",
        "ht_val": "38.5%",
        "ht_lbl": "حساسية التقاط الضغط (2.5×)",
        "accordion_title": "ℹ️ دليل بنية النماذج والمفاضلة السريرية (Model Trade-Off Guide)",
        "tradeoff_md": """### ⚖️ لماذا يدمج RetinaFlow نموذجين بطلين معاً؟
تم تدريب النماذج على **11,839 صورة سريرية حقيقية**، حيث تعاني البيانات من اختلال حاد في التوزيع:
* فئة **ارتفاع ضغط الدم (Hypertension)** نادرة جداً (88 حالة فقط في كامل قاعدة البيانات).
* **ResNet-50 (M5a):** بطل الدقة الإجمالية (**71.7%**)، ولكنه يحقق 15.4% فقط في التقاط فئة الضغط.
* **EfficientNet-B5 (M6):** بطل الحساسية العالية بحساسية تفوق **38.5%** للضغط (2.5 ضعف).
* **وضع المقارنة المزدوج:** يُشغل النموذجين بالتوازي ويرصد أي تباين تشخيصي للحالات النادرة.

*إخلاء مسؤولية: للأغراض البحثية والتعليمية والدعم التشخيصي فقط. ليس جهازاً طبياً معتمداً.*""",
        "card1_num": "01",
        "card1_title": "صورة قاع العين",
        "upload_label": "رفع صورة قاع العين (Fundus Image)",
        "mode_label": "وضع المحرك التشخيصي",
        "mode_choices": [
            "⚡ مقارنة النموذجين معاً (Dual Consensus)",
            "General-Purpose (ResNet-50)",
            "High-Sensitivity (EfficientNet-B5)",
        ],
        "mode_blurb": "**الوضع المزدوج التوافقي:** يشغل النموذجين معاً بالتوازي؛ حيث يقدم ResNet-50 الدقة العامة بينما يدقق EfficientNet-B5 في الحالات النادرة والحرجة.",
        "btn_analyze": "⛶ بدء فحص وتحليل الصورة",
        "btn_reset": "🔄 إعادة ضبط",
        "privacy_note": "🔒 يتم الاستدلال محلياً عبر محرك ONNX. الصور لا تغادر جهازك.",
        "examples_label": "اختر إحدى عينات الفحص الجاهزة للاختبار",
        "card2_num": "02",
        "card2_title": "التقييم التشخيصي للنموذج",
        "placeholder_html": """<div style="text-align: center; padding: 48px 20px; color: #808593;">
            <div style="font-size: 2.8rem; margin-bottom: 12px; opacity: 0.6;">👁️</div>
            <h3 style="color: #ffffff; font-size: 1.15rem; margin-bottom: 6px; font-weight: 600;">في انتظار تحميل الصورة وبدء الفحص</h3>
            <p style="font-size: 0.88rem; max-width: 320px; margin: 0 auto; color: #5a5e6b;">
                قم برفع صورة لقاع العين أو اختر إحدى العينات الجاهزة ثم اضغط على <strong>بدء فحص وتحليل الصورة</strong>.
            </p>
        </div>""",
        "label_m1": "احتمالات النموذج العام (ResNet-50)",
        "label_m2": "احتمالات نموذج الحساسية العالية (EfficientNet-B5)",
        "footer_text": "منصة RetinaFlow البحثية • تطوير: محمد مصطفى البسيوني • GitHub: <a href='https://github.com/markegyptian55-cloud/RetinaFlow' target='_blank' style='color:#ff7849;text-decoration:none;'>markegyptian55-cloud/RetinaFlow</a> • للأغراض البحثية والتعليمية فقط."
    }
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ============================================================================
# STANDALONE MODEL RESOLUTION & CACHING
# ============================================================================
_loaded_sessions: Dict[str, ort.InferenceSession] = {}
_result_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

def resolve_model_file(rel_path: str) -> str:
    """
    Checks if model exists locally. If not found (e.g., when app.py is uploaded 
    alone to a new Hugging Face Space), downloads it on-demand from the official Space.
    """
    if os.path.exists(rel_path):
        return rel_path

    print(f"[*] Local model '{rel_path}' not found. Downloading from Space '{HF_REPO_ID}'...")
    try:
        from huggingface_hub import hf_hub_download
        cached_file = hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="space",
            filename=rel_path
        )
        print(f"[✓] Retrieved: {cached_file}")
        return cached_file
    except Exception as e:
        raise FileNotFoundError(
            f"Could not load '{rel_path}' locally and auto-download from '{HF_REPO_ID}' failed: {e}"
        )

def get_session(model_key: str) -> ort.InferenceSession:
    if model_key not in _loaded_sessions:
        cfg = MODEL_CONFIGS[model_key]
        resolved_path = resolve_model_file(cfg["rel_path"])
        _loaded_sessions[model_key] = ort.InferenceSession(
            resolved_path, providers=['CPUExecutionProvider']
        )
    return _loaded_sessions[model_key]

def _hash_image(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()

def preprocess(image: Image.Image, size: int) -> np.ndarray:
    image = image.convert('RGB').resize((size, size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return np.expand_dims(arr, axis=0).astype(np.float32)

def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def run_inference_raw(image: Image.Image, model_key: str) -> Tuple[Dict[str, float], float]:
    config = MODEL_CONFIGS[model_key]
    session = get_session(model_key)
    input_tensor = preprocess(image, config["input_size"])
    input_name = session.get_inputs()[0].name
    
    t0 = time.perf_counter()
    logits = session.run(None, {input_name: input_tensor})[0][0]
    lat_ms = (time.perf_counter() - t0) * 1000
    
    probs = softmax(logits)
    prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    return prob_dict, lat_ms

# ============================================================================
# MCP & ZERO-GPU ENDPOINT
# ============================================================================
@spaces.GPU
def classify(image: Image.Image, model_choice: str = "General-Purpose (ResNet-50)") -> Dict[str, float]:
    """
    Public tool endpoint for Model Context Protocol (MCP) and programmatic APIs.
    Preserves exact backward-compatible signature.
    """
    if image is None:
        raise gr.Error("Please upload a fundus image before classifying.")
    
    # Normalize model choice string if passed from Arabic or alias
    if "EfficientNet" in model_choice:
        model_choice = "High-Sensitivity (EfficientNet-B5)"
    else:
        model_choice = "General-Purpose (ResNet-50)"
        
    cache_key = (_hash_image(image), model_choice)
    if cache_key in _result_cache:
        return _result_cache[cache_key]["probs"]
        
    probs, lat_ms = run_inference_raw(image, model_choice)
    _result_cache[cache_key] = {"probs": probs, "lat": lat_ms}
    return probs

# ============================================================================
# CLINICAL REPORT GENERATION (BILINGUAL)
# ============================================================================
def build_clinical_card_html(top_class: str, top_prob: float, lat_ms: float, model_name: str, consensus_note: str = "", lang: str = "en") -> str:
    data = CLINICAL_DATA.get(top_class, CLINICAL_DATA['Others'])
    lang_data = data[lang]
    pct = top_prob * 100
    
    lbl_heading = "PREDICTED SEVERITY & FINDING" if lang == "en" else "التشخيص السريري الأرجح"
    lbl_conf = "CONFIDENCE" if lang == "en" else "درجة الثقة"
    lbl_latency = "LATENCY" if lang == "en" else "سرعة المعالجة"
    lbl_engine = "ACTIVE ENGINE" if lang == "en" else "المحرك التشخيصي"
    lbl_clinical = "CLINICAL INSIGHT:" if lang == "en" else "الدلالة السريرية:"
    
    consensus_section = ""
    if consensus_note:
        consensus_section = f"""
        <div style="margin-top: 14px; padding: 12px 14px; background: #181920; 
                    border-radius: 8px; border-left: 3px solid {data['color']}; font-size: 0.88rem; line-height: 1.5;">
            {consensus_note}
        </div>
        """
        
    html = f"""
    <div class="rf-card" style="border-left: 4px solid {data['color']};">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 8px;">
            <div>
                <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1.2px; color: #808593; font-weight: 600;">
                    {lbl_heading}
                </div>
                <h2 style="margin: 4px 0 2px 0; color: #ffffff; font-size: 1.55rem; font-weight: 700; display: flex; align-items: center; gap: 8px;">
                    <span>{data['icon']}</span>
                    <span>{lang_data['name']}</span>
                </h2>
                {f'<div style="font-size: 0.88rem; color: #808593;">{top_class}</div>' if lang == 'ar' else ''}
            </div>
            <div>
                <span style="display: inline-block; background: {data['color']}18; color: {data['color']}; 
                             border: 1px solid {data['color']}44; padding: 5px 12px; border-radius: 6px; 
                             font-size: 0.75rem; font-weight: 700; letter-spacing: 0.5px;">
                    {lang_data['badge']}
                </span>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 8px; margin: 14px 0; 
                    background: #0d0e12; padding: 12px 14px; border-radius: 8px; border: 1px solid #1a1b22;">
            <div>
                <div style="font-size: 0.7rem; color: #808593; font-weight: 600; text-transform: uppercase;">{lbl_conf}</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: {data['color']}; margin-top: 2px;">{pct:.1f}%</div>
            </div>
            <div>
                <div style="font-size: 0.7rem; color: #808593; font-weight: 600; text-transform: uppercase;">{lbl_latency}</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #ff7849; margin-top: 2px;">{lat_ms:.0f} <span style="font-size: 0.75rem; font-weight: 500; color: #808593;">ms</span></div>
            </div>
            <div>
                <div style="font-size: 0.7rem; color: #808593; font-weight: 600; text-transform: uppercase;">{lbl_engine}</div>
                <div style="font-size: 0.92rem; font-weight: 700; color: #e5e7eb; margin-top: 6px;">{model_name}</div>
            </div>
        </div>

        <div style="font-size: 0.88rem; line-height: 1.55; color: #9ca3af; background: #0d0e12; 
                    padding: 12px 14px; border-radius: 8px; border: 1px solid #1a1b22;">
            <strong style="color: #ffffff;">{lbl_clinical}</strong> {lang_data['desc']}
        </div>

        {consensus_section}
    </div>
    """
    return html

# ============================================================================
# INTERACTIVE UI HANDLER
# ============================================================================
def predict_ui(image: Optional[Image.Image], mode: str, lang: str):
    if image is None:
        err_msg = "Please upload a fundus image before running analysis." if lang == "en" else "يرجى رفع صورة لقاع العين أولاً قبل بدء الفحص."
        raise gr.Error(err_msg)

    # Detect if Dual Mode is selected in either language
    is_dual = "Dual" in mode or "مقارنة" in mode

    if is_dual:
        probs_m1, lat_m1 = run_inference_raw(image, "General-Purpose (ResNet-50)")
        probs_m2, lat_m2 = run_inference_raw(image, "High-Sensitivity (EfficientNet-B5)")
        
        top1_m1 = max(probs_m1, key=probs_m1.get)
        top1_m2 = max(probs_m2, key=probs_m2.get)
        
        # Clinical consensus logic
        if top1_m1 == top1_m2:
            if lang == "en":
                note = f"""
                <span style="color: #10b981; font-weight: 700;">🤝 Model Consensus:</span>
                Both engines independently agreed on <strong>{top1_m1}</strong> 
                ({probs_m1[top1_m1]*100:.1f}% ResNet-50 · {probs_m2[top1_m2]*100:.1f}% EfficientNet-B5).
                """
            else:
                note = f"""
                <span style="color: #10b981; font-weight: 700;">🤝 توافق تشخيصي تام:</span>
                كلا النموذجين متفقان على أرجحية تشخيص <strong>{top1_m1}</strong> ({CLINICAL_DATA[top1_m1]['ar']['name']}) 
                بثقة {probs_m1[top1_m1]*100:.1f}% (ResNet) و {probs_m2[top1_m2]*100:.1f}% (EfficientNet).
                """
            primary_class = top1_m1
            primary_prob = probs_m1[top1_m1]
        else:
            if lang == "en":
                note = f"""
                <span style="color: #ff7849; font-weight: 700;">⚖️ Model Divergence Detected:</span>
                General model favors <strong>{top1_m1}</strong> ({probs_m1[top1_m1]*100:.1f}%), 
                while High-Sensitivity model flags <strong>{top1_m2}</strong> ({probs_m2[top1_m2]*100:.1f}%). 
                <em>Notice: Useful for alerting clinicians to rare high-risk cases like Hypertension.</em>
                """
            else:
                note = f"""
                <span style="color: #ff7849; font-weight: 700;">⚖️ تباين في التشخيص (Divergence):</span>
                النموذج العام يرجح <strong>{top1_m1}</strong> ({probs_m1[top1_m1]*100:.1f}%)، 
                بينما نموذج الحساسية العالية يرصد <strong>{top1_m2}</strong> ({probs_m2[top1_m2]*100:.1f}%). 
                <em>تنبيه: يفيد هذا التباين في لفت نظر الطبيب للحالات النادرة كضغط الدم.</em>
                """
            if top1_m2 in ['Hypertension', 'Glaucoma', 'Diabetic Retinopathy'] and probs_m2[top1_m2] > 0.30:
                primary_class = top1_m2
                primary_prob = probs_m2[top1_m2]
            else:
                primary_class = top1_m1
                primary_prob = probs_m1[top1_m1]

        total_lat = lat_m1 + lat_m2
        engine_label = "Dual Consensus (ResNet-50 + EfficientNet-B5)" if lang == "en" else "المحرك المزدوج التوافقي"
        card_html = build_clinical_card_html(primary_class, primary_prob, total_lat, engine_label, note, lang=lang)
        return card_html, probs_m1, probs_m2, gr.update(visible=True)

    else:
        model_key = "High-Sensitivity (EfficientNet-B5)" if "EfficientNet" in mode else "General-Purpose (ResNet-50)"
        probs, lat = run_inference_raw(image, model_key)
        top1 = max(probs, key=probs.get)
        cfg = MODEL_CONFIGS[model_key]
        card_html = build_clinical_card_html(top1, probs[top1], lat, cfg["name"], lang=lang)
        return card_html, probs, None, gr.update(visible=False)

def update_model_blurb(mode: str, lang: str) -> str:
    texts = UI_TEXTS[lang]
    if "Dual" in mode or "مقارنة" in mode:
        return texts["mode_blurb"]
    
    if "EfficientNet" in mode:
        return MODEL_CONFIGS["High-Sensitivity (EfficientNet-B5)"]["blurb_en" if lang == "en" else "blurb_ar"]
    return MODEL_CONFIGS["General-Purpose (ResNet-50)"]["blurb_en" if lang == "en" else "blurb_ar"]

def clear_all(lang: str):
    texts = UI_TEXTS[lang]
    default_mode = texts["mode_choices"][0]
    return (
        None, 
        default_mode, 
        texts["mode_blurb"], 
        texts["placeholder_html"], 
        None, 
        None, 
        gr.update(visible=False)
    )

# Language Toggle Function
def toggle_language(current_lang: str):
    new_lang = "ar" if current_lang == "en" else "en"
    t = UI_TEXTS[new_lang]
    
    # Header HTML
    header_html = f"""
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
        <div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.8rem;">👁️</span>
                <h1 style="font-size: 2rem; font-weight: 800; letter-spacing: -0.5px; margin: 0; color: #ffffff;">
                    {t['title']}
                </h1>
                <span class="rf-status-pill">{t['top_status']}</span>
            </div>
            <p style="margin: 4px 0 0 0; color: #808593; font-size: 0.95rem;">
                {t['subtitle']}
            </p>
        </div>
        <div style="display: flex; align-items: center; gap: 20px;">
            <div style="text-align: right; border-right: 1px solid #22242c; padding-right: 18px;">
                <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff;">{t['acc_val']}</div>
                <div style="font-size: 0.68rem; font-weight: 600; color: #808593; letter-spacing: 0.5px;">{t['acc_lbl']}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 1.25rem; font-weight: 800; color: #ff7849;">{t['ht_val']}</div>
                <div style="font-size: 0.68rem; font-weight: 600; color: #808593; letter-spacing: 0.5px;">{t['ht_lbl']}</div>
            </div>
        </div>
    </div>
    """

    card1_title = f"""<span style="color: #ff7849; font-weight: 700; margin-right: 6px;">{t['card1_num']}</span> <strong style="color: #ffffff; font-size: 1.1rem;">{t['card1_title']}</strong>"""
    card2_title = f"""<span style="color: #ff7849; font-weight: 700; margin-right: 6px;">{t['card2_num']}</span> <strong style="color: #ffffff; font-size: 1.1rem;">{t['card2_title']}</strong>"""

    return (
        new_lang,
        gr.update(value=t['lang_btn']),
        header_html,
        gr.update(label=t['accordion_title']),
        t['tradeoff_md'],
        card1_title,
        gr.update(label=t['upload_label']),
        gr.update(label=t['mode_label'], choices=t['mode_choices'], value=t['mode_choices'][0]),
        t['mode_blurb'],
        gr.update(value=t['btn_analyze']),
        gr.update(value=t['btn_reset']),
        f"""<div style="font-size: 0.78rem; color: #5a5e6b; text-align: center; margin-top: 8px;">{t['privacy_note']}</div>""",
        card2_title,
        t['placeholder_html'],
        gr.update(label=t['label_m1']),
        gr.update(label=t['label_m2']),
        f"""<div style="text-align: center; margin-top: 24px; padding-top: 14px; border-top: 1px solid #1a1b22; color: #525666; font-size: 0.8rem;">{t['footer_text']}</div>"""
    )

# Discover examples
EXAMPLES_DIR = "assets/samples"
def discover_samples():
    if not os.path.isdir(EXAMPLES_DIR):
        return []
    valid = ('.jpg', '.jpeg', '.png')
    try:
        return [[os.path.join(EXAMPLES_DIR, f)] for f in sorted(os.listdir(EXAMPLES_DIR)) if f.lower().endswith(valid)]
    except Exception:
        return []

SAMPLE_LIST = discover_samples()

# ============================================================================
# RETINAGRADE ULTRA-CLEAN MATTE OBSIDIAN DESIGN SYSTEM
# ============================================================================
CUSTOM_CSS = """
:root {
    --bg-base: #090a0d;
    --bg-surface: #131418;
    --bg-container: #0d0e12;
    --border-color: #1e2027;
    --border-hover: #2d303b;
    --text-primary: #ffffff;
    --text-secondary: #808593;
    --text-dim: #525666;
    --accent-orange: #ff7849;
    --accent-orange-hover: #ff895e;
    --accent-teal: #10b981;
}

body, .gradio-container {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif !important;
}

/* Outer Card Containers */
.rf-card {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.45) !important;
}

/* Header Banner */
#rf-top-header {
    background: #0d0e12 !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 14px !important;
    padding: 16px 22px !important;
    margin-bottom: 12px !important;
}

/* Status Pill */
.rf-status-pill {
    display: inline-flex;
    align-items: center;
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.25);
    color: #10b981;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 9999px;
    letter-spacing: 0.3px;
}

/* Language Toggle Button */
#rf-lang-btn {
    background: #14151a !important;
    border: 1px solid #2a2d38 !important;
    color: #ff7849 !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    padding: 6px 14px !important;
    transition: all 0.15s ease !important;
}
#rf-lang-btn:hover {
    background: #1e2027 !important;
    border-color: #ff7849 !important;
}

/* Analyze Button (RetinaGrade Coral-Orange) */
#rf-btn-analyze {
    background: var(--accent-orange) !important;
    color: #090a0d !important;
    font-weight: 700 !important;
    font-size: 0.98rem !important;
    border: none !important;
    border-radius: 9px !important;
    padding: 11px 20px !important;
    box-shadow: 0 4px 18px rgba(255, 120, 73, 0.32) !important;
    transition: all 0.15s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
#rf-btn-analyze:hover {
    background: var(--accent-orange-hover) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(255, 120, 73, 0.45) !important;
}

/* Reset Button */
#rf-btn-reset {
    background: #1a1b22 !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    transition: all 0.15s ease !important;
}
#rf-btn-reset:hover {
    background: #22242c !important;
    color: #ffffff !important;
}

/* Image Upload Container */
#rf-upload-box {
    background: var(--bg-container) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
}

/* Radio Selection */
.gr-radio .wrap {
    gap: 8px !important;
}

/* Scrollbars */
::-webkit-scrollbar { width: 7px; height: 7px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #323542; }
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    secondary_hue=gr.themes.colors.neutral,
    neutral_hue=gr.themes.colors.neutral,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#090a0d",
    background_fill_primary="#131418",
    background_fill_secondary="#0d0e12",
    border_color_primary="#1e2027",
    block_background_fill="#131418",
    block_border_color="#1e2027",
    block_label_text_color="#808593",
    body_text_color="#ffffff",
    body_text_color_subdued="#808593",
)

# ============================================================================
# APP LAYOUT (ENGLISH BY DEFAULT)
# ============================================================================
t_init = UI_TEXTS["en"]

with gr.Blocks(theme=THEME, css=CUSTOM_CSS, title="RetinaFlow — Retinal AI Classifier") as demo:

    # Global Language State
    lang_state = gr.State("en")

    # Top Control Bar (Language Toggle + Quick Links)
    with gr.Row():
        with gr.Column(scale=9):
            pass
        with gr.Column(scale=3, min_width=160):
            lang_toggle_btn = gr.Button(t_init["lang_btn"], elem_id="rf-lang-btn", size="sm")

    # Top Header Banner
    with gr.Column(elem_id="rf-top-header"):
        header_banner_html = gr.HTML(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
            <div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.8rem;">👁️</span>
                    <h1 style="font-size: 2rem; font-weight: 800; letter-spacing: -0.5px; margin: 0; color: #ffffff;">
                        {t_init['title']}
                    </h1>
                    <span class="rf-status-pill">{t_init['top_status']}</span>
                </div>
                <p style="margin: 4px 0 0 0; color: #808593; font-size: 0.95rem;">
                    {t_init['subtitle']}
                </p>
            </div>
            <div style="display: flex; align-items: center; gap: 20px;">
                <div style="text-align: right; border-right: 1px solid #22242c; padding-right: 18px;">
                    <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff;">{t_init['acc_val']}</div>
                    <div style="font-size: 0.68rem; font-weight: 600; color: #808593; letter-spacing: 0.5px;">{t_init['acc_lbl']}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.25rem; font-weight: 800; color: #ff7849;">{t_init['ht_val']}</div>
                    <div style="font-size: 0.68rem; font-weight: 600; color: #808593; letter-spacing: 0.5px;">{t_init['ht_lbl']}</div>
                </div>
            </div>
        </div>
        """)

    # Trade-off Advisory Collapsible Card
    with gr.Accordion(t_init["accordion_title"], open=False) as guide_accordion:
        guide_content = gr.Markdown(t_init["tradeoff_md"])

    # Main Two-Column Workflow (Matching RetinaGrade Mockup)
    with gr.Row(equal_height=False):
        
        # Left Panel: 01 Fundus image
        with gr.Column(scale=5):
            with gr.Column(elem_classes=["rf-card"]):
                card1_title = gr.HTML(
                    f"""<span style="color: #ff7849; font-weight: 700; margin-right: 6px;">{t_init['card1_num']}</span> <strong style="color: #ffffff; font-size: 1.1rem;">{t_init['card1_title']}</strong>"""
                )

                image_input = gr.Image(
                    type="pil", 
                    label=t_init["upload_label"],
                    elem_id="rf-upload-box"
                )

                model_selector = gr.Radio(
                    choices=t_init["mode_choices"],
                    value=t_init["mode_choices"][0],
                    label=t_init["mode_label"],
                    elem_id="rf-model-radio"
                )

                model_info = gr.Markdown(
                    t_init["mode_blurb"],
                    elem_id="rf-model-info"
                )

                with gr.Row():
                    submit_btn = gr.Button(t_init["btn_analyze"], variant="primary", elem_id="rf-btn-analyze", scale=3)
                    reset_btn = gr.Button(t_init["btn_reset"], elem_id="rf-btn-reset", scale=1)

                privacy_html = gr.HTML(
                    f"""<div style="font-size: 0.78rem; color: #5a5e6b; text-align: center; margin-top: 8px;">{t_init['privacy_note']}</div>"""
                )

            # Sample Gallery
            if SAMPLE_LIST:
                gr.Examples(
                    examples=SAMPLE_LIST,
                    inputs=image_input,
                    label=t_init["examples_label"]
                )

        # Right Panel: 02 Model assessment
        with gr.Column(scale=6):
            with gr.Column(elem_classes=["rf-card"]):
                card2_title = gr.HTML(
                    f"""<span style="color: #ff7849; font-weight: 700; margin-right: 6px;">{t_init['card2_num']}</span> <strong style="color: #ffffff; font-size: 1.1rem;">{t_init['card2_title']}</strong>"""
                )

                clinical_report_box = gr.HTML(
                    value=t_init["placeholder_html"],
                    elem_id="rf-clinical-box"
                )

                probs_m1_label = gr.Label(
                    num_top_classes=8, 
                    label=t_init["label_m1"],
                    elem_id="rf-probs-m1"
                )

                with gr.Column(visible=True) as secondary_col:
                    probs_m2_label = gr.Label(
                        num_top_classes=8, 
                        label=t_init["label_m2"],
                        elem_id="rf-probs-m2"
                    )

    # Invisible MCP bridge button
    mcp_bridge_btn = gr.Button(visible=False)
    mcp_bridge_btn.click(
        fn=classify,
        inputs=[image_input, model_selector],
        outputs=probs_m1_label,
        api_name="classify"
    )

    # Event Handlers: Language Toggle
    lang_toggle_btn.click(
        fn=toggle_language,
        inputs=[lang_state],
        outputs=[
            lang_state,
            lang_toggle_btn,
            header_banner_html,
            guide_accordion,
            guide_content,
            card1_title,
            image_input,
            model_selector,
            model_info,
            submit_btn,
            reset_btn,
            privacy_html,
            card2_title,
            clinical_report_box,
            probs_m1_label,
            probs_m2_label,
            footer_html := gr.HTML(
                f"""<div style="text-align: center; margin-top: 24px; padding-top: 14px; border-top: 1px solid #1a1b22; color: #525666; font-size: 0.8rem;">{t_init['footer_text']}</div>"""
            )
        ]
    )

    # Event Handlers: Model Selector Change
    model_selector.change(
        fn=update_model_blurb,
        inputs=[model_selector, lang_state],
        outputs=model_info,
        api_name=False
    )

    # Event Handlers: Submit
    submit_btn.click(
        fn=predict_ui,
        inputs=[image_input, model_selector, lang_state],
        outputs=[clinical_report_box, probs_m1_label, probs_m2_label, secondary_col],
        api_name=False
    )

    # Event Handlers: Reset
    reset_btn.click(
        fn=clear_all,
        inputs=[lang_state],
        outputs=[image_input, model_selector, model_info, clinical_report_box, probs_m1_label, probs_m2_label, secondary_col],
        api_name=False
    )

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": 7860,
    }
    import inspect
    if "mcp_server" in inspect.signature(demo.launch).parameters:
        launch_kwargs["mcp_server"] = True

    demo.launch(**launch_kwargs)
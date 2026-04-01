from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

def _is_low_confidence(confidence):
    try:
        return float(confidence) < 0.65
    except (TypeError, ValueError):
        return True

def _format_confidence(confidence):
    try:
        return f"{int(round(float(confidence) * 100))}%"
    except (TypeError, ValueError):
        return "Unknown"

def _quality_issues(blur_score, brightness, foreground_ratio, noise_level):
    issues = []
    # If scores are normalized 0-1, treat >0.5 blur/noise as poor.
    if blur_score > 0.5:
        issues.append("blurry")
    # Brightness expected in 0-1 range here; outside is too dark/bright.
    if not (0.3 <= brightness <= 0.8):
        if brightness < 0.3:
            issues.append("too dark")
        else:
            issues.append("too bright")
    # Foreground ratio too low means background dominates.
    if foreground_ratio < 0.5:
        issues.append("leaf/stem not clearly visible")
    if noise_level > 0.5:
        issues.append("noisy")
    return issues

@csrf_exempt
def diagnose_plant(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    
    # Extract inputs
    image_quality = data.get('image_quality', {})
    blur_score = image_quality.get('blur_score', 0)
    brightness = image_quality.get('brightness', 0.5)
    foreground_ratio = image_quality.get('foreground_ratio', 1)
    noise_level = image_quality.get('noise_level', 0)
    
    model_output = data.get('model_output', {})
    predicted_label = model_output.get('predicted_label', 'Unknown')
    confidence = model_output.get('confidence', 0)
    top_k_labels = model_output.get('top_k_labels', [])
    stage_prediction = model_output.get('stage_prediction', 'Unknown')
    
    visual_explanation = data.get('visual_explanation', {})
    grad_cam_summary = visual_explanation.get('grad_cam_summary', '')
    
    user_context = data.get('user_context', {})
    crop_type = user_context.get('crop_type', '')
    symptoms = user_context.get('symptoms', '')
    season_weather = user_context.get('season_weather', '')
    
    # Check image quality
    quality_issues = _quality_issues(blur_score, brightness, foreground_ratio, noise_level)
    if quality_issues:
        reason = ", ".join(quality_issues)
        return JsonResponse({
            "status": "reupload",
            "message": "Please upload a clearer image.",
            "reason": f"The image has issues: {reason}.",
            "advice": [
                "Take the photo in daylight",
                "Keep the camera steady",
                "Move closer so the leaf fills the frame",
                "Avoid shadows and keep the leaf unobstructed"
            ]
        })
    
    # If confidence is low or classes are close, avoid a strong claim
    if _is_low_confidence(confidence):
        possible_alternatives = top_k_labels[:3] if top_k_labels else []
        note = "Confidence is low, so this result is uncertain. Please upload a clearer photo."
        if grad_cam_summary:
            note += " The highlighted region was reviewed, but it is not strong enough for a firm diagnosis."
        return JsonResponse({
            "status": "ok",
            "disease": predicted_label,
            "stage": stage_prediction,
            "confidence": _format_confidence(confidence),
            "evidence": [
                "The model prediction is uncertain at this confidence level."
            ],
            "possible_alternatives": possible_alternatives,
            "suggestions": [
                "Retake the photo in good daylight",
                "Keep the leaf in sharp focus and fill most of the frame",
                "Capture both the front and back of the leaf if possible"
            ],
            "pesticide_guidance": [
                "If you plan to use any product, follow local agricultural advice",
                "Always follow the product label instructions"
            ],
            "note": note
        })
    
    # If image is good and confidence is strong, provide diagnosis
    disease = predicted_label
    stage = stage_prediction
    conf_percent = _format_confidence(confidence)
    
    evidence = []
    if symptoms:
        evidence.append(f"User reported symptoms: {symptoms}")
    if grad_cam_summary:
        evidence.append(f"Attention map supports: {grad_cam_summary}")
    if not evidence:
        evidence.append("Based on model prediction and visible signs.")
    
    possible_alternatives = top_k_labels[:2] if top_k_labels else []
    
    suggestions = [
        "Remove heavily infected leaves",
        "Avoid overhead watering",
        "Improve airflow around the plant",
        "Monitor nearby leaves for spread"
    ]
    
    pesticide_guidance = [
        "Use a locally recommended fungicide only if needed",
        "Follow label instructions and agricultural guidance"
    ]
    
    note = "The image quality looks usable for a prediction."
    if grad_cam_summary:
        note += " The highlighted region supports the model's decision."
    
    return JsonResponse({
        "status": "ok",
        "disease": disease,
        "stage": stage,
        "confidence": conf_percent,
        "evidence": evidence,
        "possible_alternatives": possible_alternatives,
        "suggestions": suggestions,
        "pesticide_guidance": pesticide_guidance,
        "note": note
    })

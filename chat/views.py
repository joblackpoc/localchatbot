import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .services import ask_ollama

@csrf_exempt
def api_chat(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    body = json.loads(request.body.decode("utf-8"))
    prompt = body.get("prompt", "")

    if len(prompt) < 1:
        return JsonResponse({"error": "Prompt required"}, status=400)

    reply = ask_ollama(prompt)
    return JsonResponse({"response": reply})

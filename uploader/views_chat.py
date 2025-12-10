import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

from .models import UploadedFile
from .utils import extract_text_from_file
from chat.rag import ingest_text

@csrf_exempt
def chat_upload(request):
    if request.method == "POST":
        file = request.FILES.get("file")
        obj = UploadedFile.objects.create(file=file)

        file_path = settings.MEDIA_ROOT / obj.file.name
        extracted = extract_text_from_file(str(file_path))

        obj.extracted_text = extracted
        obj.save()

        ingest_text(extracted)

        return JsonResponse({
            "filename": obj.file.name,
            "text_length": len(extracted)
        })

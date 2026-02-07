from transformers import OwlViTProcessor
try:
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    print("Has post_process_object_detection:", hasattr(processor, "post_process_object_detection"))
    print("Has image_processor:", hasattr(processor, "image_processor"))
    if hasattr(processor, "image_processor"):
        print("Image processor has post_process_object_detection:", hasattr(processor.image_processor, "post_process_object_detection"))
except Exception as e:
    print(e)

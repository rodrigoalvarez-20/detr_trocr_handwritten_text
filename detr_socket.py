from transformers import DetrForObjectDetection, DetrImageProcessor
import socket
import json
import torch
import os
import tempfile
import cv2
from supervision import Detections

DETR_MODEL = os.environ.get("DETR_MODEL", "/home/ralvarez22/Documentos/trocr_hand/trocr_llm/finetuned/detr/Ena/V_4")
TROCR_MODEL = os.environ.get("TROCR_MODEL", "/home/ralvarez22/Documentos/trocr_hand/trocr_llm/finetuned/trocr/Ena/V_1")

DEFAULT_CONFIDENCE_TH = 0.5
DEFAULT_IOU_TH = 0.2
SOCKET_PATH = os.environ.get("LLM_SOCKET_PATH", os.path.join(tempfile.gettempdir(), "detr_ocr"))
SOCKET_MAX_CONNECTIONS = 1

DEFAULT_HF_PATH = os.environ.get("HF_PATH", "")
DEFAULT_IMAGES_PATH = os.environ.get("IMAGES_PATH", "/home/ralvarez22/Documentos/trocr_hand/trocr_api/static")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

global img_proc
global detr_model
global server

try:
    os.unlink(SOCKET_PATH)
except OSError:
    if os.path.exists(SOCKET_PATH):
        raise


def _load_models():
    global img_proc
    global detr_model
    img_proc = DetrImageProcessor.from_pretrained(DETR_MODEL)
    detr_model = DetrForObjectDetection.from_pretrained(
        pretrained_model_name_or_path=DETR_MODEL,
        num_queries=100,
        ignore_mismatched_sizes=True
    ).to(DEVICE)


def _init_socket():
    global server
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(SOCKET_MAX_CONNECTIONS)

def generate_bboxes(image_path: str, confidence_th: float = None, iou_th: float = None):
    if not confidence_th:
        confidence_th = DEFAULT_CONFIDENCE_TH
    if not iou_th:
        iou_th = DEFAULT_IOU_TH
    image = cv2.imread(image_path)
    with torch.no_grad():
        inputs = img_proc(images=image, return_tensors='pt').to(DEVICE)
        outputs = detr_model(**inputs)
        target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
        results = img_proc.post_process_object_detection(
            outputs=outputs, 
            threshold=confidence_th, 
            target_sizes=target_sizes
        )[0]
    
    model_detections = Detections.from_transformers(transformers_results=results).with_nms(threshold=iou_th)

    scores = model_detections.confidence.tolist()
    labels = model_detections.class_id.tolist()
    boxes = model_detections.xyxy.tolist()
    
    return {
        "scores": scores,
        "labels": labels,
        "boxes": boxes
    }
    

def main():
    _load_models()
    _init_socket()
    print('Server is listening for incoming connections...')
    connection, client_address = server.accept()
    try:
        print('Connection from', str(connection).split(", ")[0][-4:])
        connection.send(b"\n>")
        # receive data from the client
        while True:
            data: bytes = connection.recv(2048)
            
            if len(data) == 0:
                continue
            
            print("RCV Data: {}".format(data.decode()))
            try:
                model_data = json.loads(data.decode())
                
                if model_data["type"].lower() == "detr":
                    conf_th = model_data.get("threshold", None)
                    full_image_path = os.path.join(DEFAULT_IMAGES_PATH, model_data["image"])
                    model_out = generate_bboxes(full_image_path, conf_th)
                    sck_response = {
                        "status": 200,
                        "message": "Se han generado correctamente los cuadros delimitadores",
                        "data": model_out
                    }
                    connection.send(json.dumps(sck_response).encode())
                    connection.send(b"\n>")
            except Exception as ex:
                    print("Bad JSON on data!")
                    print(ex)
                    ex_data = { "status": 500, "message": "Bad JSON format" }
                    connection.send(json.dumps(ex_data).encode())
                    connection.send(b"\n>")
    finally:
        # close the connection
        connection.close()
        # remove the socket file
        os.unlink(SOCKET_PATH)


if __name__ == "__main__":
    main()
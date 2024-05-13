import argparse
import cv2
import pickle
import glob
from fastapi import FastAPI, File, UploadFile, Form
from starlette.responses import Response
from face_d_api_class import AnimeFaceDetect

# FastAPIアプリケーションの初期化
app = FastAPI()

# AnimeFace_detectクラスのインスタンス生成
AF = AnimeFaceDetect()

# コマンドライン引数の設定
parser = argparse.ArgumentParser(description='顔検出サーバー設定')
parser.add_argument("--host", type=str, default="127.0.0.1", help="サービスを提供するIPアドレスを指定")
parser.add_argument("--port", type=int, default=50001, help="サービスを提供するポートを指定")
args = parser.parse_args()

# IPアドレスとポートの表示
print(f"IP={args.host}, PORT={args.port}")

# 基本的な顔検出を行うエンドポイント
@app.post("/face_det/")
async def face_det(image: UploadFile = File(...), confidence_level: float = Form(...)):
    file_contents = image.file.read()
    img_data = pickle.loads(file_contents)  # データを元の形式に復元
    dnum, rgb_img, predict_bbox, pre_dict_label_index, scores = AF.face_det(img_data, confidence_level)
    out_img = [dnum, predict_bbox, pre_dict_label_index, scores]
    frame_data = pickle.dumps(out_img)  # イメージデータを返送
    return Response(content=frame_data, media_type="application/octet-stream")

@app.post("/face_det_sq/")#img_data,confidence_level
def face_det_sq(image: UploadFile = File(...), confidence_level:float  = Form(...)): #file=OpenCV
    print("onfidence_level=",confidence_level)
    file_contents = image.file.read()
    img_data =(pickle.loads(file_contents))#元の形式にpickle.loadsで復元

    dnum,rgb_img, predict_bbox, pre_dict_label_index, scores = AF.face_det_sq(img_data,confidence_level)
    out_img  =[dnum, predict_bbox, pre_dict_label_index, scores]
    frame_data = pickle.dumps(out_img, 5)  # tx_dataはpklデータ、イメージのみ返送
    return Response(content=frame_data, media_type="application/octet-stream")

@app.post("/face_det_head/")#face_det_head(self,img_data,ratio,shift,confidence_level)
def face_det_head(image: UploadFile = File(...), ratio:float=Form(1.72), shift:float=Form(0.32),confidence_level:float=Form(0.5)): #file=OpenCV
    print("onfidence_level=",confidence_level)
    file_contents = image.file.read()
    img_data =(pickle.loads(file_contents))#元の形式にpickle.loadsで復元

    dnum,rgb_img, predict_bbox, pre_dict_label_index, scores = AF.face_det_head(img_data,ratio,shift,confidence_level)
    out_img  =[dnum, predict_bbox, pre_dict_label_index, scores]
    frame_data = pickle.dumps(out_img, 5)  # tx_dataはpklデータ、イメージのみ返送
    return Response(content=frame_data, media_type="application/octet-stream")

# メイン関数
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)





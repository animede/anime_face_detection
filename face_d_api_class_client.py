import cv2
import pickle
import requests

#       Return：　image,dnum,predict_bbox, pre_dict_label_index, scores
#                   omage= バウンディングbox付きイメージ
#                   dnum=  検出数（0.5以上のSCORE）
#                   bbox=  検出した顔のarrey[ＸＹ，Ｘ’Ｙ]のリスト
#                   label= 検出した顔の属性のリスト。0:girl,　1:girl_low,　2:man,　3:man_low
#                   score= 検出した顔のSCOREのリスト


class AnimeFaceDet:
    def __init__(self, url):
        # APIのURLをインスタンス変数として保存
        self.url = url

    def face_det(self, img_data, confidence_level):
        # 基本的な顔検出を行うメソッド
        data = {"confidence_level": confidence_level}
        send_img_data = pickle.dumps(img_data, 5)
        files = {"image": ("img.dat", send_img_data, "application/octet-stream")}
        response = requests.post(self.url + "/face_det/", files=files, data=data)
        if response.status_code == 200:
            received_data = pickle.loads(response.content)
            dnum, predict_bbox, pre_dict_label_index, scores = received_data
        else:
            print("エラーが発生しました")
            return None

        for i in range(dnum):
            box = predict_bbox[i]
            img_data = cv2.rectangle(img_data, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), thickness=3)
        return img_data, dnum, predict_bbox, pre_dict_label_index, scores

    def face_det_sq(self, img_data, confidence_level):
        # 正方形に調整して顔検出を行うメソッド
        data = {"confidence_level": confidence_level}
        send_img_data = pickle.dumps(img_data, 5)
        files = {"image": ("img.dat", send_img_data, "application/octet-stream")}
        response = requests.post(self.url + "/face_det_sq/", files=files, data=data)
        if response.status_code == 200:
            received_data = pickle.loads(response.content)
            dnum, predict_bbox, pre_dict_label_index, scores = received_data
        else:
            print("エラーが発生しました")
            return None

        for i in range(dnum):
            box = predict_bbox[i]
            img_data = cv2.rectangle(img_data, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), thickness=3)
        return img_data, dnum, predict_bbox, pre_dict_label_index, scores

    def face_det_head(self, img_data, ratio=1.68, shift=0.5, confidence_level=0.5):
        # 頭部を拡大して検出するメソッド
        data = {"ratio": ratio, "shift": shift, "confidence_level": confidence_level}
        send_img_data = pickle.dumps(img_data, 5)
        files = {"image": ("img.dat", send_img_data, "application/octet-stream")}
        response = requests.post(self.url + "/face_det_head/", files=files, data=data)
        if response.status_code == 200:
            received_data = pickle.loads(response.content)
            dnum, predict_bbox, pre_dict_label_index, scores = received_data
        else:
            print("エラーが発生しました")
            return None

        for i in range(dnum):
            box = predict_bbox[i]
            img_data = cv2.rectangle(img_data, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), thickness=3)
        return img_data, dnum, predict_bbox, pre_dict_label_index, scores



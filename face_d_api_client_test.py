import time
import os
import sys
import socket
import pickle
import argparse
import cv2
from  face_d_api_class_client import AnimeFaceDet


global host
global port
import requests

def main():

    parser = argparse.ArgumentParser(description='Anime-to-sketch test options.')
    parser.add_argument("--host",     type=str,  default="0.0.0.0",  help="サービスを提供するip アドレスを指定。")
    parser.add_argument("--port",   type=int,  default=50001,       help="サービスを提供するポートを指定。")
    parser.add_argument('--wg', default='./weights/ssd_best8.pth', type=str) 
    parser.add_argument('--dataroot', default='./image/a' ,type=str)
    parser.add_argument('--filename', default='./image/test2.jpg' ,type=str)
    parser.add_argument('--test', default=0, type=int)  
    parser.add_argument('--level', default=0.5, type=float)  

    args = parser.parse_args()
    level = args.level
    host  =args.host
    port  =args.port
    url="http://" + host+":"+str(port)
    print("IP=",host,"PORT=",port)
    print("url=",url)
    test=args.test

    AF=AnimeFaceDet(url)

    #****************  TEST START *****************
    #face_det
    if  test==2:
        imagefile = args.filename
        print("imagefile=",imagefile)
        img_data = cv2.imread(imagefile)
        confidence_level=0.4

        img_data, dnum, predict_bbox, pre_dict_label_index, scores =AF.face_det(img_data,confidence_level)
        print("dnum=",dnum,"bbox=",predict_bbox,"label=",pre_dict_label_index,"score=",scores)

        cv2.imshow("color",img_data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #face_det_sq
    if  test==3:
        imagefile = args.filename
        print("imagefile=",imagefile)
        img_data = cv2.imread(imagefile)
        confidence_level=0.5

        img_data, dnum, predict_bbox, pre_dict_label_index, scores =AF.face_det(img_data,confidence_level)
        print("dnum=",dnum,"bbox=",predict_bbox,"label=",pre_dict_label_index,"score=",scores)

        cv2.imshow("color",img_data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #face_det_head(self,img_data,ratio,shift,confidence_level)
    if  test==4:
        imagefile = args.filename
        print("imagefile=",imagefile)
        img_data = cv2.imread(imagefile)

        ratio=1.72
        shift=0.32
        confidence_level=0.5

        #Full val
        #img_data, dnum, predict_bbox, pre_dict_label_index, scores =AF.face_det_head(img_data, ratio, shift, confidence_level)
        #Use defolt val
        img_data, dnum, predict_bbox, pre_dict_label_index, scores =AF.face_det_head(img_data)
        print("dnum=",dnum,"bbox=",predict_bbox,"label=",pre_dict_label_index,"score=",scores)

        cv2.imshow("color",img_data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
################################# FUNCTION ############################################
# face_det(img_data,level) img_dataからアニメfaceを検出する　ssdのオリジナル出力を得る
#       img_data:　OpenCVのjpeg イメージデータ
#       level:     検出スコア：　　省略可能：デフォルト=0.5
#
#       Return：　image, dnum, predict_bbox, pre_dict_label_index, scores
#                   omage= バウンディングbox付きイメージ
#                   dnum=   result[0]   検出数（0.5以上のSCORE）
#                   bbox=   result[1]   検出した顔のarrey[ＸＹ，Ｘ’Ｙ]のリスト
#                   label=  result[2]   検出した顔の属性のリスト。0:girl,　1:girl_low,　2:man,　3:man_low
#                   score=  result[3]   検出した顔のSCOREのリスト
#
# face_det_sq(img_data,level) img_dataからアニメfaceを検出する　顎から眉毛あたりまでの正方形を得る。
#       img_data:　OpenCVのjpeg イメージデータ
#       level:     検出スコア：　　省略可能：デフォルト=0.5
#
#       Return：　image, dnum, predict_bbox, pre_dict_label_index, scores
#                   omage= バウンディングbox付きイメージ
#                   dnum=  検出数（0.5以上のSCORE）
#                   bbox=  検出した顔のarrey[ＸＹ，Ｘ’Ｙ]のリスト
#                   label= 検出した顔の属性のリスト。0:girl,　1:girl_low,　2:man,　3:man_low
#                   score= 検出した顔のSCOREのリスト
#
# face_det_head(img_data,ratio,shift,level) img_dataからアニメfaceを検出する
#       img_data:　OpenCVのjpeg イメージデータ
#       ratio:　   正方形データの拡大レシオ      　省略可能：デフォルト=1.72　ここの数字でバウンディングBOXに頭が入るように調整
#       shift:　   拡大した正方形データ上下シフト   省略可能：デフォルト=0.32　調整したBOXを上下位にずらすため　
#       level:     検出スコア：　　省略可能：デフォルト=0.5
#
#       Return：　image, dnum, predict_bbox, pre_dict_label_index, scores
#                   omage= バウンディングbox付きイメージ
#                   dnum=   result[0]   検出数（0.5以上のSCORE）
#                   bbox=   result[1]   検出した顔のarrey[ＸＹ，Ｘ’Ｙ]のリスト
#                   label=  result[2]   検出した顔の属性のリスト。0:girl,　1:girl_low,　2:man,　3:man_low
#                   score=  result[3]   検出した顔のSCOREのリスト
"""

if __name__ == "__main__":
    main()

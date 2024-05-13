import argparse
import glob
import torch
from PIL import Image
from utils.ssd_model import SSD
from utils.ssd_predict_show import SSDPredictShow
import cv2
import torchvision.transforms as transforms

def get_args():
    # コマンドライン引数をパースする関数
    parser = argparse.ArgumentParser(description='顔検出サービスの設定')
    parser.add_argument('--weights', default='./weights/ssd_best8.pth', type=str, help='重みファイルのパス')
    parser.add_argument('--dataroot', default='./image/', type=str, help='画像のルートディレクトリ')
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='サービスのIPアドレス')
    parser.add_argument('--port', type=int, default=50003, help='サービスのポート')
    return parser.parse_args()

def load_model(weights_path):
    # SSDモデルをロードする関数
    ssd_config = {
        'num_classes': 5,  # 背景クラスを含むクラス数
        'input_size': 300,  # 入力画像サイズ
        'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  # DBoxのアスペクト比
        'feature_maps': [38, 19, 10, 5, 3, 1],  # 特徴マップのサイズ
        'steps': [8, 16, 32, 64, 100, 300],  # DBoxのステップ
        'min_sizes': [21, 45, 99, 153, 207, 261],  # DBoxの最小サイズ
        'max_sizes': [45, 99, 153, 207, 261, 315],  # DBoxの最大サイズ
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],  # アスペクト比
    }
    model = SSD(phase="inference", cfg=ssd_config)
    model_weights = torch.load(weights_path, map_location={'cuda:0': 'cpu'})
    model.load_state_dict(model_weights)
    return model

def process_images(image_path, model, categories):
    # 画像処理を行う関数
    for image_file in glob.glob(image_path + '/*'):
        img_data = cv2.imread(image_file)
        ssd = SSDPredictShow(eval_categories=categories, net=model)
        rgb_img, predict_bbox, pre_dict_label_index, scores = ssd.ssd_predict(img_data)
        #print(rgb_img, predict_bbox, pre_dict_label_index, scores)
        return rgb_img, predict_bbox, pre_dict_label_index, scores

def main():
    # メイン関数
    args = get_args()
    model = load_model(args.weights)
    print('ネットワークが重みをロードして設定完了')
    voc_classes = ['girl', 'girl_low', 'man', 'man_low']  # カテゴリのリスト
    rgb_img, predict_bbox, pre_dict_label_index, scores = process_images(args.dataroot, model, voc_classes)
    print(rgb_img, predict_bbox, pre_dict_label_index, scores)

if __name__ == '__main__':
    main()

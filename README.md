# Anime_face_detection

厳選した学習データセットを用いて独自に学習したモデルを利用する高精度アニメ顔検出SSDです。

検出タグ　girl　boy　big　small　

コンフィデンスレベル

環境構築

### 参考

tkh作成後　リポジトリから作成したtkhへrequirements.txt　をコピーします

または仮想環境をアクティベート後にリポへ移動してpip install -r requirements.txt
```
python3.11 -m venv tkh
source tkh/bin/activate
cd tkh
pip install -r requirements.txt
```

checkポイントのダウンロード

https://huggingface.co/UZUKI/webapp1/tree/main

ssd_best8.pth　をダウンロードして　weightsに移動させる


ライセンス　MIT

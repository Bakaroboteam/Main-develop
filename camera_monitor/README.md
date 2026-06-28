ラズパイに繋いだUSBカメラの映像を確認できるプログラムです。

使用方法
1.VNC viewerやvscode上でSSH接続しラズパイのターミナルを開く
2.cd RasPike-ART/sdk/workspace でディレクトリをworkspaceに移動
3.Main-develop/camera_monitor/camera_monitor.sh で起動
4.起動するとターミナルにURLが出ます。パソコンのブラウザにURLを入力
  URLの例）http://(ラズパイで繋げているネットワークのIPアドレス):8000
5.ctrl+c で終了

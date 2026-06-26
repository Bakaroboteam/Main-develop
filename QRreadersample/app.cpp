#include "app.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>   // ソケット関連の関数を使うために必要
#include <sys/un.h>       // UNIXドメインソケット用のヘッダ
#include <sys/stat.h>

#include <Light.h> 
#include "spikeapi.h"

using namespace spikeapi;

// ==========================================
// ソケット通信用の定義
// ==========================================
// UNIXドメインソケット：同じマシン上のプロセス間通信に使う通信方式
// ファイルシステムのパスで識別されるため、異なるプロセス同士が通信できる

#define QR_SOCKET_PATH "/tmp/qr_socket"  // QRサーバーの所在地（ソケットファイルのパス）
#define BUFFER_SIZE 1024                  // 1回の通信で受け取るデータの最大サイズ
#define HANDSHAKE_MSG "ASP_QR_CLIENT"     // このプロセスのハンドシェイク送信メッセージ
#define HANDSHAKE_ACK "QR_SERVER"         // 期待する応答メッセージ

// ==========================================
// ハンドシェイク：プロセス間通信の初期化処理
// 2つのプロセスが正しく通信できることを確認する儀式のようなもの
// ==========================================

/**
 * QRサーバーへの接続を試みる
 * 
 * 説明：
 *   UNIXドメインソケットを作成して、QRサーバーに接続します。
 *   サーバーがまだ起動していない場合に備えて、3秒間にわたって
 *   リトライします（100msごとに10回試行）。
 * 
 * ソケットの流れ：
 *   1. socket() - ソケットを作成（電話を用意するようなもの）
 *   2. connect() - サーバーに接続（番号を入力して相手に電話をかける）
 * 
 * @return ソケットディスクリプタ（接続成功時は0以上、失敗時は-1）
 *         ディスクリプタ：リソースを管理するための整数ID
 */
int connect_to_qr_server() {
  // ステップ1：ソケット作成
  //   AF_UNIX    - UNIXドメインソケット（同じマシン上での通信）
  //   SOCK_STREAM - ストリーム型（確実な順序で受信できる通信）
  //   0          - プロトコル（自動選択）
  int sock = socket(AF_UNIX, SOCK_STREAM, 0);
  if (sock < 0) {
    perror("socket");  // エラー理由を表示
    return -1;
  }

  // ステップ2：サーバーアドレスの設定
  //   UNIXドメインソケットではファイルパスがアドレスになる
  struct sockaddr_un server_addr;
  memset(&server_addr, 0, sizeof(struct sockaddr_un));  // メモリをクリア
  server_addr.sun_family = AF_UNIX;                      // UNIXドメインソケット指定
  strncpy(server_addr.sun_path, QR_SOCKET_PATH, sizeof(server_addr.sun_path) - 1);

  // ステップ3：サーバーへの接続を試みる（リトライロジック）
  //   サーバーがまだ起動していないことを想定し、複数回試行
  for (int retry = 0; retry < 30; retry++) {
    if (connect(sock, (struct sockaddr *)&server_addr, sizeof(struct sockaddr_un)) == 0) {
      printf("Connected to QR server\n");
      return sock;  // 接続成功
    }
    if (retry < 29) {
      usleep(100000); // 100ms（= 0.1秒）待機してから次を試す
    }
  }

  printf("Failed to connect to QR server\n");
  close(sock);  // 接続失敗時はソケットを閉じる
  return -1;
}

/**
 * ハンドシェイクを実行する
 * 
 * 説明：
 *   ASPプロセスとQRサーバーが正しく相互認識できることを確認します。
 *   
 * 手順：
 *   1. このプロセスが "ASP_QR_CLIENT" メッセージを送信
 *   2. サーバーが "QR_SERVER" メッセージを返してくることを確認
 *   3. 両者が正しく応答したら通信を開始
 * 
 * @param sock ソケットディスクリプタ
 * @return 成功時は1、失敗時は0
 */
int perform_handshake(int sock) {
  // ステップ1：ハンドシェイクメッセージ送信
  //   send() - データをソケットを通じて送信
  //   strlen() - 文字列の長さを計算
  if (send(sock, HANDSHAKE_MSG, strlen(HANDSHAKE_MSG), 0) < 0) {
    perror("send handshake");
    return 0;
  }
  printf("Sent handshake message\n");

  // ステップ2：ハンドシェイク応答受信
  //   recv() - ソケットからデータを受信
  //   BUFFER_SIZE - 最大何バイト受け取るか
  char buffer[BUFFER_SIZE];
  int n = recv(sock, buffer, sizeof(buffer) - 1, 0);
  if (n < 0) {
    perror("recv handshake");
    return 0;
  }
  buffer[n] = '\0';  // 文字列の終端を明示的に設定（重要！）

  // ステップ3：応答メッセージが期待値と一致するか確認
  if (strcmp(buffer, HANDSHAKE_ACK) == 0) {
    printf("Handshake successful\n");
    return 1;
  }

  printf("Invalid handshake response: %s\n", buffer);
  return 0;
}

/**
 * メインタスク - ASP内で実行されるメイン処理
 * 
 * 説明：
 *   このタスクはASP（リアルタイムOS）の中で実行されます。
 *   QRサーバーにアクセスして、色データを受け取り、
 *   LED（Light）の色を制御する処理の中心です。
 * 
 * 処理手順：
 *   1. Lightクラスのインスタンス生成（LED制御準備）
 *   2. QRサーバーに接続
 *   3. ハンドシェイク実行
 *   4. メインループ：
 *      - QRコード（色文字列）を受信
 *      - 前回と異なる色なら、LightクラスのsetColorで変更
 *      - 100ms待機してから繰り返し
 */
void main_task(intptr_t unused) { 

  printf("Start!\n");

  // ==========================================
  // ステップ1：LED制御オブジェクトの準備
  // ==========================================
  // Light spikeapi：RaSpikeで提供される、ロボットのLED光源を操作するクラス
  // Light light;
  
  // ==========================================
  // ステップ2：QRサーバーへの接続試行
  // ==========================================
  int sock = connect_to_qr_server();
  if (sock < 0) {
    printf("Failed to connect to QR server. Running without QR input.\n");
    ext_tsk();  // タスク終了
    return;
  }

  // ==========================================
  // ステップ3：ハンドシェイク（初期設定）実行
  // ==========================================
  if (!perform_handshake(sock)) {
    printf("Handshake failed\n");
    close(sock);
    ext_tsk();  // タスク終了
    return;
  }

  // ==========================================
  // ステップ4：メインループ - QRコード受取と色変更
  // ==========================================
  // 前回受け取った色を記録（同じ色が連続する場合を検出するため）
  char buffer[BUFFER_SIZE] = "";

  while (true) {
    // QRコードプロセスからのデータを受信
    // recv() - ブロッキング受け取り（データ到着まで待つ）
    int n = recv(sock, buffer, sizeof(buffer) - 1, 0);
    
    if (n < 0) {
      perror("recv");
      break;  // エラー時はループを抜ける
    } else if (n == 0) {
      printf("QR server disconnected\n");
      break;  // サーバー切断時はループを抜ける
    }

    // 受け取ったデータを文字列として終端を設定
    buffer[n] = '\0';
    printf("Received QR data: %s\n", buffer);

    // CPU負荷軽減のため、100ms待機してから次のデータを受信
    // dly_tsk() - ASP提供のマイクロ秒単位の遅延関数
    // 100 * 1000 = 100,000マイクロ秒 = 100ms
    dly_tsk(100 * 1000);
  }

  // ==========================================
  // クリーンアップ
  // ==========================================
  close(sock);  // ソケットを閉じる
  ext_tsk();    // タスク終了
}

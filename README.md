# seisviewer2d

このリポジトリは 2D 地震データを表示するシンプルな Web アプリケーションです。ここでは主要なコードファイルの役割を説明します。

## ディレクトリ構成

```
seisviewer2d/
├── app/
│   ├── api/
│   │   └── endpoints.py
│   ├── utils/
│   │   └── utils.py
│   ├── static/
│   │   ├── index.html
│   │   └── api.js
│   └── main.py
├── Dockerfile
└── ruff.toml
```

## ファイル説明

### `app/main.py`
FastAPI アプリケーションのエントリーポイントです。静的ファイル(`static` ディレクトリ)を公開し、`api/endpoints.py` で定義されたエンドポイントを登録しています。

### `app/api/endpoints.py`
SEG-Y ファイルをアップロードしたり、指定されたキーでセクションを取得する API エンドポイントを提供します。アップロードされたファイルは `uploads` ディレクトリに保存され、`SegySectionReader` を用いてトレースを読み込みます。

### `app/utils/utils.py`
`SegySectionReader` クラスを定義します。SEG-Y ファイルからキーごとのトレースセクションを読み込む処理をカプセル化し、結果をキャッシュします。

### `app/static/index.html`
ブラウザ上でデータを表示するフロントエンドです。Plotly を用いて地震波形をプロットし、ファイルのアップロードやキー指定を行う UI を提供します。

### `app/static/api.js`
シンプルなサンプルコードとして `fetchAndPlot` 関数を定義しています。API からデータを取得してプロットするための例です。

### `Dockerfile`
開発用コンテナを構築するための設定が記述されています。必要な依存ライブラリのインストールやユーザー設定を行います。

### `ruff.toml`
コード整形・静的解析ツール Ruff の設定ファイルです。スタイルのルールや使用する Python バージョンなどを指定しています。

## 使い方
1. 依存ライブラリをインストール後、`uvicorn app.main:app --reload` を実行するとローカルサーバーが起動します。
2. ブラウザで `http://localhost:8000` にアクセスし、SEG-Y ファイルをアップロードして波形を表示できます。


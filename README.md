# マイナビ x SIGNATE Student Cup 2018 チュートリアル

## 概要
データサイエンス初心者のために、簡単なチュートリアルを書きました。

与えられたデータのみを用いて前処理、EDA、ランダムフォレストによる予測を行っています。

このnotebookを用いて予測した結果は、2018年9月27日18:00現在、LB上で0.25650のスコアを出しています。

コードは自由に改変していただいて構わないので、是非このノートブックを元にコンペティションにチャレンジしてみてください。

(ただし、最終的にはルール通りにきちんとtrain.py等を用意することを忘れないでください)

健闘を祈ります。


## 追記
2018/10/26に、J1, J2の全試合の情報をJ.LEAGUE Data Siteからスクレイピングするscriptを公開しました。

また、google検索を用いて、スタジアムの収容人数も自動で集めます。

各サイトへのアクセスは負荷の少ない時間帯に間隔を一秒以上開けたうえで実行してください。


各試合のhtmlはローカルに保存するため、初回の実行には4時間程度かかりますが、二回目以降の実行では6コア使って2分程度で終わります。


## 追記2
レポート提出時に提出したスクリプトの一部を公開しました。

公式から配布されたデータ、訓練済みのモデル、スクレイピングで取得した生のhtmlは含まれていません
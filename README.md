# Othello-AI
[機械学習練習] 2人・完全・零和ゲームAIを強化学習で作る練習

## 概要
このプロジェクトは、強化学習を用いてオセロ（リバーシ）のAIを作成する練習プロジェクトです。

## 開発フェーズ
1. **フェーズ1（完了）**: オセロが遊べる環境の構築
2. **フェーズ2（完了）**: 強化学習を用いたAIの作成
3. **フェーズ3（予定）**: AIの評価と改善

## 必要なライブラリ
```bash
pip install -r requirements.txt
```

主要なライブラリ:
- `numpy`: 数値計算
- `pygame`: GUI版ゲーム
- `torch`: 機械学習（PyTorch）※GPU対応
- `matplotlib`: 学習曲線の可視化

### GPU対応について
PyTorchがCUDA対応でインストールされている場合、自動的にGPUを使用します。
**GPU使用により学習が約3-5倍高速化します。**

**GPU版PyTorchのインストール（推奨）:**
```bash
# CUDA 11.8の場合
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1の場合
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU版のみの場合（GPU不要）
pip install torch
```

詳細は[PyTorch公式サイト](https://pytorch.org/)または[GPU使用ガイド](GPU_USAGE.md)を参照してください。

## 使い方

### オセロをプレイする（GUI版）🎮
**新機能**: pygameを使用したグラフィカルインターフェース

```bash
python -m Scripts.gui_game
```

または

```bash
python Scripts/gui_game.py
```

**特徴**:
- **マウスクリック**で駒を配置
- **グレーの円（枠のみ）**で有効な手を表示
- **緑色のボード**に黒白の駒を表示
- **ターン数**と**現在のプレイヤー**をUIで可視化（ゴールドでハイライト）
- **リアルタイムスコア**表示

プレイヤータイプを選択できます：
- **人間**: マウスクリックで着手
- **ランダムAI**: ランダムに着手を選択
- **貪欲AI**: 最も多く石を取れる手を選択
- **学習済みDQN AI**: 強化学習で訓練されたAI（フェーズ2で実装）

### オセロをプレイする（コンソール版）
人間同士、人間vsAI、AIvsAIでオセロをプレイできます。

```bash
python -m Scripts.game
```

または

```bash
python Scripts/game.py
```

プレイヤータイプを選択できます：
- **人間**: コンソールから手動で入力
- **ランダムAI**: ランダムに着手を選択
- **貪欲AI**: 最も多く石を取れる手を選択
- **学習済みDQN AI**: 強化学習で訓練されたAI（フェーズ2で実装）

### AIの学習

DQN（Deep Q-Network）を使用してAIを学習させることができます。

**基本的な使い方:**
```bash
python -m Scripts.train
```

**GPU使用を明示的に指定:**
```bash
# GPUを使用（CUDA対応PyTorchが必要）
python -m Scripts.train --device cuda

# CPUを使用
python -m Scripts.train --device cpu

# エピソード数も指定
python -m Scripts.train --device cuda --episodes 20000

# 対話なしで実行（自動化に便利）
python -m Scripts.train --device cuda --episodes 10000 --no-interactive
```

学習パラメータ:
- デフォルトエピソード数: 10,000（カスタマイズ可能）
- 学習アルゴリズム: Deep Q-Network (DQN)
- 自己対戦による学習
- **GPU使用時**: 学習速度が大幅に向上（約3-5倍高速）

学習済みモデルは `Models/` ディレクトリに保存されます。

### AIの評価

学習したAIを評価することができます。

**基本的な使い方:**
```bash
python -m Scripts.evaluate
```

**GPU使用を指定:**
```bash
# GPUで評価
python -m Scripts.evaluate --device cuda

# 評価ゲーム数も指定
python -m Scripts.evaluate --device cuda --games 200

# 対話なしで実行
python -m Scripts.evaluate --device cuda --no-interactive
```

評価内容:
- ランダムAIとの対戦（勝率を測定）
- 貪欲AIとの対戦（より強い相手との勝率）
- 平均獲得石数の計算

### デモンストレーション
AIエージェント同士の対戦デモと統計を実行できます。

```bash
python -m Scripts.demo
```

または

```bash
python Scripts/demo.py
```

## プロジェクト構成
```
Othello-AI/
├── Scripts/
│   ├── __init__.py       # パッケージ初期化
│   ├── board.py          # ボード管理とゲームロジック
│   ├── game.py           # コンソール版ゲームインターフェース
│   ├── gui_game.py       # pygame版GUIゲーム
│   ├── player.py         # プレイヤークラス（人間・AI）
│   ├── agents.py         # AIエージェント（ランダム・貪欲）
│   ├── dqn_agent.py      # DQNエージェント（強化学習）★NEW
│   ├── train.py          # 学習スクリプト ★NEW
│   ├── evaluate.py       # 評価スクリプト ★NEW
│   ├── demo.py           # デモスクリプト
│   └── self_play.py      # 自己対戦実行スクリプト
├── Models/               # 学習済みモデル保存先 ★NEW
├── Logs/                 # 学習ログ・結果保存先 ★NEW
├── Specification/        # 仕様書
│   └── General.md        # プロジェクト仕様書
├── ML_Explanation.md     # 機械学習の説明文書 ★NEW
├── requirements.txt      # 必要なライブラリ
└── README.md
```

## ゲームのルール
- ボードサイズ: 8×8
- 先手: 黒石（●）
- 後手: 白石（○）
- 相手の石を自分の石で挟むと、挟まれた石が自分の色に変わる
- 石を置いて相手の石を1つ以上挟める場所にのみ着手可能
- 着手可能な場所がない場合はパス
- 両プレイヤーが連続でパス、またはボードが埋まるとゲーム終了
- 終了時に石の数が多い方が勝利

## 今後の予定
- ~~Deep Q-Network (DQN)を用いたAIの実装~~ ✓ 完了
- ~~強化学習による学習機能の追加~~ ✓ 完了
- ~~AIの評価システムの構築~~ ✓ 完了
- AIの性能向上（より多くの学習エピソード、ハイパーパラメータ調整）
- より高度なアルゴリズム（AlphaZero風アプローチなど）の実装

## 機械学習について

実装した機械学習手法の詳細な説明は [`ML_Explanation.md`](ML_Explanation.md) を参照してください。

**主な内容**:
- DQN（Deep Q-Network）の基礎
- 実装の詳細とネットワーク構造
- 学習プロセスと結果
- 評価結果（対ランダムAI: 71%勝率）
- 今後の改善案

詳細な仕様については `Specification/General.md` を参照してください。

# ✅ SymPy で記号式を構築し、数値代入後に LightGBM で予測する方法

## 📘 前提：`func_predict(self, a, b, c)` は LightGBM による予測関数

```python
import sympy as sp
import numpy as np
import lightgbm as lgb

class MyModel:
    def __init__(self):
        # 仮の学習済みモデルを読み込み（実際には事前に学習）
        self.model = lgb.Booster(model_file="model.txt")

    def func_predict(self, a, b, c):
        data = np.array([[a, b, c]])
        predicted = self.model.predict(data)
        return predicted
```

---

## 🧩 シンボリック式を構築し、数値代入して予測する手順

```python
# 1. 記号の定義
a, b, c = sp.symbols("a b c")

# 2. 記号的な関数（評価しない）として func_predict を登録
func_symbol = sp.Function("func_predict", evaluate=False)

# 3. sympify を使って数式を記述
rhs_eq = "func_predict(a, b, c)"
locals_dict = {"a": a, "b": b, "c": c, "func_predict": func_symbol}
rhs_expr = sp.sympify(rhs_eq, locals=locals_dict)

print(f"構築された式: {rhs_expr}")
# 出力例: func_predict(a, b, c)

# 4. 値を代入して抽出（予測には使えない、単なる置換）
subs_expr = rhs_expr.subs({a: 1.2, b: 3.4, c: 5.6})
# → func_predict(1.2, 3.4, 5.6) という式になる（まだ数値ではない）

# 5. func_predict の定義を使って実際の予測を行う
model = MyModel()
if isinstance(subs_expr, sp.Function):
    args = subs_expr.args  # (1.2, 3.4, 5.6)
    result = model.func_predict(*args)
    print(f"予測結果: {result}")
else:
    raise ValueError("subs_expr は関数呼び出し式ではありません")
```

---

## 🔍 補足ポイント

| 処理                    | 説明                                            |
| --------------------- | --------------------------------------------- |
| `sympify()`           | 評価されない記号関数として `func_predict(a, b, c)` を式として保持 |
| `.subs()`             | 記号に数値を代入して `func_predict(1.2, 3.4, 5.6)` に変形  |
| `.args`               | 関数式の引数をタプルとして取り出す                             |
| `func_predict(*args)` | 実際に LightGBM モデルを使って予測値を得る                    |

---

## ✅ 結果

この流れにより、\*\*「SymPyで式を記号的に表現しつつ、あとで数値を入れて LightGBM による予測を行う」\*\*という処理を安全に実装できます。

必要であれば、複数の `rhs_expr` をループで処理してバッチ予測も可能です。

# 1. lightGBMモデルによる予測値を返すメソッドを数式の中に使いたい

## なぜエラーになるのか？
```
rhs_expr = sp.sympify(rhs_eq, locals=locals_dict)
```
この行の rhs_eq = "func_predict(a, b, c)" によって、sympify は文字列を評価しようとします。

でもそのとき a, b, c は sympy.Symbol であり、func_predict の中ではそれを np.array([[a, b, c]]) に変換し、さらに self.model.predict(...) に渡します。

➡ LightGBMの .predict() は 数値の ndarray しか受け取れないため、Symbol が入っていると 内部でエラーになります。

## 解決方法1：func_predict(a, b, c) を sympy の関数として形式的に扱う
以下のように、func_predict を直接実行するのではなく、「記号的な関数（UndefinedFunction）」として扱います。

### Function オブジェクトとして扱う方法
```
import sympy as sp

a, b, c = sp.symbols("a b c")
FuncPredict = sp.Function("func_predict")  # 記号的な関数として定義

rhs_expr = FuncPredict(a, b, c)
```
##  「評価を遅延させたい」のであれば…
SymPyに「func_predict = 実際のPython関数」として登録しつつも、それを sympify 内で評価させたくないなら、evaluate=False を使ったり、Function を明示的に使うのが正解です。

## 数値を代入して LightGBM で予測させたいときは？
```
# 記号的に構成した式
rhs_expr = FuncPredict(a, b, c)

# 数値代入
subs_dict = {a: 1.2, b: 3.4, c: 5.6}
evaluated_args = [subs_dict[sym] for sym in (a, b, c)]

# 自分で評価して呼び出す（数値を渡して実行）
result = self.func_predict(*evaluated_args)
```
## まとめ
| やりたいこと                                   | 方法                                       |
| ---------------------------------------- | ---------------------------------------- |
| `func_predict(a, b, c)` を **記号的に式に使いたい** | `sp.Function("func_predict")` を使って形式的に定義 |
| `func_predict` を **数値で呼び出したい**           | `subs_dict` に基づいて値を取り出し、明示的に関数を呼び出す      |
| `sympify` 内で関数を評価したいが、LightGBMに渡せない      | → そもそも `sympify` は不適（数式評価エンジンではない）       |


## 解決方法2：evaluate=False を使う方法

「evaluate=False を使う方法」とは、SymPyの数式を作るときに func_predict(a, b, c) を即実行せず、記号的な関数呼び出しとして保持するためのテクニックです。

今回の問題では、func_predict は LightGBM モデルを使って値を返す実関数なので、そのまま呼び出すとエラーになるわけです。
この評価を止めるには、sympy.Function を使いつつ evaluate=False を明示する方法があります。

### 解決方法：記号関数として保持する（evaluate=False）
```
import sympy as sp

a, b, c = sp.symbols("a b c")

# func_predict を未定義関数として定義（ここでは self.func_predict は使わない）
FuncPredict = sp.Function("func_predict", evaluate=False)

# 記号的な呼び出し式を作成（評価されない）
rhs_expr = FuncPredict(a, b, c)
```

### この方法の意味
FuncPredict(a, b, c) は、あくまで「記号的に func_predict(a, b, c) と書いてあるだけ」の式です。

実際には LightGBM の predict() は呼ばれません。

あとで .subs() で a, b, c に値を代入して、手動で func_predict を呼びたい場合に活用できます。


### もし sympify() に文字列を使う場合の例（evaluate=False 相当）
```
from sympy import symbols, sympify, Function

a, b, c = symbols("a b c")
locals_dict = {
    "a": a,
    "b": b,
    "c": c,
    "func_predict": Function("func_predict", evaluate=False)
}

rhs_eq = "func_predict(a, b, c)"
rhs_expr = sympify(rhs_eq, locals=locals_dict)
```


### その後、数値を代入して評価したい場合
SymPyの式から実行に使うには、自分で数値を取り出して func_predict() に渡します：
```
# 値を与える
subs_vals = {a: 1.0, b: 2.0, c: 3.0}
args = [subs_vals[s] for s in (a, b, c)]
result = self.func_predict(*args)
```

## まとめ
func_predict(a, b, c) を 記号関数として式に保持する場合
- Function("func_predict", evaluate=False) を使う

sympify() で評価させず式を構築する場合
- locals={"func_predict": Function(..., evaluate=False)}

あとで数値代入して実行する場合
- subs を使って値を取り出し、関数を呼び出す

# 2. SymPy で記号式を構築し、数値代入後に LightGBM で予測する方法

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

# 3. 複数のメソッド名に柔軟に対応したい
ympy.sympify() で構築した式に登場する文字列（例："self_func_predict"）に応じて、対応する self メソッド（例：self.func_predict）を呼び出す。
- メソッド名をパースし、存在するメソッドなら引数を渡し呼び出す

## 実装例（柔軟なメソッド呼び出し対応）
```
import sympy as sp
import numpy as np
import lightgbm as lgb

class MyModel:
    def __init__(self):
        self.model = lgb.Booster(model_file="model.txt")
    
    def func_predict(self, a, b, c):
        data = np.array([[a, b, c]])
        return self.model.predict(data)
    
    def func_other(self, a, b, c):
        return a + b + c  # 仮の別メソッド

    def evaluate_symbolic_expr(self, expr):
        if isinstance(expr, sp.Function):
            func_name = str(expr.func)  # 例: 'self_func_predict'
            if func_name.startswith("self_"):
                method_name = func_name.replace("self_", "")
                if hasattr(self, method_name):
                    method = getattr(self, method_name)
                    args = expr.args
                    return method(*args)
                else:
                    raise AttributeError(f"メソッド {method_name} は存在しません")
        raise TypeError("関数呼び出し式のみ評価可能です")

# --- 使用例 ---

a, b, c = sp.symbols("a b c")
func_symbol = sp.Function("self_func_predict", evaluate=False)
rhs_eq = "self_func_predict(a, b, c)"
locals_dict = {"a": a, "b": b, "c": c, "self_func_predict": func_symbol}
rhs_expr = sp.sympify(rhs_eq, locals=locals_dict)

# 数値を代入して関数呼び出し式に変換
subs_expr = rhs_expr.subs({a: 1.1, b: 2.2, c: 3.3})  # → self_func_predict(1.1, 2.2, 3.3)

# 実行
model = MyModel()
result = model.evaluate_symbolic_expr(subs_expr)
print(f"予測結果: {result}")
```

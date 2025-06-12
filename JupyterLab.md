# JupyterLab起動方法。PowerShellで。バックグラウンドで

```
PS C:\Users\ive\dev24\banana> Start-Process powershell -ArgumentList '-NoExit', '-Command', 'poetry run jupyter-lab *>> jupyterlab_0201c.log'
```

別のPowerShellが開き、						

Chromeのあたしいタブで Jupyter Launcher が開く						http://localhost:8888/lab

ログファイル jupyterlab_0201c.log にログが記録され始める						

PowerShellのプロンプトが空くので入力可能。

次のコマンドでもポート番号が分かる

```
PS C:\Users\ive\dev24\banana> poetry run jupyter-lab list
	Python Currently running servers:						
	http://localhost:8888/?token=1f40305607317cd3e9e5378 :: C:\Users\one\dev24\banana
```

## 止め方
```
PS C:\Users\ive\dev24\banana> poetry run jupyter-lab stop

	Python [JupyterServerStopApp] Shutting down server on 8888...						
	Could not stop server on 8888				このように表示されるが、実際は止まっている
```

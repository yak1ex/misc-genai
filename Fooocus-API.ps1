Set-Location $env:USERPROFILE\scoop\apps\stabilitymatrix\current\Data\Packages\Fooocus-API
. .\venv\Scripts\activate.ps1
python main.py --host $env:COMPUTERNAME.ToLower()

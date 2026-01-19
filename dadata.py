from dadata import Dadata

token = "0559d6b58671780f448e493b4e005b45268ecb60"
secret = "d1ed295b657caf5d8a5167c7d59f90f1b885daa6"
dadata = Dadata(token, secret)
result = dadata.clean("address", "москва сухонская 11")
print(result)
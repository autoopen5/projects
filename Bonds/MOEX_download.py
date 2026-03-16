import requests
import pandas as pd

URL = "https://iss.moex.com/iss/engines/stock/markets/bonds/securities.json"


def load_bonds():

    params = {
        "iss.meta": "off",
        "securities.columns": "SECID,SHORTNAME,MATDATE",
        "marketdata.columns": "SECID,LAST,YIELD"
    }

    r = requests.get(URL, params=params)
    data = r.json()

    securities = pd.DataFrame(
        data["securities"]["data"],
        columns=data["securities"]["columns"]
    )

    market = pd.DataFrame(
        data["marketdata"]["data"],
        columns=data["marketdata"]["columns"]
    )

    bonds = securities.merge(market, on="SECID")

    return bonds

secid = "RU000A10E5C4"

if __name__ == "__main__":
    bonds = load_bonds()
    bond = bonds[bonds["SECID"] == secid]
    # bonds = bonds.sort_values("YIELD", ascending=False)
    # print(bonds.head(20))
    # bonds.to_csv("moex_bonds.csv", index=False)
    print(bond)

"""Cost of each Azure backend, from Azure list prices and measured throughput. All arithmetic is here.

    python3 cost.py results/2026-10-12/bench-chat.json --out results/2026-10-12

Prices come from the public Azure Retail Prices API (https://prices.azure.com/api/retail/prices,
no sign-in) at run time and are written to OUT/prices.json. Throughput is the median tok/s of
each backend in a bench_chat.py JSON file, one request at a time; parallel figures, when
OUT/parallel-vm.json exists, come from compare.py run on the VM.

USD per million output tokens = USD per hour / (tok/s * 3600) * 1e6, so it assumes the VM is
busy for the whole hour. The Foundry models are billed per token and have no hourly price.
"""

import argparse
import datetime
import json
import os
import urllib.parse
import urllib.request

HOURS_PER_MONTH = 730
API = "https://prices.azure.com/api/retail/prices"
# backend -> Retail Prices API filter for its hourly meter (Linux, pay-as-you-go)
HOURLY = {
    "vm": "armSkuName eq 'Standard_NV36ads_A10_v5' and armRegionName eq 'eastus' and priceType eq 'Consumption' "
          "and meterName eq 'NV36ads A10 v5' and productName eq 'Virtual Machines NVadsA10v5 Series'",
}
# backend -> (input meter, output meter), USD per 1K tokens, global deployment in eastus
PER_TOKEN = {
    "foundry": ("armRegionName eq 'eastus' and meterName eq 'gpt 4.1 nano Inp glbl Tokens'",
                "armRegionName eq 'eastus' and meterName eq 'gpt 4.1 nano Outp glbl Tokens'"),
    "phi": ("armRegionName eq 'eastus' and productName eq 'Azure Phi Models' and meterName eq 'Phi-4-Mini-Input Tokens'",
            "armRegionName eq 'eastus' and productName eq 'Azure Phi Models' and meterName eq 'Phi-4-Mini-Output Tokens'"),
}


def price(flt: str) -> dict:
    url = API + "?" + urllib.parse.urlencode({"$filter": flt})
    with urllib.request.urlopen(url, timeout=60) as r:
        items = json.load(r)["Items"]
    if len(items) != 1:
        raise SystemExit(f"expected one price for {flt!r}, got {len(items)}")
    i = items[0]
    return {"meter": i["meterName"], "unit": i["unitOfMeasure"], "usd": i["retailPrice"], "region": i["armRegionName"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", help="bench_chat.py --json output")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    prices = {"fetched": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
              "hourly": {b: price(f) for b, f in HOURLY.items()},
              "per_token": {b: {"input": price(i), "output": price(o)} for b, (i, o) in PER_TOKEN.items()}}
    with open(os.path.join(a.out, "prices.json"), "w") as f:
        json.dump(prices, f, indent=2)
    bench = json.load(open(a.bench))
    tps = {s["backend"]: s["median_tokens_per_second"] for s in bench["summary"]}

    lines = [f"prices fetched {prices['fetched']}; bench {a.bench}", "",
             "One request at a time (bench_chat.py, 100-token answer)", "",
             "backend      tok/s  USD/hr  USD/month 24x7  USD / 1M output tokens"]
    for b, t in tps.items():
        if b in prices["hourly"]:
            hr = prices["hourly"][b]["usd"]
            lines.append(f"{b:<10} {t:>7}  {hr:>6.4f}  {hr * HOURS_PER_MONTH:>14.0f}  {hr / (t * 3600) * 1e6:>22.2f}")
        elif b in prices["per_token"]:
            p = prices["per_token"][b]
            lines.append(f"{b:<10} {t:>7}       -          0 idle  {p['output']['usd'] * 1000:>22.3f}"
                         f"  (per-token list price, input {p['input']['usd'] * 1000:.3f})")
    par = os.path.join(a.out, "parallel-vm.json")
    if os.path.exists(par) and "vm" in prices["hourly"]:
        hr = prices["hourly"]["vm"]["usd"]
        load = json.load(open(par))["load"]
        lines += ["", f"Requests in parallel on the VM (compare.py, {json.load(open(par))['settings']['load_tokens']}-token answers)",
                  "", "parallel    tok/s  USD / 1M output tokens"]
        for c, v in load.items():
            t = v["aggregate_tokens_per_second_median"]
            lines.append(f"{c:>8} {t:>8}  {hr / (t * 3600) * 1e6:>22.3f}")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(a.out, "cost.txt"), "w") as f:
        f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

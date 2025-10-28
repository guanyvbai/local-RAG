#!/usr/bin/env python3
import argparse, csv, json, os, sys, time
from typing import List, Dict, Any
import requests

NVD_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"
CPE_MATCH_API = "https://services.nvd.nist.gov/rest/json/cpematch/2.0"
API_KEY = os.getenv("NVD_API_KEY")
HEADERS = {
    "apiKey": API_KEY
} if API_KEY else {}

RATE_SLEEP = 1.2  # be nice to NVD


def fetch_cve(cve_id: str) -> Dict[str, Any]:
    params = {
        "cveId": cve_id
    }
    r = requests.get(NVD_API, params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()


def extract_keywords(cve_json: Dict[str, Any]) -> List[str]:
    """Very simple heuristic to derive keywords for CPE search from CVE data.
    Prefers vendor/product from CPEs listed in configurations if present; otherwise uses CVE title text tokens.
    """
    kws = set()
    try:
        vulns = cve_json.get("vulnerabilities", [])
        for v in vulns:
            cve = v.get("cve", {})
            # try configurations
            for node in (cve.get("configurations", {}).get("nodes", []) or []):
                for m in node.get("cpeMatch", []) or []:
                    cpe23 = m.get("criteria")
                    if cpe23:
                        parts = cpe23.split(":")
                        if len(parts) > 5:
                            vendor = parts[3]
                            product = parts[4]
                            if vendor and vendor != "*":
                                kws.add(vendor)
                            if product and product != "*":
                                kws.add(product)
        # fall back to descriptions
            descs = (cve.get("descriptions") or [])
            for d in descs:
                if d.get("lang") == "en":
                    for token in d.get("value", "").lower().replace('/', ' ').replace('\n',' ').split():
                        if token.isalpha() and len(token) > 2:
                            kws.add(token)
    except Exception:
        pass
    return list(kws)[:10]


def search_cpes_by_keywords(keywords: List[str]) -> List[Dict[str, Any]]:
    results = []
    for kw in keywords:
        params = {
            "keyword": kw
        }
        r = requests.get(CPE_MATCH_API, params=params, headers=HEADERS, timeout=60)
        if r.status_code == 429:
            time.sleep(RATE_SLEEP)
            r = requests.get(CPE_MATCH_API, params=params, headers=HEADERS, timeout=60)
        r.raise_for_status()
        data = r.json()
        for m in data.get("matches", []) or []:
            cpe23 = m.get("cpe23Uri")
            if cpe23:
                results.append({
                    "keyword": kw,
                    "cpe23Uri": cpe23,
                    "vulnerable": m.get("vulnerable"),
                })
        time.sleep(RATE_SLEEP)
    # dedupe by cpe23Uri
    seen = set()
    deduped = []
    for x in results:
        if x["cpe23Uri"] not in seen:
            seen.add(x["cpe23Uri"])
            deduped.append(x)
    return deduped


def main():
    parser = argparse.ArgumentParser(description="Map CVE IDs to candidate CPEs via NVD APIs.")
    parser.add_argument("cve_ids", help="Comma-separated CVE IDs")
    args = parser.parse_args()

    cve_ids = [x.strip() for x in args.cve_ids.split(',') if x.strip()]
    out_rows = []
    all_json = {}

    for cid in cve_ids:
        try:
            cve_json = fetch_cve(cid)
            all_json[cid] = cve_json
            kws = extract_keywords(cve_json)
            cpes = search_cpes_by_keywords(kws) if kws else []
            for c in cpes:
                out_rows.append({
                    "cve_id": cid,
                    **c
                })
        except Exception as e:
            out_rows.append({
                "cve_id": cid,
                "error": str(e)
            })

    # write json
    with open("cve_to_cpe.json", "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)

    # write csv
    with open("cve_to_cpe.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cve_id", "keyword", "cpe23Uri", "vulnerable", "error"])
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print("Wrote cve_to_cpe.json and cve_to_cpe.csv")

if __name__ == "__main__":
    main()

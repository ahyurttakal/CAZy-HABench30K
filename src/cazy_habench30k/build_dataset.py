#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UniProt-CAZy benchmark Benchmark Builder 

  python build_dataset.py \\
      --out UniProt-CAZy-benchmark-30k \\
      --target_total 30000 \\
      --min_per_family 100 \\
      --ce_min_per_family 20 \\
      --aa_min_per_family 20 \\
      --max_per_family 600 \\
      --priority_classes CE,AA
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import requests
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) UniProt-CAZy-benchmark-Builder/1.0"

CAZY_BASE = "https://www.cazy.org"
CAZY_CLASS_PAGES = {
    "GH": f"{CAZY_BASE}/Glycoside-Hydrolases.html",
    "GT": f"{CAZY_BASE}/GlycosylTransferase-family",
    "PL": f"{CAZY_BASE}/Polysaccharide-Lyases.html",
    "CE": f"{CAZY_BASE}/Carbohydrate-Esterases.html",
    "AA": f"{CAZY_BASE}/Auxiliary-Activities.html",
    "CBM": f"{CAZY_BASE}/Carbohydrate-Binding-Modules.html",
}

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"

FAM_TOKEN_RE = re.compile(r"\b(GH|GT|PL|CE|AA|CBM)\d+\b", re.IGNORECASE)

AA_STRICT = set("ACDEFGHIKLMNPQRSTVWY")
AA_WITH_X = set("ACDEFGHIKLMNPQRSTVWYX")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def parse_link_next(link_header: str) -> Optional[str]:
    if not link_header:
        return None
    m = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
    return m.group(1) if m else None


class HTTPClient:
    def __init__(self, timeout_s: int = 180, retries: int = 7):
        self.s = requests.Session()
        self.timeout_s = timeout_s
        self.retries = retries

    def get(self, url: str, *, params=None, headers=None, stream=False, allow_429=True) -> requests.Response:
        hdr = {
            "User-Agent": UA,
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }
        if headers:
            hdr.update(headers)

        last = None
        for i in range(self.retries):
            try:
                r = self.s.get(url, params=params, headers=hdr, timeout=self.timeout_s, stream=stream)
                if r.status_code == 429 and allow_429:
                    ra = r.headers.get("Retry-After")
                    wait = int(ra) if ra and ra.isdigit() else min(40, 3 + i * 3)
                    print(f"    [HTTP] 429 rate limit -> sleep {wait}s")
                    time.sleep(wait)
                    continue
                if r.status_code in (500, 502, 503, 504):
                    wait = min(40, 2 + i * 3)
                    print(f"    [HTTP] {r.status_code} server error -> retry in {wait}s ({i+1}/{self.retries})")
                    time.sleep(wait)
                    continue
                return r
            except Exception as e:
                last = e
                wait = min(40, 2 + i * 3)
                print(f"    [HTTP] error {e} -> retry in {wait}s ({i+1}/{self.retries})")
                time.sleep(wait)
        raise RuntimeError(f"GET failed for {url}. Last error: {last}")


def discover_families(client: HTTPClient) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for cls, url in CAZY_CLASS_PAGES.items():
        print(f"[DISCOVER] {cls} from {url}")
        r = client.get(url)
        if r.status_code >= 400:
            print(f"    WARN: {url} -> {r.status_code}")
            out[cls] = []
            continue
        toks = sorted({m.group(0).upper() for m in FAM_TOKEN_RE.finditer(r.text or "")})
        out[cls] = toks
        print(f"    found {len(toks)} families")
        time.sleep(0.3)
    return out


@dataclass
class QueryTemplate:
    name: str
    # function family -> query string
    build: callable


def candidate_query_templates() -> List[QueryTemplate]:
    """
    UniProt query language can vary; we try a few plausible templates and auto-select the one that returns results.
    """
    return [
        QueryTemplate(
            "database:cazy AND token",
            lambda fam: f'database:cazy AND {fam}',
        ),
        QueryTemplate(
            "database(type:cazy AND token)",
            lambda fam: f'database:(type:cazy {fam})',
        ),
        QueryTemplate(
            "database(type:cazy) AND token",
            lambda fam: f'database:(type:cazy) AND {fam}',
        ),
        QueryTemplate(
            "xref(cazy) AND token",
            lambda fam: f'(xref:cazy) AND {fam}',
        ),
        # very loose fallback: text contains token AND mentions CAZy (may be noisy)
        QueryTemplate(
            "text(CAZy) AND token (fallback)",
            lambda fam: f'cazy AND {fam}',
        ),
    ]


# [FIX-CE] CE ve AA için ek fallback sorgu şablonları.
# CAZy cross-reference sorguları CE/AA için yetersiz kalabilir çünkü:
#  - CE sekansları genellikle daha az karakterize, UniProt cross-ref eksik
#  - AA oksidoreduktazlar farklı veritabanlarında (CAT, BRENDA) listelenmiş olabilir
# Bu şablonlar sadece CE/AA family'leri için denenir.

# CE için UniProt'ta çalışan sorgu stratejileri:
#  - UniProt'ta CE family'leri "Carbohydrate esterase family X" şeklinde annotation altında geçer
#  - cc_function alanında family token aranabilir
#  - Swiss-Prot (reviewed) subset'i küçük ama güvenilir başlangıç noktası
CE_FALLBACK_TEMPLATES: List[QueryTemplate] = [
    # Strateji 1: UniProt'un cc_function/ft_binding alanlarında CE token ara
    QueryTemplate(
        "CE ft_description token",
        lambda fam: f'ft_description:{fam}',
    ),
    # Strateji 2: Protein adında "carbohydrate esterase" + family numarasını ara
    # CE1 → "carbohydrate esterase 1", CE4 → "carbohydrate esterase 4" vb.
    QueryTemplate(
        "CE protein name family",
        lambda fam: (
            f'name:"carbohydrate esterase {fam[2:]}"'
            if fam.upper().startswith("CE") and fam[2:].isdigit()
            else f'name:"{fam}"'
        ),
    ),
    # Strateji 3: reviewed:true ile Swiss-Prot kayıtları — küçük ama temiz
    QueryTemplate(
        "CE reviewed xref",
        lambda fam: f'database:cazy AND reviewed:true AND {fam}',
    ),
    # Strateji 4: taxonomy kısıtlamasız geniş metin arama (en geniş, en gürültülü)
    QueryTemplate(
        "CE text loose",
        lambda fam: f'("{fam}") AND (carbohydrate esterase OR acetylesterase OR feruloyl esterase)',
    ),
]

# AA için UniProt'ta çalışan sorgu stratejileri:
#  - AA9/AA10/AA11 → lytic polysaccharide monooxygenase (LPMO) — UniProt'ta iyi temsil
#  - AA1/AA2/AA3 → laccase/peroxidase/glucose oxidase — genel keyword'ler mevcut
#  - AA family'leri çoğunlukla "auxiliary activity" yerine enzim adıyla geçer
AA_FAMILY_KEYWORDS: Dict[str, str] = {
    "AA1":  "laccase",
    "AA2":  "manganese peroxidase OR lignin peroxidase OR versatile peroxidase",
    "AA3":  "glucose oxidase OR aryl alcohol oxidase OR cellobiose dehydrogenase",
    "AA4":  "vanillyl alcohol oxidase",
    "AA5":  "glyoxal oxidase OR galactose oxidase",
    "AA6":  "1,4-benzoquinone reductase",
    "AA7":  "glucooligosaccharide oxidase",
    "AA8":  "iron reductase",
    "AA9":  "lytic polysaccharide monooxygenase",
    "AA10": "lytic polysaccharide monooxygenase",
    "AA11": "lytic polysaccharide monooxygenase",
    "AA12": "pyrroloquinoline quinone",
    "AA13": "lytic polysaccharide monooxygenase",
    "AA14": "lytic polysaccharide monooxygenase",
    "AA15": "lytic polysaccharide monooxygenase",
    "AA16": "lytic polysaccharide monooxygenase",
    "AA17": "lytic polysaccharide monooxygenase",
}

def _aa_keyword(fam: str) -> str:
    """AA family için en iyi UniProt keyword ifadesini döner."""
    kw = AA_FAMILY_KEYWORDS.get(fam.upper(), "")
    if kw:
        return f'({kw})'
    # Bilinmeyen AA family: genel auxiliary activity
    return f'name:"auxiliary activity"'

AA_FALLBACK_TEMPLATES: List[QueryTemplate] = [
    # Strateji 1: Family'e özgü enzim adı ile ara (en hassas)
    QueryTemplate(
        "AA enzyme name keyword",
        lambda fam: f'{_aa_keyword(fam)} AND taxonomy_id:131567',
    ),
    # Strateji 2: reviewed:true Swiss-Prot + enzyme name
    QueryTemplate(
        "AA reviewed enzyme name",
        lambda fam: f'reviewed:true AND {_aa_keyword(fam)}',
    ),
    # Strateji 3: CAZy xref + enzyme name (kesişim)
    QueryTemplate(
        "AA xref + enzyme name",
        lambda fam: f'database:cazy AND {_aa_keyword(fam)}',
    ),
    # Strateji 4: ft_description'da family token
    QueryTemplate(
        "AA ft_description token",
        lambda fam: f'ft_description:{fam}',
    ),
]


def get_total_hits(client: HTTPClient, query: str) -> int:
    """
    Use size=0 to minimize payload; read x-total-results if present; otherwise count lines.
    """
    params = {"query": query, "format": "list", "size": "0"}
    r = client.get(UNIPROT_SEARCH, params=params)
    if r.status_code >= 400:
        return -1
    # UniProt commonly returns x-total-results
    xtr = r.headers.get("x-total-results") or r.headers.get("X-Total-Results")
    if xtr and xtr.isdigit():
        return int(xtr)
    # fallback: if size=0 ignored, count accessions
    txt = (r.text or "").strip()
    if not txt:
        return 0
    return len(txt.splitlines())


def get_hits_with_fallback(
    client: HTTPClient,
    fam: str,
    cls: str,
    primary_tpl: "QueryTemplate",
    min_hits: int,
) -> Tuple[int, "QueryTemplate"]:
    """
    [FIX-CE/AA] CE ve AA family'leri için birincil sorgu yeterli hit vermezse
    class'a özel fallback şablonları dener. İlk min_hits'i geçen sonucu döner.
    Geçen yoksa en yüksek hit'li fallback'i döner.
    Diğer class'lar için sadece birincil şablonu kullanır.
    """
    primary_hits = get_total_hits(client, primary_tpl.build(fam))
    if primary_hits >= min_hits:
        return primary_hits, primary_tpl

    # CE / AA için fallback dene
    fallbacks: List[QueryTemplate] = []
    if cls == "CE":
        fallbacks = CE_FALLBACK_TEMPLATES
    elif cls == "AA":
        fallbacks = AA_FALLBACK_TEMPLATES

    if not fallbacks:
        return max(primary_hits, 0), primary_tpl

    print(f"    [{cls}] {fam}: primary hits={primary_hits} < {min_hits} → fallback sorgular deneniyor...")

    best_hits = max(primary_hits, 0)
    best_tpl  = primary_tpl

    for fb_tpl in fallbacks:
        time.sleep(0.2)
        try:
            fb_hits = get_total_hits(client, fb_tpl.build(fam))
        except Exception as e:
            print(f"    [{cls}] {fam}: fallback '{fb_tpl.name}' hata → {e}")
            continue

        if fb_hits < 0:
            continue

        print(f"    [{cls}] {fam}: fallback '{fb_tpl.name}' → {fb_hits} hits")

        if fb_hits > best_hits:
            best_hits = fb_hits
            best_tpl  = fb_tpl

        # İlk yeterli sonuçta dur
        if fb_hits >= min_hits:
            return best_hits, best_tpl

    return best_hits, best_tpl


def autodetect_template(client: HTTPClient, probe_families: List[str]) -> QueryTemplate:
    """
    Pick a UniProt query template for CAZy families.

    benchmark-friendly rule:
    1) Prefer *CAZy cross-reference* specific templates (database/xref) if they produce any hits.
    2) Only fall back to loose text queries if cross-reference templates fail.

    Probe aileler GH/GT/CBM'den seçilir — bunlar UniProt'ta en iyi kapsanmış ailelerdir
    ve template seçimi için güvenilir pozitif örnekler sağlar. CE/AA fallback'leri
    ayrı get_hits_with_fallback() mekanizması ile ele alınır.
    """
    templates = candidate_query_templates()

    preferred_order = [
        "database:cazy AND token",
        "xref(cazy) AND token",
        "database(type:cazy AND token)",
        "database(type:cazy) AND token",
        "text(CAZy) AND token (fallback)",
    ]

    # Probe için CE/AA değil, UniProt'ta sağlam kapsanan GH/GT/CBM ailelerini kullan
    # CE1 veya AA9 kullanmak template seçimini bozabilir (hit sayısı düşük → yanlış template seçimi)
    safe_probes = [f for f in probe_families
                   if re.match(r'^(GH|GT|CBM)\d+$', f, re.IGNORECASE)]
    # Yeterli güvenli probe yoksa tümünü kullan
    effective_probes = (safe_probes[:6] if len(safe_probes) >= 3 else probe_families[:6])

    print(f"[UNIPROT] Template tespiti başlıyor (probe aileler: {effective_probes})...")

    scores: Dict[str, Tuple[int, int]] = {}
    tpl_by_name: Dict[str, QueryTemplate] = {t.name: t for t in templates}

    for tpl in templates:
        hits_sum = 0
        ok = 0
        for fam in effective_probes:
            q = tpl.build(fam)
            h = get_total_hits(client, q)
            if h is not None and h >= 0:
                ok += 1
                hits_sum += h
            time.sleep(0.15)

        scores[tpl.name] = (hits_sum, ok)
        print(f"    template='{tpl.name}'  probe_hits_sum={hits_sum}  ok={ok}/{len(effective_probes)}")

    for name in preferred_order:
        if name in scores:
            hits_sum, ok = scores[name]
            if ok > 0 and hits_sum > 0:
                chosen = tpl_by_name[name]
                print(f"[UNIPROT] Seçilen template (tercihli): {chosen.name}  (probe_sum={hits_sum}, ok={ok})")
                return chosen

    best = None
    best_hits = -1
    for name, (hits_sum, ok) in scores.items():
        if ok <= 0:
            continue
        if hits_sum > best_hits:
            best_hits = hits_sum
            best = tpl_by_name[name]

    if not best or best_hits <= 0:
        raise SystemExit(
            "ERROR: Çalışan UniProt query template bulunamadı.\n"
            "candidate_query_templates() fonksiyonunu güncel UniProt sözdizimi için güncelleyin."
        )

    print(f"[UNIPROT] Seçilen template (fallback-en iyi): {best.name}  (probe_sum={best_hits})")
    return best

def fetch_fasta_pages(client: HTTPClient, query: str, max_records: int, page_size: int = 500) -> str:
    """
    Download FASTA for a query using cursor pagination (Link: rel='next').
    Stops after reaching max_records (approx; we stop by counting '>' headers).
    """
    fasta_chunks: List[str] = []
    url = UNIPROT_SEARCH
    params = {"query": query, "format": "fasta", "size": str(page_size)}

    n_headers = 0
    while True:
        r = client.get(url, params=params)
        if r.status_code >= 400:
            break
        txt = r.text or ""
        if txt.strip():
            fasta_chunks.append(txt)
            n_headers += txt.count("\n>")
            if txt.startswith(">"):
                n_headers += 1  # first header
        link = r.headers.get("Link", "")
        next_url = parse_link_next(link)
        if not next_url:
            break
        if n_headers >= max_records:
            break
        url = next_url
        params = None
        time.sleep(0.2)

    return "".join(fasta_chunks)


def clean_and_label_fasta(
    fasta_text: str,
    family: str,
    min_len: int,
    max_len: int,
    allow_x: bool,
) -> List[SeqRecord]:
    """
    Parse FASTA, clean sequences, and tag record.id with original accession.
    We label with the requested family (not by header parsing), because UniProt headers vary.
    """
    tmp = "_tmp_uniprot_family.fasta"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(fasta_text)

    alpha = AA_WITH_X if allow_x else AA_STRICT
    out: List[SeqRecord] = []
    for rec in SeqIO.parse(tmp, "fasta"):
        rid = (rec.id or "").strip()
        if not rid:
            continue
        seq = str(rec.seq).upper().replace(" ", "").replace("\n", "").replace("\r", "")
        if not (min_len <= len(seq) <= max_len):
            continue
        if any(ch not in alpha for ch in seq):
            continue
        # Keep accession as ID; store label in description for traceability
        rr = SeqRecord(Seq(seq), id=rid, name=rid, description=f"{rid} | {family}")
        out.append(rr)

    try:
        os.remove(tmp)
    except Exception:
        pass
    # de-dup by id within family
    uniq = {r.id: r for r in out}
    return list(uniq.values())


def random_split(records: List[SeqRecord], test_ratio: float, seed: int) -> Tuple[List[SeqRecord], List[SeqRecord]]:
    rnd = random.Random(seed)
    recs = records[:]
    rnd.shuffle(recs)
    k = int(round(len(recs) * (1 - test_ratio)))
    return recs[:k], recs[k:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target_total", type=int, default=30000)
    ap.add_argument("--min_per_family", type=int, default=100,
                    help="GH/GT/PL/CBM için minimum UniProt hit sayısı")

    # [FIX-CE/AA] Per-class eşik override'ları
    ap.add_argument("--ce_min_per_family", type=int, default=20,
                    help="CE family'leri için minimum hit (CE az karakterize → düşük tut)")
    ap.add_argument("--aa_min_per_family", type=int, default=20,
                    help="AA family'leri için minimum hit (AA az karakterize → düşük tut)")
    ap.add_argument("--pl_min_per_family", type=int, default=50,
                    help="PL family'leri için minimum hit")

    ap.add_argument("--max_per_family", type=int, default=600,
                    help="Family başına maksimum sekans (denge için)")
    ap.add_argument("--ce_max_per_family", type=int, default=400,
                    help="CE için ayrı cap (UniProt coverage düşük olabilir)")
    ap.add_argument("--aa_max_per_family", type=int, default=400,
                    help="AA için ayrı cap")

    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_len", type=int, default=50)
    ap.add_argument("--max_len", type=int, default=3000)
    ap.add_argument("--allow_x", action="store_true")
    ap.add_argument("--page_size", type=int, default=500)

    # [NEW-2] Round-robin öncelik class'ları
    ap.add_argument("--priority_classes", type=str, default="CE,AA",
                    help="Virgüllü class listesi — round-robin'de ÖNCE bu class'lar doldurulur. "
                         "CE ve AA'nın target'a ulaşılmadan dışarıda kalmasını önler.")

    args = ap.parse_args()

    # Per-class min_per_family haritası
    class_min: Dict[str, int] = {
        "GH":  args.min_per_family,
        "GT":  args.min_per_family,
        "PL":  args.pl_min_per_family,
        "CE":  args.ce_min_per_family,
        "AA":  args.aa_min_per_family,
        "CBM": args.min_per_family,
    }
    # Per-class max_per_family haritası
    class_max: Dict[str, int] = {
        "GH":  args.max_per_family,
        "GT":  args.max_per_family,
        "PL":  args.max_per_family,
        "CE":  args.ce_max_per_family,
        "AA":  args.aa_max_per_family,
        "CBM": args.max_per_family,
    }

    priority_classes = [c.strip().upper() for c in args.priority_classes.split(",") if c.strip()]
    # Round-robin sırası: önce priority class'lar, sonra kalanlar
    ALL_CLASSES = ["GH", "GT", "PL", "CE", "AA", "CBM"]
    non_priority = [c for c in ALL_CLASSES if c not in priority_classes]
    cls_order_full = priority_classes + non_priority  # CE, AA önce

    ensure_dir(args.out)
    inter = os.path.join(args.out, "intermediate")
    ensure_dir(inter)

    client = HTTPClient(timeout_s=180, retries=7)

    # 1) CAZy family discovery
    fams_by_class = discover_families(client)
    all_fams = sorted({f for lst in fams_by_class.values() for f in lst})
    write_text(os.path.join(inter, "families_discovered.txt"), "\n".join(all_fams))
    print(f"[A] Keşfedilen family sayısı: {len(all_fams)}")
    for cls in ALL_CLASSES:
        n = len(fams_by_class.get(cls, []))
        print(f"    {cls}: {n} family")

    # 2) Auto-detect UniProt query template
    probe = []
    for candidate in ["GH13", "GH5", "GT2", "AA10", "CBM50", "CE1", "PL1"]:
        if candidate in all_fams:
            probe.append(candidate)
    probe = probe or all_fams[:10]
    tpl = autodetect_template(client, probe)

    # 3) Estimate hit counts — CE/AA için fallback ile
    print("[B] Her family için UniProt hit sayısı tahmin ediliyor...")
    print(f"    Per-class min eşikler: { {c: class_min[c] for c in ALL_CLASSES} }")
    fam_hits: List[Tuple[str, int, str, str]] = []  # (fam, hits, class, template_used)

    for cls in ALL_CLASSES:
        fams_in_cls = fams_by_class.get(cls, [])
        min_h = class_min[cls]
        print(f"\n  [{cls}] {len(fams_in_cls)} family, min_per_family={min_h}")
        for fam in fams_in_cls:
            # [FIX-CE/AA] fallback-aware hit sayımı
            h, used_tpl = get_hits_with_fallback(client, fam, cls, tpl, min_hits=min_h)
            if h is None or h < 0:
                h = 0
            fam_hits.append((fam, h, cls, used_tpl.name))
            time.sleep(0.05)

    df_hits = pd.DataFrame(fam_hits, columns=["family", "hits", "class", "template_used"])
    df_hits = df_hits.sort_values(["class", "hits"], ascending=[True, False])
    df_hits.to_csv(os.path.join(args.out, "family_hits_uniprot.csv"), index=False)

    # Per-class min eşik uygula
    eligible_parts = []
    for cls in ALL_CLASSES:
        min_h = class_min[cls]
        part = df_hits[(df_hits["class"] == cls) & (df_hits["hits"] >= min_h)].copy()
        eligible_parts.append(part)
        n_total_cls = (df_hits["class"] == cls).sum()
        print(f"  [{cls}] eligible: {len(part)}/{n_total_cls} family (hits >= {min_h})")

    eligible = pd.concat(eligible_parts, ignore_index=True)
    print(f"\n[B] Toplam eligible family: {len(eligible)}")
    if eligible.empty:
        raise SystemExit(
            "ERROR: Hiçbir eligible family bulunamadı.\n"
            "CE/AA için --ce_min_per_family 10 veya --aa_min_per_family 10 deneyin.\n"
            "family_hits_uniprot.csv dosyasını inceleyerek gerçek hit sayılarını görün."
        )

    # CE/AA özet uyarıları
    for cls in ["CE", "AA"]:
        n_elig = (eligible["class"] == cls).sum()
        if n_elig == 0:
            print(f"\n  [WARN] {cls} class'ından hiç eligible family yok!")
            print(f"         family_hits_uniprot.csv'de '{cls}' satırlarını kontrol edin.")
            print(f"         --{cls.lower()}_min_per_family değerini düşürmeyi deneyin.")
        elif n_elig < 5:
            print(f"\n  [WARN] {cls} class'ından yalnızca {n_elig} family eligible — coverage düşük.")

    # 4) [NEW-1] Priority-first round-robin download
    rnd = random.Random(args.seed)
    cls_to_fams: Dict[str, List[Tuple[str, int, str]]] = {c: [] for c in ALL_CLASSES}
    # template_used bilgisini de sakla
    tpl_map: Dict[str, str] = {}
    for _, row in eligible.iterrows():
        cls_to_fams[row["class"]].append((row["family"], int(row["hits"]), row["template_used"]))
        tpl_map[row["family"]] = row["template_used"]
    for cls in ALL_CLASSES:
        cls_to_fams[cls].sort(key=lambda x: x[1], reverse=True)

    chosen_records: List[SeqRecord] = []
    labels_rows: List[dict] = []
    seen_ids: Set[str] = set()

    print(f"\n[C] Sekans indirme başlıyor (priority: {priority_classes})...")
    idx = {c: 0 for c in ALL_CLASSES}

    # [NEW-1] İki aşamalı round-robin:
    # Aşama 1: Sadece priority class'lar (CE, AA) — kendi max'larına ulaşana kadar
    print(f"\n  [Aşama 1] Priority class'lar: {priority_classes}")
    for cls in priority_classes:
        fam_list = cls_to_fams.get(cls, [])
        max_pf = class_max[cls]
        for fam, hits, used_tpl_name in fam_list:
            remain = args.target_total - len(chosen_records)
            if remain <= 0:
                break
            want = min(max_pf, remain)
            # Doğru template'i yeniden oluştur
            all_tpls = candidate_query_templates() + CE_FALLBACK_TEMPLATES + AA_FALLBACK_TEMPLATES
            tpl_to_use = next((t for t in all_tpls if t.name == used_tpl_name), tpl)
            q = tpl_to_use.build(fam)
            print(f"    [{cls}] {fam} (template={used_tpl_name}): hits={hits} → fetch {want}")
            fasta_txt = fetch_fasta_pages(client, q, max_records=want, page_size=args.page_size)
            recs = clean_and_label_fasta(fasta_txt, fam, args.min_len, args.max_len, args.allow_x)
            kept = 0
            for r in recs:
                if r.id in seen_ids:
                    continue
                seen_ids.add(r.id)
                chosen_records.append(r)
                labels_rows.append({"id": r.id, "family": fam, "class": cls})
                kept += 1
                if len(chosen_records) >= args.target_total:
                    break
            idx[cls] += 1
            print(f"        {cls} kept {kept} (total={len(chosen_records)})")
            time.sleep(0.2)

    # Aşama 2: Tüm class'lar round-robin (priority dahil, devam eder)
    print(f"\n  [Aşama 2] Tam round-robin sırası: {cls_order_full}")
    while len(chosen_records) < args.target_total:
        progressed = False
        for cls in cls_order_full:
            fam_list = cls_to_fams.get(cls, [])
            if idx[cls] >= len(fam_list):
                continue
            fam, hits, used_tpl_name = fam_list[idx[cls]]
            idx[cls] += 1
            remain = args.target_total - len(chosen_records)
            if remain <= 0:
                break
            max_pf = class_max[cls]
            want = min(max_pf, remain)
            all_tpls = candidate_query_templates() + CE_FALLBACK_TEMPLATES + AA_FALLBACK_TEMPLATES
            tpl_to_use = next((t for t in all_tpls if t.name == used_tpl_name), tpl)
            q = tpl_to_use.build(fam)
            print(f"    [{cls}] {fam}: hits={hits} → fetch {want}")
            fasta_txt = fetch_fasta_pages(client, q, max_records=want, page_size=args.page_size)
            recs = clean_and_label_fasta(fasta_txt, fam, args.min_len, args.max_len, args.allow_x)
            kept = 0
            for r in recs:
                if r.id in seen_ids:
                    continue
                seen_ids.add(r.id)
                chosen_records.append(r)
                labels_rows.append({"id": r.id, "family": fam, "class": cls})
                kept += 1
                if len(chosen_records) >= args.target_total:
                    break
            print(f"        kept {kept} (total={len(chosen_records)})")
            progressed = progressed or (kept > 0)
            time.sleep(0.2)
            if len(chosen_records) >= args.target_total:
                break
        if not progressed:
            print("[WARN] Bir tam turda ilerleme yok — durduruluyor.")
            break
        if all(idx[c] >= len(cls_to_fams.get(c, [])) for c in ALL_CLASSES):
            break

    if not chosen_records:
        raise SystemExit("ERROR: 0 sekans indirildi. UniProt sorgu şablonu yanlış olabilir.")

    # 5) Çıktılar
    all_fa = os.path.join(args.out, "all.fasta")
    with open(all_fa, "w", encoding="utf-8") as f:
        SeqIO.write(chosen_records, f, "fasta")

    labels = pd.DataFrame(labels_rows).drop_duplicates(subset=["id"])
    labels.to_csv(os.path.join(args.out, "labels.csv"), index=False)
    fam_counts = labels.groupby(["class", "family"])["id"].nunique().sort_values(ascending=False)
    fam_counts.to_csv(os.path.join(args.out, "family_counts.csv"), header=["count"])

    train, test = random_split(chosen_records, test_ratio=args.test_ratio, seed=args.seed)
    with open(os.path.join(args.out, "train.fasta"), "w", encoding="utf-8") as f:
        SeqIO.write(train, f, "fasta")
    with open(os.path.join(args.out, "test.fasta"), "w", encoding="utf-8") as f:
        SeqIO.write(test, f, "fasta")

    # [NEW-3] Per-class coverage raporu
    class_summary = labels.groupby("class").agg(
        n_families=("family", "nunique"),
        n_sequences=("id", "count"),
    ).reset_index()
    class_summary["pct_of_total"] = (class_summary["n_sequences"] / len(labels) * 100).round(1)
    class_summary.to_csv(os.path.join(args.out, "class_coverage.csv"), index=False)

    report = {
        "target_total":          args.target_total,
        "selected_total":        len(chosen_records),
        "unique_ids":            len(seen_ids),
        "families_used":         int(labels["family"].nunique()),
        "classes_used":          int(labels["class"].nunique()),
        "per_class_min":         class_min,
        "per_class_max":         class_max,
        "priority_classes":      priority_classes,
        "split": {
            "train": len(train), "test": len(test), "test_ratio": args.test_ratio
        },
        "query_template":        tpl.name,
        "class_coverage":        class_summary.to_dict(orient="records"),
        "notes": [
            "v2: CE ve AA için per-class min/max eşikleri ve priority round-robin.",
            "Homology-aware split için homology.py kullanın (MMseqs2 linclust).",
        ],
    }
    write_text(os.path.join(args.out, "report.json"), json.dumps(report, indent=2))

    # Özet
    print("\n" + "="*60)
    print("  TAMAMLANDI")
    print("="*60)
    print(f"  all.fasta    : {len(chosen_records)} sekans")
    print(f"  labels.csv   : {labels.shape[0]} satır")
    print(f"  train/test   : {len(train)}/{len(test)}")
    print(f"\n  Per-class coverage:")
    for _, row in class_summary.iterrows():
        flag = " ← EKLENDİ" if row["class"] in ["CE", "AA"] else ""
        print(f"    {row['class']:4s}: {row['n_families']:3d} family, "
              f"{row['n_sequences']:5d} seq ({row['pct_of_total']:.1f}%){flag}")

    # CE/AA uyarısı
    for cls in ["CE", "AA"]:
        n = int(class_summary[class_summary["class"] == cls]["n_sequences"].sum()) if cls in class_summary["class"].values else 0
        if n == 0:
            print(f"\n  [WARN] {cls} hâlâ 0 sekans! family_hits_uniprot.csv'i inceleyin.")
        elif n < 500:
            print(f"\n  [WARN] {cls}: {n} seq — az. --{cls.lower()}_min_per_family'i düşürmeyi deneyin.")

    print("\n  Sonraki adım: python homology.py --dataset_dir <out>")
    print("="*60)


if __name__ == "__main__":
    main()

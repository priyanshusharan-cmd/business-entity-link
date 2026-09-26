import re
import unicodedata
from typing import Tuple, Optional, Dict, List, Any
import polars as pl

LEGAL_SUFFIX_MAP = [
    (r'\b(प्राइवेट\s+लिमिटेड|प्रा\.?\s*लि\.?|प्राइवेट\s+लि\.?)\b', 'pvt_ltd'),
    (r'\b(लिमिटेड|लि\.?)\b', 'ltd'),
    (r'\b(एलएलपी)\b', 'llp'),
    (r'\b(private\s+limited|pvt\.?\s*ltd\.?|pvt\s+limited)\b', 'pvt_ltd'),
    (r'\b(public\s+limited|pub\.?\s*ltd\.?)\b', 'pub_ltd'),
    (r'\b(limited|ltd\.?)\b', 'ltd'),
    (r'\b(incorporated|inc\.?)\b', 'inc'),
    (r'\b(corporation|corp\.?)\b', 'corp'),
    (r'\b(limited\s+liability\s+company|l\.?l\.?c\.?|llc)\b', 'llc'),
    (r'\b(limited\s+liability\s+partnership|l\.?l\.?p\.?|llp)\b', 'llp'),
    (r'\b(company|co\.?)\b', 'co'),
    (r'\b(societe\s+a\s+responsabilite\s+limitee|s\.?a\.?r\.?l\.?|sarl)\b', 'sarl'),
    (r'\b(societe\s+par\s+actions\s+simplifiee|s\.?a\.?s\.?|sas)\b', 'sas'),
    (r'\b(societe\s+anonyme|s\.?a\.?|sa)\b', 'sa'),
    (r'\b(societe\s+civile\s+immobiliere|s\.?c\.?i\.?|sci)\b', 'sci'),
    (r'\b(entreprise\s+unipersonnelle\s+a\s+responsabilite\s+limitee|e\.?u\.?r\.?l\.?|eurl)\b', 'eurl'),
    (r'\b(enterprise|ent\.?)\b', 'enterprise'),
]

ADDRESS_ABBREV_MAP = [
    (r'\b(rd\.?|road)\b', 'road'),
    (r'\b(st\.?|street)\b', 'street'),
    (r'\b(ave\.?|avenue)\b', 'avenue'),
    (r'\b(blvd\.?|boulevard)\b', 'boulevard'),
    (r'\b(dr\.?|drive)\b', 'drive'),
    (r'\b(ct\.?|court)\b', 'court'),
    (r'\b(ln\.?|lane)\b', 'lane'),
    (r'\b(pkwy\.?|parkway)\b', 'parkway'),
    (r'\b(ste\.?|suite)\b', 'suite'),
    (r'\b(apt\.?|apartment)\b', 'apartment'),
    (r'\b(bldg\.?|building)\b', 'building'),
    (r'\b(fl\.?|floor)\b', 'floor'),
    (r'\b(hwy\.?|highway)\b', 'highway'),
    (r'\b(opp\.?|opposite)\b', 'opposite'),
    (r'\b(nr\.?|near)\b', 'near'),
    (r'\b(r\.?|rue)\b', 'rue'),
    (r'\b(bd\.?|boulevard)\b', 'boulevard'),
    (r'\b(av\.?|avenue)\b', 'avenue'),
    (r'\b(all\.?|allee)\b', 'allee'),
    (r'\b(pl\.?|place)\b', 'place'),
    (r'\b(imp\.?|impasse)\b', 'impasse'),
]

class EntityNormalizer:
    def __init__(self):
        self.suffix_regexes = [(re.compile(pat, re.IGNORECASE), repl) for pat, repl in LEGAL_SUFFIX_MAP]
        self.address_regexes = [(re.compile(pat, re.IGNORECASE), repl) for pat, repl in ADDRESS_ABBREV_MAP]

    @staticmethod
    def remove_latin_accents(text: str) -> str:
        if not text:
            return ""
        res = []
        for c in text:
            if 'LATIN' in unicodedata.name(c, ''):
                nfkd = unicodedata.normalize('NFKD', c)
                res.append(''.join([ch for ch in nfkd if not unicodedata.combining(ch)]))
            else:
                res.append(c)
        return ''.join(res)

    def extract_legal_suffix(self, name: str) -> Tuple[str, str]:
        if not name or not str(name).strip():
            return "", "none"
        text = self.remove_latin_accents(str(name)).lower()
        found_suffix = "none"
        for reg, norm_suffix in self.suffix_regexes:
            if reg.search(text):
                found_suffix = norm_suffix
                text = reg.sub("", text)
                break
        clean_text = re.sub(r'[^\w\s]', ' ', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        return clean_text, found_suffix

    def normalize_name(self, name: Optional[str]) -> Dict[str, str]:
        if not name or not str(name).strip():
            return {"name_clean": "", "legal_suffix": "none"}
        clean_name, suffix = self.extract_legal_suffix(name)
        return {"name_clean": clean_name, "legal_suffix": suffix}

    def normalize_address(self, address: Optional[str]) -> Dict[str, Any]:
        if not address or not str(address).strip():
            return {"address_clean": "", "has_address": False, "numbers": []}
        text = self.remove_latin_accents(str(address)).lower()
        for reg, repl in self.address_regexes:
            text = reg.sub(repl, text)
        numbers = re.findall(r'\b\d+\b', text)
        clean_text = re.sub(r'[^\w\s]', ' ', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        return {"address_clean": clean_text, "has_address": True, "numbers": numbers}

    def normalize_polars_df(self, df: pl.DataFrame) -> pl.DataFrame:
        names = df["business_name"].to_list()
        addresses = df["business_address"].to_list()
        cleaned_names, suffixes, cleaned_addrs, has_addrs = [], [], [], []
        for n, a in zip(names, addresses):
            n_res = self.normalize_name(n)
            a_res = self.normalize_address(a)
            cleaned_names.append(n_res["name_clean"])
            suffixes.append(n_res["legal_suffix"])
            cleaned_addrs.append(a_res["address_clean"])
            has_addrs.append(a_res["has_address"])
        return df.with_columns([
            pl.Series("name_clean", cleaned_names, dtype=pl.String),
            pl.Series("legal_suffix", suffixes, dtype=pl.String),
            pl.Series("address_clean", cleaned_addrs, dtype=pl.String),
            pl.Series("has_address", has_addrs, dtype=pl.Boolean)
        ])

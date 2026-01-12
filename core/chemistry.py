import re
import pandas as pd

# Element -> oxide mass conversion factors: (molar mass oxide) / (molar mass element(s))
_OXIDE_FACTORS = {
    "SnO2": 150.71 / 118.71,
    "Ta2O5": 441.89 / (2 * 180.95),
    "Nb2O5": 265.81 / (2 * 92.91),
    "ZrO2": 123.22 / 91.22,
    "Rb2O": 186.94 / (2 * 85.47),
    "Cs2O": 281.81 / (2 * 132.91),
    "SrO": 103.62 / 87.62,
}

# Canonical names for common variants
_CANONICAL = {
    "SNO2": "SnO2",
    "TA2O5": "Ta2O5",
    "NB2O5": "Nb2O5",
    "ZRO2": "ZrO2",
    "RB2O": "Rb2O",
    "CS2O": "Cs2O",
    "SRO": "SrO",

    "SN": "Sn",
    "TA": "Ta",
    "NB": "Nb",
    "ZR": "Zr",
    "RB": "Rb",
    "CS": "Cs",
    "SR": "Sr",

    "LAT": "lat",
    "LATITUDE": "lat",
    "LON": "lon",
    "LONG": "lon",
    "LONGITUDE": "lon",

    "INTERVAL": "INTERVAL",

    "HOLE": "hole_id",
    "HOLEID": "hole_id",
    "HOLE_ID": "hole_id",
    "DRILLHOLE": "hole_id",
    "DHID": "hole_id",
    "PIT": "hole_id",
    "PITID": "hole_id",
    "PIT_ID": "hole_id",
}

ELEMENT_TO_OXIDE = {
    "Sn": "SnO2",
    "Ta": "Ta2O5",
    "Nb": "Nb2O5",
    "Zr": "ZrO2",
    "Rb": "Rb2O",
    "Cs": "Cs2O",
    "Sr": "SrO",
}

OXIDE_KEYS = list(_OXIDE_FACTORS.keys())


def _norm(col: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]", "", str(col))
    return s.upper()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {}
    for c in out.columns:
        k = _norm(c)
        if k in _CANONICAL:
            rename[c] = _CANONICAL[k]
    return out.rename(columns=rename)


def ensure_oxides(
    df: pd.DataFrame,
    prefer_existing_oxides: bool = True,
    allow_element_to_oxide: bool = True,
) -> pd.DataFrame:
    """
    Ensures oxide columns exist where possible.

    - If oxides already exist: keep them.
    - If only elements exist (Sn, Ta, Zr...): optionally create oxide equivalents.
    - Mixed inputs are supported.

    IMPORTANT:
    - Element->oxide conversion is mass-based and assumes comparable basis.
      For ICP-MS/AAS (ppm elements), you may prefer using elements directly.
    """
    out = standardize_columns(df)

    # If oxides exist and we prefer them, do nothing
    have_oxides = any(k in out.columns for k in OXIDE_KEYS)
    if have_oxides and prefer_existing_oxides:
        return out

    if not allow_element_to_oxide:
        return out

    # Create missing oxides from elements when possible
    for el, ox in ELEMENT_TO_OXIDE.items():
        if ox not in out.columns and el in out.columns and ox in _OXIDE_FACTORS:
            out[ox] = pd.to_numeric(out[el], errors="coerce") * _OXIDE_FACTORS[ox]

    return out

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def normalise_day(series: pd.Series) -> pd.Series:
    """Return string formatted YYYY-MM-DD regardless of input type."""
    raw = series.copy()
    if pd.api.types.is_numeric_dtype(raw):
        raw = raw.astype("Int64").astype(str).str.zfill(8)
    else:
        raw = raw.astype(str).str.strip()

    parsed = pd.to_datetime(raw, errors="coerce")
    missing_mask = parsed.isna()
    if missing_mask.any():
        parsed.loc[missing_mask] = pd.to_datetime(
            raw[missing_mask], format="%Y%m%d", errors="coerce"
        )

    if parsed.isna().any():
        missing = parsed.isna().sum()
        sample = raw[parsed.isna()].head().tolist()
        raise ValueError(
            f"No se pudieron convertir {missing} fechas. Ejemplos problematicos: {sample}"
        )

    return parsed.dt.strftime("%Y-%m-%d")


def export_sheet(excel_path: Path, sheet: str, output_path: Path) -> None:
    df = pd.read_excel(excel_path, sheet_name=sheet)
    if "day" not in df.columns:
        raise KeyError(f"La hoja '{sheet}' del archivo '{excel_path.name}' no contiene la columna 'day'.")

    df = df.copy()
    df["day"] = normalise_day(df["day"])

    # Asegura nombres de columnas exactos a los esperados por el notebook
    column_map = {col: col.strip().lower() for col in df.columns}
    df.rename(columns=column_map, inplace=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    rel_path = output_path.resolve().relative_to(Path.cwd())
    print(f"Generated {rel_path}")


def main() -> None:
    base_dir = Path("Gategroup 2025-20251025T180038Z-1-001") / "Gategroup 2025"
    catalog: list[dict[str, str]] = []

    workbook = base_dir / "Hackatlon Consumption and Estimation - Gategroup Dataset 1 de 2.xlsx"
    if not workbook.exists():
        raise FileNotFoundError(f"No se encontro el archivo esperado: {workbook}")

    sheets: Iterable[str] = ("Plant 1", "Plant 2")
    data_dir = Path("data")

    for sheet in sheets:
        output_name = f"{sheet}.csv"
        output_path = data_dir / output_name
        export_sheet(workbook, sheet, output_path)
        catalog.append(
            {
                "sheet": sheet,
                "output": str(output_path.resolve().relative_to(Path.cwd())),
            }
        )

    catalog_path = data_dir / "plant_catalog.json"
    catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(f"Saved catalog: {catalog_path.resolve().relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()

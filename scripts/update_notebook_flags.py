from pathlib import Path

import nbformat


def main() -> None:
    nb_path = Path("notebooks/global_flights_forecasting.ipynb")
    nb = nbformat.read(nb_path, as_version=4)
    changed = False

    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", "")
        if isinstance(source, list):
            lines = source
        else:
            lines = source.split("\n")

        new_lines = []
        local_change = False
        for line in lines:
            if "infer_datetime_format=True" in line:
                line = line.replace(
                    ", errors='coerce', infer_datetime_format=True",
                    ", errors='coerce'",
                )
                local_change = True
            new_lines.append(line)

        if local_change:
            changed = True
            cell["source"] = new_lines if isinstance(source, list) else "\n".join(new_lines)

    if changed:
        nbformat.write(nb, nb_path)
        print("Removed deprecated infer_datetime_format flag.")
    else:
        print("No updates were necessary.")


if __name__ == "__main__":
    main()

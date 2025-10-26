from pathlib import Path

import nbformat


def main() -> None:
    nb_path = Path("notebooks/global_flights_forecasting.ipynb")
    nb = nbformat.read(nb_path, as_version=4)

    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        outputs = cell.get("outputs", [])
        for output in outputs:
            if output.get("output_type") == "stream":
                text = output.get("text", "")
                if text.strip():
                    print(text.strip())
            elif output.get("output_type") in {"execute_result", "display_data"}:
                data = output.get("data", {})
                text = data.get("text/plain")
                if text:
                    print(text.strip())


if __name__ == "__main__":
    main()

# Real-World Outbreak Dataset: 1978 English Boarding School Influenza

This directory contains real-world epidemiological benchmark time-series data for testing and validating the **Fractional SIR** model in **FracID**.

## Dataset Description
- **File**: `boarding_school_influenza_1978.csv`
- **Epidemic Event**: Influenza A (H1N1) outbreak in an isolated English boarding school during January–February 1978.
- **Population**: $N = 763$ boys (almost completely isolated residential population).
- **Duration**: 15 consecutive days ($t = 0, 1, \dots, 14$).

## Variables & Units
| Column | Description | Units |
|---|---|---|
| `t` | Elapsed observation day since outbreak detection | Days ($d$) |
| `S` | Susceptible individuals ($S(t) = N - I(t) - R(t)$) | Count of persons |
| `I` | Active, bed-ridden infectious individuals | Count of persons |
| `R` | Recovered / convalescent individuals | Count of persons |

## Provenance and Open Licence
- **Original Source**: Anonymous, *"Influenza in a boarding school"*, **British Medical Journal**, 1978, 1(6112): 587–588. [doi:10.1136/bmj.1.6112.587](https://doi.org/10.1136/bmj.1.6112.587).
- **Licence**: Public Domain / Open Access historical medical reporting data.
- **Standard Benchmark**: Extensively analyzed in mathematical biology and fractional epidemic modeling literature (e.g. Murray, *Mathematical Biology*, Springer, 2002).

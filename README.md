# IF3170_G11_ThreeAgents

Implementasi Tugas Besar 1 IF3170 Inteligensi Artifisial: pencarian konfigurasi paket pada truk menggunakan Hill-climbing, Simulated Annealing, dan Genetic Algorithm.

## Persiapan

- Python **3.10 atau lebih baru**.
- Install dependency visualisasi:

```bash
python -m pip install -r requirements.txt
```

Jalankan tes fondasi dari root repository:

```bash
python -m unittest discover -s tests -v
```

Hill-climbing dan Simulated Annealing masih dikembangkan. Genetic Algorithm sudah memiliki loop, runner eksperimen, serta visualisasi.

## Struktur direktori

```text
.
├── src/
│   ├── model.py                # Model Package, Truck, Placement, dan State
│   ├── validation.py           # Pemeriksaan seluruh constraint state satu truk
│   ├── neighbors.py            # State awal acak dan move neighbor yang valid
│   ├── contracts.py            # Interface pemanggilan algoritma dan objective
│   ├── results.py              # SearchResult dan metrik khusus tiap algoritma
│   ├── algorithms/
│   │   └── genetic_algorithm/
│   │       ├── population.py    # Pembentukan populasi awal
│   │       ├── selection.py     # Roulette-wheel selection
│   │       ├── crossover.py     # Single-point crossover dan repair
│   │       ├── mutation.py      # Mutasi satu gen
│   │       ├── _placement.py    # Helper placement valid untuk operator
│   │       └── algorithm.py     # Loop Genetic Algorithm
│   ├── experiments/
│   │   ├── data.py              # Loader problem dari JSON
│   │   ├── run_ga.py            # CLI runner
│   │   ├── genetic_algorithm.py # Dua sweep parameter dan hasil per run
│   │   └── analysis.py          # Ringkasan lintas run
│   └── visualization/
│       └── genetic_algorithm.py # Plot riwayat dan tampilan state 3D
├── tests/                      # Tes model, validator, neighbor, dan kontrak
├── docs/
│   └── laporan-akhir/          # Sumber dan hasil laporan
├── output/                     # Hasil run, data eksperimen, dan visualisasi
└── README.md
```

Folder `target/` berisi bahan referensi lokal dan diabaikan Git; jangan gunakan folder tersebut untuk file yang perlu dikirim melalui repository.

## Panduan pengembangan

### Kontrak algoritma

Semua algoritma mengikuti interface di `src/contracts.py`:

```python
algorithm(
    initial_state,
    *,
    seed,
    objective_fn=None,
    **algorithm_parameters,
) -> SearchResult
```

- `initial_state` adalah state valid yang dibangkitkan secara acak. HC dan SA memulai pencarian dari state tersebut; GA menggunakannya sebagai anggota pertama populasi awal acak.
- `seed` wajib diberikan agar proses acak dapat diulang. Catat nilainya di `SearchResult.seed`.
- `objective_fn` menerima state dan mengembalikan nilai numerik yang dimaksimalkan. Jika tidak diberikan, gunakan objective bawaan melalui `resolve_objective`.
- Simpan parameter algoritma yang digunakan di `SearchResult.parameters`.
- Kembalikan `SearchResult` dengan state awal/akhir terbaik, nilai objective, riwayat objective per iterasi atau generasi, durasi, dan metrik khusus algoritma.

### Fondasi yang tersedia

- `model.py`: orientasi direpresentasikan oleh permutasi indeks dimensi dan dimensi terpasang. Paket di luar truk memiliki penempatan `None`.
- `validation.py`: `validate_state(state)` mengembalikan daftar pelanggaran; daftar kosong berarti state valid.
- `neighbors.py`: `random_initial_state(packages, truck, rng)` dan `random_neighbor(state, rng)` membentuk state valid. Neighbor menggunakan swap, perpindahan, atau rotasi 90°.
- `results.py`: `SearchResult` memakai metrik `HillClimbingMetrics`, `SimulatedAnnealingMetrics`, atau `GeneticAlgorithmMetrics`.

### Menjalankan eksperimen GA

Runner membaca JSON problem dengan `truck` (dimensi dan kapasitas) serta `packages` (ID, dimensi, value, weight, fragile, ETA). Jalankan kedua sweep wajib dengan tiga nilai untuk setiap parameter dan tiga seed:

```bash
python -m src.experiments.run_ga --input examples/ga-demo.json --generation-values 4 6 8 --population-values 4 6 8 --fixed-population-size 6 --fixed-generations 6 --seeds 41 42 43 --output output/ga-demo
```

Contoh tersebut memakai **data sintetis** untuk demo. Untuk eksperimen laporan, siapkan JSON dengan data paket tugas yang sebenarnya, pilih nilai parameter lewat pilot, lalu jalankan dengan lokasi input dan output yang sesuai. Runner menghasilkan 18 run, `results.json`, `summary.json` (mean dan simpangan baku), plot ringkasan tiap sweep, serta JSON dan visualisasi state/objective untuk setiap run.

### Alur kerja

1. Buat branch sesuai jenis perubahan: `feat/...` untuk fitur, `fix/...` untuk perbaikan, `docs/...` untuk dokumentasi, atau `test/...` untuk tes. Contoh: `feat/hill-climbing`, `feat/simulated-annealing`, `docs/readme`.
2. Implementasikan algoritma di `src/algorithms/` dan gunakan kontrak fondasi di atas; jangan membuat representasi state atau validator alternatif.
3. Tambahkan atau sesuaikan tes di `tests/`.
4. Jalankan seluruh suite dengan `python -m unittest discover -s tests -v` sebelum menggabungkan perubahan.
5. Simpan data eksperimen dan visualisasi run di `output/`; catat seed serta parameter agar hasil dapat direproduksi.

Gunakan semantic commit yang sesuai dengan branch, misalnya `feat: add hill-climbing` atau `docs: update development guide`. Pastikan hasil eksperimen, pembahasan, dan pembagian tugas diperbarui pada laporan di `docs/laporan-akhir/`.

## Pembagian tugas

- **Aufar:** kontrak dan fondasi bersama, Genetic Algorithm, serta bonus multi-truk.
- **Arin:** Hill-climbing wajib dan seluruh variannya sebagai bonus.
- **Mike:** Simulated Annealing wajib dan objective function alternatif sebagai bonus.

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
│   ├── model.py                # Model Package, Truck, Placement, State, dan FleetState
│   ├── validation.py           # Pemeriksaan constraint state satu truk dan armada
│   ├── neighbors.py            # State awal acak dan move neighbor yang valid
│   ├── contracts.py            # Interface pemanggilan algoritma dan objective
│   ├── search_problems.py       # Adapter operasi State dan FleetState ke SearchProblem
│   ├── results.py              # SearchResult dan metrik khusus tiap algoritma
│   ├── algorithms/
│   │   └── genetic_algorithm/
│   │       ├── fleet.py         # Inisialisasi, crossover, mutasi, dan GA multi-truk
│   │       ├── population.py    # Pembentukan populasi awal
│   │       ├── selection.py     # Roulette-wheel selection
│   │       ├── crossover.py     # Single-point crossover dan repair
│   │       ├── mutation.py      # Mutasi satu gen
│   │       ├── _placement.py    # Helper placement valid untuk operator
│   │       └── algorithm.py     # Loop Genetic Algorithm
│   ├── experiments/
│   │   ├── data.py              # Loader problem dari JSON
│   │   ├── run_ga.py            # CLI runner
│   │   ├── run_fleet_ga.py      # CLI GA multi-truk
│   │   ├── run_ga_ablation.py   # CLI studi ablasi GA
│   │   ├── genetic_algorithm.py # Dua sweep parameter dan hasil per run
│   │   └── analysis.py          # Ringkasan lintas run
│   └── visualization/
│       └── genetic_algorithm.py # Plot riwayat dan tampilan state 3D
├── tests/                      # Tes fondasi, eksperimen, dan kasus GA kompleks
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

Untuk state/domain baru, pasangkan operasi domain lewat `SearchProblem` satu kali, lalu gunakan kontrak yang sama pada HC, SA, atau GA:

```python
from src.contracts import SearchProblem

problem = SearchProblem(
    objective_fn=score,
    is_valid=is_valid,
    random_state=random_state,
    neighbor=neighbor,
    crossover=crossover,  # opsional; dipakai GA
    mutate=mutate,        # opsional; dipakai GA
)
result = simulated_annealing(initial_state, seed=41, search_problem=problem, ...)
```

`search_problem_for_state(state)` menyediakan adapter bawaan untuk `State` dan `FleetState`. Karena itu state armada dapat dipakai langsung oleh HC dan SA; GA memakai operasi crossover/mutasi armada dari adapter yang sama. Panggilan lama tanpa `search_problem` tetap didukung untuk state bawaan.

### Fondasi yang tersedia

- `model.py`: orientasi direpresentasikan oleh permutasi indeks dimensi dan dimensi terpasang. Paket di luar truk memiliki penempatan `None`.
- `validation.py`: `validate_state(state)` mengembalikan daftar pelanggaran; daftar kosong berarti state valid.
- `neighbors.py`: `random_initial_state(packages, truck, rng)` dan `random_neighbor(state, rng)` membentuk state valid. Neighbor menggunakan swap, perpindahan, atau rotasi 90°.
- `results.py`: `SearchResult` memakai metrik `HillClimbingMetrics`, `SimulatedAnnealingMetrics`, atau `GeneticAlgorithmMetrics`.

### Menjalankan eksperimen GA

Runner membaca JSON problem dengan `truck` (dimensi dan kapasitas) serta `packages` (ID, dimensi, value, weight, fragile, ETA). Jalankan kedua sweep wajib dengan tiga nilai untuk setiap parameter dan tiga seed:

```bash
python -m src.experiments.run_ga --input tests/fixtures/demo.json --generation-values 4 6 8 --population-values 4 6 8 --fixed-population-size 6 --fixed-generations 6 --seeds 41 42 43 --output output/ga-demo
```

Contoh tersebut memakai **data sintetis** untuk demo. Untuk eksperimen laporan, siapkan JSON dengan data paket tugas yang sebenarnya, pilih nilai parameter lewat pilot, lalu jalankan dengan lokasi input dan output yang sesuai. Runner menghasilkan 18 run, `results.json`, `summary.json` (mean dan simpangan baku), plot ringkasan tiap sweep, serta JSON dan visualisasi state/objective untuk setiap run.

#### Bonus multi-truk

Jalankan GA untuk tiga salinan dari truk pada JSON lama:

```bash
python -m src.experiments.run_fleet_ga --input tests/fixtures/demo.json --truck-count 3 --population-size 20 --generations 50 --seed 41 --output output/fleet-ga-demo
```

Untuk dimensi dan kapasitas yang berbeda per truk, gunakan `trucks` sebagai daftar pada JSON, dengan format atribut truk yang sama:

```json
"trucks": [
  {"dimensions": [10, 20, 5], "max_capacity": 500},
  {"dimensions": [6, 8, 4], "max_capacity": 180}
]
```

Jangan berikan `--truck-count` jika JSON sudah memakai daftar `trucks`. GA mencari konfigurasi dengan total value armada setinggi mungkin secara heuristik (tidak menjamin global optimum) dan memastikan tiap paket paling banyak masuk ke satu truk. Operasi tambahan yang terlibat: inisialisasi mencari truk yang muat untuk tiap paket; gen GA menyimpan assignment truk beserta koordinat/orientasinya; crossover mencoba mempertahankan placement warisan, memperbaikinya di truk asal, lalu mencoba truk lain; mutasi dapat memindah, merotasi, memindahkan paket ke truk lain, atau mengeluarkannya. Constraint ruang, support, kerapuhan, dan kapasitas tetap divalidasi per truk. Paket yang tidak mendapat penempatan valid tetap di luar armada. Hasil JSON mencatat muatan per truk, total value, serta state awal/akhir; dua plot disimpan di folder output.

HC dan SA juga dapat dijalankan satu kali pada format JSON armada yang sama:

```bash
python -m src.experiments.run_fleet_hc --input tests/fixtures/demo.json --truck-count 3 --variants steepest random_restart --seeds 41 42 43 --objectives total_value eta_weighted_value compact_value --output output/fleet-hc-demo
python -m src.experiments.run_fleet_sa --input tests/fixtures/demo.json --truck-count 3 --max-iterations 2000 --seeds 41 42 43 --objectives total_value eta_weighted_value compact_value --output output/fleet-sa-demo
```

Keduanya menerima daftar `trucks` heterogen atau `--truck-count` untuk menduplikasi `truck` lama. `--seeds` menjalankan tiga seed berbeda; jika tidak diberikan, `--seed` (default 41) menjalankan satu kali. Beberapa objective menghasilkan ringkasan dan perbandingan total value. Objective armada menghitung nilai/ETA untuk semua paket yang termuat; `compact_value` menambahkan bonus kekompakan berbobot volume lintas truk. Hasil mencakup JSON per run dan ringkasan, plot state awal/akhir, serta riwayat objective; SA juga menyimpan grafik probabilitas penerimaan.

#### Fixture dan benchmark lintas algoritma

`tests/fixtures/` berisi instance per kategori metodologi laporan: `base.json` (kasus dasar), `tight-space.json` dan `tight-capacity.json` (kendala ketat, diturunkan dari base dengan mengubah satu faktor), `infeasible.json` (tidak ada paket yang bisa dimuat), `boundary/` (kasus batas), dan `invalid/` (masukan yang harus ditolak loader). `tests/test_fixtures.py` memeriksa sifat tiap fixture. `hard.json` dan `demo.json` adalah instance lama yang masih dipakai tes lain.

```bash
python -m src.experiments.run_benchmark --output output/benchmark
```

Perintah ini menjalankan empat varian HC, SA, dan GA pada semua fixture valid dengan seed 11-15 (state awal dan seed yang sama untuk semua algoritma), memvalidasi tiap state akhir, mencatat jumlah evaluasi objective dan durasi, serta memeriksa bahwa fixture `invalid/` ditolak. Hasil: `runs.json`, `summary.json`, dan `summary.png`.

#### Studi ablasi operator GA

```bash
python -m src.experiments.run_ga_ablation --input tests/fixtures/demo.json --output output/ga-ablation-demo
```

Runner membandingkan baseline GA dengan tiga ablasi satu-komponen: tanpa mutasi, tanpa crossover, dan seleksi uniform sebagai pembanding roulette-wheel. Secara default tiap varian dijalankan dengan 5 seed yang sama, populasi 20, 50 generasi, dan mutation rate baseline 0,05. Parameter bersama bisa diubah lewat opsi CLI. Hasilnya mencakup `runs.json`, `summary.json` (mean dan simpangan baku objective/durasi per varian), dan plot objective maksimum serta rata-rata populasi. Gunakan dataset dan parameter tetap untuk membandingkan semua varian. Contoh di atas memakai data sintetis demo; hasil laporan harus memakai problem kelompok. Ablasi ini eksperimen tambahan dan tidak menggantikan 18 run sweep wajib.

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

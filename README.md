# IF3170_G11_ThreeAgents

Implementasi Tugas Besar 1 IF3170 Inteligensi Artifisial: pencarian konfigurasi paket pada truk menggunakan Hill-climbing, Simulated Annealing, dan Genetic Algorithm.

## Persiapan

- Python **3.10 atau lebih baru**.
- Tidak ada dependency eksternal untuk fondasi saat ini.

Jalankan tes fondasi dari root repository:

```bash
python -m unittest discover -s tests -v
```

Algoritma pencarian dan entry point aplikasi dikembangkan terpisah. Instruksi untuk menjalankan eksperimen akan ditambahkan setelah runner algoritma tersedia.

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
│   │   └── genetic_algorithm.py # Fondasi populasi awal GA; operator/loop menyusul
│   ├── experiments/            # Runner eksperimen dan pengumpulan hasil
│   └── visualization/          # Visualisasi state dan hasil eksperimen
├── tests/                      # Tes model, validator, neighbor, dan kontrak
├── docs/
│   └── laporan-akhir/          # Sumber dan hasil laporan
├── output/                     # Hasil run, data eksperimen, dan visualisasi
└── README.md
```

Folder `algorithms`, `experiments`, dan `visualization` memiliki `.gitkeep` sampai implementasinya ditambahkan. Folder `target/` berisi bahan referensi lokal dan diabaikan Git; jangan gunakan folder tersebut untuk file yang perlu dikirim melalui repository.

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

- `initial_state` adalah state valid. HC dan SA memulai pencarian dari state tersebut; GA menggunakannya sebagai kandidat dalam populasi awal.
- `seed` wajib diberikan agar proses acak dapat diulang. Catat nilainya di `SearchResult.seed`.
- `objective_fn` menerima state dan mengembalikan nilai numerik yang dimaksimalkan. Jika tidak diberikan, gunakan objective bawaan melalui `resolve_objective`.
- Simpan parameter algoritma yang digunakan di `SearchResult.parameters`.
- Kembalikan `SearchResult` dengan state awal/akhir terbaik, nilai objective, riwayat objective per iterasi atau generasi, durasi, dan metrik khusus algoritma.

### Fondasi yang tersedia

- `model.py`: orientasi direpresentasikan oleh permutasi indeks dimensi dan dimensi terpasang. Paket di luar truk memiliki penempatan `None`.
- `validation.py`: `validate_state(state)` mengembalikan daftar pelanggaran; daftar kosong berarti state valid.
- `neighbors.py`: `random_initial_state(packages, truck, rng)` dan `random_neighbor(state, rng)` membentuk state valid. Neighbor menggunakan swap, perpindahan, atau rotasi 90°.
- `results.py`: `SearchResult` memakai metrik `HillClimbingMetrics`, `SimulatedAnnealingMetrics`, atau `GeneticAlgorithmMetrics`.

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

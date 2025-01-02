from itertools import combinations

# Data barang
barang = [
    {"nama": "cabe", "berat": 300, "harga": 90000},
    {"nama": "kentang", "berat": 250, "harga": 80000},
    {"nama": "wortel", "berat": 300, "harga": 60000},
    {"nama": "pisang", "berat": 150, "harga": 25000},
    {"nama": "terong", "berat": 100, "harga": 10000},
    {"nama": "kubis", "berat": 100, "harga": 70000},
]

kapasitas = 1250

def cari_kombinasi_optimal(barang, kapasitas):
    max_nilai = 0
    kombinasi_terbaik = []
    for i in range(1, len(barang) + 1):
        for kombinasi in combinations(barang, i):
            total_berat = sum(item['berat'] for item in kombinasi)
            if total_berat <= kapasitas:
                total_harga = sum(item['berat'] * item['harga'] for item in kombinasi)
                if total_harga > max_nilai:
                    max_nilai = total_harga
                    kombinasi_terbaik = kombinasi
    return kombinasi_terbaik, max_nilai

kombinasi, nilai = cari_kombinasi_optimal(barang, kapasitas)
print("Kombinasi terbaik:", [(item['nama'], item['berat']) for item in kombinasi])
print("Nilai maksimum:", nilai)

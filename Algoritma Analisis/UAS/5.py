def hitung_koin(nominal, koin):
    koin.sort(reverse=True)
    hasil = []
    for nilai in koin:
        jumlah = nominal // nilai
        hasil.append((nilai, jumlah))
        nominal %= nilai
    return hasil


nominal = 2750
koin = [1000, 500, 100]
hasil = hitung_koin(nominal, koin)
print("Jumlah koin:", hasil)

# Membaca file pidato
file_path = "Dataset/Pidato Presiden Prabowo 2024.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

# Memecah teks menjadi daftar string berdasarkan paragraf
data = [para.strip() for para in text.split("\n\n") if para.strip()]

# Menampilkan hasil dalam format yang diinginkan
print("data = [")
for para in data:
    print(f'    "{para}",')
print("]")

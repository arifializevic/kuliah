from bertopic import BERTopic

# Membaca dataset dari file
with open("Dataset/Pidato Presiden Prabowo 2024_convert .txt", "r", encoding="utf-8") as file:
    text = file.read()

# Memecah teks menjadi paragraf atau bagian
# Memecah berdasarkan paragraf (dengan 2 baris kosong)
documents = text.split("\n\n")

# Inisialisasi model BERTopic
topic_model = BERTopic(language="indonesian", verbose=True)

# Fit model ke dokumen
topics, probs = topic_model.fit_transform(documents)

# Menampilkan hasil topik
print("\nTopik yang dihasilkan:")
print(topic_model.get_topic_info())

# Visualisasi topik
topic_model.visualize_topics()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# from ucimlrepo import list_available_datasets
# (list_available_datasets)

# Mengimpor dataset
haberman = pd.read_csv('haberman.csv')

# Mengakses data
x = haberman[['Age','Op_Year','axil_nodes']]
y = haberman['Surv_status']

# 1. Membagi data menjadi data latih dan data uji dengan test_size=0.25.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Membuat klasifikasi menggunakan RandomForestClassifier
#refrence
# #https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
classifier = RandomForestClassifier(n_estimators=45, random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
# 2. Berapa score rata2 akurasi dari model tersebut?
akurasi = accuracy_score(y_test, y_pred)
print("Skor Akurasi:", akurasi)

#3. Prediksi data test dengan model yang telah kalian buat!
# y_pred = classifier.predict(X_test)
# print("skor prediksi dari model:", y_pred)


#4. Bagaimana hasil confusion matrix dari hasil prediksi tersebut?
konfusi_matrik = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", konfusi_matrik)

#5. Bagaimana classification report dari hasil prediksi tersebut?
klasifikasi = classification_report(y_test, y_pred)
print("Klasifikasi:\n", klasifikasi)

# 6. Seberapa baik model anda dalam memprediksi seorang pasien mempunyai status positive?

positif = classification_report(y_test, y_pred, output_dict=True)['1']['recall']
print(f"positif: {positif}")

# 7. Seberapa baik model anda dalam memprediksi seorang pasien mempunyai status negatif?

negatif = classification_report(y_test, y_pred, output_dict=True)['2']['recall']
print(f"negatif: {negatif}")

#8. Buatlah analisis sederhana mengapa anda mendapatkan nilai akurasi tsb?
print("Harus Bagus Dari Model dan Kualitas Data Set")


# 9. Jelaskan alternatif cara untuk meningkatkan hasil akurasi
print("Penambahan Nilai Epoch atau Learning Rate")







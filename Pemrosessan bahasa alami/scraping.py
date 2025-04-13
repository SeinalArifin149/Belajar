from selenium import webdriver

# Buat instance driver Firefox
driver = webdriver.Firefox()

# Buka Google untuk uji coba
driver.get("https://www.google.com")

print("Judul halaman:", driver.title)  # Cetak judul halaman
driver.quit()

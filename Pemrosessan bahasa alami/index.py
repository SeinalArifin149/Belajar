from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
import time
import csv

# Setup Firefox
options = Options()
options.add_argument("--headless")  # Hapus ini kalau mau lihat browsernya
driver = webdriver.Firefox(options=options)

# Cari di Google Maps
query = "Rumah Sakit Dr. Soetomo Surabaya"
maps_url = f"https://www.google.com/maps/search/{query.replace(' ', '+')}"
driver.get(maps_url)

time.sleep(5)

# Klik hasil pertama
first_result = driver.find_element(By.CLASS_NAME, 'hfpxzc')
first_result.click()

time.sleep(5)

# Klik tombol ulasan
reviews_button = driver.find_element(By.XPATH, "//button[contains(@aria-label, 'ulasan')]")
reviews_button.click()

time.sleep(5)

# Scroll ke bawah untuk load semua ulasan
scrollable_div = driver.find_element(By.XPATH, "//div[@class='m6QErb DxyBCb kA9KIf dS8AEf ecceSd']")

for _ in range(100):  # atur scroll sesuai kebutuhan
    driver.execute_script('arguments[0].scrollTop = arguments[0].scrollHeight', scrollable_div)


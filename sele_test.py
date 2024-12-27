from selenium import webdriver
from selenium.webdriver.common.by import By


def setup(expolorer="chrome"):
    if expolorer == "chrome":
        driver = webdriver.Chrome()
    elif expolorer == "firefox":
        driver = webdriver.Firefox()
    elif expolorer == "edge":
        driver = webdriver.Edge()
    else:
        raise ValueError("Not supported browser")
    driver.get("https://www.selenium.dev/selenium/web/web-form.html")
    return driver

def teardown(driver):
    driver.quit()

def test_eight_components():
    driver = setup(expolorer="edge")

    title = driver.title
    assert title == "Web form"
    url = driver.current_url
    print(url)

    driver.implicitly_wait(0.5)

    text_box = driver.find_element(by=By.NAME, value="my-text")
    submit_button = driver.find_element(by=By.CSS_SELECTOR, value="button")

    text_box.send_keys("Selenium")
    driver.implicitly_wait(3)
    submit_button.click()

    message = driver.find_element(by=By.ID, value="message")
    value = message.text
    assert value == "Received!"

    # teardown(driver)


if __name__ == "__main__":
    test_eight_components()
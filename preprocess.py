from bs4 import BeautifulSoup

def parse_HTML(path):
    with open(path, 'r',encoding="utf-8") as fd:
        text = fd.read()

    soup = BeautifulSoup(text, 'html.parser')

    return soup

if __name__ == "__main__":
    PATH = "scraped\\file.htm"
    text = parse_HTML(PATH)
    print(text)
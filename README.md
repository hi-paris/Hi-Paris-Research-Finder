# Hi! Paris Expert Finder

The Hi! Paris Expert Finder platform is a smart research engine designed to search for researchers affiliated with Hi! Paris who work in a specific field and closely related areas.



## Installation

Clone the repository:

```bash
git clone https://github.com/hi-paris/Hi-Paris-Research-Finder.git
cd hi-paris-expert-finder
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

Create a `.streamlit/secrets.toml` file:

```toml
APP_PASSWORD="Add_Password"
google_sheet_url="link_to_csv_list"
```

---

## Run the App

```bash
streamlit run app.py
```

---

## Project Structure

```
.
├── app.py
├── logo/
│   ├── icon_hi_search.png
│   └── logo_hi_paris.png         
├── requirements.txt
└── README.md
```

---


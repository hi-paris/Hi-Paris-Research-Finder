# Hi! Paris Expert Finder

The Hi! Paris Expert Finder platform is a smart research engine designed to search for researchers affiliated with Hi! Paris who work in a specific field and closely related areas.



## Installation

## 1. Install uv (if not already installed):

```bash
pip install uv
```

## 2. Clone the repository:

```bash
git clone https://github.com/hi-paris/Hi-Paris-Research-Finder.git
cd Hi-Paris-Research-Finder
```
## 3. Create and activate the virtual environment: 

- Create a uv virtual environment

```bash
uv venv
```

- Activate it:

```bash
.\.venv\Scripts\activate
```

## 4. Install dependencies from pyproject.toml: 

```bash
uv sync
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
python -m streamlit run app.py
```

---

## Project Structure

```
.
├── app.py             
├── search.py         
├── utils.py            
├── pyproject.toml     
├── logo/
│   ├── icon_hi_search.png
│   └── logo_hi_paris.png
└── README.md
```
---


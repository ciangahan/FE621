Getting Started: setting up virtual env and locally installing FE621 python package

```{bash}
git clone git@github.com:ciangahan/FE621.git
cd FE621

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

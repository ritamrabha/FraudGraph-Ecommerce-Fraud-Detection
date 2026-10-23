# Dataset Information

The benchmark datasets used in this project are large graph datasets:
- **Amazon.mat** (~212 MB): Multi-relational e-commerce user graph (11,944 users, 25 behavioral features, 3 relations).
- **YelpChi.mat** (~198 MB): Multi-relational hotel and restaurant review graph (45,954 reviews, 32 features, 3 relations).

### Automatic Download
You do **not** need to manually download these files. When you run:
```bash
python run_experiments.py --dataset Amazon
```
or run `streamlit run app.py`, the loader in [`src/data.py`](../src/data.py) will automatically download and extract the dataset from DGL's official repository:
- `Amazon`: https://data.dgl.ai/dataset/FraudAmazon.zip
- `YelpChi`: https://data.dgl.ai/dataset/FraudYelp.zip

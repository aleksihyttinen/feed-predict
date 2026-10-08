# feed-predict

Small web app (in Finnish) that predicts the next baby feeding time from historical feeding data.  

Running in [predict.leksi.dev](https://predict.leksi.dev)

Built with:

- Python
- scikit-learn
- FastAPI  

The model uses **time of day** and **day of week** as features. It was evaluated at roughly **50 minutes MAE** on future data, then retrained on all available data for production.

## Run locally

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8000
```

## Deploy

The production setup runs FastAPI behind an existing Caddy instance:

```text
predict.leksi.dev
        ↓
Caddy
        ↓
feed-predict:8000
```

The app joins the shared Docker `web` network so it can be served by the existing Caddy container.

## Retrain

After updating `data.csv`:

```bash
python misc/save.py
```

Then rebuild the container:

```bash
docker compose up -d --build
```

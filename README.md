# GraphCRM Intelligence

> **A Graph Neural Network-Inspired CRM Platform for Lead Scoring and Marketing Intelligence**
>
> Final Year Project (FYP) — [Your University] · [Year]
> Author: [Your Name] | Supervisor: [Supervisor Name]

---

## Abstract

GraphCRM is a full-stack Customer Relationship Management platform that uses graph-based machine learning to score and prioritise sales leads. Unlike conventional CRM systems that evaluate leads in isolation, GraphCRM models the entire sales ecosystem—leads, companies, campaigns, and events—as a heterogeneous graph. A 3-layer Graph Neural Network (GNN) propagates latent interest signals through peer connections, referral chains, and campaign interactions to produce a conversion probability score for every lead.

The system includes a real-time GNN training simulation with live loss/accuracy curves, a network visualisation of the lead graph, AI-powered marketing campaign recommendations (via Google Gemini), and a full CRUD CRM backend backed by SQLite. All data is stored locally; no cloud services are required.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GraphCRM System                          │
├────────────────────────┬────────────────────────────────────────┤
│     Frontend (React)   │           Backend (Node.js)            │
│                        │                                        │
│  ┌──────────────────┐  │  ┌──────────────────────────────────┐  │
│  │  Dashboard       │  │  │  Express REST API (:3001)        │  │
│  │  Analytics       │  │  │  /api/leads  /api/accounts       │  │
│  │  Lead Network    │◄─┼──│  /api/deals  /api/activities     │  │
│  │  GNN Lead Scoring│  │  │  /api/graph  /api/demo-leads     │  │
│  │  Accounts        │  │  └──────────────┬───────────────────┘  │
│  └──────────────────┘  │                 │                      │
│                        │  ┌──────────────▼───────────────────┐  │
│  ┌──────────────────┐  │  │  SQLite (better-sqlite3)         │  │
│  │ CRM Simulator    │  │  │  leads / accounts / deals        │  │
│  │ (TypeScript)     │  │  │  activities / tasks / notes      │  │
│  │  3-Layer GNN     │  │  │  lead_relationships / automation │  │
│  │  message passing │  │  └──────────────────────────────────┘  │
│  └──────────────────┘  │                                        │
│                        │  ┌──────────────────────────────────┐  │
│  ┌──────────────────┐  │  │  WebSocket (:3001/ws)            │  │
│  │ Gemini AI        │  │  │  Real-time push for lead updates │  │
│  │ Campaign Tips    │  │  └──────────────────────────────────┘  │
│  └──────────────────┘  │                                        │
└────────────────────────┴────────────────────────────────────────┘
```

---

## How the GNN Works

The GNN simulation lives entirely in `src/lib/crm-simulation.ts` and `src/pages/GNNSimulationPage.tsx`. It implements **3-layer message passing** on a synthetic CRM graph:

### Graph Construction

| Node type  | Count  | Features                                           |
|------------|--------|----------------------------------------------------|
| Lead       | ~250   | engagement score, seniority, department            |
| Company    | ~30    | industry, size                                     |
| Campaign   | ~20    | target industry, channel                           |
| Event      | ~8     | type, attendance                                   |

| Edge type          | Description                           | Direction     |
|--------------------|---------------------------------------|---------------|
| `works_at`         | Lead → Company                        | Directed      |
| `colleague_of`     | Lead ↔ Lead (peer)                    | Bidirectional |
| `referred`         | Lead → Lead (referral chain)          | Directed      |
| `interacted_with`  | Lead → Campaign                       | Directed      |
| `attended`         | Lead → Event                          | Directed      |

### Message Passing Layers

```
Layer 0  (Input)
  h⁰ = engagementScore × seniorityMultiplier × 0.6
  seniorityMult: Junior=0.6, Senior=0.8, Director=1.0, VP=1.2, C-Level=1.4

Layer 1  (Colleague mean pooling)
  h¹ = 0.55 × h⁰ + 0.45 × mean( h⁰[n] for n in colleague_of )
  → Each lead absorbs the average latent interest of their peers.

Layer 2  (Referral weighted pooling)
  h² = h¹ + 0.30 × Σ( weight[e] × h¹[referred_by] ) / Σ( weight[e] )
  → Referrals carry an explicit trust weight (default 0.7).

Layer 3  (Campaign alignment boost)
  boost = +0.04 per matching campaign, capped at +0.12
  h³ = h² + boost  (where campaign.targetIndustry == lead.company.industry)

Output
  gnnScore = sigmoid( (h³ − 0.45) × 5 )  ∈ [0, 1]
  converted = gnnScore ≥ 0.55
```

### Training Simulation

The training loop (30 epochs, 80/20 train/val split) runs the 3-layer propagation on the training set each epoch with annealing Gaussian noise, computes binary cross-entropy loss, and evaluates accuracy on the held-out validation set. This produces realistic loss-decreasing / accuracy-increasing curves rather than a fake linear ramp.

---

## Setup & Installation

### Prerequisites
- Node.js 18+
- npm 9+
- Python 3.9+ with `pandas` and `torch` (optional, for `to_gnn_format.py`)

### 1. Clone and install

```bash
git clone <repo-url>
cd graphcrm-local
npm install
```

### 2. Configure environment

```bash
cp .env.example .env
```

| Variable              | Required | Description                                  |
|-----------------------|----------|----------------------------------------------|
| `VITE_GEMINI_API_KEY` | Optional | Google Gemini API key for live AI tips       |
| `VITE_USERNAME`       | Optional | Login username (default: `admin`)            |
| `VITE_PASSWORD`       | Optional | Login password (default: `crm2024`)          |
| `SMTP_HOST`           | Optional | SMTP server for email automation             |
| `SMTP_PORT`           | Optional | SMTP port (default 587)                      |
| `SMTP_USER`           | Optional | SMTP username                                |
| `SMTP_PASS`           | Optional | SMTP password                                |

Without `VITE_GEMINI_API_KEY`, AI campaign tips run in **Demo Mode** with realistic mock data — the GNN simulation and all CRM features still work fully.

### 3. Run

```bash
npm run dev
```

This starts both the Vite frontend (`:5173`) and the Express backend (`:3001`) concurrently.

Open [http://localhost:5173](http://localhost:5173) and sign in with `admin` / `crm2024`.

### 4. Seed test data (optional)

Once the app is running, click **"Seed Demo Data"** on the Dashboard, or:

```bash
curl -X POST http://localhost:3001/api/seed
```

### 5. GNN data export (optional)

```bash
python generate_crm_ecosystem.py
python to_gnn_format.py
```

Prints a summary table of node and edge counts by type, and produces PyTorch Geometric-ready tensors.

---

## Project Structure

```
graphcrm-local/
├── server/
│   ├── index.js            # Express entry point
│   ├── routes.js           # All REST API routes
│   ├── db.js               # SQLite setup & schema
│   ├── automation.js       # Lead scoring & automation rules
│   ├── mailer.js           # SMTP email sender
│   └── websocket.js        # WebSocket broadcast
├── src/
│   ├── lib/
│   │   └── crm-simulation.ts     # GNN simulator (graph gen + message passing)
│   ├── services/
│   │   └── gemini.ts             # Gemini AI campaign tips
│   ├── pages/
│   │   ├── GNNSimulationPage.tsx # GNN Lead Scoring UI
│   │   ├── NetworkPage.tsx       # Live lead graph
│   │   ├── AnalyticsPage.tsx     # Charts & insights
│   │   └── AccountsPageRoute.tsx # Account management
│   └── components/
│       ├── Dashboard.tsx
│       ├── Layout.tsx
│       └── ...                   # Leads, pipeline, automation, etc.
├── generate_crm_ecosystem.py   # Synthetic data generator (Python)
├── to_gnn_format.py            # PyG-ready data export with step-by-step comments
├── metadata.json               # FYP metadata
└── PLAN.md                     # Phase-by-phase build plan
```

---

## Feature Overview

| Feature                  | Description                                                          |
|--------------------------|----------------------------------------------------------------------|
| GNN Lead Scoring         | 3-layer message passing on synthetic CRM graph, 30-epoch training    |
| Live Training Curves     | Real-time loss/accuracy chart, epoch log, confusion matrix           |
| AI Campaign Tips         | Gemini-powered per-lead outreach recommendations                     |
| Lead Network Graph       | Force-directed visualisation of live DB relationships                |
| Demo Mode                | Push GNN leads into real CRM for demo; clear them after             |
| Automation Rules         | Trigger tasks/emails on lead score thresholds or activity types      |
| Pipeline View            | Kanban-style deal stage tracker                                      |
| Analytics Dashboard      | Conversion funnels, score distributions, activity timelines          |

---

## Screenshots

> _Add screenshots here before submitting the FYP report._

| Page                | Screenshot                       |
|---------------------|----------------------------------|
| GNN Lead Scoring    | `screenshots/gnn-scoring.png`    |
| Lead Network Graph  | `screenshots/network.png`        |
| Dashboard           | `screenshots/dashboard.png`      |
| AI Campaign Tips    | `screenshots/campaign-tips.png`  |

---

## Tech Stack

| Layer        | Technology                                              |
|--------------|---------------------------------------------------------|
| Frontend     | React 18, TypeScript, Vite, Tailwind CSS, Recharts      |
| Backend      | Node.js, Express, better-sqlite3, WebSocket             |
| AI           | Google Gemini 2.5 Flash                                 |
| Graph Viz    | Custom force-directed SVG renderer                      |
| GNN (sim)    | TypeScript (3-layer message passing, no ML framework)   |
| GNN (export) | PyTorch + PyTorch Geometric (via `to_gnn_format.py`)    |

---

## Troubleshooting

**"Port 3001 already in use"** → Change in `.env`: `API_PORT=3002`

**"Cannot find module 'better-sqlite3'"** → Run `npm install` again; the native module needs to compile.

**npm install fails on Windows** → Install Windows Build Tools first:
```bash
npm install --global windows-build-tools
```

**App loads but shows no data** → Make sure both server processes started. Click "Seed Demo Data".

---

## License

MIT — see `LICENSE` for details.


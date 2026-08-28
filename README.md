# Abuse-Ring Sentinel

**Track:** AI Risk Manager - Razorpay AI Buildathon

Detects coordinated multi-account abuse (collusion rings) that individually-looking
transactions and single-account fraud models miss.

## Status
In progress - build log below.

## Problem
Merchants lose money not just to lone bad actors, but to coordinated abuse:
multiple accounts working together, each looking individually clean, but
collectively draining money through referral fraud, return abuse, or promo abuse.

## Approach
1. Model accounts and their shared attributes (device, IP, card, address) as a graph.
2. Extract graph-based features (cluster size, shared-attribute density, timing).
3. Train a classifier to flag ring membership using graph + transaction features.
4. Report precision/recall/F1 on a held-out set of rings, plus honest false-positive cost.
5. Every flag comes with a human-readable explanation and audit log - strictly
   defense-only, no automated account actions.

## Setup
pip install -r requirements.txt

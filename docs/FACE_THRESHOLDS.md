# Face Recognition Thresholds Documentation

> **Last Updated**: January 18, 2026  
> **Purpose**: Document all face recognition threshold changes for quick reference

---

## 📍 File Locations Summary

| File | Purpose |
|------|---------|
| `config/default.yaml` | YAML configuration (loaded at runtime) |
| `core/config.py` | Python dataclass defaults |
| `identity/binding.py` | Binding state machine logic (hardcoded defaults) |

---

## 🔧 Changes Made

### 1. `config/default.yaml` (Lines 178-186)

**Location**: `governance.binding.confirmation` section

```yaml
# BEFORE:
confirmation:
  min_samples_strong: 3               # 3 strong samples (dist < 0.75) to confirm
  min_samples_weak: 5                 # 5 weak samples (dist < 0.88) to confirm
  window_seconds: 3.0                 # All samples within 3 seconds
  min_avg_score: 0.75                 # Average distance must be < 0.75

# AFTER:
confirmation:
  min_samples_strong: 1               # 1 strong sample to confirm (instant recognition)
  min_samples_weak: 3                 # 3 weak samples to confirm
  window_seconds: 3.0                 # All samples within 3 seconds
  min_avg_score: 0.45                 # Score must be >= 0.45 (relaxed for quick recognition)
  min_avg_margin: 0.02                # Margin must be >= 0.02 (very relaxed)
  min_quality_for_strong: 0.40        # Quality threshold for strong sample
```

---

### 2. `core/config.py` (Lines 105-108)

**Location**: `BindingConfirmationRules` dataclass

```python
# BEFORE:
@dataclass
class BindingConfirmationRules:
    """Rules for confirming identity (Phase C)"""
    min_samples_strong: int = 3

# AFTER:
@dataclass
class BindingConfirmationRules:
    """Rules for confirming identity (Phase C)"""
    min_samples_strong: int = 1
```

---

### 3. `identity/binding.py` (Lines 380-395)

**Location**: `_handle_unknown_state()` method - Strong Sample Tier Logic

```python
# BEFORE (Lines 380-389):
# Tier 1: Strong absolute match
if (e.score >= self._get_threshold('confirmation.min_avg_score', 0.60) and 
    e.margin >= self._get_threshold('confirmation.min_avg_margin', 0.05)):
    is_connected = True
    
# Tier 2: Weaker match but highly distinct (High Margin)
elif (e.score >= 0.50 and e.margin >= 0.20):
    is_connected = True

# AFTER:
# Tier 1: Strong absolute match (RELAXED for faster recognition)
if (e.score >= self._get_threshold('confirmation.min_avg_score', 0.45) and 
    e.margin >= self._get_threshold('confirmation.min_avg_margin', 0.02)):
    is_connected = True
    
# Tier 2: Weaker match but highly distinct (High Margin)
elif (e.score >= 0.45 and e.margin >= 0.10):
    is_connected = True
```

---

### 4. `identity/binding.py` (Line 392)

**Location**: Quality gate for strong samples

```python
# BEFORE:
if is_connected and e.quality >= self._get_threshold('confirmation.min_quality_for_strong', 0.50):

# AFTER:
if is_connected and e.quality >= self._get_threshold('confirmation.min_quality_for_strong', 0.40):
```

---

### 5. `identity/binding.py` (Line 395)

**Location**: min_samples default value

```python
# BEFORE:
min_samples = self._get_threshold('confirmation.min_samples_strong', 3)

# AFTER:
min_samples = self._get_threshold('confirmation.min_samples_strong', 1)
```

---

## 📊 Complete Threshold Reference Table

| Threshold | Before | After | Effect |
|-----------|--------|-------|--------|
| `min_samples_strong` | 3 | **1** | Instant recognition (1 sample) |
| `min_samples_weak` | 5 | **3** | Faster weak confirmation |
| `min_avg_score` (Tier 1) | 0.60 | **0.45** | Accept 45%+ matches as strong |
| `min_avg_margin` (Tier 1) | 0.05 | **0.02** | Very low margin required |
| `min_score` (Tier 2) | 0.50 | **0.45** | Accept 45%+ matches |
| `min_margin` (Tier 2) | 0.20 | **0.10** | Lower margin for Tier 2 |
| `min_quality_for_strong` | 0.50 | **0.40** | Accept lower quality faces |

---

## 🔍 How the Recognition Pipeline Works

```
Frame → Face Detection → Quality Check → Embedding → Gallery Match → Score + Margin
                                                                          ↓
                                                              Strong Sample Check
                                                                          ↓
                                                    ┌─────────────────────────────────┐
                                                    │ Tier 1: Score ≥ 0.45            │
                                                    │         AND Margin ≥ 0.02       │
                                                    ├─────────────────────────────────┤
                                                    │ Tier 2: Score ≥ 0.45            │
                                                    │         AND Margin ≥ 0.10       │
                                                    └─────────────────────────────────┘
                                                                          ↓
                                                              Quality ≥ 0.40?
                                                                          ↓
                                                              Count as STRONG
                                                                          ↓
                                                    strong_samples ≥ 1? → TRANSITION!
```

---

## 🎨 State Machine States

| State | Color (RGB) | Meaning |
|-------|-------------|---------|
| UNKNOWN | Yellow (255, 200, 0) | Scanning, no recognition yet |
| PENDING | Light Green (144, 238, 144) | Recognition started, gathering evidence |
| LOCKED | Green (0, 255, 0) | Confirmed identity |
| STALE | Gray (128, 128, 128) | No recent evidence (8+ seconds) |

---

## 🚀 Quick Tuning Guide

### To make recognition FASTER (less strict):
```yaml
# In config/default.yaml
confirmation:
  min_samples_strong: 1      # Lower = faster
  min_avg_score: 0.40        # Lower = accepts weaker matches
  min_avg_margin: 0.01       # Lower = accepts less distinct matches
  min_quality_for_strong: 0.30  # Lower = accepts blurry faces
```

### To make recognition MORE STRICT (fewer false positives):
```yaml
# In config/default.yaml
confirmation:
  min_samples_strong: 3      # Higher = requires more evidence
  min_avg_score: 0.60        # Higher = requires stronger matches
  min_avg_margin: 0.10       # Higher = requires more distinct matches
  min_quality_for_strong: 0.60  # Higher = requires clearer faces
```

---

## ⚠️ Important Notes

1. **YAML vs Hardcoded**: The `binding.py` file has hardcoded defaults in `self._get_threshold()` calls. 
   If YAML doesn't have a value, the hardcoded default is used.

2. **Multiple Locations**: `min_samples_strong` appears in:
   - `config/default.yaml` (line ~179)
   - `core/config.py` (line ~107)
   - `identity/binding.py` (line ~395, 469, 566, 606, 945, 970)
   
3. **Score vs Distance**: 
   - Score = 1 - distance (higher is better)
   - 50% score = 0.50 distance to gallery embedding

4. **Margin**: The gap between best match and second-best match.
   - High margin (0.20+) = Very distinct, clearly this person
   - Low margin (0.02) = Similar to other people in gallery

---

## 📁 Files Cleaned Up

The following duplicate/old files were **deleted** from `chimeric_identity/`:
- `fusion_engine.py` (48KB) - Moved to `_deprecated/`
- `logging_utils.py` (32KB) - Moved to `_deprecated/`
- `runner_standalone.py` (41KB) - Moved to `_deprecated/`

These were causing Pylance import errors because they referenced non-existent modules.
